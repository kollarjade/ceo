const std = @import("std");
const yaml = @import("yaml.zig");

pub const Config = struct {
    model: ModelConfig,
    training: TrainingConfig,
    runtime: RuntimeConfig,
    data: DataConfig,
    checkpoint: CheckpointConfig,
    telemetry: TelemetryConfig,

    const Self = @This();

    pub fn parseFromFile(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
        defer allocator.free(content);

        return parseFromYaml(allocator, content);
    }

    pub fn parseFromYaml(allocator: std.mem.Allocator, content: []const u8) !Self {
        var parser = yaml.YamlParser.init(allocator, content);
        defer parser.deinit();

        var root = try parser.parse();
        defer root.deinit();

        return .{
            .model = try ModelConfig.parse(allocator, root.getMap("model") orelse return error.MissingModelConfig),
            .training = try TrainingConfig.parse(allocator, root.getMap("training") orelse return error.MissingTrainingConfig),
            .runtime = try RuntimeConfig.parse(allocator, root.getMap("runtime") orelse return error.MissingRuntimeConfig),
            .data = try DataConfig.parse(allocator, root.getMap("data") orelse return error.MissingDataConfig),
            .checkpoint = try CheckpointConfig.parse(allocator, root.getMap("checkpoint") orelse return error.MissingCheckpointConfig),
            .telemetry = try TelemetryConfig.parse(allocator, root.getMap("telemetry") orelse return error.MissingTelemetryConfig),
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.training.deinit(allocator);
        self.data.deinit(allocator);
        self.checkpoint.deinit(allocator);
    }

    /// Create default 1T parameter model config
    pub fn default1T(allocator: std.mem.Allocator) Self {
        return .{
            .model = ModelConfig.default1T(allocator),
            .training = TrainingConfig.default(),
            .runtime = RuntimeConfig.default8GPU(),
            .data = DataConfig.default(allocator),
            .checkpoint = CheckpointConfig.default(allocator),
            .telemetry = TelemetryConfig.default(),
        };
    }
};

/// Model architecture configuration
pub const ModelConfig = struct {
    /// Model name/identifier
    name: []const u8,
    /// Vocabulary size
    vocab_size: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of layers
    num_layers: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Number of key-value heads (for GQA)
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Intermediate dimension for MLP
    intermediate_dim: usize,
    /// Maximum sequence length (target)
    max_seq_len: usize,
    /// EFLA configuration
    efla: EflaConfig,
    /// PRISM configuration
    prism: PrismConfig,
    /// Normalization type
    norm_type: NormType,
    /// Activation function
    activation: ActivationType,
    /// Use tied embeddings
    tie_embeddings: bool,
    /// Dropout rate
    dropout: f32,
    /// Precision
    dtype: @import("../tensor/dtype.zig").DType,
    /// Parameter count target for validation
    target_params: u64,

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        return .{
            .name = try allocator.dupe(u8, map.getString("name") orelse "efla-1t"),
            .vocab_size = map.getInt(usize, "vocab_size") orelse 65536,
            .hidden_dim = map.getInt(usize, "hidden_dim") orelse 16384,
            .num_layers = map.getInt(usize, "num_layers") orelse 80,
            .num_heads = map.getInt(usize, "num_heads") orelse 128,
            .num_kv_heads = map.getInt(usize, "num_kv_heads") orelse 16,
            .head_dim = map.getInt(usize, "head_dim") orelse 128,
            .intermediate_dim = map.getInt(usize, "intermediate_dim") orelse 65536,
            .max_seq_len = map.getInt(usize, "max_seq_len") orelse 50000000,
            .efla = if (map.getMap("efla")) |m| try EflaConfig.parse(allocator, m) else EflaConfig.default(),
            .prism = if (map.getMap("prism")) |m| try PrismConfig.parse(allocator, m) else PrismConfig.default(),
            .norm_type = if (map.getString("norm_type")) |s| std.meta.stringToEnum(NormType, s) orelse .rmsnorm else .rmsnorm,
            .activation = if (map.getString("activation")) |s| std.meta.stringToEnum(ActivationType, s) orelse .gelu else .gelu,
            .tie_embeddings = map.getBool("tie_embeddings") orelse false,
            .dropout = map.getFloat(f32, "dropout") orelse 0.0,
            .dtype = if (map.getString("dtype")) |s| parseDtype(s) else .bf16,
            .target_params = map.getInt(u64, "target_params") orelse 1_000_000_000_000,
            .allocator = allocator,
        };
    }

    pub fn default1T(allocator: std.mem.Allocator) Self {
        return .{
            .name = allocator.dupe(u8, "efla-1t") catch unreachable,
            .vocab_size = 131072,
            .hidden_dim = 16384,
            .num_layers = 80,
            .num_heads = 128,
            .num_kv_heads = 16,
            .head_dim = 128,
            .intermediate_dim = 65536,
            .max_seq_len = 50_000_000,
            .efla = EflaConfig.default(),
            .prism = PrismConfig.default(),
            .norm_type = .rmsnorm,
            .activation = .gelu,
            .tie_embeddings = false,
            .dropout = 0.0,
            .dtype = .bf16,
            .target_params = 1_000_000_000_000,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.efla.deinit(allocator);
        self.prism.deinit(allocator);
    }

    /// Validate configuration
    pub fn validate(self: *Self) !void {
        if (self.hidden_dim == 0) return error.InvalidHiddenDim;
        if (self.num_layers == 0) return error.InvalidNumLayers;
        if (self.num_heads == 0) return error.InvalidNumHeads;
        if (self.head_dim == 0) return error.InvalidHeadDim;
        if (self.hidden_dim % self.num_heads != 0) return error.HiddenDimNotDivisibleByHeads;

        const param_count = try self.countParameters();
        const tolerance: f64 = 0.05;
        const ratio = @as(f64, @floatFromInt(param_count)) / @as(f64, @floatFromInt(self.target_params));

        if (ratio < (1.0 - tolerance)) {
            std.log.warn("Parameter count {d} is below target {d} by more than 5%", .{ param_count, self.target_params });
        } else if (ratio > (1.0 + tolerance)) {
            std.log.warn("Parameter count {d} exceeds target {d} by more than 5%", .{ param_count, self.target_params });
        }
    }

    /// Count total parameters
    pub fn countParameters(self: *Self) !u64 {
        var total: u64 = 0;

        // Embedding layer
        total += @as(u64, self.vocab_size) * self.hidden_dim;

        // Per layer
        const layer_params = self.countLayerParameters();
        total += layer_params * self.num_layers;

        // Output layer (if not tied)
        if (!self.tie_embeddings) {
            total += @as(u64, self.vocab_size) * self.hidden_dim;
        }

        // Final norm
        total += self.hidden_dim;

        return total;
    }

    fn countLayerParameters(self: *Self) u64 {
        var params: u64 = 0;

        // EFLA components
        // QKV projection
        params += @as(u64, self.hidden_dim) * (self.hidden_dim + 2 * self.num_kv_heads * self.head_dim);
        // Output projection
        params += @as(u64, self.hidden_dim) * self.hidden_dim;

        // PRISM components
        for (0..self.prism.num_iterations) |_| {
            // W_beta, W_k, W_p projections
            params += 3 * @as(u64, self.hidden_dim) * self.head_dim;
        }

        // MLP
        params += @as(u64, self.hidden_dim) * self.intermediate_dim * 2;

        // Norms
        params += self.hidden_dim * 2;

        return params;
    }

    /// Estimate memory requirements per GPU (bytes)
    pub fn estimateMemory(self: *Self) !u64 {
        const param_bytes = (try self.countParameters()) * 2; // BF16
        const gradient_bytes = param_bytes;
        const optimizer_bytes = param_bytes * 2; // Adam state

        // Activation memory (rough estimate for 50M context)
        const seq_len = @min(self.max_seq_len, 100000); // Cap estimate
        const activation_bytes = @as(u64, seq_len) * self.hidden_dim * self.num_layers * 4;

        return (param_bytes + gradient_bytes + optimizer_bytes + activation_bytes) / 8; // Divided by 8 GPUs
    }
};

fn parseDtype(s: []const u8) @import("../tensor/dtype.zig").DType {
    if (std.mem.eql(u8, s, "fp32")) return .fp32;
    if (std.mem.eql(u8, s, "fp16")) return .fp16;
    if (std.mem.eql(u8, s, "bf16")) return .bf16;
    if (std.mem.eql(u8, s, "fp8")) return .fp8_e4m3;
    return .bf16;
}

pub const NormType = enum {
    layernorm,
    rmsnorm,
};

pub const ActivationType = enum {
    relu,
    gelu,
    silu,
    swiglu,
};

/// EFLA configuration
pub const EflaConfig = struct {
    /// Enable EFLA
    enabled: bool,
    /// Number of EFLA heads
    num_heads: usize,
    /// State dimension
    state_dim: usize,
    /// Chunk size for parallel processing
    chunk_size: usize,
    /// Use learned beta
    learned_beta: bool,
    /// Initial beta value
    initial_beta: f32,
    /// Beta schedule
    beta_schedule: BetaSchedule,
    /// Use chunked scan
    use_chunked_scan: bool,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        _ = allocator;
        return .{
            .enabled = map.getBool("enabled") orelse true,
            .num_heads = map.getInt(usize, "num_heads") orelse 128,
            .state_dim = map.getInt(usize, "state_dim") orelse 128,
            .chunk_size = map.getInt(usize, "chunk_size") orelse 4096,
            .learned_beta = map.getBool("learned_beta") orelse true,
            .initial_beta = map.getFloat(f32, "initial_beta") orelse 1.0,
            .beta_schedule = if (map.getString("beta_schedule")) |s| std.meta.stringToEnum(BetaSchedule, s) orelse .constant else .constant,
            .use_chunked_scan = map.getBool("use_chunked_scan") orelse true,
        };
    }

    pub fn default() Self {
        return .{
            .enabled = true,
            .num_heads = 128,
            .state_dim = 128,
            .chunk_size = 4096,
            .learned_beta = true,
            .initial_beta = 1.0,
            .beta_schedule = .constant,
            .use_chunked_scan = true,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

pub const BetaSchedule = enum {
    constant,
    linear_warmup,
    cosine_decay,
};

/// PRISM configuration
pub const PrismConfig = struct {
    /// Enable PRISM
    enabled: bool,
    /// Number of iterations
    num_iterations: usize,
    /// ShortConv window size
    shortconv_window: usize,
    /// Use input-anchored proxy
    use_proxy: bool,
    /// Forget factor
    forget_factor: f32,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        _ = allocator;
        return .{
            .enabled = map.getBool("enabled") orelse true,
            .num_iterations = map.getInt(usize, "num_iterations") orelse 3,
            .shortconv_window = map.getInt(usize, "shortconv_window") orelse 64,
            .use_proxy = map.getBool("use_proxy") orelse true,
            .forget_factor = map.getFloat(f32, "forget_factor") orelse 0.99,
        };
    }

    pub fn default() Self {
        return .{
            .enabled = true,
            .num_iterations = 3,
            .shortconv_window = 64,
            .use_proxy = true,
            .forget_factor = 0.99,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

/// Training configuration
pub const TrainingConfig = struct {
    /// Total training steps
    total_steps: usize,
    /// Warmup steps
    warmup_steps: usize,
    /// Micro-batch size
    micro_batch_size: usize,
    /// Global batch size
    global_batch_size: usize,
    /// Gradient accumulation steps
    gradient_accumulation_steps: usize,
    /// Learning rate
    learning_rate: f32,
    /// Min learning rate
    min_learning_rate: f32,
    /// Weight decay
    weight_decay: f32,
    /// Gradient clipping
    gradient_clip: f32,
    /// Label smoothing
    label_smoothing: f32,
    /// Optimizer
    optimizer: OptimizerType,
    /// Lion beta1
    lion_beta1: f32,
    /// Lion beta2
    lion_beta2: f32,
    /// Muon momentum
    muon_momentum: f32,
    /// Muon iteration count
    muon_iterations: usize,
    /// LR schedule
    lr_schedule: LRSchedule,
    /// Mixed precision
    mixed_precision: bool,
    /// FP8 training
    fp8_training: bool,
    /// Loss scaling
    loss_scale: f32,
    /// Dynamic loss scaling
    dynamic_loss_scale: bool,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        _ = allocator;
        return .{
            .total_steps = map.getInt(usize, "total_steps") orelse 1000000,
            .warmup_steps = map.getInt(usize, "warmup_steps") orelse 2000,
            .micro_batch_size = map.getInt(usize, "micro_batch_size") orelse 1,
            .global_batch_size = map.getInt(usize, "global_batch_size") orelse 512,
            .gradient_accumulation_steps = map.getInt(usize, "gradient_accumulation_steps") orelse 64,
            .learning_rate = map.getFloat(f32, "learning_rate") orelse 1e-4,
            .min_learning_rate = map.getFloat(f32, "min_learning_rate") orelse 1e-5,
            .weight_decay = map.getFloat(f32, "weight_decay") orelse 0.1,
            .gradient_clip = map.getFloat(f32, "gradient_clip") orelse 1.0,
            .label_smoothing = map.getFloat(f32, "label_smoothing") orelse 0.0,
            .optimizer = if (map.getString("optimizer")) |s| std.meta.stringToEnum(OptimizerType, s) orelse .lion_muon else .lion_muon,
            .lion_beta1 = map.getFloat(f32, "lion_beta1") orelse 0.95,
            .lion_beta2 = map.getFloat(f32, "lion_beta2") orelse 0.98,
            .muon_momentum = map.getFloat(f32, "muon_momentum") orelse 0.95,
            .muon_iterations = map.getInt(usize, "muon_iterations") orelse 5,
            .lr_schedule = if (map.getString("lr_schedule")) |s| std.meta.stringToEnum(LRSchedule, s) orelse .cosine else .cosine,
            .mixed_precision = map.getBool("mixed_precision") orelse true,
            .fp8_training = map.getBool("fp8_training") orelse true,
            .loss_scale = map.getFloat(f32, "loss_scale") orelse 65536.0,
            .dynamic_loss_scale = map.getBool("dynamic_loss_scale") orelse true,
        };
    }

    pub fn default() Self {
        return .{
            .total_steps = 1000000,
            .warmup_steps = 2000,
            .micro_batch_size = 1,
            .global_batch_size = 512,
            .gradient_accumulation_steps = 64,
            .learning_rate = 1e-4,
            .min_learning_rate = 1e-5,
            .weight_decay = 0.1,
            .gradient_clip = 1.0,
            .label_smoothing = 0.0,
            .optimizer = .lion_muon,
            .lion_beta1 = 0.95,
            .lion_beta2 = 0.98,
            .muon_momentum = 0.95,
            .muon_iterations = 5,
            .lr_schedule = .cosine,
            .mixed_precision = true,
            .fp8_training = true,
            .loss_scale = 65536.0,
            .dynamic_loss_scale = true,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn validate(self: *Self) !void {
        if (self.global_batch_size % self.micro_batch_size != 0) {
            return error.InvalidBatchSize;
        }
        if (self.gradient_accumulation_steps == 0) {
            return error.InvalidGradientAccumulation;
        }
    }
};

pub const OptimizerType = enum {
    lion,
    muon,
    lion_muon,
    adamw,
};

pub const LRSchedule = enum {
    constant,
    linear_warmup,
    cosine,
    linear_warmup_cosine,
};

/// Runtime configuration
pub const RuntimeConfig = struct {
    /// World size
    world_size: usize,
    /// Rank
    rank: usize,
    /// Tensor parallel size
    tensor_parallel_size: usize,
    /// Pipeline parallel size
    pipeline_parallel_size: usize,
    /// ZeRO stage
    zero_stage: u8,
    /// Enable CPU offload
    cpu_offload: bool,
    /// Enable NVMe offload
    nvme_offload: bool,
    /// NVMe path
    nvme_path: []const u8,
    /// Seed
    seed: u64,
    /// Deterministic
    deterministic: bool,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        return .{
            .world_size = map.getInt(usize, "world_size") orelse 8,
            .rank = map.getInt(usize, "rank") orelse 0,
            .tensor_parallel_size = map.getInt(usize, "tensor_parallel_size") orelse 8,
            .pipeline_parallel_size = map.getInt(usize, "pipeline_parallel_size") orelse 1,
            .zero_stage = @intCast(map.getInt(usize, "zero_stage") orelse 2),
            .cpu_offload = map.getBool("cpu_offload") orelse false,
            .nvme_offload = map.getBool("nvme_offload") orelse false,
            .nvme_path = try allocator.dupe(u8, map.getString("nvme_path") orelse "/tmp/offload"),
            .seed = map.getInt(u64, "seed") orelse 42,
            .deterministic = map.getBool("deterministic") orelse true,
        };
    }

    pub fn default8GPU() Self {
        return .{
            .world_size = 8,
            .rank = 0,
            .tensor_parallel_size = 8,
            .pipeline_parallel_size = 1,
            .zero_stage = 2,
            .cpu_offload = false,
            .nvme_offload = false,
            .nvme_path = "/tmp/offload",
            .seed = 42,
            .deterministic = true,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.nvme_path);
    }
};

/// Data configuration
pub const DataConfig = struct {
    /// Data path
    path: []const u8,
    /// Tokenizer path
    tokenizer_path: []const u8,
    /// Sequence length
    seq_len: usize,
    /// Shuffle buffer size
    shuffle_buffer_size: usize,
    /// Prefetch factor
    prefetch_factor: usize,
    /// Number of workers
    num_workers: usize,
    /// Pack sequences
    pack_sequences: bool,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        return .{
            .path = try allocator.dupe(u8, map.getString("path") orelse "data/train.bin"),
            .tokenizer_path = try allocator.dupe(u8, map.getString("tokenizer_path") orelse "tokenizer.bin"),
            .seq_len = map.getInt(usize, "seq_len") orelse 8192,
            .shuffle_buffer_size = map.getInt(usize, "shuffle_buffer_size") orelse 10000,
            .prefetch_factor = map.getInt(usize, "prefetch_factor") orelse 2,
            .num_workers = map.getInt(usize, "num_workers") orelse 4,
            .pack_sequences = map.getBool("pack_sequences") orelse true,
        };
    }

    pub fn default(allocator: std.mem.Allocator) Self {
        return .{
            .path = allocator.dupe(u8, "data/train.bin") catch unreachable,
            .tokenizer_path = allocator.dupe(u8, "tokenizer.bin") catch unreachable,
            .seq_len = 8192,
            .shuffle_buffer_size = 10000,
            .prefetch_factor = 2,
            .num_workers = 4,
            .pack_sequences = true,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.tokenizer_path);
    }
};

/// Checkpoint configuration
pub const CheckpointConfig = struct {
    /// Checkpoint directory
    dir: []const u8,
    /// Save interval (steps)
    save_interval: usize,
    /// Keep last N checkpoints
    keep_last_n: usize,
    /// Save optimizer state
    save_optimizer: bool,
    /// Compression
    compression: bool,
    /// Async save
    async_save: bool,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        return .{
            .dir = try allocator.dupe(u8, map.getString("dir") orelse "checkpoints"),
            .save_interval = map.getInt(usize, "save_interval") orelse 1000,
            .keep_last_n = map.getInt(usize, "keep_last_n") orelse 5,
            .save_optimizer = map.getBool("save_optimizer") orelse true,
            .compression = map.getBool("compression") orelse true,
            .async_save = map.getBool("async_save") orelse true,
        };
    }

    pub fn default(allocator: std.mem.Allocator) Self {
        return .{
            .dir = allocator.dupe(u8, "checkpoints") catch unreachable,
            .save_interval = 1000,
            .keep_last_n = 5,
            .save_optimizer = true,
            .compression = true,
            .async_save = true,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.dir);
    }
};

/// Telemetry configuration
pub const TelemetryConfig = struct {
    /// Log level
    log_level: LogLevel,
    /// Log file
    log_file: []const u8,
    /// Metrics interval (steps)
    metrics_interval: usize,
    /// Enable Prometheus
    enable_prometheus: bool,
    /// Prometheus port
    prometheus_port: u16,
    /// Track memory
    track_memory: bool,
    /// Track throughput
    track_throughput: bool,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        _ = allocator;
        return .{
            .log_level = if (map.getString("log_level")) |s| std.meta.stringToEnum(LogLevel, s) orelse .info else .info,
            .log_file = map.getString("log_file") orelse "training.log",
            .metrics_interval = map.getInt(usize, "metrics_interval") orelse 10,
            .enable_prometheus = map.getBool("enable_prometheus") orelse false,
            .prometheus_port = @intCast(map.getInt(usize, "prometheus_port") orelse 9090),
            .track_memory = map.getBool("track_memory") orelse true,
            .track_throughput = map.getBool("track_throughput") orelse true,
        };
    }

    pub fn default() Self {
        return .{
            .log_level = .info,
            .log_file = "training.log",
            .metrics_interval = 10,
            .enable_prometheus = false,
            .prometheus_port = 9090,
            .track_memory = true,
            .track_throughput = true,
        };
    }
};

pub const LogLevel = enum {
    debug,
    info,
    warn,
    err,
};

test "Config parse" {
    const yaml_content =
        \\model:
        \\  name: test-model
        \\  hidden_dim: 256
        \\  num_layers: 4
        \\training:
        \\  learning_rate: 0.001
        \\runtime:
        \\  world_size: 1
        \\data:
        \\  path: test.bin
        \\checkpoint:
        \\  dir: checkpoints
        \\telemetry:
        \\  log_level: info
    ;

    const allocator = std.testing.allocator;
    var cfg = try Config.parseFromYaml(allocator, yaml_content);
    defer cfg.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 256), cfg.model.hidden_dim);
    try std.testing.expectEqual(@as(usize, 4), cfg.model.num_layers);
}
