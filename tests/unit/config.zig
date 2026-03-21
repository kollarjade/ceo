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

        const stat = try file.stat();
        const max_size: usize = 16 * 1024 * 1024;
        if (stat.size > max_size) return error.ConfigTooLarge;

        const content = try file.readToEndAlloc(allocator, max_size);
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
        self.runtime.deinit(allocator);
        self.data.deinit(allocator);
        self.checkpoint.deinit(allocator);
        self.telemetry.deinit(allocator);
    }

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

pub const ModelConfig = struct {
    name: []const u8,
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    max_seq_len: usize,
    efla: EflaConfig,
    prism: PrismConfig,
    norm_type: NormType,
    activation: ActivationType,
    tie_embeddings: bool,
    dropout: f32,
    dtype: @import("../tensor/dtype.zig").DType,
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

    pub fn countParameters(self: *Self) !u64 {
        var total: u64 = 0;
        total += @as(u64, self.vocab_size) * self.hidden_dim;
        total += self.countLayerParameters() * self.num_layers;
        if (!self.tie_embeddings) {
            total += @as(u64, self.vocab_size) * self.hidden_dim;
        }
        total += self.hidden_dim;
        return total;
    }

    fn countLayerParameters(self: *Self) u64 {
        var params: u64 = 0;
        params += @as(u64, self.hidden_dim) * (self.hidden_dim + 2 * self.num_kv_heads * self.head_dim);
        params += @as(u64, self.hidden_dim) * self.hidden_dim;
        for (0..self.prism.num_iterations) |_| {
            params += 3 * @as(u64, self.hidden_dim) * self.head_dim;
        }
        params += @as(u64, self.hidden_dim) * self.intermediate_dim * 2;
        params += self.hidden_dim * 2;
        return params;
    }

    pub fn estimateMemory(self: *Self) !u64 {
        const param_bytes = (try self.countParameters()) * 2;
        const gradient_bytes = param_bytes;
        const optimizer_bytes = param_bytes * 2;
        const seq_len = @min(self.max_seq_len, 100000);
        const activation_bytes = @as(u64, seq_len) * self.hidden_dim * self.num_layers * 4;
        return (param_bytes + gradient_bytes + optimizer_bytes + activation_bytes) / 8;
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

pub const EflaConfig = struct {
    enabled: bool,
    num_heads: usize,
    state_dim: usize,
    chunk_size: usize,
    learned_beta: bool,
    initial_beta: f32,
    beta_schedule: BetaSchedule,
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

pub const PrismConfig = struct {
    enabled: bool,
    num_iterations: usize,
    shortconv_window: usize,
    use_proxy: bool,
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

pub const TrainingConfig = struct {
    total_steps: usize,
    warmup_steps: usize,
    micro_batch_size: usize,
    global_batch_size: usize,
    gradient_accumulation_steps: usize,
    learning_rate: f32,
    min_learning_rate: f32,
    weight_decay: f32,
    gradient_clip: f32,
    label_smoothing: f32,
    optimizer: OptimizerType,
    lion_beta1: f32,
    lion_beta2: f32,
    muon_momentum: f32,
    muon_iterations: usize,
    lr_schedule: LRSchedule,
    mixed_precision: bool,
    fp8_training: bool,
    loss_scale: f32,
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

pub const RuntimeConfig = struct {
    world_size: usize,
    rank: usize,
    tensor_parallel_size: usize,
    pipeline_parallel_size: usize,
    zero_stage: u8,
    cpu_offload: bool,
    nvme_offload: bool,
    nvme_path: []const u8,
    seed: u64,
    deterministic: bool,
    owns_nvme_path: bool,

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
            .owns_nvme_path = true,
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
            .owns_nvme_path = false,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.owns_nvme_path) allocator.free(self.nvme_path);
    }
};

pub const DataConfig = struct {
    path: []const u8,
    tokenizer_path: []const u8,
    seq_len: usize,
    shuffle_buffer_size: usize,
    prefetch_factor: usize,
    num_workers: usize,
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

pub const CheckpointConfig = struct {
    dir: []const u8,
    save_interval: usize,
    keep_last_n: usize,
    save_optimizer: bool,
    compression: bool,
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

pub const TelemetryConfig = struct {
    log_level: LogLevel,
    log_file: []const u8,
    metrics_interval: usize,
    enable_prometheus: bool,
    prometheus_port: u16,
    track_memory: bool,
    track_throughput: bool,
    owns_log_file: bool,

    const Self = @This();

    pub fn parse(allocator: std.mem.Allocator, map: *yaml.YamlMap) !Self {
        return .{
            .log_level = if (map.getString("log_level")) |s| std.meta.stringToEnum(LogLevel, s) orelse .info else .info,
            .log_file = try allocator.dupe(u8, map.getString("log_file") orelse "training.log"),
            .metrics_interval = map.getInt(usize, "metrics_interval") orelse 10,
            .enable_prometheus = map.getBool("enable_prometheus") orelse false,
            .prometheus_port = @intCast(map.getInt(usize, "prometheus_port") orelse 9090),
            .track_memory = map.getBool("track_memory") orelse true,
            .track_throughput = map.getBool("track_throughput") orelse true,
            .owns_log_file = true,
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
            .owns_log_file = false,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.owns_log_file) allocator.free(self.log_file);
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
