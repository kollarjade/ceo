const std = @import("std");
const config_mod = @import("../util/config.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const data_mod = @import("../data/dataset.zig");
const checkpoint_mod = @import("../checkpoint/manager.zig");
const model_mod = @import("../model/model.zig");

pub const Config = config_mod.Config;
pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;

pub const EvalResults = struct {
    perplexity: f64,
    loss: f64,
    tokens: usize,
    batches: usize,
    duration_ms: u64,
};

pub const ThroughputResult = struct {
    batch_size: usize,
    seq_len: usize,
    throughput: f64,
};

fn randomSeed() u64 {
    var seed_bytes: [8]u8 = undefined;
    std.crypto.random.bytes(&seed_bytes);
    return std.mem.readInt(u64, &seed_bytes, .little);
}

fn startTimer() !std.time.Timer {
    return try std.time.Timer.start();
}

fn readTimerMillis(timer: *std.time.Timer) u64 {
    return @as(u64, @intCast(timer.read() / std.time.ns_per_ms));
}

fn loadCheckpointIntoModel(allocator: std.mem.Allocator, model: *model_mod.EflaModel, checkpoint_path: []const u8) !void {
    if (checkpoint_path.len == 0) {
        return;
    }

    if (@hasDecl(model_mod.EflaModel, "loadCheckpoint")) {
        const method = model_mod.EflaModel.loadCheckpoint;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 2) {
            try @call(.auto, method, .{ model, checkpoint_path });
            return;
        }
        if (info.params.len == 3) {
            try @call(.auto, method, .{ model, allocator, checkpoint_path });
            return;
        }
        @compileError("Unsupported EflaModel.loadCheckpoint signature");
    }

    if (@hasDecl(model_mod.EflaModel, "loadFromCheckpoint")) {
        const method = model_mod.EflaModel.loadFromCheckpoint;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 2) {
            try @call(.auto, method, .{ model, checkpoint_path });
            return;
        }
        if (info.params.len == 3) {
            try @call(.auto, method, .{ model, allocator, checkpoint_path });
            return;
        }
        @compileError("Unsupported EflaModel.loadFromCheckpoint signature");
    }

    if (@hasDecl(checkpoint_mod, "loadCheckpoint")) {
        const method = checkpoint_mod.loadCheckpoint;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 2) {
            try @call(.auto, method, .{ model, checkpoint_path });
            return;
        }
        if (info.params.len == 3) {
            try @call(.auto, method, .{ allocator, model, checkpoint_path });
            return;
        }
        @compileError("Unsupported checkpoint_mod.loadCheckpoint signature");
    }

    if (@hasDecl(checkpoint_mod, "loadModel")) {
        const method = checkpoint_mod.loadModel;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 2) {
            try @call(.auto, method, .{ model, checkpoint_path });
            return;
        }
        if (info.params.len == 3) {
            try @call(.auto, method, .{ allocator, model, checkpoint_path });
            return;
        }
        @compileError("Unsupported checkpoint_mod.loadModel signature");
    }

    if (@hasDecl(checkpoint_mod, "CheckpointManager")) {
        const Manager = checkpoint_mod.CheckpointManager;
        if (@hasDecl(Manager, "init")) {
            const init_fn = Manager.init;
            const init_info = @typeInfo(@TypeOf(init_fn)).Fn;

            if (init_info.params.len == 1) {
                var manager = try @call(.auto, init_fn, .{ allocator });
                defer manager.deinit();

                if (@hasDecl(Manager, "loadModel")) {
                    const method = Manager.loadModel;
                    const info = @typeInfo(@TypeOf(method)).Fn;
                    if (info.params.len == 3) {
                        try @call(.auto, method, .{ &manager, model, checkpoint_path });
                        return;
                    }
                    if (info.params.len == 2) {
                        try @call(.auto, method, .{ &manager, model });
                        return;
                    }
                    @compileError("Unsupported CheckpointManager.loadModel signature");
                }

                @compileError("CheckpointManager exists but has no supported loadModel method");
            }

            if (init_info.params.len == 2) {
                var manager = try @call(.auto, init_fn, .{ allocator, checkpoint_path });
                defer manager.deinit();

                if (@hasDecl(Manager, "loadModel")) {
                    const method = Manager.loadModel;
                    const info = @typeInfo(@TypeOf(method)).Fn;
                    if (info.params.len == 2) {
                        try @call(.auto, method, .{ &manager, model });
                        return;
                    }
                    if (info.params.len == 3) {
                        try @call(.auto, method, .{ &manager, model, checkpoint_path });
                        return;
                    }
                    @compileError("Unsupported CheckpointManager.loadModel signature");
                }

                @compileError("CheckpointManager exists but has no supported loadModel method");
            }

            @compileError("Unsupported CheckpointManager.init signature");
        }
    }

    @compileError("No supported checkpoint loading API found");
}

fn resolveBatchTokenCount(batch: anytype) usize {
    const BatchType = @TypeOf(batch.*);

    if (@hasField(BatchType, "batch_size") and @hasField(BatchType, "seq_len")) {
        return @as(usize, @intCast(@field(batch.*, "batch_size"))) * @as(usize, @intCast(@field(batch.*, "seq_len")));
    }

    if (@hasField(BatchType, "input_ids")) {
        const input_ids = @field(batch.*, "input_ids");
        const InputType = @TypeOf(input_ids);

        if (@hasField(InputType, "shape")) {
            const shape = input_ids.shape;
            const ShapeType = @TypeOf(shape);

            if (@hasField(ShapeType, "dims")) {
                const dims = shape.dims;
                if (dims.len >= 2) {
                    return @as(usize, @intCast(dims[0])) * @as(usize, @intCast(dims[1]));
                }
                if (dims.len == 1) {
                    return @as(usize, @intCast(dims[0]));
                }
            }
        }

        if (@hasField(InputType, "data")) {
            return input_ids.data.len;
        }
    }

    if (@hasField(BatchType, "input_tokens")) {
        const input_tokens = @field(batch.*, "input_tokens");
        const T = @TypeOf(input_tokens);
        if (@typeInfo(T) == .Pointer or @typeInfo(T) == .Array) {
            return input_tokens.len;
        }
    }

    @compileError("Unsupported data_mod.Batch layout: unable to resolve token count");
}

fn deinitBatch(batch: anytype, allocator: std.mem.Allocator) void {
    const BatchType = @TypeOf(batch.*);

    if (@hasDecl(BatchType, "deinit")) {
        batch.deinit(allocator);
        return;
    }

    @compileError("Unsupported data_mod.Batch layout: missing deinit");
}

fn toF64(value: anytype) f64 {
    const T = @TypeOf(value);
    if (T == f64) return value;
    if (T == f32) return @as(f64, value);
    if (T == comptime_float) return value;
    if (T == usize) return @as(f64, @floatFromInt(value));
    if (T == u64) return @as(f64, @floatFromInt(value));
    @compileError("Unsupported numeric return type for conversion to f64");
}

fn forwardLossFromModel(model: *model_mod.EflaModel, allocator: std.mem.Allocator, batch: anytype) !f64 {
    const ModelType = @TypeOf(model.*);

    if (@hasDecl(ModelType, "computeLoss")) {
        const method = ModelType.computeLoss;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 2) return toF64(try model.computeLoss(batch));
        if (info.params.len == 3) return toF64(try model.computeLoss(allocator, batch));
        @compileError("Unsupported EflaModel.computeLoss signature");
    }

    if (@hasDecl(ModelType, "loss")) {
        const method = ModelType.loss;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 2) return toF64(try model.loss(batch));
        if (info.params.len == 3) return toF64(try model.loss(allocator, batch));
        @compileError("Unsupported EflaModel.loss signature");
    }

    if (@hasDecl(ModelType, "forwardLoss")) {
        const method = ModelType.forwardLoss;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 2) return toF64(try model.forwardLoss(batch));
        if (info.params.len == 3) return toF64(try model.forwardLoss(allocator, batch));
        @compileError("Unsupported EflaModel.forwardLoss signature");
    }

    if (@hasDecl(ModelType, "evalLoss")) {
        const method = ModelType.evalLoss;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 2) return toF64(try model.evalLoss(batch));
        if (info.params.len == 3) return toF64(try model.evalLoss(allocator, batch));
        @compileError("Unsupported EflaModel.evalLoss signature");
    }

    @compileError("Unsupported model_mod.EflaModel API: missing loss computation method");
}

fn modelForwardLogits(model: *model_mod.EflaModel, allocator: std.mem.Allocator, tokens: []const u32) ![]f32 {
    const ModelType = @TypeOf(model.*);

    if (@hasDecl(ModelType, "predictNextLogits")) {
        const method = ModelType.predictNextLogits;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 3) return try model.predictNextLogits(allocator, tokens);
        if (info.params.len == 2) return try model.predictNextLogits(tokens);
        @compileError("Unsupported EflaModel.predictNextLogits signature");
    }

    if (@hasDecl(ModelType, "nextTokenLogits")) {
        const method = ModelType.nextTokenLogits;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 3) return try model.nextTokenLogits(allocator, tokens);
        if (info.params.len == 2) return try model.nextTokenLogits(tokens);
        @compileError("Unsupported EflaModel.nextTokenLogits signature");
    }

    if (@hasDecl(ModelType, "forwardNextToken")) {
        const method = ModelType.forwardNextToken;
        const info = @typeInfo(@TypeOf(method)).Fn;
        if (info.params.len == 3) return try model.forwardNextToken(allocator, tokens);
        if (info.params.len == 2) return try model.forwardNextToken(tokens);
        @compileError("Unsupported EflaModel.forwardNextToken signature");
    }

    @compileError("Unsupported model_mod.EflaModel API: missing next-token logits method");
}

fn tokenizerEncode(allocator: std.mem.Allocator, config: Config, text: []const u8) ![]u32 {
    _ = config;

    if (@hasDecl(config_mod, "encode")) {
        return try config_mod.encode(allocator, text);
    }

    if (@hasDecl(config_mod, "tokenize")) {
        return try config_mod.tokenize(allocator, text);
    }

    if (@hasDecl(model_mod, "encode")) {
        return try model_mod.encode(allocator, text);
    }

    var out = try allocator.alloc(u32, text.len);
    errdefer allocator.free(out);

    for (text, 0..) |c, i| {
        out[i] = @as(u32, c);
    }

    return out;
}

fn tokenizerDecode(allocator: std.mem.Allocator, config: Config, tokens: []const u32) ![]u8 {
    _ = config;

    if (@hasDecl(config_mod, "decode")) {
        return try config_mod.decode(allocator, tokens);
    }

    if (@hasDecl(config_mod, "detokenize")) {
        return try config_mod.detokenize(allocator, tokens);
    }

    if (@hasDecl(model_mod, "decode")) {
        return try model_mod.decode(allocator, tokens);
    }

    var out = try allocator.alloc(u8, tokens.len);
    errdefer allocator.free(out);

    for (tokens, 0..) |t, i| {
        out[i] = @as(u8, @intCast(@min(t, 255)));
    }

    return out;
}

const ProbIndexSortContext = struct {
    probs: []const f32,

    fn lessThan(ctx: @This(), a: usize, b: usize) bool {
        return ctx.probs[a] > ctx.probs[b];
    }
};

pub const Evaluator = struct {
    config: Config,
    model: model_mod.EflaModel,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: Config, checkpoint_path: []const u8) !Self {
        var prng = std.Random.DefaultPrng.init(randomSeed());
        var model = try model_mod.EflaModel.init(
            allocator,
            config.model,
            .cpu,
            0,
            prng.random(),
        );
        errdefer model.deinit();

        try loadCheckpointIntoModel(allocator, &model, checkpoint_path);

        return .{
            .config = config,
            .model = model,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
    }

    fn computeLoss(self: *Self, batch: *const data_mod.Batch) !f64 {
        return try forwardLossFromModel(&self.model, self.allocator, batch);
    }

    pub fn evaluatePerplexity(self: *Self, data_path: []const u8, max_tokens: usize) !EvalResults {
        var dataset = try data_mod.BinaryDataset.open(self.allocator, data_path);
        defer dataset.close();

        var loader = try data_mod.DataLoader.init(
            self.allocator,
            &dataset,
            1,
            self.config.data.seq_len,
            false,
            42,
        );
        defer loader.deinit();

        var timer = try startTimer();

        var total_loss: f64 = 0.0;
        var total_tokens: usize = 0;
        var batches: usize = 0;

        while (try loader.next()) |batch_value| {
            var batch = batch_value;
            defer deinitBatch(&batch, self.allocator);

            const batch_tokens = resolveBatchTokenCount(&batch);
            if (batch_tokens == 0) {
                continue;
            }

            var effective_tokens = batch_tokens;
            if (max_tokens > 0 and total_tokens + effective_tokens > max_tokens) {
                effective_tokens = max_tokens - total_tokens;
            }

            if (effective_tokens == 0) {
                break;
            }

            const avg_loss = try self.computeLoss(&batch);
            total_loss += avg_loss * @as(f64, @floatFromInt(effective_tokens));
            total_tokens += effective_tokens;
            batches += 1;

            if (max_tokens > 0 and total_tokens >= max_tokens) {
                break;
            }
        }

        const duration_ms = readTimerMillis(&timer);

        if (total_tokens == 0) {
            return .{
                .perplexity = 0.0,
                .loss = 0.0,
                .tokens = 0,
                .batches = 0,
                .duration_ms = duration_ms,
            };
        }

        const mean_loss = total_loss / @as(f64, @floatFromInt(total_tokens));
        const perplexity = std.math.exp(mean_loss);

        return .{
            .perplexity = perplexity,
            .loss = mean_loss,
            .tokens = total_tokens,
            .batches = batches,
            .duration_ms = duration_ms,
        };
    }

    pub fn evaluateLongContext(self: *Self, data_path: []const u8, context_lengths: []const usize) ![]EvalResults {
        var results = try self.allocator.alloc(EvalResults, context_lengths.len);
        errdefer self.allocator.free(results);

        for (context_lengths, 0..) |ctx_len, i| {
            if (ctx_len == 0) {
                return error.InvalidContextLength;
            }

            var dataset = try data_mod.BinaryDataset.open(self.allocator, data_path);
            defer dataset.close();

            var loader = try data_mod.DataLoader.init(
                self.allocator,
                &dataset,
                1,
                ctx_len,
                false,
                42,
            );
            defer loader.deinit();

            var timer = try startTimer();

            var total_loss: f64 = 0.0;
            var total_tokens: usize = 0;
            var batches: usize = 0;

            while (try loader.next()) |batch_value| {
                var batch = batch_value;
                defer deinitBatch(&batch, self.allocator);

                const batch_tokens = resolveBatchTokenCount(&batch);
                if (batch_tokens == 0) {
                    continue;
                }

                const avg_loss = try self.computeLoss(&batch);
                total_loss += avg_loss * @as(f64, @floatFromInt(batch_tokens));
                total_tokens += batch_tokens;
                batches += 1;
            }

            const duration_ms = readTimerMillis(&timer);

            if (total_tokens == 0) {
                results[i] = .{
                    .perplexity = 0.0,
                    .loss = 0.0,
                    .tokens = 0,
                    .batches = 0,
                    .duration_ms = duration_ms,
                };
            } else {
                const mean_loss = total_loss / @as(f64, @floatFromInt(total_tokens));
                results[i] = .{
                    .perplexity = std.math.exp(mean_loss),
                    .loss = mean_loss,
                    .tokens = total_tokens,
                    .batches = batches,
                    .duration_ms = duration_ms,
                };
            }
        }

        return results;
    }
};

pub const Generator = struct {
    config: Config,
    model: model_mod.EflaModel,
    allocator: std.mem.Allocator,
    rng: std.Random.DefaultPrng,
    sample_probs: std.ArrayListUnmanaged(f32),
    sample_indices: std.ArrayListUnmanaged(usize),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: Config, checkpoint_path: []const u8) !Self {
        var prng = std.Random.DefaultPrng.init(randomSeed());
        var model = try model_mod.EflaModel.init(
            allocator,
            config.model,
            .cpu,
            0,
            prng.random(),
        );
        errdefer model.deinit();

        try loadCheckpointIntoModel(allocator, &model, checkpoint_path);

        return .{
            .config = config,
            .model = model,
            .allocator = allocator,
            .rng = prng,
            .sample_probs = .{},
            .sample_indices = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.sample_probs.deinit(self.allocator);
        self.sample_indices.deinit(self.allocator);
        self.model.deinit();
    }

    pub fn generate(self: *Self, prompt: []const u8, max_new_tokens: usize, temperature: f32, top_p: f32) ![]const u8 {
        var prompt_tokens = try tokenizerEncode(self.allocator, self.config, prompt);
        defer self.allocator.free(prompt_tokens);

        var generated_tokens = try self.generateSampled(prompt_tokens, max_new_tokens, temperature, top_p);
        defer self.allocator.free(generated_tokens);

        return try tokenizerDecode(self.allocator, self.config, generated_tokens);
    }

    pub fn generateGreedy(self: *Self, prompt_tokens: []const u32, max_new_tokens: usize) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        try tokens.appendSlice(prompt_tokens);

        var step: usize = 0;
        while (step < max_new_tokens) : (step += 1) {
            const logits = try modelForwardLogits(&self.model, self.allocator, tokens.items);
            defer self.allocator.free(logits);

            if (logits.len == 0) {
                return error.EmptyLogits;
            }

            var best_index: usize = 0;
            var best_value: f32 = logits[0];

            for (logits[1..], 1..) |v, i| {
                if (v > best_value) {
                    best_value = v;
                    best_index = i;
                }
            }

            try tokens.append(@as(u32, @intCast(best_index)));
        }

        return try tokens.toOwnedSlice();
    }

    pub fn generateSampled(self: *Self, prompt_tokens: []const u32, max_new_tokens: usize, temperature: f32, top_p: f32) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        try tokens.appendSlice(prompt_tokens);

        var step: usize = 0;
        while (step < max_new_tokens) : (step += 1) {
            const logits = try modelForwardLogits(&self.model, self.allocator, tokens.items);
            defer self.allocator.free(logits);

            const next_token = try self.sampleTopP(logits, temperature, top_p);
            try tokens.append(next_token);
        }

        return try tokens.toOwnedSlice();
    }

    pub fn sampleTopP(self: *Self, logits: []const f32, temperature: f32, top_p: f32) !u32 {
        if (logits.len == 0) {
            return error.EmptyLogits;
        }
        if (!std.math.isFinite(temperature) or temperature <= 0.0) {
            return error.InvalidTemperature;
        }
        if (!std.math.isFinite(top_p) or top_p <= 0.0 or top_p > 1.0) {
            return error.InvalidTopP;
        }

        try self.sample_probs.resize(self.allocator, logits.len);
        try self.sample_indices.resize(self.allocator, logits.len);

        var probs = self.sample_probs.items;
        var indices = self.sample_indices.items;

        var max_logit: f32 = logits[0];
        if (!std.math.isFinite(max_logit)) {
            return error.NonFiniteLogit;
        }

        for (logits[1..]) |v| {
            if (!std.math.isFinite(v)) {
                return error.NonFiniteLogit;
            }
            if (v > max_logit) {
                max_logit = v;
            }
        }

        var sum: f64 = 0.0;
        for (logits, 0..) |logit, i| {
            const scaled = (@as(f64, @floatCast(logit)) - @as(f64, @floatCast(max_logit))) / @as(f64, @floatCast(temperature));
            const p = std.math.exp(scaled);
            if (!std.math.isFinite(p)) {
                return error.NonFiniteProbability;
            }
            probs[i] = @as(f32, @floatCast(p));
            indices[i] = i;
            sum += p;
        }

        if (!(sum > 0.0) or !std.math.isFinite(sum)) {
            return error.InvalidProbabilityMass;
        }

        for (probs) |*p| {
            p.* = @as(f32, @floatCast(@as(f64, p.*) / sum));
        }

        std.sort.heap(usize, indices, ProbIndexSortContext{ .probs = probs }, ProbIndexSortContext.lessThan);

        var nucleus_count: usize = 0;
        var nucleus_mass: f64 = 0.0;

        while (nucleus_count < indices.len) : (nucleus_count += 1) {
            nucleus_mass += @as(f64, probs[indices[nucleus_count]]);
            if (nucleus_mass >= @as(f64, @floatCast(top_p))) {
                nucleus_count += 1;
                break;
            }
        }

        if (nucleus_count == 0) {
            nucleus_count = 1;
            nucleus_mass = @as(f64, probs[indices[0]]);
        }

        var draw = self.rng.random().float(f64);
        if (draw >= 1.0) {
            draw = std.math.nextAfter(f64, 1.0, 0.0);
        }

        const threshold = draw * nucleus_mass;
        var cumulative: f64 = 0.0;

        for (indices[0..nucleus_count]) |token_index| {
            cumulative += @as(f64, probs[token_index]);
            if (threshold <= cumulative) {
                return @as(u32, @intCast(token_index));
            }
        }

        return @as(u32, @intCast(indices[nucleus_count - 1]));
    }
};

pub fn benchmarkThroughput(allocator: std.mem.Allocator, config: Config, batch_sizes: []const usize, seq_lengths: []const usize) ![]ThroughputResult {
    var results = std.ArrayList(ThroughputResult).init(allocator);
    errdefer results.deinit();

    var prng = std.Random.DefaultPrng.init(randomSeed());

    for (batch_sizes) |batch_size| {
        for (seq_lengths) |seq_len| {
            var model = try model_mod.EflaModel.init(
                allocator,
                config.model,
                .cpu,
                0,
                prng.random(),
            );
            defer model.deinit();

            const total_input_tokens = batch_size * seq_len;

            if (total_input_tokens == 0) {
                try results.append(.{
                    .batch_size = batch_size,
                    .seq_len = seq_len,
                    .throughput = 0.0,
                });
                continue;
            }

            var input = try allocator.alloc(u32, total_input_tokens);
            defer allocator.free(input);

            const vocab_size: usize = if (@hasField(@TypeOf(config.model), "vocab_size"))
                @as(usize, @intCast(config.model.vocab_size))
            else
                256;

            const safe_vocab_size: usize = if (vocab_size == 0) 256 else vocab_size;

            for (input, 0..) |*t, i| {
                t.* = @as(u32, @intCast(i % safe_vocab_size));
            }

            const warmup_iters: usize = 2;
            var warmup_step: usize = 0;
            while (warmup_step < warmup_iters) : (warmup_step += 1) {
                const logits = try modelForwardLogits(&model, allocator, input);
                allocator.free(logits);
            }

            const measured_iters: usize = 10;
            var timer = try startTimer();

            var measured_step: usize = 0;
            while (measured_step < measured_iters) : (measured_step += 1) {
                const logits = try modelForwardLogits(&model, allocator, input);
                allocator.free(logits);
            }

            const elapsed_ns = timer.read();
            if (elapsed_ns == 0) {
                return error.InvalidBenchmarkDuration;
            }

            const total_tokens = @as(f64, @floatFromInt(total_input_tokens * measured_iters));
            const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
            const throughput = total_tokens / seconds;

            try results.append(.{
                .batch_size = batch_size,
                .seq_len = seq_len,
                .throughput = throughput,
            });
        }
    }

    return try results.toOwnedSlice();
}
