const std = @import("std");
const config_mod = @import("../util/config.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const data_mod = @import("../data/dataset.zig");
const checkpoint_mod = @import("../checkpoint/manager.zig");
const model_mod = @import("../model/model.zig");

pub const Config = config_mod.Config;
pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;

/// Evaluation results
pub const EvalResults = struct {
    perplexity: f64,
    loss: f64,
    tokens: usize,
    batches: usize,
    duration_ms: usize,
};

/// Model evaluator
pub const Evaluator = struct {
    config: Config,
    model: *model_mod.EflaModel,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        config: Config,
        checkpoint_path: []const u8,
    ) !Self {
        _ = checkpoint_path;

        var prng = std.Random.DefaultPrng.init(42);

        var model = try model_mod.EflaModel.init(
            allocator,
            config.model,
            .cpu, // Evaluator can run on CPU
            0,
            prng.random(),
        );

        return .{
            .config = config,
            .model = model,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
    }

    /// Evaluate perplexity on a dataset
    pub fn evaluatePerplexity(self: *Self, data_path: []const u8, max_tokens: usize) !EvalResults {
        var dataset = try data_mod.BinaryDataset.open(self.allocator, data_path);
        defer dataset.close();

        var loader = try data_mod.DataLoader.init(
            self.allocator,
            &dataset,
            1, // batch size
            self.config.data.seq_len,
            false, // don't shuffle
            42,
        );
        defer loader.deinit();

        const start_time = std.time.milliTimestamp();
        var total_loss: f64 = 0.0;
        var total_tokens: usize = 0;
        var batches: usize = 0;

        while (try loader.next()) |batch| {
            defer batch.deinit(self.allocator);

            // Compute loss
            const loss = try self.computeLoss(&batch);
            total_loss += loss * @as(f64, @floatFromInt(batch.batch_size * batch.seq_len));
            total_tokens += batch.batch_size * batch.seq_len;
            batches += 1;

            if (max_tokens > 0 and total_tokens >= max_tokens) {
                break;
            }
        }

        const end_time = std.time.milliTimestamp();
        const avg_loss = total_loss / @as(f64, @floatFromInt(total_tokens));
        const perplexity = std.math.exp(avg_loss);

        return .{
            .perplexity = perplexity,
            .loss = avg_loss,
            .tokens = total_tokens,
            .batches = batches,
            .duration_ms = @intCast(end_time - start_time),
        };
    }

    fn computeLoss(self: *Self, batch: *const data_mod.Batch) !f64 {
        _ = self;
        _ = batch;

        // Placeholder - would compute cross-entropy loss
        return 10.0;
    }

    /// Evaluate long-context performance
    pub fn evaluateLongContext(
        self: *Self,
        data_path: []const u8,
        context_lengths: []const usize,
    ) ![]EvalResults {
        var results = try self.allocator.alloc(EvalResults, context_lengths.len);
        errdefer self.allocator.free(results);

        for (context_lengths, 0..) |ctx_len, i| {
            std.log.info("Evaluating at context length {d}", .{ctx_len});

            results[i] = try self.evaluatePerplexity(data_path, ctx_len);
        }

        return results;
    }
};

/// Text generator
pub const Generator = struct {
    config: Config,
    model: *model_mod.EflaModel,
    allocator: std.mem.Allocator,
    rng: std.Random.DefaultPrng,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        config: Config,
        checkpoint_path: []const u8,
    ) !Self {
        _ = checkpoint_path;

        var prng = std.Random.DefaultPrng.init(42);

        var model = try model_mod.EflaModel.init(
            allocator,
            config.model,
            .cpu,
            0,
            prng.random(),
        );

        return .{
            .config = config,
            .model = model,
            .allocator = allocator,
            .rng = prng,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
    }

    /// Generate text from a prompt
    pub fn generate(
        self: *Self,
        prompt: []const u8,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) ![]const u8 {
        _ = prompt;
        _ = temperature;
        _ = top_p;

        // Placeholder output
        var output = std.ArrayList(u8).init(self.allocator);
        defer output.deinit();

        for (0..max_new_tokens) |_| {
            // Would sample from model output
            try output.append(' ');
        }

        return output.toOwnedSlice();
    }

    /// Greedy generation
    pub fn generateGreedy(
        self: *Self,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
    ) ![]u32 {
        _ = self;

        var tokens = try self.allocator.alloc(u32, prompt_tokens.len + max_new_tokens);
        @memcpy(tokens[0..prompt_tokens.len], prompt_tokens);

        // Fill with placeholder tokens
        for (prompt_tokens.len..tokens.len) |i| {
            tokens[i] = 0;
        }

        return tokens;
    }

    /// Nucleus (top-p) sampling
    pub fn sampleTopP(
        self: *Self,
        logits: []const f32,
        temperature: f32,
        top_p: f32,
    ) !u32 {
        _ = self;

        // Apply temperature
        var scaled = try self.allocator.alloc(f32, logits.len);
        defer self.allocator.free(scaled);

        for (logits, scaled) |l, *s| {
            s.* = l / temperature;
        }

        // Softmax
        var max_val: f32 = -std.math.inf(f32);
        for (scaled) |s| {
            if (s > max_val) max_val = s;
        }

        var sum: f64 = 0.0;
        for (scaled) |*s| {
            s.* = @exp(s.* - max_val);
            sum += s.*;
        }

        for (scaled) |*s| {
            s.* /= @floatCast(sum);
        }

        // Sort by probability (descending)
        // Simplified - just sample randomly for now

        const r = self.rng.random().float(f64);
        var cumsum: f64 = 0.0;

        for (scaled, 0..) |prob, i| {
            cumsum += prob;
            if (cumsum >= r * top_p) {
                return @intCast(i);
            }
        }

        return @intCast(scaled.len - 1);
    }
};

/// Benchmark throughput
pub fn benchmarkThroughput(
    allocator: std.mem.Allocator,
    config: Config,
    batch_sizes: []const usize,
    seq_lengths: []const usize,
) ![]struct { batch_size: usize, seq_len: usize, throughput: f64 } {
    var results = std.ArrayList(struct { batch_size: usize, seq_len: usize, throughput: f64 }).init(allocator);
    defer results.deinit();

    var prng = std.Random.DefaultPrng.init(42);

    for (batch_sizes) |batch_size| {
        for (seq_lengths) |seq_len| {
            std.log.info("Benchmarking batch_size={d}, seq_len={d}", .{ batch_size, seq_len });

            var model = try model_mod.EflaModel.init(
                allocator,
                config.model,
                .cpu,
                0,
                prng.random(),
            );
            defer model.deinit();

            const num_iterations = 10;
            const start = std.time.nanoTimestamp();

            for (0..num_iterations) |_| {
                // Would run forward pass
            }

            const end = std.time.nanoTimestamp();
            const duration_ns = @as(f64, @floatFromInt(end - start));
            const tokens_per_iter = @as(f64, @floatFromInt(batch_size * seq_len));
            const throughput = tokens_per_iter * @as(f64, @floatFromInt(num_iterations)) / (duration_ns / 1e9);

            try results.append(.{
                .batch_size = batch_size,
                .seq_len = seq_len,
                .throughput = throughput,
            });
        }
    }

    return results.toOwnedSlice();
}
