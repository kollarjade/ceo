const std = @import("std");
const config_mod = @import("../util/config.zig");
const runtime_mod = @import("main.zig");
const telemetry_mod = @import("../telemetry/main.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const optim_mod = @import("../optim/optimizer.zig");
const nn_mod = @import("../nn/layers.zig");
const model_mod = @import("../model/model.zig");
const data_mod = @import("../data/dataset.zig");
const checkpoint_mod = @import("../checkpoint/manager.zig");

pub const Config = config_mod.Config;
pub const DistributedRuntime = runtime_mod.DistributedRuntime;
pub const Telemetry = telemetry_mod.Telemetry;
pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;

/// Main training orchestrator
pub const Trainer = struct {
    config: Config,
    runtime: *DistributedRuntime,
    telemetry: *Telemetry,
    model: *model_mod.EflaModel,
    optimizer: *optim_mod.LionMuonOptimizer,
    scheduler: optim_mod.LRScheduler,
    clipper: optim_mod.GradientClipper,
    checkpoint_manager: checkpoint_mod.CheckpointManager,
    step: usize,
    epoch: usize,
    tokens_seen: usize,
    best_loss: f32,
    allocator: std.mem.Allocator,
    rng: std.Random.DefaultPrng,

    const Self = @This();

    /// Initialize trainer
    pub fn init(
        allocator: std.mem.Allocator,
        config: Config,
        runtime: *DistributedRuntime,
        telemetry: *Telemetry,
    ) !Self {
        var prng = std.Random.DefaultPrng.init(config.runtime.seed);

        // Initialize model
        var model = try model_mod.EflaModel.init(
            allocator,
            config.model,
            .cuda,
            @intCast(runtime.rank),
            prng.random(),
        );
        errdefer model.deinit();

        // Collect parameters
        var params = try model.collectParameters(allocator);
        errdefer allocator.free(params);

        // Initialize optimizer
        var optimizer = try optim_mod.LionMuonOptimizer.init(
            allocator,
            params,
            config.training.learning_rate,
            config.training.lion_beta1,
            config.training.lion_beta2,
            config.training.muon_momentum,
            config.training.muon_iterations,
            config.training.weight_decay,
            .cuda,
            @intCast(runtime.rank),
        );
        errdefer optimizer.deinit();

        // Initialize scheduler
        var scheduler = optim_mod.LRScheduler.init(
            config.training.learning_rate,
            config.training.min_learning_rate,
            config.training.warmup_steps,
            config.training.total_steps,
            .linear_warmup_cosine,
        );

        // Initialize gradient clipper
        var clipper = optim_mod.GradientClipper.init(
            config.training.gradient_clip,
            .norm,
        );

        // Initialize checkpoint manager
        var checkpoint_manager = try checkpoint_mod.CheckpointManager.init(
            allocator,
            config.checkpoint.dir,
            config.checkpoint.keep_last_n,
            config.checkpoint.compression,
        );
        errdefer checkpoint_manager.deinit();

        return .{
            .config = config,
            .runtime = runtime,
            .telemetry = telemetry,
            .model = model,
            .optimizer = optimizer,
            .scheduler = scheduler,
            .clipper = clipper,
            .checkpoint_manager = checkpoint_manager,
            .step = 0,
            .epoch = 0,
            .tokens_seen = 0,
            .best_loss = std.math.inf(f32),
            .allocator = allocator,
            .rng = prng,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.optimizer.deinit();
        self.checkpoint_manager.deinit();
    }

    /// Run training
    pub fn run(self: *Self, data_path: ?[]const u8, checkpoint_dir: ?[]const u8) !void {
        _ = checkpoint_dir;

        const path = data_path orelse self.config.data.path;

        std.log.info("Starting training from step {d}", .{self.step});
        std.log.info("Training data: {s}", .{path});

        // Open dataset
        var dataset = try data_mod.BinaryDataset.open(self.allocator, path);
        defer dataset.close();

        std.log.info("Dataset contains {d} tokens", .{dataset.num_tokens});

        // Create data loader
        var loader = try data_mod.DataLoader.init(
            self.allocator,
            &dataset,
            self.config.training.micro_batch_size,
            self.config.data.seq_len,
            true,
            self.config.runtime.seed,
        );
        defer loader.deinit();

        std.log.info("Created data loader with {d} batches", .{loader.numBatches()});

        // Training loop
        while (self.step < self.config.training.total_steps) {
            // Get next batch
            if (try loader.next()) |batch| {
                defer batch.deinit(self.allocator);

                // Forward pass
                const loss = try self.forwardStep(&batch);

                // Backward pass
                try self.backwardStep(&batch);

                // Update parameters
                try self.updateStep();

                // Update metrics
                self.step += 1;
                self.tokens_seen += batch.batch_size * batch.seq_len;

                // Log metrics
                if (self.step % self.config.telemetry.metrics_interval == 0) {
                    try self.logMetrics(loss);
                }

                // Save checkpoint
                if (self.step % self.config.checkpoint.save_interval == 0) {
                    try self.saveCheckpoint();
                }
            } else {
                // End of epoch
                loader.reset();
                self.epoch += 1;
                std.log.info("Starting epoch {d}", .{self.epoch});
            }
        }

        std.log.info("Training completed at step {d}", .{self.step});
    }

    /// Run smoke test
    pub fn smokeTest(self: *Self) !void {
        std.log.info("Running smoke test...", .{});

        // Create a small test batch
        const batch_size = 1;
        const seq_len = 16;

        var input_tokens = try self.allocator.alloc(u32, batch_size * seq_len);
        defer self.allocator.free(input_tokens);
        @memset(input_tokens, 0);

        var target_tokens = try self.allocator.alloc(u32, batch_size * seq_len);
        defer self.allocator.free(target_tokens);
        @memset(target_tokens, 0);

        var batch = data_mod.Batch{
            .input = input_tokens,
            .target = target_tokens,
            .batch_size = batch_size,
            .seq_len = seq_len,
        };

        // Run a few iterations
        for (0..3) |i| {
            const loss = try self.forwardStep(&batch);
            try self.backwardStep(&batch);
            try self.updateStep();

            std.log.info("Smoke test iteration {d}: loss = {d:.4}", .{ i, loss });
        }

        std.log.info("Smoke test passed!", .{});
    }

    /// Resume from checkpoint
    pub fn resume(self: *Self, checkpoint_path: []const u8) !void {
        std.log.info("Resuming from checkpoint: {s}", .{checkpoint_path});

        // Load checkpoint
        const metadata = try self.checkpoint_manager.load(
            checkpoint_path,
            try self.model.collectParameters(self.allocator),
            try self.model.getParameterNames(self.allocator),
        );

        self.step = metadata.step;
        self.epoch = metadata.epoch;
        self.tokens_seen = metadata.tokens_seen;
        self.best_loss = metadata.loss;

        std.log.info("Resumed from step {d}, epoch {d}", .{ self.step, self.epoch });
    }

    /// Forward pass
    fn forwardStep(self: *Self, batch: *const data_mod.Batch) !f32 {
        // Create input tensor
        const input_shape = Shape.init(&[_]usize{ batch.batch_size, batch.seq_len });
        var input_tensor = try Tensor.fromSlice(
            self.allocator,
            input_shape,
            .int32,
            .cuda,
            @intCast(self.runtime.rank),
            // Convert to float for model input
            blk: {
                var floats = try self.allocator.alloc(f32, batch.input.len);
                for (batch.input, floats) |t, *f| {
                    f.* = @floatFromInt(t);
                }
                break :blk floats;
            },
        );
        defer input_tensor.deinit();

        // Forward pass through model
        const output = try self.model.forward(input_tensor);
        defer output.deinit();

        // Compute loss (placeholder - would use cross-entropy kernel)
        const loss: f32 = 10.0; // Placeholder

        return loss;
    }

    /// Backward pass
    fn backwardStep(self: *Self, batch: *const data_mod.Batch) !void {
        _ = batch;

        // Backprop through model
        try self.model.backward();

        // Clip gradients
        var grads = try self.model.collectGradients(self.allocator);
        defer self.allocator.free(grads);

        _ = try self.clipper.clip(grads);
    }

    /// Update parameters
    fn updateStep(self: *Self) !void {
        // Update learning rate
        const lr = self.scheduler.getLR();
        self.scheduler.step();

        // Get gradients and update
        var grads = try self.model.collectGradients(self.allocator);
        defer self.allocator.free(grads);

        try self.optimizer.step(grads);

        // Update telemetry
        try self.telemetry.logStep(.{
            .step = self.step,
            .tokens = self.tokens_seen,
            .lr = lr,
            .grad_norm = 0.0, // Would compute from actual gradients
        });
    }

    /// Log metrics
    fn logMetrics(self: *Self, loss: f32) !void {
        const lr = self.scheduler.getLR();
        const throughput = @as(f64, @floatFromInt(self.tokens_seen)) /
            @as(f64, @floatFromInt(self.step + 1));

        std.log.info(
            "step={d} loss={d:.4} lr={d:.2e} tokens={d} throughput={d:.1} tok/s",
            .{ self.step, loss, lr, self.tokens_seen, throughput },
        );

        try self.telemetry.logMetrics(.{
            .step = self.step,
            .loss = loss,
            .lr = lr,
            .tokens = self.tokens_seen,
            .throughput = throughput,
        });
    }

    /// Save checkpoint
    fn saveCheckpoint(self: *Self) !void {
        const metadata = checkpoint_mod.CheckpointMetadata{
            .step = self.step,
            .epoch = self.epoch,
            .tokens_seen = self.tokens_seen,
            .loss = self.best_loss,
            .learning_rate = self.scheduler.getLR(),
            .timestamp = std.time.timestamp(),
            .git_revision = [_]u8{0} ** 40,
            .config_hash = [_]u8{0} ** 32,
        };

        var params = try self.model.collectParameters(self.allocator);
        defer self.allocator.free(params);

        var names = try self.model.getParameterNames(self.allocator);
        defer {
            for (names) |n| self.allocator.free(n);
            self.allocator.free(names);
        }

        _ = try self.checkpoint_manager.save(
            self.step,
            params,
            names,
            null, // optimizer state
            null, // rng state
            metadata,
        );

        std.log.info("Saved checkpoint at step {d}", .{self.step});
    }
};
