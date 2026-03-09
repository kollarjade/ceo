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
const dtype_mod = @import("../tensor/dtype.zig");

pub const Config = config_mod.Config;
pub const DistributedRuntime = runtime_mod.DistributedRuntime;
pub const Telemetry = telemetry_mod.Telemetry;
pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const BF16 = dtype_mod.BF16;

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
    cached_params: ?[]*Tensor,
    cached_grads: ?[]*Tensor,
    cached_param_names: ?[][]const u8,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: Config,
        runtime: *DistributedRuntime,
        telemetry: *Telemetry,
    ) !Self {
        var prng = std.Random.DefaultPrng.init(cfg.runtime.seed);

        var model = try model_mod.EflaModel.init(
            allocator,
            cfg.model,
            .cuda,
            @intCast(runtime.rank),
            &prng,
        );
        errdefer model.deinit();

        const params = try model.collectParameters(allocator);
        errdefer allocator.free(params);

        var optimizer = try optim_mod.LionMuonOptimizer.init(
            allocator,
            params,
            cfg.training.learning_rate,
            cfg.training.lion_beta1,
            cfg.training.lion_beta2,
            cfg.training.muon_momentum,
            cfg.training.muon_iterations,
            cfg.training.weight_decay,
            .cuda,
            @intCast(runtime.rank),
        );
        errdefer optimizer.deinit();

        const scheduler = optim_mod.LRScheduler.init(
            cfg.training.learning_rate,
            cfg.training.min_learning_rate,
            cfg.training.warmup_steps,
            cfg.training.total_steps,
            .linear_warmup_cosine,
        );

        const clipper = optim_mod.GradientClipper.init(
            cfg.training.gradient_clip,
            .norm,
        );

        var checkpoint_manager = try checkpoint_mod.CheckpointManager.init(
            allocator,
            cfg.checkpoint.dir,
            cfg.checkpoint.keep_last_n,
            cfg.checkpoint.compression,
        );
        errdefer checkpoint_manager.deinit();

        return .{
            .config = cfg,
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
            .cached_params = params,
            .cached_grads = null,
            .cached_param_names = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.cached_params) |p| {
            self.allocator.free(p);
        }
        if (self.cached_grads) |g| {
            self.allocator.free(g);
        }
        if (self.cached_param_names) |names| {
            for (names) |n| self.allocator.free(n);
            self.allocator.free(names);
        }
        self.model.deinit();
        self.optimizer.deinit();
        self.checkpoint_manager.deinit();
    }

    pub fn run(self: *Self, data_path: ?[]const u8) !void {
        const path = data_path orelse self.config.data.path;

        std.log.info("Starting training from step {d}", .{self.step});
        std.log.info("Training data: {s}", .{path});

        var dataset = try data_mod.BinaryDataset.open(self.allocator, path);
        defer dataset.close();

        std.log.info("Dataset contains {d} tokens", .{dataset.num_tokens});

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

        const start_time = std.time.milliTimestamp();

        while (self.step < self.config.training.total_steps) {
            if (try loader.next()) |batch| {
                defer batch.deinit(self.allocator);

                const loss = try self.trainStep(&batch);

                self.step += 1;
                self.tokens_seen += batch.batch_size * batch.seq_len;

                if (loss < self.best_loss) {
                    self.best_loss = loss;
                }

                if (self.step % self.config.telemetry.metrics_interval == 0) {
                    const elapsed_ms = std.time.milliTimestamp() - start_time;
                    const elapsed_s = @as(f64, @floatFromInt(elapsed_ms)) / 1000.0;
                    const tput = if (elapsed_s > 0.0)
                        @as(f64, @floatFromInt(self.tokens_seen)) / elapsed_s
                    else
                        0.0;
                    try self.logMetrics(loss, tput);
                }

                if (self.step % self.config.checkpoint.save_interval == 0) {
                    try self.saveCheckpoint();
                }
            } else {
                loader.reset();
                self.epoch += 1;
                std.log.info("Starting epoch {d}", .{self.epoch});
            }
        }

        std.log.info("Training completed at step {d}", .{self.step});
    }

    fn trainStep(self: *Self, batch: *const data_mod.Batch) !f32 {
        const loss = try self.forwardStep(batch);

        try self.backwardStep();

        const grads = try self.model.collectGradients(self.allocator);
        defer self.allocator.free(grads);

        const grad_norm = try self.clipper.clip(grads);

        const lr = self.scheduler.getLR();
        self.optimizer.setLR(lr);

        try self.optimizer.step(grads);

        self.scheduler.step();

        const now = std.time.timestamp();
        const elapsed_s = if (self.step > 0)
            @as(f64, @floatFromInt(self.tokens_seen)) / @as(f64, @floatFromInt(now))
        else
            0.0;

        try self.telemetry.logStep(.{
            .step = self.step,
            .tokens = self.tokens_seen,
            .loss = loss,
            .lr = lr,
            .grad_norm = grad_norm,
            .throughput = elapsed_s,
            .memory_used = 0,
            .memory_total = 0,
            .timestamp = now,
        });

        return loss;
    }

    pub fn smokeTest(self: *Self) !void {
        std.log.info("Running smoke test...", .{});

        const batch_size = 1;
        const seq_len = 16;

        var input_tokens = try self.allocator.alloc(u32, batch_size * seq_len);
        defer self.allocator.free(input_tokens);
        @memset(input_tokens, 1);

        var target_tokens = try self.allocator.alloc(u32, batch_size * seq_len);
        defer self.allocator.free(target_tokens);
        @memset(target_tokens, 2);

        const batch = data_mod.Batch{
            .input = input_tokens,
            .target = target_tokens,
            .batch_size = batch_size,
            .seq_len = seq_len,
        };

        for (0..3) |i| {
            const loss = try self.trainStep(&batch);
            std.log.info("Smoke test iteration {d}: loss = {d:.4}", .{ i, loss });
        }

        std.log.info("Smoke test passed!", .{});
    }

    pub fn resume(self: *Self, checkpoint_path: []const u8) !void {
        std.log.info("Resuming from checkpoint: {s}", .{checkpoint_path});

        const params = try self.model.collectParameters(self.allocator);
        defer self.allocator.free(params);

        const names = try self.model.getParameterNames(self.allocator);
        defer {
            for (names) |n| self.allocator.free(n);
            self.allocator.free(names);
        }

        const metadata = try self.checkpoint_manager.load(
            checkpoint_path,
            params,
            names,
        );

        self.step = metadata.step;
        self.epoch = metadata.epoch;
        self.tokens_seen = metadata.tokens_seen;
        self.best_loss = metadata.loss;

        std.log.info("Resumed from step {d}, epoch {d}", .{ self.step, self.epoch });
    }

    fn forwardStep(self: *Self, batch: *const data_mod.Batch) !f32 {
        const input_shape = Shape.init(&[_]usize{ batch.batch_size, batch.seq_len });

        var input_tensor = try Tensor.init(
            self.allocator,
            input_shape,
            .uint32,
            .cuda,
            @intCast(self.runtime.rank),
        );
        defer input_tensor.deinit();

        const input_ptr = input_tensor.typedPtr(u32) orelse return error.InvalidDType;
        @memcpy(input_ptr[0..batch.input.len], batch.input);

        const target_shape = Shape.init(&[_]usize{ batch.batch_size, batch.seq_len });
        var target_tensor = try Tensor.init(
            self.allocator,
            target_shape,
            .uint32,
            .cuda,
            @intCast(self.runtime.rank),
        );
        defer target_tensor.deinit();

        const target_ptr = target_tensor.typedPtr(u32) orelse return error.InvalidDType;
        @memcpy(target_ptr[0..batch.target.len], batch.target);

        const output = try self.model.forward(input_tensor);
        defer output.deinit();

        const loss = try self.model.computeLoss(output, target_tensor);

        return loss;
    }

    fn backwardStep(self: *Self) !void {
        try self.model.backward();
    }

    fn logMetrics(self: *Self, loss: f32, throughput: f64) !void {
        const lr = self.scheduler.getLR();

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
            .grad_norm = 0.0,
            .memory_used = 0,
            .memory_total = 0,
            .timestamp = std.time.timestamp(),
        });
    }

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

        const params = try self.model.collectParameters(self.allocator);
        defer self.allocator.free(params);

        const names = try self.model.getParameterNames(self.allocator);
        defer {
            for (names) |n| self.allocator.free(n);
            self.allocator.free(names);
        }

        _ = try self.checkpoint_manager.save(
            self.step,
            params,
            names,
            null,
            null,
            metadata,
        );

        std.log.info("Saved checkpoint at step {d}", .{self.step});
    }
};
