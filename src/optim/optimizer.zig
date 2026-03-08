const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const kernels = @import("../kernels/optim_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;
pub const BF16 = dtype_mod.BF16;

/// Optimizer parameter group
pub const ParamGroup = struct {
    params: []*Tensor,
    lr: f32,
    weight_decay: f32,
    optimizer_type: OptimizerType,
};

pub const OptimizerType = enum {
    lion,
    muon,
    lion_muon,
};

/// Lion Optimizer
///
/// Implementation of Lion (Evolved Sign Momentum) optimizer:
///
/// v_t = β1 * m_{t-1} + (1-β1) * g_t
/// m_t = β2 * m_{t-1} + (1-β2) * g_t
/// x_{t+1} = x_t - η * sign(v_t) - η * λ * x_t (decoupled weight decay)
///
/// Where:
/// - β1 = 0.95 (momentum for update)
/// - β2 = 0.98 (momentum for EMA)
/// - η = learning rate
/// - λ = weight decay
pub const Lion = struct {
    /// Parameters being optimized
    params: []*Tensor,
    /// Learning rate
    lr: f32,
    /// Beta1 (momentum for update direction)
    beta1: f32,
    /// Beta2 (momentum for EMA of gradients)
    beta2: f32,
    /// Weight decay
    weight_decay: f32,
    /// Momentum buffer (m)
    momentum: []*Tensor,
    /// Step count
    step: usize,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    /// Initialize Lion optimizer
    pub fn init(
        allocator: std.mem.Allocator,
        params: []*Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*Lion {
        const self = try allocator.create(Lion);
        errdefer allocator.destroy(self);

        // Initialize momentum buffers
        var momentum = try allocator.alloc(*Tensor, params.len);
        errdefer allocator.free(momentum);

        for (params, 0..) |param, i| {
            momentum[i] = try Tensor.zeros(allocator, param.shape, .bf16, device, device_id);
        }

        // Copy params slice
        var params_copy = try allocator.alloc(*Tensor, params.len);
        @memcpy(params_copy, params);

        self.* = .{
            .params = params_copy,
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .weight_decay = weight_decay,
            .momentum = momentum,
            .step = 0,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *Lion) void {
        for (self.momentum) |m| m.deinit();
        self.allocator.free(self.momentum);
        self.allocator.free(self.params);
        self.allocator.destroy(self);
    }

    /// Perform a single optimization step
    pub fn step(self: *Lion, grads: []*Tensor) !void {
        std.debug.assert(grads.len == self.params.len);

        self.step += 1;

        for (self.params, grads, self.momentum) |param, grad, m| {
            try self.stepParam(param, grad, m);
        }
    }

    fn stepParam(self: *Lion, param: *Tensor, grad: *Tensor, m: *Tensor) !void {
        const numel = param.shape.numel();

        if (self.device == .cuda) {
            try kernels.lionStepCuda(
                param.ptr(),
                grad.ptr(),
                m.ptr(),
                numel,
                self.lr,
                self.beta1,
                self.beta2,
                self.weight_decay,
            );
        } else {
            try self.stepParamCpu(param, grad, m);
        }
    }

    fn stepParamCpu(self: *Lion, param: *Tensor, grad: *Tensor, m: *Tensor) !void {
        const numel = param.shape.numel();

        const param_ptr = param.typedPtr(BF16).?;
        const grad_ptr = grad.typedPtr(BF16).?;
        const m_ptr = m.typedPtr(BF16).?;

        const one_minus_beta1 = 1.0 - self.beta1;
        const one_minus_beta2 = 1.0 - self.beta2;

        for (0..numel) |i| {
            const g = grad_ptr[i].toFloat32();
            const m_old = m_ptr[i].toFloat32();
            const p = param_ptr[i].toFloat32();

            // v_t = β1 * m_{t-1} + (1-β1) * g_t
            const v = self.beta1 * m_old + one_minus_beta1 * g;

            // m_t = β2 * m_{t-1} + (1-β2) * g_t
            const m_new = self.beta2 * m_old + one_minus_beta2 * g;

            // Update: x = x - lr * sign(v) - lr * weight_decay * x
            const sign_v: f32 = if (v > 0) 1.0 else if (v < 0) -1.0 else 0.0;
            const update = self.lr * sign_v + self.lr * self.weight_decay * p;
            const p_new = p - update;

            param_ptr[i] = BF16.fromFloat32(p_new);
            m_ptr[i] = BF16.fromFloat32(m_new);
        }
    }

    /// Zero gradients
    pub fn zeroGrad(self: *Lion) !void {
        for (self.params) |param| {
            if (param.grad) |g| {
                try g.zero_();
            }
        }
    }
};

/// Muon Optimizer (Matrix Sign Momentum)
///
/// Implementation for matrix parameters:
///
/// G_t = (1/B) Σ_b G_t^b
/// B_t = β * B_{t-1} + G_t
/// O_t = NewtonSchulz(B_t)
/// X_{t+1} = X_t - η * O_t
///
/// The Newton-Schulz iteration computes the matrix sign / orthogonal factor:
/// Y_0 = B / ||B||_F
/// Y_{k+1} = 0.5 * Y_k * (3I - Y_k^2)
pub const Muon = struct {
    /// Matrix parameters being optimized
    params: []*Tensor,
    /// Learning rate
    lr: f32,
    /// Momentum coefficient
    beta: f32,
    /// Newton-Schulz iteration count
    ns_iterations: usize,
    /// Momentum buffer (B)
    momentum: []*Tensor,
    /// Step count
    step: usize,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    /// Initialize Muon optimizer
    pub fn init(
        allocator: std.mem.Allocator,
        params: []*Tensor,
        lr: f32,
        beta: f32,
        ns_iterations: usize,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*Muon {
        const self = try allocator.create(Muon);
        errdefer allocator.destroy(self);

        // Initialize momentum buffers
        var momentum = try allocator.alloc(*Tensor, params.len);
        errdefer allocator.free(momentum);

        for (params, 0..) |param, i| {
            momentum[i] = try Tensor.zeros(allocator, param.shape, .bf16, device, device_id);
        }

        // Copy params slice
        var params_copy = try allocator.alloc(*Tensor, params.len);
        @memcpy(params_copy, params);

        self.* = .{
            .params = params_copy,
            .lr = lr,
            .beta = beta,
            .ns_iterations = ns_iterations,
            .momentum = momentum,
            .step = 0,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *Muon) void {
        for (self.momentum) |m| m.deinit();
        self.allocator.free(self.momentum);
        self.allocator.free(self.params);
        self.allocator.destroy(self);
    }

    /// Perform optimization step
    pub fn step(self: *Muon, grads: []*Tensor) !void {
        std.debug.assert(grads.len == self.params.len);

        self.step += 1;

        for (self.params, grads, self.momentum) |param, grad, m| {
            try self.stepParam(param, grad, m);
        }
    }

    fn stepParam(self: *Muon, param: *Tensor, grad: *Tensor, m: *Tensor) !void {
        // param is shape (M, N), treat as matrix
        const M = param.shape.dim(0);
        const N = param.shape.dim(1);

        if (self.device == .cuda) {
            try kernels.muonStepCuda(
                param.ptr(),
                grad.ptr(),
                m.ptr(),
                M,
                N,
                self.lr,
                self.beta,
                self.ns_iterations,
            );
        } else {
            try self.stepParamCpu(param, grad, m, M, N);
        }
    }

    fn stepParamCpu(self: *Muon, param: *Tensor, grad: *Tensor, m: *Tensor, M: usize, N: usize) !void {
        const param_ptr = param.typedPtr(BF16).?;
        const grad_ptr = grad.typedPtr(BF16).?;
        const m_ptr = m.typedPtr(BF16).?;

        // Update momentum: B_t = β * B_{t-1} + G_t
        for (0..M * N) |i| {
            const m_val = m_ptr[i].toFloat32();
            const g_val = grad_ptr[i].toFloat32();
            m_ptr[i] = BF16.fromFloat32(self.beta * m_val + g_val);
        }

        // Compute Newton-Schulz iteration
        // Y_0 = B / ||B||_F
        var frobenius_sq: f32 = 0.0;
        for (0..M * N) |i| {
            const val = m_ptr[i].toFloat32();
            frobenius_sq += val * val;
        }
        const frobenius = @sqrt(frobenius_sq);

        if (frobenius < 1e-8) {
            // Skip if gradient is too small
            return;
        }

        const scale = 1.0 / frobenius;

        // Y_0 = B / ||B||_F
        // We'll store Y in a temporary buffer
        var Y = try self.allocator.alloc(f32, M * N);
        defer self.allocator.free(Y);

        for (0..M * N) |i| {
            Y[i] = m_ptr[i].toFloat32() * scale;
        }

        // Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3I - Y_k^T Y_k)
        // Simplified for now - full implementation needs matrix multiply
        for (0..self.ns_iterations) |_| {
            // Compute Y_k^T Y_k (N x N matrix)
            var YtY = try self.allocator.alloc(f32, N * N);
            defer self.allocator.free(YtY);
            @memset(YtY, 0.0);

            // Y_k^T Y_k
            for (0..M) |i| {
                for (0..N) |j| {
                    for (0..N) |k| {
                        YtY[j * N + k] += Y[i * N + j] * Y[i * N + k];
                    }
                }
            }

            // Y_new = 0.5 * Y * (3I - Y^T Y)
            var Y_new = try self.allocator.alloc(f32, M * N);
            defer self.allocator.free(Y_new);

            for (0..M) |i| {
                for (0..N) |j| {
                    var sum: f32 = 0.0;
                    for (0..N) |k| {
                        const identity_factor: f32 = if (j == k) 3.0 else 0.0;
                        sum += Y[i * N + k] * (identity_factor - YtY[k * N + j]);
                    }
                    Y_new[i * N + j] = 0.5 * sum;
                }
            }

            @memcpy(Y, Y_new);
        }

        // Update: X = X - lr * O_t (where O_t is the orthogonal factor)
        for (0..M * N) |i| {
            const p = param_ptr[i].toFloat32();
            param_ptr[i] = BF16.fromFloat32(p - self.lr * Y[i]);
        }
    }

    pub fn zeroGrad(self: *Muon) !void {
        for (self.params) |param| {
            if (param.grad) |g| {
                try g.zero_();
            }
        }
    }
};

/// Combined Lion + Muon Optimizer
/// Applies Muon to matrix parameters and Lion to vector/scalar parameters
pub const LionMuonOptimizer = struct {
    lion: *Lion,
    muon: *Muon,
    matrix_param_indices: []usize,
    vector_param_indices: []usize,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        params: []*Tensor,
        lr: f32,
        lion_beta1: f32,
        lion_beta2: f32,
        muon_beta: f32,
        muon_iterations: usize,
        weight_decay: f32,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*LionMuonOptimizer {
        const self = try allocator.create(LionMuonOptimizer);
        errdefer allocator.destroy(self);

        // Separate params into matrix and vector
        var matrix_params = std.ArrayList(*Tensor).init(allocator);
        defer matrix_params.deinit();
        var vector_params = std.ArrayList(*Tensor).init(allocator);
        defer vector_params.deinit();

        var matrix_indices = std.ArrayList(usize).init(allocator);
        var vector_indices = std.ArrayList(usize).init(allocator);

        for (params, 0..) |param, idx| {
            if (param.shape.ndim >= 2) {
                try matrix_params.append(param);
                try matrix_indices.append(idx);
            } else {
                try vector_params.append(param);
                try vector_indices.append(idx);
            }
        }

        const lion = try Lion.init(
            allocator,
            try vector_params.toOwnedSlice(),
            lr,
            lion_beta1,
            lion_beta2,
            weight_decay,
            device,
            device_id,
        );
        errdefer lion.deinit();

        const muon = try Muon.init(
            allocator,
            try matrix_params.toOwnedSlice(),
            lr,
            muon_beta,
            muon_iterations,
            device,
            device_id,
        );
        errdefer muon.deinit();

        self.* = .{
            .lion = lion,
            .muon = muon,
            .matrix_param_indices = try matrix_indices.toOwnedSlice(),
            .vector_param_indices = try vector_indices.toOwnedSlice(),
            .allocator = allocator,
        };

        return self;
    }

    pub fn deinit(self: *LionMuonOptimizer) void {
        self.lion.deinit();
        self.muon.deinit();
        self.allocator.free(self.matrix_param_indices);
        self.allocator.free(self.vector_param_indices);
        self.allocator.destroy(self);
    }

    pub fn step(self: *LionMuonOptimizer, grads: []*Tensor) !void {
        // Split gradients
        var matrix_grads = try self.allocator.alloc(*Tensor, self.muon.params.len);
        defer self.allocator.free(matrix_grads);
        var vector_grads = try self.allocator.alloc(*Tensor, self.lion.params.len);
        defer self.allocator.free(vector_grads);

        for (self.matrix_param_indices, 0..) |idx, i| {
            matrix_grads[i] = grads[idx];
        }
        for (self.vector_param_indices, 0..) |idx, i| {
            vector_grads[i] = grads[idx];
        }

        try self.muon.step(matrix_grads);
        try self.lion.step(vector_grads);
    }

    pub fn zeroGrad(self: *LionMuonOptimizer) !void {
        try self.lion.zeroGrad();
        try self.muon.zeroGrad();
    }
};

/// Learning rate scheduler
pub const LRScheduler = struct {
    base_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    schedule_type: ScheduleType,
    current_step: usize,

    pub const ScheduleType = enum {
        constant,
        linear_warmup,
        cosine,
        linear_warmup_cosine,
    };

    pub fn init(
        base_lr: f32,
        min_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
        schedule_type: ScheduleType,
    ) LRScheduler {
        return .{
            .base_lr = base_lr,
            .min_lr = min_lr,
            .warmup_steps = warmup_steps,
            .total_steps = total_steps,
            .schedule_type = schedule_type,
            .current_step = 0,
        };
    }

    /// Get current learning rate
    pub fn getLR(self: *LRScheduler) f32 {
        return switch (self.schedule_type) {
            .constant => self.base_lr,
            .linear_warmup => self.linearWarmup(),
            .cosine => self.cosine(),
            .linear_warmup_cosine => self.linearWarmupCosine(),
        };
    }

    /// Step the scheduler
    pub fn step(self: *LRScheduler) void {
        self.current_step += 1;
    }

    fn linearWarmup(self: *LRScheduler) f32 {
        if (self.current_step < self.warmup_steps) {
            return self.base_lr * @as(f32, @floatFromInt(self.current_step)) /
                @as(f32, @floatFromInt(self.warmup_steps));
        }
        return self.base_lr;
    }

    fn cosine(self: *LRScheduler) f32 {
        const progress = @as(f64, @floatFromInt(self.current_step)) /
            @as(f64, @floatFromInt(self.total_steps));
        const cosine_factor = 0.5 * (1.0 + @cos(std.math.pi * progress));
        return self.min_lr + (self.base_lr - self.min_lr) * @floatCast(cosine_factor);
    }

    fn linearWarmupCosine(self: *LRScheduler) f32 {
        if (self.current_step < self.warmup_steps) {
            return self.linearWarmup();
        }
        return self.cosine();
    }
};

/// Gradient clipping
pub const GradientClipper = struct {
    max_norm: f32,
    clip_type: ClipType,

    pub const ClipType = enum {
        norm, // Clip by global norm
        value, // Clip by value
    };

    pub fn init(max_norm: f32, clip_type: ClipType) GradientClipper {
        return .{
            .max_norm = max_norm,
            .clip_type = clip_type,
        };
    }

    /// Clip gradients in place
    pub fn clip(self: GradientClipper, grads: []*Tensor) !f32 {
        return switch (self.clip_type) {
            .norm => self.clipByNorm(grads),
            .value => self.clipByValue(grads),
        };
    }

    fn clipByNorm(self: GradientClipper, grads: []*Tensor) !f32 {
        // Compute global norm
        var global_norm_sq: f64 = 0.0;
        for (grads) |grad| {
            const numel = grad.shape.numel();
            const ptr = grad.typedPtr(BF16).?;
            for (0..numel) |i| {
                const val = ptr[i].toFloat32();
                global_norm_sq += val * val;
            }
        }
        const global_norm = @sqrt(global_norm_sq);

        // Clip if necessary
        if (global_norm > self.max_norm) {
            const scale = self.max_norm / global_norm;
            for (grads) |grad| {
                const numel = grad.shape.numel();
                const ptr = grad.typedPtr(BF16).?;
                for (0..numel) |i| {
                    ptr[i] = BF16.fromFloat32(ptr[i].toFloat32() * scale);
                }
            }
        }

        return @floatCast(global_norm);
    }

    fn clipByValue(self: GradientClipper, grads: []*Tensor) !f32 {
        var max_val: f32 = 0.0;
        for (grads) |grad| {
            const numel = grad.shape.numel();
            const ptr = grad.typedPtr(BF16).?;
            for (0..numel) |i| {
                var val = ptr[i].toFloat32();
                if (val > self.max_norm) {
                    ptr[i] = BF16.fromFloat32(self.max_norm);
                    max_val = self.max_norm;
                } else if (val < -self.max_norm) {
                    ptr[i] = BF16.fromFloat32(-self.max_norm);
                    max_val = self.max_norm;
                } else if (@abs(val) > max_val) {
                    max_val = @abs(val);
                }
            }
        }
        return max_val;
    }
};

test "Lion optimizer step" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    const shape = Shape.init(&[_]usize{ 4, 8 });
    var param = try Tensor.randNormal(gpa.allocator(), shape, .bf16, .cpu, 0, &rng, 0.0, 1.0);
    defer param.deinit();

    var params = [_]*Tensor{param};

    var opt = try Lion.init(gpa.allocator(), &params, 0.001, 0.95, 0.98, 0.01, .cpu, 0);
    defer opt.deinit();

    var grad = try Tensor.randNormal(gpa.allocator(), shape, .bf16, .cpu, 0, &rng, 0.0, 0.1);
    defer grad.deinit();

    var grads = [_]*Tensor{grad};

    try opt.step(&grads);
    try std.testing.expectEqual(@as(usize, 1), opt.step);
}

test "LRScheduler cosine" {
    var scheduler = LRScheduler.init(1e-4, 1e-5, 1000, 10000, .cosine);

    scheduler.current_step = 0;
    try std.testing.expectApproxEqRel(@as(f32, 1e-4), scheduler.getLR(), 0.01);

    scheduler.current_step = 5000;
    const mid_lr = scheduler.getLR();
    try std.testing.expect(mid_lr > 1e-5 and mid_lr < 1e-4);

    scheduler.current_step = 10000;
    try std.testing.expectApproxEqRel(@as(f32, 1e-5), scheduler.getLR(), 0.01);
}
