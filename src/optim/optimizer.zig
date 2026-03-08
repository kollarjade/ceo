const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const kernels = @import("../kernels/optim_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;
pub const BF16 = dtype_mod.BF16;

pub const OptimError = error{
    UnsupportedDtype,
    ShapeMismatch,
    InvalidArgument,
    InvalidDevice,
    DeviceMismatch,
    NotContiguous,
};

pub const Lion = struct {
    params: []*Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    momentum: []*Tensor,
    stepCount: usize,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        params: []*Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
    ) !*Lion {
        const self = try allocator.create(Lion);
        errdefer allocator.destroy(self);

        var momentum = try allocator.alloc(*Tensor, params.len);
        errdefer allocator.free(momentum);

        var initialized_count: usize = 0;
        errdefer {
            for (0..initialized_count) |i| {
                momentum[i].deinit();
            }
        }

        for (params, 0..) |param, i| {
            momentum[i] = try Tensor.zeros(allocator, param.shape, .f32, param.device, param.device_id);
            initialized_count += 1;
        }

        const params_copy = try allocator.alloc(*Tensor, params.len);
        errdefer allocator.free(params_copy);
        @memcpy(params_copy, params);

        self.* = .{
            .params = params_copy,
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .weight_decay = weight_decay,
            .momentum = momentum,
            .stepCount = 0,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Lion) void {
        for (self.momentum) |m| {
            m.deinit();
        }
        self.allocator.free(self.momentum);
        self.allocator.free(self.params);
        self.allocator.destroy(self);
    }

    pub fn step(self: *Lion, grads: []*Tensor) !void {
        if (grads.len != self.params.len) return OptimError.InvalidArgument;
        self.stepCount += 1;
        for (self.params, grads, self.momentum) |param, grad, m| {
            if (!param.shape.eql(grad.shape)) return OptimError.ShapeMismatch;
            if (!param.shape.eql(m.shape)) return OptimError.ShapeMismatch;
            try self.stepParam(param, grad, m);
        }
    }

    fn stepParam(self: *Lion, param: *Tensor, grad: *Tensor, m: *Tensor) !void {
        if (param.device != grad.device or param.device != m.device) return OptimError.DeviceMismatch;

        if (param.device == .cuda) {
            try kernels.lionStepCuda(
                param.ptr(),
                grad.ptr(),
                m.ptr(),
                param.shape.numel(),
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
        if (param.dtype != .bf16 or grad.dtype != .bf16 or m.dtype != .f32) return OptimError.UnsupportedDtype;

        const numel = param.shape.numel();
        const param_ptr = param.typedPtr(BF16) orelse return OptimError.NotContiguous;
        const grad_ptr = grad.typedPtr(BF16) orelse return OptimError.NotContiguous;
        const m_ptr = m.typedPtr(f32) orelse return OptimError.NotContiguous;

        const one_minus_beta1 = 1.0 - self.beta1;
        const one_minus_beta2 = 1.0 - self.beta2;

        for (0..numel) |i| {
            const g = grad_ptr[i].toFloat32();
            const m_old = m_ptr[i];
            const p = param_ptr[i].toFloat32();

            const v = self.beta1 * m_old + one_minus_beta1 * g;
            const m_new = self.beta2 * m_old + one_minus_beta2 * g;

            var sign_v: f32 = 0.0;
            if (v > 1e-7) {
                sign_v = 1.0;
            } else if (v < -1e-7) {
                sign_v = -1.0;
            }

            const update_val = self.lr * sign_v + self.lr * self.weight_decay * p;
            const p_new = p - update_val;

            param_ptr[i] = BF16.fromFloat32(p_new);
            m_ptr[i] = m_new;
        }
    }

    pub fn zeroGrad(self: *Lion) !void {
        for (self.params) |param| {
            if (param.grad) |g| {
                try g.zero_();
            }
        }
    }
};

pub const Muon = struct {
    params: []*Tensor,
    lr: f32,
    beta: f32,
    ns_iterations: usize,
    momentum: []*Tensor,
    stepCount: usize,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        params: []*Tensor,
        lr: f32,
        beta: f32,
        ns_iterations: usize,
    ) !*Muon {
        const self = try allocator.create(Muon);
        errdefer allocator.destroy(self);

        var momentum = try allocator.alloc(*Tensor, params.len);
        errdefer allocator.free(momentum);

        var initialized_count: usize = 0;
        errdefer {
            for (0..initialized_count) |i| {
                momentum[i].deinit();
            }
        }

        for (params, 0..) |param, i| {
            momentum[i] = try Tensor.zeros(allocator, param.shape, .f32, param.device, param.device_id);
            initialized_count += 1;
        }

        const params_copy = try allocator.alloc(*Tensor, params.len);
        errdefer allocator.free(params_copy);
        @memcpy(params_copy, params);

        self.* = .{
            .params = params_copy,
            .lr = lr,
            .beta = beta,
            .ns_iterations = ns_iterations,
            .momentum = momentum,
            .stepCount = 0,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Muon) void {
        for (self.momentum) |m| {
            m.deinit();
        }
        self.allocator.free(self.momentum);
        self.allocator.free(self.params);
        self.allocator.destroy(self);
    }

    pub fn step(self: *Muon, grads: []*Tensor) !void {
        if (grads.len != self.params.len) return OptimError.InvalidArgument;
        self.stepCount += 1;
        for (self.params, grads, self.momentum) |param, grad, m| {
            if (!param.shape.eql(grad.shape)) return OptimError.ShapeMismatch;
            if (!param.shape.eql(m.shape)) return OptimError.ShapeMismatch;
            if (param.shape.ndim != 2) return OptimError.InvalidArgument;
            try self.stepParam(param, grad, m);
        }
    }

    fn stepParam(self: *Muon, param: *Tensor, grad: *Tensor, m: *Tensor) !void {
        if (param.device != grad.device or param.device != m.device) return OptimError.DeviceMismatch;
        
        const M = param.shape.dim(0);
        const N = param.shape.dim(1);

        if (param.device == .cuda) {
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
        if (param.dtype != .bf16 or grad.dtype != .bf16 or m.dtype != .f32) return OptimError.UnsupportedDtype;

        const param_ptr = param.typedPtr(BF16) orelse return OptimError.NotContiguous;
        const grad_ptr = grad.typedPtr(BF16) orelse return OptimError.NotContiguous;
        const m_ptr = m.typedPtr(f32) orelse return OptimError.NotContiguous;

        for (0..M * N) |i| {
            const m_val = m_ptr[i];
            const g_val = grad_ptr[i].toFloat32();
            m_ptr[i] = self.beta * m_val + g_val;
        }

        var frobenius_sq: f64 = 0.0;
        for (0..M * N) |i| {
            const val = @as(f64, @floatCast(m_ptr[i]));
            frobenius_sq += val * val;
        }
        const frobenius = @as(f32, @floatCast(@sqrt(frobenius_sq)));
        
        if (frobenius < 1e-8) {
            for (0..M * N) |i| {
                const p = param_ptr[i].toFloat32();
                param_ptr[i] = BF16.fromFloat32(p - self.lr * m_ptr[i]);
            }
            return;
        }

        const scale = 1.0 / frobenius;
        var Y = try self.allocator.alloc(f32, M * N);
        defer self.allocator.free(Y);
        for (0..M * N) |i| {
            Y[i] = m_ptr[i] * scale;
        }

        var YtY = try self.allocator.alloc(f32, N * N);
        defer self.allocator.free(YtY);
        
        var Y_new = try self.allocator.alloc(f32, M * N);
        defer self.allocator.free(Y_new);

        var iter: usize = 0;
        while (iter < self.ns_iterations) : (iter += 1) {
            @memset(YtY, 0.0);
            for (0..N) |j| {
                for (0..N) |k| {
                    var sum: f32 = 0.0;
                    for (0..M) |i| {
                        sum += Y[i * N + j] * Y[i * N + k];
                    }
                    YtY[j * N + k] = sum;
                }
            }

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

            const temp = Y;
            Y = Y_new;
            Y_new = temp;
        }

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
    ) !*LionMuonOptimizer {
        const self = try allocator.create(LionMuonOptimizer);
        errdefer allocator.destroy(self);

        var matrix_params = std.ArrayList(*Tensor).init(allocator);
        defer matrix_params.deinit();
        var vector_params = std.ArrayList(*Tensor).init(allocator);
        defer vector_params.deinit();

        var matrix_indices = std.ArrayList(usize).init(allocator);
        errdefer matrix_indices.deinit();
        var vector_indices = std.ArrayList(usize).init(allocator);
        errdefer vector_indices.deinit();

        for (params, 0..) |param, idx| {
            if (param.shape.ndim == 2) {
                try matrix_params.append(param);
                try matrix_indices.append(idx);
            } else {
                try vector_params.append(param);
                try vector_indices.append(idx);
            }
        }

        const lion = try Lion.init(
            allocator,
            vector_params.items,
            lr,
            lion_beta1,
            lion_beta2,
            weight_decay,
        );
        errdefer lion.deinit();

        const muon = try Muon.init(
            allocator,
            matrix_params.items,
            lr,
            muon_beta,
            muon_iterations,
        );
        errdefer muon.deinit();

        const matrix_idx_slice = try matrix_indices.toOwnedSlice();
        errdefer allocator.free(matrix_idx_slice);
        
        const vector_idx_slice = try vector_indices.toOwnedSlice();
        errdefer allocator.free(vector_idx_slice);

        self.* = .{
            .lion = lion,
            .muon = muon,
            .matrix_param_indices = matrix_idx_slice,
            .vector_param_indices = vector_idx_slice,
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
        const total_params = self.matrix_param_indices.len + self.vector_param_indices.len;
        if (grads.len != total_params) return OptimError.InvalidArgument;

        var matrix_grads = try self.allocator.alloc(*Tensor, self.muon.params.len);
        defer self.allocator.free(matrix_grads);
        
        var vector_grads = try self.allocator.alloc(*Tensor, self.lion.params.len);
        defer self.allocator.free(vector_grads);
        
        for (self.matrix_param_indices, 0..) |idx, i| {
            if (idx >= grads.len) return OptimError.InvalidArgument;
            matrix_grads[i] = grads[idx];
        }
        for (self.vector_param_indices, 0..) |idx, i| {
            if (idx >= grads.len) return OptimError.InvalidArgument;
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

    pub fn getLR(self: *LRScheduler) f32 {
        return switch (self.schedule_type) {
            .constant => self.base_lr,
            .linear_warmup => self.linearWarmup(),
            .cosine => self.cosine(),
            .linear_warmup_cosine => self.linearWarmupCosine(),
        };
    }

    pub fn step(self: *LRScheduler) void {
        self.current_step += 1;
    }

    fn linearWarmup(self: *LRScheduler) f32 {
        if (self.warmup_steps == 0) return self.base_lr;
        if (self.current_step < self.warmup_steps) {
            const step_clamped = @max(@as(usize, 1), self.current_step);
            const num = @as(f32, @floatFromInt(step_clamped));
            const denom = @as(f32, @floatFromInt(self.warmup_steps));
            return self.base_lr * num / denom;
        }
        return self.base_lr;
    }

    fn cosine(self: *LRScheduler) f32 {
        if (self.total_steps == 0) return self.min_lr;
        const progress = @as(f64, @floatFromInt(self.current_step)) /
            @as(f64, @floatFromInt(self.total_steps));
        const clamped = @min(@max(progress, 0.0), 1.0);
        const cosine_factor = 0.5 * (1.0 + @cos(std.math.pi * @floatCast(clamped)));
        return self.min_lr + (self.base_lr - self.min_lr) * @floatCast(cosine_factor);
    }

    fn linearWarmupCosine(self: *LRScheduler) f32 {
        if (self.current_step < self.warmup_steps) {
            return self.linearWarmup();
        }
        const adjusted_total = if (self.total_steps > self.warmup_steps) self.total_steps - self.warmup_steps else 1;
        const adjusted_step = if (self.current_step > self.warmup_steps) self.current_step - self.warmup_steps else 0;
        if (adjusted_total == 0) return self.min_lr;
        
        const progress = @as(f64, @floatFromInt(adjusted_step)) /
            @as(f64, @floatFromInt(adjusted_total));
        const clamped = @min(@max(progress, 0.0), 1.0);
        const cosine_factor = 0.5 * (1.0 + @cos(std.math.pi * @floatCast(clamped)));
        return self.min_lr + (self.base_lr - self.min_lr) * @floatCast(cosine_factor);
    }
};

pub const GradientClipper = struct {
    max_norm: f32,
    clip_type: ClipType,

    pub const ClipType = enum {
        norm,
        value,
    };

    pub fn init(max_norm: f32, clip_type: ClipType) GradientClipper {
        return .{
            .max_norm = max_norm,
            .clip_type = clip_type,
        };
    }

    pub fn clip(self: GradientClipper, grads: []*Tensor) !f32 {
        return switch (self.clip_type) {
            .norm => self.clipByNorm(grads),
            .value => self.clipByValue(grads),
        };
    }

    fn clipByNorm(self: GradientClipper, grads: []*Tensor) !f32 {
        var global_norm_sq: f64 = 0.0;
        for (grads) |grad| {
            if (grad.dtype != .bf16) return OptimError.UnsupportedDtype;
            const numel = grad.shape.numel();
            const ptr = grad.typedPtr(BF16) orelse return OptimError.NotContiguous;
            for (0..numel) |i| {
                const val = @as(f64, @floatCast(ptr[i].toFloat32()));
                global_norm_sq += val * val;
            }
        }
        const global_norm = @as(f32, @floatCast(@sqrt(global_norm_sq)));
        if (global_norm > self.max_norm and global_norm > 1e-8) {
            const scale = self.max_norm / global_norm;
            for (grads) |grad| {
                const numel = grad.shape.numel();
                const ptr = grad.typedPtr(BF16) orelse return OptimError.NotContiguous;
                for (0..numel) |i| {
                    ptr[i] = BF16.fromFloat32(ptr[i].toFloat32() * scale);
                }
            }
        }
        return global_norm;
    }

    fn clipByValue(self: GradientClipper, grads: []*Tensor) !f32 {
        var max_val: f32 = 0.0;
        for (grads) |grad| {
            if (grad.dtype != .bf16) return OptimError.UnsupportedDtype;
            const numel = grad.shape.numel();
            const ptr = grad.typedPtr(BF16) orelse return OptimError.NotContiguous;
            for (0..numel) |i| {
                const val = ptr[i].toFloat32();
                const abs_val = @abs(val);
                if (abs_val > max_val) {
                    max_val = abs_val;
                }
                
                if (val > self.max_norm) {
                    ptr[i] = BF16.fromFloat32(self.max_norm);
                } else if (val < -self.max_norm) {
                    ptr[i] = BF16.fromFloat32(-self.max_norm);
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

    const shape = Shape.init(&[_]usize{4, 8});
    var param = try Tensor.randNormal(gpa.allocator(), shape, .bf16, .cpu, 0, &rng, 0.0, 1.0);
    defer param.deinit();

    var params = [_]*Tensor{&param};

    var opt = try Lion.init(gpa.allocator(), &params, 0.001, 0.95, 0.98, 0.01);
    defer opt.deinit();

    var grad = try Tensor.randNormal(gpa.allocator(), shape, .bf16, .cpu, 0, &rng, 0.0, 0.1);
    defer grad.deinit();

    var grads = [_]*Tensor{&grad};

    try opt.step(&grads);
    try std.testing.expectEqual(@as(usize, 1), opt.stepCount);
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

