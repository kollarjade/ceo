const std = @import("std");

pub const EflaDType = c_int;
pub const EflaFp8Format = c_int;
pub const EflaMemoryKind = c_int;
pub const CudaStream = ?*anyopaque;

pub extern fn lion_step_cuda(
    param: ?*anyopaque,
    grad: ?*const anyopaque,
    momentum: ?*anyopaque,
    numel: usize,
    tensor_dtype: EflaDType,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn muon_step_cuda(
    param: ?*anyopaque,
    grad: ?*const anyopaque,
    momentum: ?*anyopaque,
    m: usize,
    n: usize,
    tensor_dtype: EflaDType,
    lr: f32,
    beta: f32,
    ns_iterations: usize,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn adamw_step_cuda(
    param: ?*anyopaque,
    grad: ?*const anyopaque,
    exp_avg: ?*anyopaque,
    exp_avg_sq: ?*anyopaque,
    numel: usize,
    tensor_dtype: EflaDType,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn clip_grad_norm_cuda(
    grads: [*]?*anyopaque,
    numels: [*]const usize,
    num_params: usize,
    tensor_dtype: EflaDType,
    max_norm: f32,
    global_norm: *f32,
    global_norm_memory_kind: EflaMemoryKind,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn quantize_fp8_cuda(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    scale: *f32,
    numel: usize,
    input_dtype: EflaDType,
    output_format: EflaFp8Format,
    scale_memory_kind: EflaMemoryKind,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn dequantize_fp8_cuda(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    scale: f32,
    numel: usize,
    input_format: EflaFp8Format,
    output_dtype: EflaDType,
    stream: CudaStream,
) callconv(.C) c_int;

fn checkedMul(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch @panic("dimension overflow");
}

fn requireEqual(actual: usize, expected: usize, name: []const u8) void {
    if (actual != expected) std.debug.panic("invalid {s}", .{name});
}

fn requirePositive(value: f32, name: []const u8) void {
    if (!(value > 0.0) or value != value) std.debug.panic("invalid {s}", .{name});
}

fn requireNonNegative(value: f32, name: []const u8) void {
    if (value < 0.0 or value != value) std.debug.panic("invalid {s}", .{name});
}

fn requireBeta(value: f32, name: []const u8) void {
    if (value < 0.0 or value >= 1.0 or value != value) std.debug.panic("invalid {s}", .{name});
}

pub fn lionStepCpu(
    param: []f32,
    grad: []const f32,
    momentum: []f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
) void {
    requireEqual(grad.len, param.len, "grad length");
    requireEqual(momentum.len, param.len, "momentum length");
    requireNonNegative(lr, "lr");
    requireBeta(beta1, "beta1");
    requireBeta(beta2, "beta2");
    requireNonNegative(weight_decay, "weight_decay");

    const one_minus_beta1 = @as(f32, 1.0) - beta1;
    const one_minus_beta2 = @as(f32, 1.0) - beta2;

    for (param, grad, momentum) |*p, g, *m| {
        const v = beta1 * m.* + one_minus_beta1 * g;
        m.* = beta2 * m.* + one_minus_beta2 * g;
        const sign_v: f32 = if (v > 0.0) 1.0 else if (v < 0.0) -1.0 else 0.0;
        p.* -= lr * sign_v + lr * weight_decay * p.*;
    }
}

pub fn adamWStepCpu(
    param: []f32,
    grad: []const f32,
    exp_avg: []f32,
    exp_avg_sq: []f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
) void {
    requireEqual(grad.len, param.len, "grad length");
    requireEqual(exp_avg.len, param.len, "exp_avg length");
    requireEqual(exp_avg_sq.len, param.len, "exp_avg_sq length");
    requireNonNegative(lr, "lr");
    requireBeta(beta1, "beta1");
    requireBeta(beta2, "beta2");
    requirePositive(eps, "eps");
    requireNonNegative(weight_decay, "weight_decay");
    if (step == 0) @panic("invalid step");

    const step_f = @as(f32, @floatFromInt(step));
    const beta1_t = std.math.pow(f32, beta1, step_f);
    const beta2_t = std.math.pow(f32, beta2, step_f);
    const bias_correction1 = @as(f32, 1.0) / (@as(f32, 1.0) - beta1_t);
    const bias_correction2 = @as(f32, 1.0) / (@as(f32, 1.0) - beta2_t);

    for (param, grad, exp_avg, exp_avg_sq) |*p, g, *ea, *eas| {
        ea.* = beta1 * ea.* + (@as(f32, 1.0) - beta1) * g;
        eas.* = beta2 * eas.* + (@as(f32, 1.0) - beta2) * g * g;

        const avg = ea.* * bias_correction1;
        const avg_sq = eas.* * bias_correction2;
        const denom = @sqrt(avg_sq) + eps;

        p.* -= lr * (avg / denom + weight_decay * p.*);
    }
}

pub fn newtonSchulzIteration(
    Y: []f32,
    temp: []f32,
    m: usize,
    n: usize,
) void {
    const mn = checkedMul(m, n);
    const nn = checkedMul(n, n);

    if (Y.len < mn) @panic("invalid Y length");
    if (temp.len < nn + mn) @panic("invalid temp length");

    const gram = temp[0..nn];
    const Y_new = temp[nn .. nn + mn];

    for (0..n) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..m) |k| {
                sum += Y[k * n + i] * Y[k * n + j];
            }
            gram[i * n + j] = sum;
        }
    }

    for (0..n) |i| {
        for (0..n) |j| {
            const identity: f32 = if (i == j) 3.0 else 0.0;
            gram[i * n + j] = identity - gram[i * n + j];
        }
    }

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..n) |k| {
                sum += Y[i * n + k] * gram[k * n + j];
            }
            Y_new[i * n + j] = @as(f32, 0.5) * sum;
        }
    }

    @memcpy(Y[0..mn], Y_new[0..mn]);
}

pub fn computeGradNorm(grads: []const []const f32) f64 {
    var norm_sq: f64 = 0.0;

    for (grads) |grad| {
        for (grad) |g| {
            const g64: f64 = g;
            norm_sq += g64 * g64;
        }
    }

    return @sqrt(norm_sq);
}

pub fn clipGradNormCpu(grads: []const []f32, max_norm: f32) f32 {
    requireNonNegative(max_norm, "max_norm");

    var norm_sq: f64 = 0.0;
    for (grads) |grad| {
        for (grad) |g| {
            const g64: f64 = g;
            norm_sq += g64 * g64;
        }
    }

    const norm = @sqrt(norm_sq);

    if (norm > @as(f64, max_norm) and norm > 0.0) {
        const scale = max_norm / @as(f32, @floatCast(norm));
        for (grads) |grad| {
            for (grad) |*g| {
                g.* *= scale;
            }
        }
    }

    return @as(f32, @floatCast(norm));
}

test "lionStepCpu" {
    var param = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const grad = [_]f32{ 0.1, 0.1, 0.1, 0.1 };
    var momentum = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    lionStepCpu(param[0..], grad[0..], momentum[0..], 0.01, 0.9, 0.99, 0.0);

    try std.testing.expect(param[0] != 1.0);
    try std.testing.expect(momentum[0] != 0.0);
}

test "adamWStepCpu" {
    var param = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const grad = [_]f32{ 0.1, 0.1, 0.1, 0.1 };
    var exp_avg = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var exp_avg_sq = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    adamWStepCpu(param[0..], grad[0..], exp_avg[0..], exp_avg_sq[0..], 0.01, 0.9, 0.999, 1e-8, 0.0, 1);

    try std.testing.expect(param[0] != 1.0);
}

test "clipGradNormCpu" {
    var grad1 = [_]f32{ 3.0, 4.0 };
    var grad2 = [_]f32{ 6.0, 8.0 };
    var grads = [_][]f32{ grad1[0..], grad2[0..] };

    const norm = clipGradNormCpu(grads[0..], 5.0);

    try std.testing.expectApproxEqRel(@as(f32, 11.18), norm, @as(f32, 0.01));

    var new_norm_sq: f32 = 0.0;
    for (grads) |grad| {
        for (grad) |g| {
            new_norm_sq += g * g;
        }
    }
    try std.testing.expectApproxEqRel(@as(f32, 5.0), @sqrt(new_norm_sq), @as(f32, 0.01));
}
