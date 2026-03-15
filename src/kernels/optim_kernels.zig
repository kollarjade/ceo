const std = @import("std");

pub extern "cuda_optim" fn lionStepCuda(
    param: ?*anyopaque,
    grad: ?*const anyopaque,
    momentum: ?*anyopaque,
    numel: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
) callconv(.C) c_int;

pub extern "cuda_optim" fn muonStepCuda(
    param: ?*anyopaque,
    grad: ?*const anyopaque,
    momentum: ?*anyopaque,
    m: usize,
    n: usize,
    lr: f32,
    beta: f32,
    ns_iterations: usize,
) callconv(.C) c_int;

pub extern "cuda_optim" fn adamWStepCuda(
    param: ?*anyopaque,
    grad: ?*const anyopaque,
    exp_avg: ?*anyopaque,
    exp_avg_sq: ?*anyopaque,
    numel: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
) callconv(.C) c_int;

pub extern "cuda_optim" fn clipGradNormCuda(
    grads: [*]?*anyopaque,
    num_params: usize,
    numels: [*]const usize,
    max_norm: f32,
    global_norm: *f32,
) callconv(.C) c_int;

pub extern "cuda_optim" fn clipGradValueCuda(
    grad: ?*anyopaque,
    numel: usize,
    max_value: f32,
) callconv(.C) c_int;

pub extern "cuda_optim" fn quantizeFP8Cuda(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    scale: *f32,
    numel: usize,
) callconv(.C) c_int;

pub extern "cuda_optim" fn dequantizeFP8Cuda(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    scale: f32,
    numel: usize,
) callconv(.C) c_int;

pub fn lionStepCpu(
    param: []f32,
    grad: []const f32,
    momentum: []f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
) void {
    const len = @min(param.len, @min(grad.len, momentum.len));
    const one_minus_beta1 = 1.0 - beta1;
    const one_minus_beta2 = 1.0 - beta2;

    for (0..len) |idx| {
        const p = param[idx];
        const g = grad[idx];
        const m = momentum[idx];

        const v = beta1 * m + one_minus_beta1 * g;

        momentum[idx] = beta2 * m + one_minus_beta2 * g;

        const sign_v: f32 = if (v > 0.0) @as(f32, 1.0) else if (v < 0.0) @as(f32, -1.0) else @as(f32, 0.0);
        param[idx] = p - lr * sign_v - lr * weight_decay * p;
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
    if (step == 0) return;
    const len = @min(param.len, @min(grad.len, @min(exp_avg.len, exp_avg_sq.len)));
    const step_f: f32 = @floatFromInt(step);
    const beta1_t = std.math.pow(f32, beta1, step_f);
    const beta2_t = std.math.pow(f32, beta2, step_f);
    const bias_correction1 = @as(f32, 1.0) / (@as(f32, 1.0) - beta1_t);
    const bias_correction2 = @as(f32, 1.0) / (@as(f32, 1.0) - beta2_t);

    for (0..len) |idx| {
        const g = grad[idx];

        exp_avg[idx] = beta1 * exp_avg[idx] + (@as(f32, 1.0) - beta1) * g;

        exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (@as(f32, 1.0) - beta2) * g * g;

        const avg = exp_avg[idx] * bias_correction1;
        const avg_sq = exp_avg_sq[idx] * bias_correction2;

        const denom = @sqrt(avg_sq) + eps;
        param[idx] -= lr * (avg / denom + weight_decay * param[idx]);
    }
}

pub fn newtonSchulzIteration(
    Y: []f32,
    temp: []f32,
    m: usize,
    n: usize,
    allocator: std.mem.Allocator,
) void {
    if (m * n == 0) return;
    if (temp.len < n * n) return;
    if (Y.len < m * n) return;

    for (0..n) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..m) |k| {
                sum += Y[k * n + i] * Y[k * n + j];
            }
            temp[i * n + j] = sum;
        }
    }

    for (0..n) |i| {
        for (0..n) |j| {
            const identity: f32 = if (i == j) @as(f32, 3.0) else @as(f32, 0.0);
            temp[i * n + j] = identity - temp[i * n + j];
        }
    }

    const y_new = allocator.alloc(f32, m * n) catch return;
    defer allocator.free(y_new);

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..n) |k| {
                sum += Y[i * n + k] * temp[k * n + j];
            }
            y_new[i * n + j] = 0.5 * sum;
        }
    }

    @memcpy(Y[0 .. m * n], y_new[0 .. m * n]);
}

pub fn computeGradNorm(grads: []const []const f32) f64 {
    var norm_sq: f64 = 0.0;

    for (grads) |grad| {
        for (grad) |g| {
            norm_sq += @as(f64, g) * @as(f64, g);
        }
    }

    return @sqrt(norm_sq);
}

pub fn clipGradNormCpu(grads: [][]f32, max_norm: f32) f32 {
    var norm_sq: f64 = 0.0;
    for (grads) |grad| {
        for (grad) |g| {
            norm_sq += @as(f64, g) * @as(f64, g);
        }
    }
    const norm: f64 = @sqrt(norm_sq);

    if (norm > @as(f64, max_norm)) {
        const scale: f32 = max_norm / @as(f32, @floatCast(norm));
        for (grads) |grad| {
            for (grad) |*g| {
                g.* *= scale;
            }
        }
    }

    return @floatCast(norm);
}

test "lionStepCpu" {
    var param = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const grad = [_]f32{ 0.1, 0.1, 0.1, 0.1 };
    var momentum = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    lionStepCpu(&param, &grad, &momentum, 0.01, 0.9, 0.99, 0.0);

    try std.testing.expect(param[0] != 1.0);

    try std.testing.expect(momentum[0] != 0.0);
}

test "adamWStepCpu" {
    var param = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const grad = [_]f32{ 0.1, 0.1, 0.1, 0.1 };
    var exp_avg = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var exp_avg_sq = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    adamWStepCpu(&param, &grad, &exp_avg, &exp_avg_sq, 0.01, 0.9, 0.999, 1e-8, 0.0, 1);

    try std.testing.expect(param[0] != 1.0);
}

test "clipGradNormCpu" {
    var grad1 = [_]f32{ 3.0, 4.0 };
    var grad2 = [_]f32{ 6.0, 8.0 };
    var grads = [_][]f32{ grad1[0..], grad2[0..] };

    const norm = clipGradNormCpu(grads[0..], 5.0);

    try std.testing.expectApproxEqRel(@as(f32, 11.18), norm, 0.01);

    var new_norm_sq: f32 = 0.0;
    for (grads) |grad| {
        for (grad) |g| {
            new_norm_sq += g * g;
        }
    }
    try std.testing.expectApproxEqRel(@as(f32, 5.0), @sqrt(new_norm_sq), 0.01);
}
