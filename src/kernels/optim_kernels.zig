const std = @import("std");

/// Optimizer kernel declarations (implemented in CUDA)

/// Lion optimizer step kernel
pub extern "cuda_optim" fn @"lion_step_cuda"(
    param: ?*anyopaque,
    grad: ?*const anyopaque,
    momentum: ?*anyopaque,
    numel: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    stream: ?*anyopaque,
) callconv(.C) anyerror!void;

/// Muon optimizer step kernel
pub extern "cuda_optim" fn @"muon_step_cuda"(
    param: ?*anyopaque,
    grad: ?*const anyopaque,
    momentum: ?*anyopaque,
    m: usize,
    n: usize,
    lr: f32,
    beta: f32,
    ns_iterations: usize,
    stream: ?*anyopaque,
) callconv(.C) anyerror!void;

/// AdamW optimizer step kernel
pub extern "cuda_optim" fn @"adamw_step_cuda"(
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
    stream: ?*anyopaque,
) callconv(.C) anyerror!void;

/// Gradient clipping by norm kernel
pub extern "cuda_optim" fn @"clip_grad_norm_cuda"(
    grads: [*]?*anyopaque,
    numels: [*]const usize,
    num_params: usize,
    max_norm: f32,
    global_norm: *f32,
    stream: ?*anyopaque,
) callconv(.C) anyerror!void;

/// Gradient clipping by value kernel
pub extern "cuda_optim" fn clipGradValueCuda(
    grad: ?*anyopaque,
    numel: usize,
    max_value: f32,
) callconv(.C) anyerror!void;

/// FP8 quantization kernel
pub extern "cuda_optim" fn @"quantize_fp8_cuda"(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    scale: *f32,
    numel: usize,
    input_format: c_int,
    output_dtype: c_int,
    stream: ?*anyopaque,
) callconv(.C) anyerror!void;

/// FP8 dequantization kernel
pub extern "cuda_optim" fn @"dequantize_fp8_cuda"(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    scale: f32,
    numel: usize,
    input_format: c_int,
    output_dtype: c_int,
    stream: ?*anyopaque,
) callconv(.C) anyerror!void;


pub const lionStepCuda = @"lion_step_cuda";
pub const muonStepCuda = @"muon_step_cuda";
pub const adamWStepCuda = @"adamw_step_cuda";
pub const clipGradNormCuda = @"clip_grad_norm_cuda";
pub const quantizeFP8Cuda = @"quantize_fp8_cuda";
pub const dequantizeFP8Cuda = @"dequantize_fp8_cuda";

/// CPU reference implementations

/// Lion step (CPU reference)
pub fn lionStepCpu(
    param: []f32,
    grad: []const f32,
    momentum: []f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
) void {
    const one_minus_beta1 = 1.0 - beta1;
    const one_minus_beta2 = 1.0 - beta2;

    for (param, grad, momentum) |*p, g, *m| {
        // v_t = β1 * m_{t-1} + (1-β1) * g_t
        const v = beta1 * m.* + one_minus_beta1 * g;

        // m_t = β2 * m_{t-1} + (1-β2) * g_t
        m.* = beta2 * m.* + one_minus_beta2 * g;

        // Update: x = x - lr * sign(v) - lr * weight_decay * x
        const sign_v: f32 = if (v > 0) 1.0 else if (v < 0) -1.0 else 0.0;
        p.* -= lr * sign_v + lr * weight_decay * p.*;
    }
}

/// AdamW step (CPU reference)
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
    const beta1_t = std.math.pow(f32, beta1, @floatFromInt(step));
    const beta2_t = std.math.pow(f32, beta2, @floatFromInt(step));
    const bias_correction1 = 1.0 / (1.0 - beta1_t);
    const bias_correction2 = 1.0 / (1.0 - beta2_t);

    for (param, grad, exp_avg, exp_avg_sq) |*p, g, *ea, *eas| {
        // Update biased first moment estimate
        ea.* = beta1 * ea.* + (1.0 - beta1) * g;

        // Update biased second raw moment estimate
        eas.* = beta2 * eas.* + (1.0 - beta2) * g * g;

        // Compute bias-corrected estimates
        const avg = ea.* * bias_correction1;
        const avg_sq = eas.* * bias_correction2;

        // Update parameters
        const denom = @sqrt(avg_sq) + eps;
        p.* -= lr * (avg / denom + weight_decay * p.*);
    }
}

/// Newton-Schulz iteration for orthogonalization (CPU reference)
pub fn newtonSchulzIteration(
    Y: []f32,
    temp: []f32,
    m: usize,
    n: usize,
) void {
    // Y_{k+1} = 0.5 * Y_k * (3I - Y_k^T Y_k)

    // Compute Y_k^T Y_k
    for (0..n) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..m) |k| {
                sum += Y[k * n + i] * Y[k * n + j];
            }
            temp[i * n + j] = sum;
        }
    }

    // Compute 3I - Y_k^T Y_k
    for (0..n) |i| {
        for (0..n) |j| {
            const identity: f32 = if (i == j) 3.0 else 0.0;
            temp[i * n + j] = identity - temp[i * n + j];
        }
    }

    // Compute Y * (3I - Y^T Y)
    var Y_new = std.mem.zeroes([4096]f32);

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..n) |k| {
                sum += Y[i * n + k] * temp[k * n + j];
            }
            Y_new[i * n + j] = 0.5 * sum;
        }
    }

    @memcpy(Y, Y_new[0 .. m * n]);
}

/// Compute gradient norm
pub fn computeGradNorm(grads: []const []const f32) f64 {
    var norm_sq: f64 = 0.0;

    for (grads) |grad| {
        for (grad) |g| {
            norm_sq += @as(f64, g) * @as(f64, g);
        }
    }

    return @sqrt(norm_sq);
}

/// Clip gradients by norm
pub fn clipGradNormCpu(grads: []][]f32, max_norm: f32) f32 {
    // Compute norm
    var norm_sq: f64 = 0.0;
    for (grads) |grad| {
        for (grad) |g| {
            norm_sq += @as(f64, g) * @as(f64, g);
        }
    }
    const norm = @sqrt(norm_sq);

    // Clip if necessary
    if (norm > max_norm) {
        const scale = max_norm / @as(f32, @floatCast(norm));
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

    // Params should have changed
    try std.testing.expect(param[0] != 1.0);

    // Momentum should have been updated
    try std.testing.expect(momentum[0] != 0.0);
}

test "adamWStepCpu" {
    var param = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const grad = [_]f32{ 0.1, 0.1, 0.1, 0.1 };
    var exp_avg = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var exp_avg_sq = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    adamWStepCpu(&param, &grad, &exp_avg, &exp_avg_sq, 0.01, 0.9, 0.999, 1e-8, 0.0, 1);

    // Params should have changed
    try std.testing.expect(param[0] != 1.0);
}

test "clipGradNormCpu" {
    var grad1 = [_]f32{ 3.0, 4.0 };
    var grad2 = [_]f32{ 6.0, 8.0 };
    var grads = [_][]f32{ &grad1, &grad2 };

    const norm = clipGradNormCpu(&grads, 5.0);

    // Original norm should be sqrt(9+16+36+64) = sqrt(125) ≈ 11.18
    try std.testing.expectApproxEqRel(@as(f32, 11.18), norm, 0.01);

    // After clipping, norm should be 5.0
    var new_norm_sq: f32 = 0.0;
    for (grads) |grad| {
        for (grad) |g| {
            new_norm_sq += g * g;
        }
    }
    try std.testing.expectApproxEqRel(@as(f32, 5.0), @sqrt(new_norm_sq), 0.01);
}
