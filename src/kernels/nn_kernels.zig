const std = @import("std");

pub extern "cuda_nn" fn rmsNormForwardCuda(
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    output: ?*anyopaque,
    numel: usize,
    normalized_shape: usize,
    eps: f32,
) callconv(.C) c_int;

pub extern "cuda_nn" fn rmsNormBackwardCuda(
    grad_output: ?*const anyopaque,
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    grad_input: ?*anyopaque,
    grad_weight: ?*anyopaque,
    numel: usize,
    normalized_shape: usize,
    eps: f32,
) callconv(.C) c_int;

pub extern "cuda_nn" fn layerNormForwardCuda(
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    bias: ?*const anyopaque,
    output: ?*anyopaque,
    numel: usize,
    normalized_shape: usize,
    eps: f32,
) callconv(.C) c_int;

pub extern "cuda_nn" fn geluForwardCuda(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    numel: usize,
    approximate: bool,
) callconv(.C) c_int;

pub extern "cuda_nn" fn softmaxForwardCuda(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    numel: usize,
    dim_size: usize,
) callconv(.C) c_int;

pub extern "cuda_nn" fn gemmForwardCuda(
    a: ?*const anyopaque,
    b: ?*const anyopaque,
    bias: ?*const anyopaque,
    c: ?*anyopaque,
    m: usize,
    k: usize,
    n: usize,
) callconv(.C) c_int;

pub extern "cuda_nn" fn gemmBackwardCuda(
    grad_c: ?*const anyopaque,
    a: ?*const anyopaque,
    b: ?*const anyopaque,
    grad_a: ?*anyopaque,
    grad_b: ?*anyopaque,
    grad_bias: ?*anyopaque,
    m: usize,
    k: usize,
    n: usize,
) callconv(.C) c_int;

pub extern "cuda_nn" fn crossEntropyForwardCuda(
    logits: ?*const anyopaque,
    targets: ?*const anyopaque,
    loss: ?*anyopaque,
    batch_size: usize,
    vocab_size: usize,
    label_smoothing: f32,
) callconv(.C) c_int;

pub extern "cuda_nn" fn crossEntropyBackwardCuda(
    grad_loss: ?*const anyopaque,
    logits: ?*const anyopaque,
    targets: ?*const anyopaque,
    grad_logits: ?*anyopaque,
    batch_size: usize,
    vocab_size: usize,
    label_smoothing: f32,
) callconv(.C) c_int;

pub extern "cuda_nn" fn embeddingForwardCuda(
    indices: ?*const anyopaque,
    weight: ?*const anyopaque,
    output: ?*anyopaque,
    num_indices: usize,
    embedding_dim: usize,
) callconv(.C) c_int;

pub fn rmsNormForwardCpu(
    input: []const f32,
    weight: []const f32,
    output: []f32,
    normalized_shape: usize,
    eps: f32,
) void {
    if (normalized_shape == 0) return;
    const n = input.len / normalized_shape;

    for (0..n) |i| {
        var sum_sq: f32 = 0.0;
        for (0..normalized_shape) |j| {
            const val = input[i * normalized_shape + j];
            sum_sq += val * val;
        }

        const mean_sq = sum_sq / @as(f32, @floatFromInt(normalized_shape));
        const rsqrt = @as(f32, 1.0) / @sqrt(mean_sq + eps);

        for (0..normalized_shape) |j| {
            output[i * normalized_shape + j] = input[i * normalized_shape + j] * rsqrt * weight[j];
        }
    }
}

pub fn layerNormForwardCpu(
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    output: []f32,
    normalized_shape: usize,
    eps: f32,
) void {
    if (normalized_shape == 0) return;
    const n = input.len / normalized_shape;
    const ns_f: f32 = @floatFromInt(normalized_shape);

    for (0..n) |i| {
        var sum: f32 = 0.0;
        var sum_sq: f32 = 0.0;

        for (0..normalized_shape) |j| {
            const val = input[i * normalized_shape + j];
            sum += val;
            sum_sq += val * val;
        }

        const mean = sum / ns_f;
        const variance = sum_sq / ns_f - mean * mean;
        const safe_variance = if (variance < 0.0) @as(f32, 0.0) else variance;
        const inv_std = @as(f32, 1.0) / @sqrt(safe_variance + eps);

        for (0..normalized_shape) |j| {
            const normalized = (input[i * normalized_shape + j] - mean) * inv_std;
            output[i * normalized_shape + j] = normalized * weight[j] + bias[j];
        }
    }
}

pub fn softmaxForwardCpu(input: []const f32, output: []f32, dim_size: usize) void {
    if (dim_size == 0) return;
    const n = input.len / dim_size;

    for (0..n) |i| {
        var max_val: f32 = -std.math.inf(f32);
        for (0..dim_size) |j| {
            if (input[i * dim_size + j] > max_val) {
                max_val = input[i * dim_size + j];
            }
        }

        var sum: f32 = 0.0;
        for (0..dim_size) |j| {
            output[i * dim_size + j] = @exp(input[i * dim_size + j] - max_val);
            sum += output[i * dim_size + j];
        }

        if (sum == 0.0) {
            const uniform = @as(f32, 1.0) / @as(f32, @floatFromInt(dim_size));
            for (0..dim_size) |j| {
                output[i * dim_size + j] = uniform;
            }
        } else {
            for (0..dim_size) |j| {
                output[i * dim_size + j] /= sum;
            }
        }
    }
}

pub fn crossEntropyForwardCpu(
    logits: []const f32,
    targets: []const u32,
    loss: []f32,
    vocab_size: usize,
    label_smoothing: f32,
) void {
    const batch_size = targets.len;
    if (batch_size == 0) return;
    if (vocab_size == 0) return;

    for (0..batch_size) |i| {
        const target = targets[i];
        if (target >= vocab_size) {
            loss[i] = 0.0;
            continue;
        }

        var max_logit: f32 = -std.math.inf(f32);
        for (0..vocab_size) |j| {
            if (logits[i * vocab_size + j] > max_logit) {
                max_logit = logits[i * vocab_size + j];
            }
        }

        var sum_exp: f64 = 0.0;
        for (0..vocab_size) |j| {
            const shifted: f64 = @as(f64, logits[i * vocab_size + j]) - @as(f64, max_logit);
            sum_exp += @exp(shifted);
        }
        const log_sum_exp: f64 = @log(sum_exp);

        const log_prob_f64: f64 = @as(f64, logits[i * vocab_size + target]) - @as(f64, max_logit) - log_sum_exp;
        const log_prob: f32 = @floatCast(log_prob_f64);

        var sample_loss: f32 = -log_prob;

        if (label_smoothing > 0.0) {
            const smooth_loss = @log(@as(f32, @floatFromInt(vocab_size)));
            sample_loss = (1.0 - label_smoothing) * sample_loss + label_smoothing * smooth_loss;
        }

        loss[i] = sample_loss;
    }
}

test "rmsNormForwardCpu" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var output = [_]f32{0} ** 8;

    rmsNormForwardCpu(&input, &weight, &output, 4, 1e-6);

    var sum_sq: f32 = 0.0;
    for (output[0..4]) |v| {
        sum_sq += v * v;
    }
    const mean_sq = sum_sq / 4.0;
    try std.testing.expectApproxEqRel(@as(f32, 1.0), mean_sq, 0.01);
}

test "softmaxForwardCpu" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 1.0, 1.0, 1.0 };
    var output = [_]f32{0} ** 6;

    softmaxForwardCpu(&input, &output, 3);

    var sum1: f32 = 0.0;
    for (output[0..3]) |v| {
        sum1 += v;
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum1, 0.001);

    var sum2: f32 = 0.0;
    for (output[3..6]) |v| {
        sum2 += v;
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum2, 0.001);
}
