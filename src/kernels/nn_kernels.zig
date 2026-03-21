const std = @import("std");

pub const EflaDType = c_int;
pub const EflaReduction = c_int;
pub const CudaStream = ?*anyopaque;

pub extern fn rmsnorm_forward_cuda(
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    output: ?*anyopaque,
    numel: usize,
    normalized_shape: usize,
    tensor_dtype: EflaDType,
    eps: f32,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn rmsnorm_backward_cuda(
    grad_output: ?*const anyopaque,
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    grad_input: ?*anyopaque,
    grad_weight: ?*anyopaque,
    numel: usize,
    normalized_shape: usize,
    tensor_dtype: EflaDType,
    grad_weight_dtype: EflaDType,
    eps: f32,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn layernorm_forward_cuda(
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    bias: ?*const anyopaque,
    output: ?*anyopaque,
    numel: usize,
    normalized_shape: usize,
    tensor_dtype: EflaDType,
    eps: f32,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn gelu_forward_cuda(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    numel: usize,
    tensor_dtype: EflaDType,
    approximate: bool,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn softmax_forward_cuda(
    input: ?*const anyopaque,
    output: ?*anyopaque,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    tensor_dtype: EflaDType,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn gemm_forward_cuda(
    a: ?*const anyopaque,
    b: ?*const anyopaque,
    bias: ?*const anyopaque,
    c: ?*anyopaque,
    m: usize,
    k: usize,
    n: usize,
    dtype_a: EflaDType,
    dtype_b: EflaDType,
    dtype_c: EflaDType,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn gemm_backward_cuda(
    grad_c: ?*const anyopaque,
    a: ?*const anyopaque,
    b: ?*const anyopaque,
    grad_a: ?*anyopaque,
    grad_b: ?*anyopaque,
    grad_bias: ?*anyopaque,
    m: usize,
    k: usize,
    n: usize,
    grad_c_dtype: EflaDType,
    a_dtype: EflaDType,
    b_dtype: EflaDType,
    grad_a_dtype: EflaDType,
    grad_b_dtype: EflaDType,
    grad_bias_dtype: EflaDType,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn cross_entropy_forward_cuda(
    logits: ?*const anyopaque,
    targets: ?*const i32,
    loss: ?*anyopaque,
    batch_size: usize,
    vocab_size: usize,
    logits_dtype: EflaDType,
    loss_dtype: EflaDType,
    label_smoothing: f32,
    reduction: EflaReduction,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn cross_entropy_backward_cuda(
    grad_loss: ?*const anyopaque,
    logits: ?*const anyopaque,
    targets: ?*const i32,
    grad_logits: ?*anyopaque,
    batch_size: usize,
    vocab_size: usize,
    grad_loss_dtype: EflaDType,
    logits_dtype: EflaDType,
    grad_logits_dtype: EflaDType,
    label_smoothing: f32,
    reduction: EflaReduction,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn embedding_forward_cuda(
    indices: ?*const i32,
    weight: ?*const anyopaque,
    output: ?*anyopaque,
    num_indices: usize,
    embedding_dim: usize,
    weight_dtype: EflaDType,
    output_dtype: EflaDType,
    stream: CudaStream,
) callconv(.C) c_int;

fn requireEqual(actual: usize, expected: usize, name: []const u8) void {
    if (actual != expected) std.debug.panic("invalid {s}", .{name});
}

fn requireNonZero(value: usize, name: []const u8) void {
    if (value == 0) std.debug.panic("invalid {s}", .{name});
}

fn requireMultiple(value: usize, divisor: usize, name: []const u8) void {
    requireNonZero(divisor, name);
    if (value % divisor != 0) std.debug.panic("invalid {s}", .{name});
}

fn requireNonNegative(value: f32, name: []const u8) void {
    if (value < 0.0 or value != value) std.debug.panic("invalid {s}", .{name});
}

fn requireProbability(value: f32, name: []const u8) void {
    if (value < 0.0 or value > 1.0 or value != value) std.debug.panic("invalid {s}", .{name});
}

pub fn rmsNormForwardCpu(
    input: []const f32,
    weight: []const f32,
    output: []f32,
    normalized_shape: usize,
    eps: f32,
) void {
    requireNonZero(normalized_shape, "normalized_shape");
    requireMultiple(input.len, normalized_shape, "input length");
    requireEqual(weight.len, normalized_shape, "weight length");
    requireEqual(output.len, input.len, "output length");
    requireNonNegative(eps, "eps");

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
    requireNonZero(normalized_shape, "normalized_shape");
    requireMultiple(input.len, normalized_shape, "input length");
    requireEqual(weight.len, normalized_shape, "weight length");
    requireEqual(bias.len, normalized_shape, "bias length");
    requireEqual(output.len, input.len, "output length");
    requireNonNegative(eps, "eps");

    const n = input.len / normalized_shape;

    for (0..n) |i| {
        var sum: f32 = 0.0;
        var sum_sq: f32 = 0.0;

        for (0..normalized_shape) |j| {
            const val = input[i * normalized_shape + j];
            sum += val;
            sum_sq += val * val;
        }

        const mean = sum / @as(f32, @floatFromInt(normalized_shape));
        const variance = sum_sq / @as(f32, @floatFromInt(normalized_shape)) - mean * mean;
        const clamped_variance = if (variance > 0.0) variance else 0.0;
        const inv_std = @as(f32, 1.0) / @sqrt(clamped_variance + eps);

        for (0..normalized_shape) |j| {
            const normalized = (input[i * normalized_shape + j] - mean) * inv_std;
            output[i * normalized_shape + j] = normalized * weight[j] + bias[j];
        }
    }
}

pub fn softmaxForwardCpu(input: []const f32, output: []f32, dim_size: usize) void {
    requireNonZero(dim_size, "dim_size");
    requireMultiple(input.len, dim_size, "input length");
    requireEqual(output.len, input.len, "output length");

    const n = input.len / dim_size;
    const neg_inf = -std.math.inf(f32);

    for (0..n) |i| {
        const row_start = i * dim_size;
        var max_val: f32 = neg_inf;

        for (0..dim_size) |j| {
            const val = input[row_start + j];
            if (val != val) @panic("input contains nan");
            if (val > max_val) {
                max_val = val;
            }
        }

        if (max_val == neg_inf) {
            const uniform = @as(f32, 1.0) / @as(f32, @floatFromInt(dim_size));
            for (output[row_start .. row_start + dim_size]) |*out| {
                out.* = uniform;
            }
            continue;
        }

        var sum: f32 = 0.0;
        for (0..dim_size) |j| {
            const value = @exp(input[row_start + j] - max_val);
            output[row_start + j] = value;
            sum += value;
        }

        if (sum == 0.0 or sum != sum) @panic("invalid softmax normalization");

        for (0..dim_size) |j| {
            output[row_start + j] /= sum;
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
    requireNonZero(vocab_size, "vocab_size");
    requireEqual(loss.len, targets.len, "loss length");
    requireEqual(logits.len, targets.len * vocab_size, "logits length");
    requireProbability(label_smoothing, "label_smoothing");

    const batch_size = targets.len;

    for (0..batch_size) |i| {
        const target: usize = @intCast(targets[i]);
        if (target >= vocab_size) @panic("target out of range");

        var max_logit: f32 = -std.math.inf(f32);
        for (0..vocab_size) |j| {
            const logit = logits[i * vocab_size + j];
            if (logit != logit) @panic("logits contain nan");
            if (logit > max_logit) {
                max_logit = logit;
            }
        }

        var sum_exp: f64 = 0.0;
        for (0..vocab_size) |j| {
            const exp_val: f32 = @exp(logits[i * vocab_size + j] - max_logit);
            sum_exp += @as(f64, exp_val);
        }

        const log_sum_exp: f32 = @as(f32, @floatCast(@log(sum_exp)));

        var target_log_prob: f32 = 0.0;
        var sum_log_probs: f32 = 0.0;

        for (0..vocab_size) |j| {
            const log_prob = logits[i * vocab_size + j] - max_logit - log_sum_exp;
            sum_log_probs += log_prob;
            if (j == target) {
                target_log_prob = log_prob;
            }
        }

        const smoothing_per_class = label_smoothing / @as(f32, @floatFromInt(vocab_size));
        loss[i] = -((@as(f32, 1.0) - label_smoothing) * target_log_prob + smoothing_per_class * sum_log_probs);
    }
}

test "rmsNormForwardCpu" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var output = [_]f32{0} ** 8;

    rmsNormForwardCpu(input[0..], weight[0..], output[0..], 4, @as(f32, 1e-6));

    var sum_sq: f32 = 0.0;
    for (output[0..4]) |v| {
        sum_sq += v * v;
    }
    const mean_sq = sum_sq / @as(f32, 4.0);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), mean_sq, @as(f32, 0.01));
}

test "softmaxForwardCpu" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 1.0, 1.0, 1.0 };
    var output = [_]f32{0} ** 6;

    softmaxForwardCpu(input[0..], output[0..], 3);

    var sum1: f32 = 0.0;
    for (output[0..3]) |v| {
        sum1 += v;
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum1, @as(f32, 0.001));

    var sum2: f32 = 0.0;
    for (output[3..6]) |v| {
        sum2 += v;
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum2, @as(f32, 0.001));
}
