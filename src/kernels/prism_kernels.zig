const std = @import("std");

pub extern "cuda_prism" fn prismForwardCuda(
    u: ?*const anyopaque,
    v: ?*const anyopaque,
    prev_state: ?*const anyopaque,
    new_state: ?*anyopaque,
    output: ?*anyopaque,
    w_beta: [*]const ?*const anyopaque,
    w_k: [*]const ?*const anyopaque,
    w_p: [*]const ?*const anyopaque,
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    num_iterations: usize,
    alpha: f32,
) callconv(.C) c_int;

pub extern "cuda_prism" fn prismBackwardCuda(
    grad_output: ?*const anyopaque,
    u: ?*const anyopaque,
    v: ?*const anyopaque,
    state: ?*const anyopaque,
    grad_u: ?*anyopaque,
    grad_v: ?*anyopaque,
    grad_state: ?*anyopaque,
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_iterations: usize,
) callconv(.C) c_int;

pub extern "cuda_prism" fn shortConvForwardCuda(
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    output: ?*anyopaque,
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    window_size: usize,
) callconv(.C) c_int;

pub extern "cuda_prism" fn shortConvBackwardCuda(
    grad_output: ?*const anyopaque,
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    grad_input: ?*anyopaque,
    grad_weight: ?*anyopaque,
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    window_size: usize,
) callconv(.C) c_int;

fn checkedMul(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch @panic("dimension overflow");
}

fn requireAtLeast(actual: usize, needed: usize, name: []const u8) void {
    if (actual < needed) std.debug.panic("invalid {s}", .{name});
}

pub fn geluForward(x: f32, approximate: bool) f32 {
    if (approximate) {
        const sqrt_2_over_pi: f32 = 0.7978845608028654;
        const coeff: f32 = 0.044715;
        const x3 = x * x * x;
        return @as(f32, 0.5) * x * (@as(f32, 1.0) + std.math.tanh(sqrt_2_over_pi * (x + coeff * x3)));
    } else {
        const sqrt2: f32 = @sqrt(@as(f32, 2.0));
        const cdf = @as(f32, 0.5) * (@as(f32, 1.0) + std.math.erf(x / sqrt2));
        return x * cdf;
    }
}

pub fn geluBackward(x: f32, approximate: bool) f32 {
    if (approximate) {
        const sqrt_2_over_pi: f32 = 0.7978845608028654;
        const coeff: f32 = 0.044715;
        const tanh_arg = sqrt_2_over_pi * (x + coeff * x * x * x);
        const tanh_val = std.math.tanh(tanh_arg);
        const sech_sq = @as(f32, 1.0) - tanh_val * tanh_val;
        const inner_deriv = sqrt_2_over_pi * (@as(f32, 1.0) + @as(f32, 3.0) * coeff * x * x);
        return @as(f32, 0.5) * (@as(f32, 1.0) + tanh_val) + @as(f32, 0.5) * x * sech_sq * inner_deriv;
    } else {
        const sqrt2: f32 = @sqrt(@as(f32, 2.0));
        const sqrt_two_pi: f32 = @sqrt(@as(f32, 2.0) * @as(f32, std.math.pi));
        const cdf = @as(f32, 0.5) * (@as(f32, 1.0) + std.math.erf(x / sqrt2));
        const pdf = @exp(-@as(f32, 0.5) * x * x) / sqrt_two_pi;
        return cdf + x * pdf;
    }
}

pub fn outerProduct(a: []const f32, b: []const f32, result: []f32) void {
    const m = a.len;
    const n = b.len;
    requireAtLeast(result.len, checkedMul(m, n), "result length");

    for (0..m) |i| {
        for (0..n) |j| {
            result[i * n + j] = a[i] * b[j];
        }
    }
}

pub fn rank1Update(a: []f32, x: []const f32, y: []const f32, alpha: f32, m: usize, n: usize) void {
    requireAtLeast(a.len, checkedMul(m, n), "matrix length");
    requireAtLeast(x.len, m, "x length");
    requireAtLeast(y.len, n, "y length");

    for (0..m) |i| {
        for (0..n) |j| {
            a[i * n + j] += alpha * x[i] * y[j];
        }
    }
}

test "geluForward" {
    try std.testing.expectApproxEqRel(@as(f32, 0.0), geluForward(0.0, true), @as(f32, 0.01));
    try std.testing.expectApproxEqRel(@as(f32, 0.841), geluForward(1.0, true), @as(f32, 0.01));
    try std.testing.expectApproxEqRel(@as(f32, -0.159), geluForward(-1.0, true), @as(f32, 0.02));
}

test "outerProduct" {
    const a = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{ 3.0, 4.0, 5.0 };
    var result = [_]f32{0} ** 6;

    outerProduct(a[0..], b[0..], result[0..]);

    try std.testing.expectEqual(@as(f32, 3.0), result[0]);
    try std.testing.expectEqual(@as(f32, 4.0), result[1]);
    try std.testing.expectEqual(@as(f32, 5.0), result[2]);
    try std.testing.expectEqual(@as(f32, 6.0), result[3]);
    try std.testing.expectEqual(@as(f32, 8.0), result[4]);
    try std.testing.expectEqual(@as(f32, 10.0), result[5]);
}
