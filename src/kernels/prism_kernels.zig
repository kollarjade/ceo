const std = @import("std");

/// PRISM kernel function declarations (implemented in CUDA)

/// PRISM forward pass kernel
pub extern "cuda_prism" fn prismForwardCuda(
    u: ?*const anyopaque, // Proxy: (batch, seq_len, hidden_dim)
    v: ?*const anyopaque, // Values: (batch, seq_len, hidden_dim)
    prev_state: ?*const anyopaque, // Previous state
    new_state: ?*anyopaque, // New state
    output: ?*anyopaque, // Output: (batch, seq_len, hidden_dim)
    w_beta: [*]const ?*const anyopaque, // Beta weights per iteration
    w_k: [*]const ?*const anyopaque, // Key weights per iteration
    w_p: [*]const ?*const anyopaque, // Proxy weights per iteration
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    num_iterations: usize,
    alpha: f32,
) callconv(.C) anyerror!void;

/// PRISM backward pass kernel
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
) callconv(.C) anyerror!void;

/// ShortConv forward kernel
pub extern "cuda_prism" fn shortConvForwardCuda(
    input: ?*const anyopaque, // (batch, seq_len, hidden_dim)
    weight: ?*const anyopaque, // (hidden_dim, window_size)
    output: ?*anyopaque, // (batch, seq_len, hidden_dim)
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    window_size: usize,
) callconv(.C) anyerror!void;

/// ShortConv backward kernel
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
) callconv(.C) anyerror!void;

/// GELU activation kernel
pub fn geluForward(x: f32, approximate: bool) f32 {
    if (approximate) {
        // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const sqrt_2_over_pi = 0.7978845608028654;
        const x3 = x * x * x;
        return 0.5 * x * (1.0 + std.math.tanh(sqrt_2_over_pi * (x + 0.044715 * x3)));
    } else {
        // Exact: x * Φ(x)
        const sqrt2 = std.math.sqrt(2.0);
        const cdf = 0.5 * (1.0 + std.math.erf(x / sqrt2));
        return x * cdf;
    }
}

/// GELU backward kernel
pub fn geluBackward(x: f32, approximate: bool) f32 {
    if (approximate) {
        const sqrt_2_over_pi = 0.7978845608028654;
        const x3 = x * x * x;
        const tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x3);
        const tanh_val = std.math.tanh(tanh_arg);
        const sech_sq = 1.0 - tanh_val * tanh_val;
        const inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
        return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech_sq * inner_deriv;
    } else {
        const sqrt2 = std.math.sqrt(2.0);
        const cdf = 0.5 * (1.0 + std.math.erf(x / sqrt2));
        const pdf = @exp(-0.5 * x * x) / @sqrt(2.0 * std.math.pi);
        return cdf + x * pdf;
    }
}

/// Outer product: a ⊗ b
pub fn outerProduct(a: []const f32, b: []const f32, result: []f32) void {
    const m = a.len;
    const n = b.len;

    for (0..m) |i| {
        for (0..n) |j| {
            result[i * n + j] = a[i] * b[j];
        }
    }
}

/// Rank-1 update: A = A + alpha * x * y^T
pub fn rank1Update(a: []f32, x: []const f32, y: []const f32, alpha: f32, m: usize, n: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            a[i * n + j] += alpha * x[i] * y[j];
        }
    }
}

test "geluForward" {
    // GELU(0) ≈ 0
    try std.testing.expectApproxEqRel(@as(f32, 0.0), geluForward(0.0, true), 0.01);

    // GELU(1) ≈ 0.841
    try std.testing.expectApproxEqRel(@as(f32, 0.841), geluForward(1.0, true), 0.01);

    // GELU(-1) ≈ -0.159
    try std.testing.expectApproxEqRel(@as(f32, -0.159), geluForward(-1.0, true), 0.02);
}

test "outerProduct" {
    const a = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{ 3.0, 4.0, 5.0 };
    var result = [_]f32{0} ** 6;

    outerProduct(&a, &b, &result);

    try std.testing.expectEqual(@as(f32, 3.0), result[0]);
    try std.testing.expectEqual(@as(f32, 4.0), result[1]);
    try std.testing.expectEqual(@as(f32, 5.0), result[2]);
    try std.testing.expectEqual(@as(f32, 6.0), result[3]);
    try std.testing.expectEqual(@as(f32, 8.0), result[4]);
    try std.testing.expectEqual(@as(f32, 10.0), result[5]);
}
