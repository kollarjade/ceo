const std = @import("std");

/// EFLA kernel function declarations (implemented in CUDA)
/// These are the extern declarations for the CUDA kernels

/// EFLA forward pass kernel
/// Computes the exact closed-form state update
pub extern "cuda_efla" fn eflaForwardCuda(
    k: ?*const anyopaque, // Keys: (batch, seq_len, num_heads, head_dim)
    v: ?*const anyopaque, // Values: (batch, seq_len, num_heads, head_dim)
    state: ?*anyopaque, // State: (num_heads, head_dim, head_dim)
    output: ?*anyopaque, // Output: (batch, seq_len, num_heads, head_dim)
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    beta: f32,
    chunk_size: usize,
) callconv(.C) anyerror!void;

/// EFLA backward pass kernel
pub extern "cuda_efla" fn eflaBackwardCuda(
    grad_output: ?*const anyopaque,
    k: ?*const anyopaque,
    v: ?*const anyopaque,
    state: ?*const anyopaque,
    grad_k: ?*anyopaque,
    grad_v: ?*anyopaque,
    grad_state: ?*anyopaque,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    beta: f32,
) callconv(.C) anyerror!void;

/// EFLA chunked scan kernel
pub extern "cuda_efla" fn eflaChunkedScanCuda(
    chunk_states: [*]?*anyopaque,
    num_chunks: usize,
    num_heads: usize,
    head_dim: usize,
) callconv(.C) anyerror!void;

/// Reference implementations for testing (CPU)

/// Compute the coefficient c_t = (1 - exp(-beta * lambda)) / lambda
/// with numerical stability for small lambda
pub fn computeCoefficient(lambda: f32, beta: f32) f32 {
    if (lambda < 1e-6) {
        // Use Taylor series expansion for numerical stability
        var c: f32 = beta;
        const beta_lambda = beta * lambda;
        c -= 0.5 * beta * beta_lambda;
        c += (1.0 / 6.0) * beta * beta * beta_lambda * lambda;
        c -= (1.0 / 24.0) * beta * beta * beta * beta_lambda * lambda * lambda;
        return c;
    }

    return (1.0 - @exp(-beta * lambda)) / lambda;
}

/// EFLA state update: S_t = (I - c_t * k * k^T) * S_{t-1} + c_t * k * v^T
pub fn eflaStateUpdate(
    state: []f32,
    k: []const f32,
    v: []const f32,
    head_dim: usize,
    beta: f32,
) void {
    var lambda: f32 = 0.0;
    for (k) |k_val| {
        lambda += k_val * k_val;
    }

    const c_t = computeCoefficient(lambda, beta);

    var k_t_s = std.mem.zeroes([128]f32);

    for (0..head_dim) |j| {
        for (0..head_dim) |i| {
            k_t_s[j] += k[i] * state[i * head_dim + j];
        }
    }

    for (0..head_dim) |i| {
        for (0..head_dim) |j| {
            state[i * head_dim + j] -= c_t * k[i] * k_t_s[j];
            state[i * head_dim + j] += c_t * k[i] * v[j];
        }
    }
}

/// EFLA output computation: o_t = S_t * k_t
pub fn eflaComputeOutput(
    state: []const f32,
    k: []const f32,
    output: []f32,
    head_dim: usize,
) void {
    for (0..head_dim) |i| {
        var sum: f32 = 0.0;
        for (0..head_dim) |j| {
            sum += state[i * head_dim + j] * k[j];
        }
        output[i] = sum;
    }
}

test "computeCoefficient" {
    const c1 = computeCoefficient(1e-8, 1.0);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), c1, 0.001);

    const c2 = computeCoefficient(1.0, 1.0);
    const expected = (1.0 - @exp(-1.0)) / 1.0;
    try std.testing.expectApproxEqRel(expected, c2, 0.001);
}
