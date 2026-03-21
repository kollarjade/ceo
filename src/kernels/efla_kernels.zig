const std = @import("std");

pub const EflaDType = c_int;
pub const CudaStream = ?*anyopaque;

pub extern fn efla_forward_cuda(
    k: ?*const anyopaque,
    v: ?*const anyopaque,
    initial_state: ?*const anyopaque,
    final_state: ?*anyopaque,
    output: ?*anyopaque,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    tensor_dtype: EflaDType,
    beta: f32,
    lambda: f32,
    chunk_size: usize,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn efla_backward_cuda(
    grad_output: ?*const anyopaque,
    k: ?*const anyopaque,
    v: ?*const anyopaque,
    initial_state: ?*const anyopaque,
    final_state: ?*const anyopaque,
    grad_k: ?*anyopaque,
    grad_v: ?*anyopaque,
    grad_initial_state: ?*anyopaque,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    tensor_dtype: EflaDType,
    beta: f32,
    lambda: f32,
    chunk_size: usize,
    stream: CudaStream,
) callconv(.C) c_int;

pub extern fn efla_chunked_scan_cuda(
    chunk_states: [*]?*anyopaque,
    num_chunks: usize,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    state_dtype: EflaDType,
    stream: CudaStream,
) callconv(.C) c_int;

fn squareLen(head_dim: usize) usize {
    return std.math.mul(usize, head_dim, head_dim) catch @panic("head_dim overflow");
}

fn requireEqual(actual: usize, expected: usize) void {
    if (actual != expected) @panic("invalid dimension");
}

pub fn computeCoefficient(lambda: f32, beta: f32) f32 {
    const x = beta * lambda;
    if (@abs(x) < @as(f32, 1e-4)) {
        const lambda2 = lambda * lambda;
        const beta2 = beta * beta;
        const beta3 = beta2 * beta;
        const beta4 = beta3 * beta;
        return beta
            - @as(f32, 0.5) * beta2 * lambda
            + (@as(f32, 1.0) / @as(f32, 6.0)) * beta3 * lambda2
            - (@as(f32, 1.0) / @as(f32, 24.0)) * beta4 * lambda2 * lambda;
    }
    if (lambda == 0.0) {
        return beta;
    }
    return (@as(f32, 1.0) - @exp(-x)) / lambda;
}

pub fn eflaStateUpdate(
    state: []f32,
    k: []const f32,
    v: []const f32,
    head_dim: usize,
    beta: f32,
) void {
    requireEqual(k.len, head_dim);
    requireEqual(v.len, head_dim);
    requireEqual(state.len, squareLen(head_dim));

    var lambda: f32 = 0.0;
    for (0..head_dim) |i| {
        lambda += k[i] * k[i];
    }

    const c_t = computeCoefficient(lambda, beta);

    for (0..head_dim) |j| {
        var projection: f32 = 0.0;
        for (0..head_dim) |i| {
            projection += k[i] * state[i * head_dim + j];
        }
        for (0..head_dim) |i| {
            const idx = i * head_dim + j;
            state[idx] = state[idx] - c_t * k[i] * projection + c_t * k[i] * v[j];
        }
    }
}

pub fn eflaComputeOutput(
    state: []const f32,
    k: []const f32,
    output: []f32,
    head_dim: usize,
) void {
    requireEqual(state.len, squareLen(head_dim));
    requireEqual(k.len, head_dim);
    requireEqual(output.len, head_dim);

    for (0..head_dim) |j| {
        var sum: f32 = 0.0;
        for (0..head_dim) |i| {
            sum += k[i] * state[i * head_dim + j];
        }
        output[j] = sum;
    }
}

test "computeCoefficient" {
    const c1 = computeCoefficient(@as(f32, 1e-8), @as(f32, 1.0));
    try std.testing.expectApproxEqRel(@as(f32, 1.0), c1, @as(f32, 0.001));

    const c2 = computeCoefficient(@as(f32, 1.0), @as(f32, 1.0));
    const expected: f32 = (@as(f32, 1.0) - @exp(@as(f32, -1.0))) / @as(f32, 1.0);
    try std.testing.expectApproxEqRel(expected, c2, @as(f32, 0.001));
}

test "computeCoefficient zero lambda" {
    const c = computeCoefficient(@as(f32, 0.0), @as(f32, 2.5));
    try std.testing.expectApproxEqRel(@as(f32, 2.5), c, @as(f32, 1e-6));
}

test "eflaStateUpdate and eflaComputeOutput" {
    var state = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const k = [_]f32{ 1.0, 0.0 };
    const v = [_]f32{ 2.0, 3.0 };

    eflaStateUpdate(state[0..], k[0..], v[0..], 2, @as(f32, 1.0));

    const c: f32 = (@as(f32, 1.0) - @exp(@as(f32, -1.0)));

    try std.testing.expectApproxEqRel(@as(f32, 1.0) + c, state[0], @as(f32, 1e-6));
    try std.testing.expectApproxEqRel(@as(f32, 3.0) * c, state[1], @as(f32, 1e-6));
    try std.testing.expectApproxEqRel(@as(f32, 0.0), state[2], @as(f32, 1e-6));
    try std.testing.expectApproxEqRel(@as(f32, 1.0), state[3], @as(f32, 1e-6));

    var output = [_]f32{ 0.0, 0.0 };
    eflaComputeOutput(state[0..], k[0..], output[0..], 2);

    try std.testing.expectApproxEqRel(@as(f32, 1.0) + c, output[0], @as(f32, 1e-6));
    try std.testing.expectApproxEqRel(@as(f32, 0.0), output[1], @as(f32, 1e-6));
}
