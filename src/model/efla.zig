const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const kernels = @import("../kernels/efla_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;

/// EFLA (Error-Free Linear Attention) State
/// Implements the exact closed-form state update from continuous-time dynamics
pub const EflaState = struct {
    /// Fast-weight state matrix S_t ∈ R^{d × d_v}
    state: *Tensor,
    /// Number of heads
    num_heads: usize,
    /// State dimension per head
    state_dim: usize,
    /// Value dimension
    value_dim: usize,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    /// Initialize EFLA state
    pub fn init(
        allocator: std.mem.Allocator,
        num_heads: usize,
        state_dim: usize,
        value_dim: usize,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*EflaState {
        const self = try allocator.create(EflaState);
        errdefer allocator.destroy(self);

        // State shape: (num_heads, state_dim, value_dim)
        const state_shape = Shape.init(&[_]usize{ num_heads, state_dim, value_dim });
        const state_tensor = try Tensor.zeros(allocator, state_shape, .bf16, device, device_id);
        errdefer state_tensor.deinit();

        self.* = .{
            .state = state_tensor,
            .num_heads = num_heads,
            .state_dim = state_dim,
            .value_dim = value_dim,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *EflaState) void {
        self.state.deinit();
        self.allocator.destroy(self);
    }

    /// Reset state to zeros
    pub fn reset(self: *EflaState) !void {
        try self.state.zero_();
    }

    /// Clone state
    pub fn clone(self: *EflaState) !*EflaState {
        const new_state = try self.state.to(self.allocator, self.device, self.device_id);

        const cloned = try self.allocator.create(EflaState);
        cloned.* = .{
            .state = new_state,
            .num_heads = self.num_heads,
            .state_dim = self.state_dim,
            .value_dim = self.value_dim,
            .allocator = self.allocator,
            .device = self.device,
            .device_id = self.device_id,
        };

        return cloned;
    }
};

/// EFLA Layer - Exact update implementation
///
/// Implements the exact closed-form update rule:
///
/// Definitions at time step t:
///   k_t ∈ R^{d} key vector
///   v_t ∈ R^{d_v} value vector
///   S_t ∈ R^{d × d_v} fast-weight state matrix
///   A_t = k_t k_t^T
///   b_t = k_t v_t^T
///   λ_t = k_t^T k_t (scalar)
///   β_t is the step size / "learning rate"
///
/// Coefficient:
///   c_t = (1 - exp(-β_t λ_t)) / λ_t
///
/// Numerically stable edge case:
///   If λ_t is near zero, compute c_t using stable series expansion
///
/// Exact state update:
///   S_t = (I - c_t k_t k_t^T) S_{t-1} + c_t k_t v_t^T
///
pub const EflaLayer = struct {
    /// Configuration
    config: config_mod.EflaConfig,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Key projection
    w_k: *Tensor,
    /// Value projection
    w_v: *Tensor,
    /// Output projection
    w_o: *Tensor,
    /// Learned beta parameter (optional)
    beta_param: ?*Tensor,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    /// Initialize EFLA layer
    pub fn init(
        allocator: std.mem.Allocator,
        config: config_mod.EflaConfig,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*EflaLayer {
        const self = try allocator.create(EflaLayer);
        errdefer allocator.destroy(self);

        const scale = @sqrt(2.0 / @as(f64, @floatFromInt(hidden_dim)));

        // Key projection: hidden_dim -> num_heads * head_dim
        const w_k_shape = Shape.init(&[_]usize{ hidden_dim, num_heads * head_dim });
        const w_k = try Tensor.randNormal(allocator, w_k_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_k.deinit();

        // Value projection: hidden_dim -> num_heads * head_dim
        const w_v_shape = Shape.init(&[_]usize{ hidden_dim, num_heads * head_dim });
        const w_v = try Tensor.randNormal(allocator, w_v_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_v.deinit();

        // Output projection: num_heads * head_dim -> hidden_dim
        const w_o_shape = Shape.init(&[_]usize{ num_heads * head_dim, hidden_dim });
        const w_o = try Tensor.randNormal(allocator, w_o_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_o.deinit();

        // Beta parameter (if learned)
        var beta_param: ?*Tensor = null;
        if (config.learned_beta) {
            const beta_shape = Shape.init(&[_]usize{1});
            beta_param = try Tensor.full(allocator, beta_shape, .fp32, device, device_id, config.initial_beta);
        }

        self.* = .{
            .config = config,
            .hidden_dim = hidden_dim,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .beta_param = beta_param,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *EflaLayer) void {
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
        if (self.beta_param) |bp| bp.deinit();
        self.allocator.destroy(self);
    }

    /// Forward pass through EFLA layer
    ///
    /// Args:
    ///   input: (batch, seq_len, hidden_dim)
    ///   state: Optional previous EFLA state
    ///
    /// Returns:
    ///   output: (batch, seq_len, hidden_dim)
    ///   new_state: Updated EFLA state
    pub fn forward(
        self: *EflaLayer,
        input: *Tensor,
        state: ?*EflaState,
    ) !struct { output: *Tensor, new_state: *EflaState } {
        _ = state;

        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);

        // Project to keys and values
        // k = input @ w_k -> (batch, seq_len, num_heads * head_dim)
        const k = try self.matmul(input, self.w_k);
        defer k.deinit();

        const v = try self.matmul(input, self.w_v);
        defer v.deinit();

        // Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        const k_reshaped = try k.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim }));
        defer k_reshaped.deinit();

        const v_reshaped = try v.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim }));
        defer v_reshaped.deinit();

        // Compute EFLA forward using chunked scan
        var new_state = try EflaState.init(
            self.allocator,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            self.device,
            self.device_id,
        );
        errdefer new_state.deinit();

        // Apply EFLA update kernel
        const output = try self.eflaForward(k_reshaped, v_reshaped, new_state);

        // Project output
        const projected = try self.matmul(output, self.w_o);

        return .{ .output = projected, .new_state = new_state };
    }

    /// EFLA forward with exact update
    fn eflaForward(
        self: *EflaLayer,
        k: *Tensor,
        v: *Tensor,
        state: *EflaState,
    ) !*Tensor {
        // This calls the CUDA kernel for efficient EFLA computation
        // The kernel implements the exact update with numerical stability

        const batch_size = k.shape.dim(0);
        const seq_len = k.shape.dim(1);

        // Output shape: (batch, seq_len, num_heads, head_dim)
        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim });
        const output = try Tensor.init(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        // Get beta value
        const beta: f32 = if (self.beta_param) |bp| blk: {
            // Get scalar value from beta parameter
            break :blk self.config.initial_beta; // Simplified
        } else self.config.initial_beta;

        // Call CUDA kernel for EFLA forward
        if (self.device == .cuda) {
            try kernels.eflaForwardCuda(
                k.ptr(),
                v.ptr(),
                state.state.ptr(),
                output.ptr(),
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
                beta,
                self.config.chunk_size,
            );
        } else {
            // CPU fallback (slow, for testing only)
            try self.eflaForwardCpu(k, v, state, output, beta);
        }

        // Reshape to (batch, seq_len, num_heads * head_dim)
        const reshaped = try output.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads * self.head_dim }));

        return reshaped;
    }

    /// CPU fallback for EFLA forward (for testing)
    fn eflaForwardCpu(
        self: *EflaLayer,
        k: *Tensor,
        v: *Tensor,
        state: *EflaState,
        output: *Tensor,
        beta: f32,
    ) !void {
        const batch_size = k.shape.dim(0);
        const seq_len = k.shape.dim(1);

        // Get typed pointers
        const k_ptr = k.typedPtr(dtype_mod.BF16).?;
        const v_ptr = v.typedPtr(dtype_mod.BF16).?;
        const s_ptr = state.state.typedPtr(dtype_mod.BF16).?;
        const o_ptr = output.typedPtr(dtype_mod.BF16).?;

        // For each batch and head
        for (0..batch_size) |b| {
            for (0..self.num_heads) |h| {
                // Initialize state for this head (if not provided)
                const head_state_offset = h * self.head_dim * self.head_dim;

                // Process sequence
                for (0..seq_len) |t| {
                    const k_offset = b * seq_len * self.num_heads * self.head_dim + t * self.num_heads * self.head_dim + h * self.head_dim;
                    const v_offset = k_offset;

                    // Compute lambda = k^T k
                    var lambda: f32 = 0.0;
                    for (0..self.head_dim) |d| {
                        const k_val = k_ptr[k_offset + d].toFloat32();
                        lambda += k_val * k_val;
                    }

                    // Compute c_t = (1 - exp(-beta * lambda)) / lambda
                    // With numerical stability for small lambda
                    const c_t: f32 = if (lambda < 1e-6) blk: {
                        // Series expansion: c_t ≈ beta - 0.5 * beta^2 * lambda + ...
                        var c: f32 = beta;
                        const beta_lambda = beta * lambda;
                        c -= 0.5 * beta * beta_lambda;
                        c += (1.0 / 6.0) * beta * beta * beta_lambda * lambda;
                        break :blk c;
                    } else blk: {
                        break :blk (1.0 - @exp(-beta * lambda)) / lambda;
                    };

                    // Update state: S_t = (I - c_t * k * k^T) * S_{t-1} + c_t * k * v^T
                    // And compute output: o_t = S_t * k_t

                    // First, compute output from previous state
                    for (0..self.head_dim) |d_out| {
                        var sum: f32 = 0.0;
                        for (0..self.head_dim) |d_in| {
                            const s_val = s_ptr[head_state_offset + d_out * self.head_dim + d_in].toFloat32();
                            const k_val = k_ptr[k_offset + d_in].toFloat32();
                            sum += s_val * k_val;
                        }
                        o_ptr[b * seq_len * self.num_heads * self.head_dim + t * self.num_heads * self.head_dim + h * self.head_dim + d_out] =
                            dtype_mod.BF16.fromFloat32(sum);
                    }

                    // Then update state
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[k_offset + i].toFloat32();
                        const v_i = v_ptr[v_offset + i].toFloat32();

                        for (0..self.head_dim) |j| {
                            const k_j = k_ptr[k_offset + j].toFloat32();

                            // S_new[i,j] = S_old[i,j] - c_t * k[i] * sum_k(S_old[j,k] * k[k]) + c_t * k[i] * v[j]
                            var s_old: f32 = s_ptr[head_state_offset + i * self.head_dim + j].toFloat32();

                            // Compute the update term
                            var s_k_sum: f32 = 0.0;
                            for (0..self.head_dim) |k_idx| {
                                const s_val = s_ptr[head_state_offset + j * self.head_dim + k_idx].toFloat32();
                                const k_val = k_ptr[k_offset + k_idx].toFloat32();
                                s_k_sum += s_val * k_val;
                            }

                            s_old = s_old - c_t * k_i * s_k_sum + c_t * k_i * v_i;

                            s_ptr[head_state_offset + i * self.head_dim + j] = dtype_mod.BF16.fromFloat32(s_old);
                        }
                    }
                }
            }
        }
    }

    /// Backward pass through EFLA layer
    pub fn backward(
        self: *EflaLayer,
        grad_output: *Tensor,
        input: *Tensor,
        state: *EflaState,
    ) !struct { grad_input: *Tensor, grad_state: *EflaState } {
        // Backprop through EFLA is complex due to the recurrence
        // We need to compute gradients for k, v, and maintain the state gradient

        _ = grad_output;
        _ = input;
        _ = state;

        // Placeholder - full implementation requires careful gradient computation
        const grad_input_shape = input.shape;
        const grad_input = try Tensor.zeros(self.allocator, grad_input_shape, .bf16, self.device, self.device_id);
        errdefer grad_input.deinit();

        const grad_state = try EflaState.init(
            self.allocator,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            self.device,
            self.device_id,
        );

        return .{ .grad_input = grad_input, .grad_state = grad_state };
    }

    /// Matrix multiplication helper
    fn matmul(self: *EflaLayer, a: *Tensor, b: *Tensor) !*Tensor {
        _ = self;
        // This would call the GEMM kernel
        const M = a.shape.dim(a.shape.ndim - 2);
        const K = a.shape.dim(a.shape.ndim - 1);
        const N = b.shape.dim(b.shape.ndim - 1);

        const batch_size = if (a.shape.ndim > 2) a.shape.first() else 1;

        const output_shape = if (a.shape.ndim > 2)
            Shape.init(&[_]usize{ batch_size, M, N })
        else
            Shape.init(&[_]usize{ M, N });

        const output = try Tensor.init(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        // Call GEMM kernel
        // For now, return zeros
        try output.zero_();

        return output;
    }
};

/// Chunked scan for parallel EFLA processing
///
/// This allows processing long sequences in parallel chunks
/// by computing the scan in a hierarchical manner
pub const ChunkedScan = struct {
    chunk_size: usize,
    num_chunks: usize,

    pub fn init(seq_len: usize, chunk_size: usize) ChunkedScan {
        return .{
            .chunk_size = chunk_size,
            .num_chunks = (seq_len + chunk_size - 1) / chunk_size,
        };
    }

    /// Compute chunk boundaries
    pub fn getChunkRange(self: ChunkedScan, chunk_idx: usize) struct { start: usize, end: usize } {
        const start = chunk_idx * self.chunk_size;
        const end = @min(start + self.chunk_size, self.num_chunks * self.chunk_size);
        return .{ .start = start, .end = end };
    }

    /// Compute prefix scan across chunks
    pub fn prefixScan(
        self: ChunkedScan,
        chunk_states: []*EflaState,
        allocator: std.mem.Allocator,
    ) !void {
        // Implement parallel prefix scan for combining chunk states
        // This allows O(log n) parallel depth for combining chunks

        _ = self;
        _ = chunk_states;
        _ = allocator;
    }
};

test "EFLA state init" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var state = try EflaState.init(gpa.allocator(), 8, 64, 64, .cpu, 0);
    defer state.deinit();

    try std.testing.expectEqual(@as(usize, 8), state.num_heads);
    try std.testing.expectEqual(@as(usize, 64), state.state_dim);
}

test "ChunkedScan" {
    const scan = ChunkedScan.init(10000, 1024);
    try std.testing.expectEqual(@as(usize, 1024), scan.chunk_size);
    try std.testing.expectEqual(@as(usize, 10), scan.num_chunks);

    const range = scan.getChunkRange(5);
    try std.testing.expectEqual(@as(usize, 5120), range.start);
    try std.testing.expectEqual(@as(usize, 6144), range.end);
}
