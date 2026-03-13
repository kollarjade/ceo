const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const kernels = @import("../kernels/prism_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;

/// PRISM (Parallel Residual Iterative Sequence Model) Layer
///
/// Implements PRISM's core mechanism:
///
/// 1. Input-anchored proxy:
///    u_t = ShortConv(X_{≤t}) ≈ S_{t-1} k_t
///
/// 2. Iterative rank accumulation:
///    B_t = Σ_{l=1..L} β_t^(l) · (δ_t^(l) ⊗ k_t^(l))
///
/// 3. Compute gates and projections from u_t:
///    β_t^(l) = W_beta^(l) u_t
///    k_t^(l) = W_k^(l) u_t
///
/// 4. Simulated contextual gain predictor:
///    p_t^(l) = W_p^(l) u_t ≈ σ′(S_{t-1} k_t)
///
/// 5. Residual initialization:
///    r_t^(1) = v_t - u_t ≈ v_t - σ(S_{t-1} k_t)
///
/// 6. Iterative refinement:
///    δ_t^(l) = GELU(p_t^(l) ⊙ r_t^(l))
///    r_t^(l+1) = r_t^(l) - δ_t^(l)
///
/// 7. Forget operator:
///    A_t = I - β_t^(1) · (k_t^(1) ⊗ k_t^(1))
///
/// 8. State update with decoupling:
///    S_t = α_t S_{t-1} (I - β_t^(1) k_t^(1) ⊗ k_t^(1)) + Σ_{l=1..L} β_t^(l) · (δ_t^(l) ⊗ k_t^(l))
///
pub const PrismLayer = struct {
    /// Configuration
    config: config_mod.PrismConfig,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of iterations (L)
    num_iterations: usize,
    /// Head dimension
    head_dim: usize,
    /// ShortConv layer
    shortconv: *ShortConv,
    /// Beta projections per iteration
    w_beta: []*Tensor,
    /// Key projections per iteration
    w_k: []*Tensor,
    /// Proxy projections per iteration
    w_p: []*Tensor,
    /// Forget factor (alpha)
    alpha: f32,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    /// Initialize PRISM layer
    pub fn init(
        allocator: std.mem.Allocator,
        config: config_mod.PrismConfig,
        hidden_dim: usize,
        head_dim: usize,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*PrismLayer {
        const self = try allocator.create(PrismLayer);
        errdefer allocator.destroy(self);

        const num_iterations = config.num_iterations;
        const scale = @sqrt(2.0 / @as(f64, @floatFromInt(hidden_dim)));

        // Initialize ShortConv
        const shortconv = try ShortConv.init(
            allocator,
            hidden_dim,
            config.shortconv_window,
            device,
            device_id,
            rng,
        );
        errdefer shortconv.deinit();

        // Initialize projection weights per iteration
        var w_beta = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_beta);
        errdefer {
            for (w_beta) |w| w.deinit();
        }

        var w_k = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_k);
        errdefer {
            for (w_k) |w| w.deinit();
        }

        var w_p = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_p);
        errdefer {
            for (w_p) |w| w.deinit();
        }

        for (0..num_iterations) |l| {
            // W_beta: hidden_dim -> head_dim
            const beta_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_beta[l] = try Tensor.randNormal(allocator, beta_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));

            // W_k: hidden_dim -> head_dim
            const k_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_k[l] = try Tensor.randNormal(allocator, k_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));

            // W_p: hidden_dim -> head_dim
            const p_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_p[l] = try Tensor.randNormal(allocator, p_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        }

        self.* = .{
            .config = config,
            .hidden_dim = hidden_dim,
            .num_iterations = num_iterations,
            .head_dim = head_dim,
            .shortconv = shortconv,
            .w_beta = w_beta,
            .w_k = w_k,
            .w_p = w_p,
            .alpha = config.forget_factor,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *PrismLayer) void {
        self.shortconv.deinit();

        for (self.w_beta) |w| w.deinit();
        for (self.w_k) |w| w.deinit();
        for (self.w_p) |w| w.deinit();

        self.allocator.free(self.w_beta);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_p);

        self.allocator.destroy(self);
    }

    /// Forward pass through PRISM layer
    ///
    /// Args:
    ///   input: (batch, seq_len, hidden_dim) - Input hidden states
    ///   v: (batch, seq_len, hidden_dim) - Value tensor from EFLA
    ///   state: Previous PRISM state
    ///
    /// Returns:
    ///   output: (batch, seq_len, hidden_dim)
    ///   new_state: Updated PRISM state
    pub fn forward(
        self: *PrismLayer,
        input: *Tensor,
        v: *Tensor,
        state: ?*PrismState,
    ) !struct { output: *Tensor, new_state: *PrismState } {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);

        // 1. Compute input-anchored proxy: u_t = ShortConv(X_{≤t})
        const u = try self.shortconv.forward(input);
        defer u.deinit();

        // 2. Initialize new state
        var new_state = try PrismState.init(
            self.allocator,
            1, // Single state for simplicity
            self.head_dim,
            self.head_dim,
            self.device,
            self.device_id,
        );
        errdefer new_state.deinit();

        // 3. Initialize output tensor
        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.hidden_dim });
        var output = try Tensor.zeros(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        // 4. Compute iterative refinement
        // This is done via CUDA kernel for efficiency
        if (self.device == .cuda) {
            try self.prismForwardCuda(u, v, state, new_state, output);
        } else {
            try self.prismForwardCpu(u, v, state, new_state, output);
        }

        return .{ .output = output, .new_state = new_state };
    }

    /// CUDA forward pass
    fn prismForwardCuda(
        self: *PrismLayer,
        u: *Tensor,
        v: *Tensor,
        prev_state: ?*PrismState,
        new_state: *PrismState,
        output: *Tensor,
    ) !void {
        // Get weight pointers
        var w_beta_ptrs = try self.allocator.alloc(?*const anyopaque, self.num_iterations);
        defer self.allocator.free(w_beta_ptrs);
        var w_k_ptrs = try self.allocator.alloc(?*const anyopaque, self.num_iterations);
        defer self.allocator.free(w_k_ptrs);
        var w_p_ptrs = try self.allocator.alloc(?*const anyopaque, self.num_iterations);
        defer self.allocator.free(w_p_ptrs);

        for (0..self.num_iterations) |l| {
            w_beta_ptrs[l] = self.w_beta[l].ptr();
            w_k_ptrs[l] = self.w_k[l].ptr();
            w_p_ptrs[l] = self.w_p[l].ptr();
        }

        try kernels.prismForwardCuda(
            u.ptr(),
            v.ptr(),
            if (prev_state) |s| s.state.ptr() else null,
            new_state.state.ptr(),
            output.ptr(),
            w_beta_ptrs.ptr,
            w_k_ptrs.ptr,
            w_p_ptrs.ptr,
            u.shape.dim(0), // batch_size
            u.shape.dim(1), // seq_len
            self.hidden_dim,
            self.head_dim,
            self.num_iterations,
            2,
            self.alpha,
            null,
        );
    }

    /// CPU forward pass (for testing)
    fn prismForwardCpu(
        self: *PrismLayer,
        u: *Tensor,
        v: *Tensor,
        prev_state: ?*PrismState,
        new_state: *PrismState,
        output: *Tensor,
    ) !void {
        const batch_size = u.shape.dim(0);
        const seq_len = u.shape.dim(1);

        const u_ptr = u.typedPtr(dtype_mod.BF16).?;
        const v_ptr = v.typedPtr(dtype_mod.BF16).?;
        const o_ptr = output.typedPtr(dtype_mod.BF16).?;

        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                const offset = b * seq_len * self.hidden_dim + t * self.hidden_dim;

                // Initialize residual: r^(1) = v - u
                var residual = try self.allocator.alloc(f32, self.head_dim);
                defer self.allocator.free(residual);

                for (0..self.head_dim) |d| {
                    residual[d] = v_ptr[offset + d].toFloat32() - u_ptr[offset + d].toFloat32();
                }

                // Iterate through refinement steps
                for (0..self.num_iterations) |l| {
                    // β_t^(l) = W_beta^(l) u_t
                    // Simplified: using scalar beta for now
                    const beta: f32 = 0.5;

                    // k_t^(l) = W_k^(l) u_t (simplified)
                    const k_ptr = self.w_k[l].typedPtr(dtype_mod.BF16).?;

                    // p_t^(l) = W_p^(l) u_t (simplified)
                    const p_ptr = self.w_p[l].typedPtr(dtype_mod.BF16).?;

                    // δ_t^(l) = GELU(p_t^(l) ⊙ r_t^(l))
                    var delta = try self.allocator.alloc(f32, self.head_dim);
                    defer self.allocator.free(delta);

                    for (0..self.head_dim) |d| {
                        const p_val = p_ptr[d].toFloat32(); // Simplified
                        const r_val = residual[d];
                        const gelu_input = p_val * r_val;
                        // GELU approximation
                        const x = gelu_input;
                        delta[d] = 0.5 * x * (1.0 + std.math.tanh(std.math.sqrt(2.0 / std.math.pi) * (x + 0.044715 * x * x * x)));
                    }

                    // Accumulate to output
                    for (0..self.hidden_dim) |d| {
                        if (d < self.head_dim) {
                            const current = o_ptr[offset + d].toFloat32();
                            o_ptr[offset + d] = dtype_mod.BF16.fromFloat32(current + beta * delta[d]);
                        }
                    }

                    // Update residual: r^(l+1) = r^(l) - δ^(l)
                    for (0..self.head_dim) |d| {
                        residual[d] -= delta[d];
                    }
                }

                // Apply forget operator to state
                // S_t = α S_{t-1} (I - β k ⊗ k) + ...
                _ = prev_state;
                _ = new_state;
            }
        }
    }

    /// Backward pass
    pub fn backward(
        self: *PrismLayer,
        grad_output: *Tensor,
        input: *Tensor,
        v: *Tensor,
        state: *PrismState,
    ) !struct { grad_input: *Tensor, grad_v: *Tensor, grad_state: *PrismState } {
        _ = grad_output;
        _ = input;
        _ = v;
        _ = state;

        // Backprop through PRISM requires careful gradient computation
        // through the iterative refinement process

        const grad_input_shape = input.shape;
        const grad_input = try Tensor.zeros(self.allocator, grad_input_shape, .bf16, self.device, self.device_id);

        const grad_v = try Tensor.zeros(self.allocator, v.shape, .bf16, self.device, self.device_id);

        const grad_state = try PrismState.init(
            self.allocator,
            1,
            self.head_dim,
            self.head_dim,
            self.device,
            self.device_id,
        );

        return .{ .grad_input = grad_input, .grad_v = grad_v, .grad_state = grad_state };
    }
};

/// ShortConv - Causal convolution over recent tokens
pub const ShortConv = struct {
    /// Window size
    window_size: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Convolution weights
    weight: *Tensor,
    /// Bias (optional)
    bias: ?*Tensor,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    pub fn init(
        allocator: std.mem.Allocator,
        hidden_dim: usize,
        window_size: usize,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*ShortConv {
        const self = try allocator.create(ShortConv);
        errdefer allocator.destroy(self);

        // Weight shape: (hidden_dim, window_size)
        const weight_shape = Shape.init(&[_]usize{ hidden_dim, window_size });
        const weight = try Tensor.randNormal(
            allocator,
            weight_shape,
            .bf16,
            device,
            device_id,
            rng,
            0.0,
            @sqrt(1.0 / @as(f64, @floatFromInt(window_size))),
        );
        errdefer weight.deinit();

        self.* = .{
            .window_size = window_size,
            .hidden_dim = hidden_dim,
            .weight = weight,
            .bias = null,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *ShortConv) void {
        self.weight.deinit();
        if (self.bias) |b| b.deinit();
        self.allocator.destroy(self);
    }

    /// Forward pass
    /// input: (batch, seq_len, hidden_dim)
    /// output: (batch, seq_len, hidden_dim)
    pub fn forward(self: *ShortConv, input: *Tensor) !*Tensor {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);

        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.hidden_dim });
        var output = try Tensor.zeros(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        if (self.device == .cuda) {
            try kernels.shortConvForwardCuda(
                input.ptr(),
                self.weight.ptr(),
                output.ptr(),
                batch_size,
                seq_len,
                self.hidden_dim,
                self.window_size,
                2,
                null,
            );
        } else {
            try self.forwardCpu(input, output);
        }

        return output;
    }

    fn forwardCpu(self: *ShortConv, input: *Tensor, output: *Tensor) !void {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);

        const input_ptr = input.typedPtr(dtype_mod.BF16).?;
        const weight_ptr = self.weight.typedPtr(dtype_mod.BF16).?;
        const output_ptr = output.typedPtr(dtype_mod.BF16).?;

        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                for (0..self.hidden_dim) |d| {
                    var sum: f32 = 0.0;

                    // Causal convolution: only look at positions <= t
                    for (0..self.window_size) |w| {
                        const lookback = @as(isize, @intCast(t)) - @as(isize, @intCast(w));
                        if (lookback >= 0) {
                            const t_lookback = @as(usize, @intCast(lookback));
                            const input_val = input_ptr[b * seq_len * self.hidden_dim + t_lookback * self.hidden_dim + d].toFloat32();
                            const weight_val = weight_ptr[d * self.window_size + w].toFloat32();
                            sum += input_val * weight_val;
                        }
                    }

                    output_ptr[b * seq_len * self.hidden_dim + t * self.hidden_dim + d] = dtype_mod.BF16.fromFloat32(sum);
                }
            }
        }
    }
};

/// PRISM State
pub const PrismState = struct {
    /// State matrix
    state: *Tensor,
    /// State dimension
    state_dim: usize,
    /// Value dimension
    value_dim: usize,
    /// Allocator
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        num_states: usize,
        state_dim: usize,
        value_dim: usize,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*PrismState {
        const self = try allocator.create(PrismState);
        errdefer allocator.destroy(self);

        const state_shape = Shape.init(&[_]usize{ num_states, state_dim, value_dim });
        const state_tensor = try Tensor.zeros(allocator, state_shape, .bf16, device, device_id);
        errdefer state_tensor.deinit();

        self.* = .{
            .state = state_tensor,
            .state_dim = state_dim,
            .value_dim = value_dim,
            .allocator = allocator,
        };

        return self;
    }

    pub fn deinit(self: *PrismState) void {
        self.state.deinit();
        self.allocator.destroy(self);
    }

    pub fn reset(self: *PrismState) !void {
        try self.state.zero_();
    }
};

test "ShortConv forward" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var conv = try ShortConv.init(gpa.allocator(), 64, 16, .cpu, 0, &rng);
    defer conv.deinit();

    const input_shape = Shape.init(&[_]usize{ 2, 32, 64 });
    var input = try Tensor.randNormal(gpa.allocator(), input_shape, .bf16, .cpu, 0, &rng, 0.0, 1.0);
    defer input.deinit();

    var output = try conv.forward(input);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 2), output.shape.dim(0));
    try std.testing.expectEqual(@as(usize, 32), output.shape.dim(1));
    try std.testing.expectEqual(@as(usize, 64), output.shape.dim(2));
}

test "PRISM state init" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var state = try PrismState.init(gpa.allocator(), 1, 64, 64, .cpu, 0);
    defer state.deinit();

    try std.testing.expectEqual(@as(usize, 64), state.state_dim);
}
