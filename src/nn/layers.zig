const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const kernels = @import("../kernels/nn_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;
pub const BF16 = dtype_mod.BF16;

/// RMSNorm Layer
///
/// Computes: y = x * rsqrt(mean(x^2) + eps) * weight
pub const RMSNorm = struct {
    /// Normalized shape (size of last dimension)
    normalized_shape: usize,
    /// Weight parameter
    weight: *Tensor,
    /// Epsilon for numerical stability
    eps: f32,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Device
    device: tensor_mod.Device,
    /// Device ID
    device_id: i32,

    pub fn init(
        allocator: std.mem.Allocator,
        normalized_shape: usize,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*RMSNorm {
        const self = try allocator.create(RMSNorm);
        errdefer allocator.destroy(self);

        // Initialize weight to ones
        const weight_shape = Shape.init(&[_]usize{normalized_shape});
        const weight = try Tensor.full(allocator, weight_shape, .bf16, device, device_id, 1.0);
        errdefer weight.deinit();

        self.* = .{
            .normalized_shape = normalized_shape,
            .weight = weight,
            .eps = 1e-6,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *RMSNorm) void {
        self.weight.deinit();
        self.allocator.destroy(self);
    }

    /// Forward pass
    /// input: (..., normalized_shape)
    /// output: (..., normalized_shape)
    pub fn forward(self: *RMSNorm, input: *Tensor) !*Tensor {
        var output = try Tensor.init(self.allocator, input.shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        if (self.device == .cuda) {
            try kernels.rmsNormForwardCuda(
                input.ptr(),
                self.weight.ptr(),
                output.ptr(),
                input.shape.numel(),
                self.normalized_shape,
                self.eps,
            );
        } else {
            try self.forwardCpu(input, output);
        }

        return output;
    }

    fn forwardCpu(self: *RMSNorm, input: *Tensor, output: *Tensor) !void {
        const numel = input.shape.numel();
        const n = numel / self.normalized_shape;
        const ndim = input.shape.ndim;
        const last_dim = input.shape.dims[ndim - 1];

        const input_ptr = input.typedPtr(BF16).?;
        const weight_ptr = self.weight.typedPtr(BF16).?;
        const output_ptr = output.typedPtr(BF16).?;

        for (0..n) |i| {
            // Compute mean of squares
            var sum_sq: f32 = 0.0;
            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                sum_sq += val * val;
            }
            const mean_sq = sum_sq / @as(f32, @floatFromInt(last_dim));
            const rsqrt = 1.0 / @sqrt(mean_sq + self.eps);

            // Normalize and scale
            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                output_ptr[i * last_dim + j] = BF16.fromFloat32(val * rsqrt * w);
            }
        }
    }

    /// Backward pass
    pub fn backward(
        self: *RMSNorm,
        grad_output: *Tensor,
        input: *Tensor,
    ) !*Tensor {
        var grad_input = try Tensor.init(self.allocator, input.shape, .bf16, self.device, self.device_id);
        errdefer grad_input.deinit();

        if (self.device == .cuda) {
            try kernels.rmsNormBackwardCuda(
                grad_output.ptr(),
                input.ptr(),
                self.weight.ptr(),
                grad_input.ptr(),
                null, // grad_weight
                input.shape.numel(),
                self.normalized_shape,
                self.eps,
            );
        } else {
            try self.backwardCpu(grad_output, input, grad_input);
        }

        return grad_input;
    }

    fn backwardCpu(self: *RMSNorm, grad_output: *Tensor, input: *Tensor, grad_input: *Tensor) !void {
        const numel = input.shape.numel();
        const n = numel / self.normalized_shape;
        const last_dim = input.shape.dims[input.shape.ndim - 1];

        const grad_out_ptr = grad_output.typedPtr(BF16).?;
        const input_ptr = input.typedPtr(BF16).?;
        const weight_ptr = self.weight.typedPtr(BF16).?;
        const grad_in_ptr = grad_input.typedPtr(BF16).?;

        for (0..n) |i| {
            // Recompute normalization constant
            var sum_sq: f32 = 0.0;
            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                sum_sq += val * val;
            }
            const mean_sq = sum_sq / @as(f32, @floatFromInt(last_dim));
            const rsqrt = 1.0 / @sqrt(mean_sq + self.eps);

            // Compute sum of weighted gradients
            var sum_grad_weighted: f32 = 0.0;
            for (0..last_dim) |j| {
                const g = grad_out_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                const x = input_ptr[i * last_dim + j].toFloat32();
                sum_grad_weighted += g * w * x;
            }

            // Backprop through RMSNorm
            const norm_factor = rsqrt / @as(f32, @floatFromInt(last_dim));
            for (0..last_dim) |j| {
                const g = grad_out_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                const x = input_ptr[i * last_dim + j].toFloat32();
                const grad = w * rsqrt * g - x * norm_factor * sum_grad_weighted;
                grad_in_ptr[i * last_dim + j] = BF16.fromFloat32(grad);
            }
        }
    }
};

/// LayerNorm Layer
pub const LayerNorm = struct {
    normalized_shape: usize,
    weight: *Tensor,
    bias: *Tensor,
    eps: f32,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(
        allocator: std.mem.Allocator,
        normalized_shape: usize,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*LayerNorm {
        const self = try allocator.create(LayerNorm);
        errdefer allocator.destroy(self);

        const shape = Shape.init(&[_]usize{normalized_shape});

        const weight = try Tensor.full(allocator, shape, .bf16, device, device_id, 1.0);
        errdefer weight.deinit();

        const bias = try Tensor.zeros(allocator, shape, .bf16, device, device_id);
        errdefer bias.deinit();

        self.* = .{
            .normalized_shape = normalized_shape,
            .weight = weight,
            .bias = bias,
            .eps = 1e-5,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *LayerNorm) void {
        self.weight.deinit();
        self.bias.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *LayerNorm, input: *Tensor) !*Tensor {
        var output = try Tensor.init(self.allocator, input.shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        if (self.device == .cuda) {
            try kernels.layerNormForwardCuda(
                input.ptr(),
                self.weight.ptr(),
                self.bias.ptr(),
                output.ptr(),
                input.shape.numel(),
                self.normalized_shape,
                self.eps,
            );
        } else {
            try self.forwardCpu(input, output);
        }

        return output;
    }

    fn forwardCpu(self: *LayerNorm, input: *Tensor, output: *Tensor) !void {
        const numel = input.shape.numel();
        const n = numel / self.normalized_shape;
        const last_dim = input.shape.dims[input.shape.ndim - 1];

        const input_ptr = input.typedPtr(BF16).?;
        const weight_ptr = self.weight.typedPtr(BF16).?;
        const bias_ptr = self.bias.typedPtr(BF16).?;
        const output_ptr = output.typedPtr(BF16).?;

        for (0..n) |i| {
            // Compute mean and variance
            var sum: f32 = 0.0;
            var sum_sq: f32 = 0.0;

            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                sum += val;
                sum_sq += val * val;
            }

            const mean = sum / @as(f32, @floatFromInt(last_dim));
            const variance = sum_sq / @as(f32, @floatFromInt(last_dim)) - mean * mean;
            const inv_std = 1.0 / @sqrt(variance + self.eps);

            // Normalize
            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                const b = bias_ptr[j].toFloat32();
                const normalized = (val - mean) * inv_std;
                output_ptr[i * last_dim + j] = BF16.fromFloat32(normalized * w + b);
            }
        }
    }
};

/// GELU activation function
pub const GELU = struct {
    approximate: bool, // Use fast approximation

    pub fn init(approximate: bool) GELU {
        return .{ .approximate = approximate };
    }

    /// Forward pass
    pub fn forward(self: GELU, allocator: std.mem.Allocator, input: *Tensor) !*Tensor {
        var output = try Tensor.init(allocator, input.shape, .bf16, input.device, input.device_id);
        errdefer output.deinit();

        if (input.device == .cuda) {
            try kernels.geluForwardCuda(input.ptr(), output.ptr(), input.shape.numel(), self.approximate);
        } else {
            try self.forwardCpu(input, output);
        }

        return output;
    }

    fn forwardCpu(self: GELU, input: *Tensor, output: *Tensor) !void {
        const input_ptr = input.typedPtr(BF16).?;
        const output_ptr = output.typedPtr(BF16).?;
        const numel = input.shape.numel();

        for (0..numel) |i| {
            const x = input_ptr[i].toFloat32();
            const y = if (self.approximate)
                geluApproximate(x)
            else
                geluExact(x);
            output_ptr[i] = BF16.fromFloat32(y);
        }
    }

    /// Exact GELU: x * Φ(x) where Φ is the CDF of standard normal
    fn geluExact(x: f32) f32 {
        const sqrt2 = std.math.sqrt(2.0);
        const cdf = 0.5 * (1.0 + std.math.erf(x / sqrt2));
        return x * cdf;
    }

    /// Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    fn geluApproximate(x: f32) f32 {
        const sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/π)
        const x3 = x * x * x;
        return 0.5 * x * (1.0 + std.math.tanh(sqrt_2_over_pi * (x + 0.044715 * x3)));
    }
};

/// SiLU (Swish) activation: x * sigmoid(x)
pub const SiLU = struct {
    pub fn forward(allocator: std.mem.Allocator, input: *Tensor) !*Tensor {
        var output = try Tensor.init(allocator, input.shape, .bf16, input.device, input.device_id);
        errdefer output.deinit();

        const input_ptr = input.typedPtr(BF16).?;
        const output_ptr = output.typedPtr(BF16).?;
        const numel = input.shape.numel();

        for (0..numel) |i| {
            const x = input_ptr[i].toFloat32();
            const sigmoid = 1.0 / (1.0 + @exp(-x));
            output_ptr[i] = BF16.fromFloat32(x * sigmoid);
        }

        return output;
    }
};

/// SwiGLU activation: Swish(gate) * linear
pub const SwiGLU = struct {
    pub fn forward(
        allocator: std.mem.Allocator,
        gate: *Tensor,
        linear: *Tensor,
    ) !*Tensor {
        std.debug.assert(gate.shape.equalTo(linear.shape));

        var output = try Tensor.init(allocator, gate.shape, .bf16, gate.device, gate.device_id);
        errdefer output.deinit();

        const gate_ptr = gate.typedPtr(BF16).?;
        const linear_ptr = linear.typedPtr(BF16).?;
        const output_ptr = output.typedPtr(BF16).?;
        const numel = gate.shape.numel();

        for (0..numel) |i| {
            const g = gate_ptr[i].toFloat32();
            const l = linear_ptr[i].toFloat32();
            const sigmoid = 1.0 / (1.0 + @exp(-g));
            const swish = g * sigmoid;
            output_ptr[i] = BF16.fromFloat32(swish * l);
        }

        return output;
    }
};

/// Softmax layer
pub const Softmax = struct {
    dim: usize,

    pub fn init(dim: usize) Softmax {
        return .{ .dim = dim };
    }

    pub fn forward(self: Softmax, allocator: std.mem.Allocator, input: *Tensor) !*Tensor {
        var output = try Tensor.init(allocator, input.shape, .bf16, input.device, input.device_id);
        errdefer output.deinit();

        if (input.device == .cuda) {
            try kernels.softmaxForwardCuda(
                input.ptr(),
                output.ptr(),
                input.shape.numel(),
                input.shape.dims[self.dim],
            );
        } else {
            try self.forwardCpu(input, output);
        }

        return output;
    }

    fn forwardCpu(self: Softmax, input: *Tensor, output: *Tensor) !void {
        const ndim = input.shape.ndim;
        const dim_size = input.shape.dims[self.dim];

        // Compute outer size (all dims before self.dim)
        var outer_size: usize = 1;
        for (input.shape.dims[0..self.dim]) |d| {
            outer_size *= d;
        }

        // Compute inner size (all dims after self.dim)
        var inner_size: usize = 1;
        for (input.shape.dims[self.dim + 1 .. ndim]) |d| {
            inner_size *= d;
        }

        const input_ptr = input.typedPtr(BF16).?;
        const output_ptr = output.typedPtr(BF16).?;

        for (0..outer_size) |i| {
            for (0..inner_size) |j| {
                const offset = i * dim_size * inner_size + j;

                // Find max for numerical stability
                var max_val: f32 = -std.math.inf(f32);
                for (0..dim_size) |k| {
                    const val = input_ptr[offset + k * inner_size].toFloat32();
                    if (val > max_val) max_val = val;
                }

                // Compute exp and sum
                var sum: f32 = 0.0;
                for (0..dim_size) |k| {
                    const val = input_ptr[offset + k * inner_size].toFloat32();
                    const exp_val = @exp(val - max_val);
                    output_ptr[offset + k * inner_size] = BF16.fromFloat32(exp_val);
                    sum += exp_val;
                }

                // Normalize
                for (0..dim_size) |k| {
                    const val = output_ptr[offset + k * inner_size].toFloat32();
                    output_ptr[offset + k * inner_size] = BF16.fromFloat32(val / sum);
                }
            }
        }
    }
};

/// Dropout layer
pub const Dropout = struct {
    p: f32, // Dropout probability
    training: bool,

    pub fn init(p: f32) Dropout {
        return .{ .p = p, .training = true };
    }

    pub fn forward(
        self: Dropout,
        allocator: std.mem.Allocator,
        input: *Tensor,
        rng: ?*std.Random,
    ) !*Tensor {
        if (!self.training or self.p == 0.0) {
            return input.to(allocator, input.device, input.device_id);
        }

        var output = try Tensor.init(allocator, input.shape, .bf16, input.device, input.device_id);
        errdefer output.deinit();

        const input_ptr = input.typedPtr(BF16).?;
        const output_ptr = output.typedPtr(BF16).?;
        const numel = input.shape.numel();

        const scale = 1.0 / (1.0 - self.p);

        if (rng) |r| {
            for (0..numel) |i| {
                const keep = r.float(f32) > self.p;
                const val = if (keep) input_ptr[i].toFloat32() * scale else 0.0;
                output_ptr[i] = BF16.fromFloat32(val);
            }
        } else {
            // No RNG provided, just scale
            for (0..numel) |i| {
                output_ptr[i] = BF16.fromFloat32(input_ptr[i].toFloat32() * scale);
            }
        }

        return output;
    }
};

/// Linear (dense) layer
pub const Linear = struct {
    in_features: usize,
    out_features: usize,
    weight: *Tensor, // Shape: (in_features, out_features)
    bias: ?*Tensor, // Shape: (out_features,)
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*Linear {
        const self = try allocator.create(Linear);
        errdefer allocator.destroy(self);

        const scale = @sqrt(2.0 / @as(f64, @floatFromInt(in_features)));

        const weight_shape = Shape.init(&[_]usize{ in_features, out_features });
        const weight = try Tensor.randNormal(allocator, weight_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer weight.deinit();

        var bias: ?*Tensor = null;
        if (has_bias) {
            const bias_shape = Shape.init(&[_]usize{out_features});
            bias = try Tensor.zeros(allocator, bias_shape, .bf16, device, device_id);
        }

        self.* = .{
            .in_features = in_features,
            .out_features = out_features,
            .weight = weight,
            .bias = bias,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *Linear) void {
        self.weight.deinit();
        if (self.bias) |b| b.deinit();
        self.allocator.destroy(self);
    }

    /// Forward: y = x @ W + b
    /// input: (..., in_features)
    /// output: (..., out_features)
    pub fn forward(self: *Linear, input: *Tensor) !*Tensor {
        // Compute output shape
        var out_shape = input.shape;
        out_shape.dims[out_shape.ndim - 1] = self.out_features;

        var output = try Tensor.zeros(self.allocator, out_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        // This would call the GEMM kernel
        // For now, simplified implementation
        if (self.device == .cuda) {
            try kernels.gemmForwardCuda(
                input.ptr(),
                self.weight.ptr(),
                if (self.bias) |b| b.ptr() else null,
                output.ptr(),
                input.shape.numel() / self.in_features,
                self.in_features,
                self.out_features,
            );
        } else {
            try self.forwardCpu(input, output);
        }

        return output;
    }

    fn forwardCpu(self: *Linear, input: *Tensor, output: *Tensor) !void {
        const batch_size = input.shape.numel() / self.in_features;

        const input_ptr = input.typedPtr(BF16).?;
        const weight_ptr = self.weight.typedPtr(BF16).?;
        const bias_ptr = if (self.bias) |b| b.typedPtr(BF16) else null;
        const output_ptr = output.typedPtr(BF16).?;

        for (0..batch_size) |b| {
            for (0..self.out_features) |o| {
                var sum: f32 = 0.0;
                for (0..self.in_features) |i| {
                    const x = input_ptr[b * self.in_features + i].toFloat32();
                    const w = weight_ptr[i * self.out_features + o].toFloat32();
                    sum += x * w;
                }
                if (bias_ptr) |bp| {
                    sum += bp[o].toFloat32();
                }
                output_ptr[b * self.out_features + o] = BF16.fromFloat32(sum);
            }
        }
    }
};

/// Embedding layer
pub const Embedding = struct {
    num_embeddings: usize, // Vocabulary size
    embedding_dim: usize,
    weight: *Tensor, // Shape: (num_embeddings, embedding_dim)
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(
        allocator: std.mem.Allocator,
        num_embeddings: usize,
        embedding_dim: usize,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*Embedding {
        const self = try allocator.create(Embedding);
        errdefer allocator.destroy(self);

        const scale = @sqrt(1.0 / @as(f64, @floatFromInt(embedding_dim)));

        const weight_shape = Shape.init(&[_]usize{ num_embeddings, embedding_dim });
        const weight = try Tensor.randNormal(allocator, weight_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer weight.deinit();

        self.* = .{
            .num_embeddings = num_embeddings,
            .embedding_dim = embedding_dim,
            .weight = weight,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };

        return self;
    }

    pub fn deinit(self: *Embedding) void {
        self.weight.deinit();
        self.allocator.destroy(self);
    }

    /// Forward: lookup embeddings for input indices
    /// input: (...,) containing indices
    /// output: (..., embedding_dim)
    pub fn forward(self: *Embedding, input: *Tensor) !*Tensor {
        // Input is indices (int32)
        const indices = input.typedPtr(u32).?;
        const weight_ptr = self.weight.typedPtr(BF16).?;

        var out_shape = input.shape;
        out_shape.dims[out_shape.ndim] = self.embedding_dim;
        out_shape.ndim += 1;

        var output = try Tensor.zeros(self.allocator, out_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        const output_ptr = output.typedPtr(BF16).?;
        const num_indices = input.shape.numel();

        for (0..num_indices) |i| {
            const idx = indices[i];
            if (idx >= self.num_embeddings) {
                return error.IndexOutOfRange;
            }

            for (0..self.embedding_dim) |d| {
                output_ptr[i * self.embedding_dim + d] = weight_ptr[idx * self.embedding_dim + d];
            }
        }

        return output;
    }
};

test "RMSNorm forward" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var norm = try RMSNorm.init(gpa.allocator(), 64, .cpu, 0);
    defer norm.deinit();

    const shape = Shape.init(&[_]usize{ 2, 32, 64 });
    var input = try Tensor.full(gpa.allocator(), shape, .bf16, .cpu, 0, 1.0);
    defer input.deinit();

    var output = try norm.forward(input);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 64), output.shape.last());
}

test "GELU forward" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const gelu = GELU.init(true);

    const shape = Shape.init(&[_]usize{ 4, 8 });
    var input = try Tensor.zeros(gpa.allocator(), shape, .bf16, .cpu, 0);
    defer input.deinit();

    var output = try gelu.forward(gpa.allocator(), input);
    defer output.deinit();

    // GELU(0) should be close to 0
    const ptr = output.typedPtr(BF16).?;
    try std.testing.expectApproxEqRel(@as(f32, 0.0), ptr[0].toFloat32(), 0.01);
}

test "Softmax forward" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const softmax = Softmax.init(1);

    const shape = Shape.init(&[_]usize{ 2, 4 });
    var input = try Tensor.full(gpa.allocator(), shape, .bf16, .cpu, 0, 1.0);
    defer input.deinit();

    var output = try softmax.forward(gpa.allocator(), input);
    defer output.deinit();

    // Each row should sum to 1
    const ptr = output.typedPtr(BF16).?;
    var sum: f32 = 0.0;
    for (0..4) |i| {
        sum += ptr[i].toFloat32();
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum, 0.01);
}
