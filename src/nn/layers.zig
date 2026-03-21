const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const kernels = @import("../kernels/nn_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;
pub const BF16 = dtype_mod.BF16;

fn cloneTensor(allocator: std.mem.Allocator, src: *Tensor) !*Tensor {
    return try src.to(allocator, src.device, src.device_id);
}

fn storeGrad(param: *Tensor, allocator: std.mem.Allocator, grad: *Tensor) !void {
    if (@hasField(@TypeOf(param.*), "grad")) {
        if (param.grad) |existing| existing.deinit();
        param.grad = try cloneTensor(allocator, grad);
    }
}

pub const RMSNorm = struct {
    normalized_shape: usize,
    weight: *Tensor,
    eps: f32,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, normalized_shape: usize, device: tensor_mod.Device, device_id: i32) !*RMSNorm {
        std.debug.assert(normalized_shape > 0);
        const self = try allocator.create(RMSNorm);
        errdefer allocator.destroy(self);
        const weight_shape = Shape.init(&[_]usize{normalized_shape});
        const weight = try Tensor.full(allocator, weight_shape, .bf16, device, device_id, 1.0);
        errdefer weight.deinit();
        self.* = .{ .normalized_shape = normalized_shape, .weight = weight, .eps = 1e-6, .allocator = allocator, .device = device, .device_id = device_id };
        return self;
    }

    pub fn deinit(self: *RMSNorm) void {
        self.weight.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *RMSNorm, input: *Tensor) !*Tensor {
        std.debug.assert(input.shape.ndim > 0);
        std.debug.assert(input.shape.dims[input.shape.ndim - 1] == self.normalized_shape);
        if (input.device != self.device or input.device_id != self.device_id) return error.InvalidDevice;
        var output = try Tensor.init(self.allocator, input.shape, input.dtype, self.device, self.device_id);
        errdefer output.deinit();
        if (self.device == .cuda) {
            try kernels.rmsNormForwardCuda(input.ptr(), self.weight.ptr(), output.ptr(), input.shape.numel(), self.normalized_shape, self.eps);
        } else {
            try self.forwardCpu(input, output);
        }
        return output;
    }

    fn forwardCpu(self: *RMSNorm, input: *Tensor, output: *Tensor) !void {
        const numel = input.shape.numel();
        const last_dim = self.normalized_shape;
        const n = numel / last_dim;
        const input_ptr = input.typedPtr(BF16) orelse return error.InvalidDType;
        const weight_ptr = self.weight.typedPtr(BF16) orelse return error.InvalidDType;
        const output_ptr = output.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..n) |i| {
            var sum_sq: f32 = 0.0;
            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                sum_sq += val * val;
            }
            const mean_sq = sum_sq / @as(f32, @floatFromInt(last_dim));
            const rsqrt = 1.0 / @sqrt(mean_sq + self.eps);
            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                output_ptr[i * last_dim + j] = BF16.fromFloat32(val * rsqrt * w);
            }
        }
    }

    pub fn backward(self: *RMSNorm, grad_output: *Tensor, input: *Tensor) !struct { grad_input: *Tensor, grad_weight: *Tensor } {
        std.debug.assert(input.shape.ndim > 0);
        std.debug.assert(input.shape.dims[input.shape.ndim - 1] == self.normalized_shape);
        if (grad_output.device != self.device or input.device != self.device) return error.InvalidDevice;
        var grad_input = try Tensor.init(self.allocator, input.shape, input.dtype, self.device, self.device_id);
        errdefer grad_input.deinit();
        const weight_shape = Shape.init(&[_]usize{self.normalized_shape});
        var grad_weight = try Tensor.zeros(self.allocator, weight_shape, .bf16, self.device, self.device_id);
        errdefer grad_weight.deinit();
        if (self.device == .cuda) {
            try kernels.rmsNormBackwardCuda(grad_output.ptr(), input.ptr(), self.weight.ptr(), grad_input.ptr(), grad_weight.ptr(), input.shape.numel(), self.normalized_shape, self.eps);
        } else {
            try self.backwardCpu(grad_output, input, grad_input, grad_weight);
        }
        try storeGrad(self.weight, self.allocator, grad_weight);
        return .{ .grad_input = grad_input, .grad_weight = grad_weight };
    }

    fn backwardCpu(self: *RMSNorm, grad_output: *Tensor, input: *Tensor, grad_input: *Tensor, grad_weight: *Tensor) !void {
        const numel = input.shape.numel();
        const last_dim = self.normalized_shape;
        const n = numel / last_dim;
        const grad_out_ptr = grad_output.typedPtr(BF16) orelse return error.InvalidDType;
        const input_ptr = input.typedPtr(BF16) orelse return error.InvalidDType;
        const weight_ptr = self.weight.typedPtr(BF16) orelse return error.InvalidDType;
        const grad_in_ptr = grad_input.typedPtr(BF16) orelse return error.InvalidDType;
        const grad_w_ptr = grad_weight.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..n) |i| {
            var sum_sq: f32 = 0.0;
            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                sum_sq += val * val;
            }
            const mean_sq = sum_sq / @as(f32, @floatFromInt(last_dim));
            const rsqrt = 1.0 / @sqrt(mean_sq + self.eps);
            var sum_grad_weighted: f32 = 0.0;
            for (0..last_dim) |j| {
                const g = grad_out_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                const x = input_ptr[i * last_dim + j].toFloat32();
                sum_grad_weighted += g * w * x;
                const normalized = x * rsqrt;
                grad_w_ptr[j] = BF16.fromFloat32(grad_w_ptr[j].toFloat32() + g * normalized);
            }
            const norm_factor = rsqrt * rsqrt * rsqrt / @as(f32, @floatFromInt(last_dim));
            for (0..last_dim) |j| {
                const g = grad_out_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                const x = input_ptr[i * last_dim + j].toFloat32();
                grad_in_ptr[i * last_dim + j] = BF16.fromFloat32(w * rsqrt * g - x * norm_factor * sum_grad_weighted);
            }
        }
    }
};

pub const LayerNorm = struct {
    normalized_shape: usize,
    weight: *Tensor,
    bias: *Tensor,
    eps: f32,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, normalized_shape: usize, device: tensor_mod.Device, device_id: i32) !*LayerNorm {
        std.debug.assert(normalized_shape > 0);
        const self = try allocator.create(LayerNorm);
        errdefer allocator.destroy(self);
        const shape = Shape.init(&[_]usize{normalized_shape});
        const weight = try Tensor.full(allocator, shape, .bf16, device, device_id, 1.0);
        errdefer weight.deinit();
        const bias = try Tensor.zeros(allocator, shape, .bf16, device, device_id);
        errdefer bias.deinit();
        self.* = .{ .normalized_shape = normalized_shape, .weight = weight, .bias = bias, .eps = 1e-5, .allocator = allocator, .device = device, .device_id = device_id };
        return self;
    }

    pub fn deinit(self: *LayerNorm) void {
        self.weight.deinit();
        self.bias.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *LayerNorm, input: *Tensor) !*Tensor {
        std.debug.assert(input.shape.ndim > 0);
        std.debug.assert(input.shape.dims[input.shape.ndim - 1] == self.normalized_shape);
        if (input.device != self.device or input.device_id != self.device_id) return error.InvalidDevice;
        var output = try Tensor.init(self.allocator, input.shape, input.dtype, self.device, self.device_id);
        errdefer output.deinit();
        if (self.device == .cuda) {
            try kernels.layerNormForwardCuda(input.ptr(), self.weight.ptr(), self.bias.ptr(), output.ptr(), input.shape.numel(), self.normalized_shape, self.eps);
        } else {
            try self.forwardCpu(input, output);
        }
        return output;
    }

    fn forwardCpu(self: *LayerNorm, input: *Tensor, output: *Tensor) !void {
        const numel = input.shape.numel();
        const last_dim = self.normalized_shape;
        const n = numel / last_dim;
        const input_ptr = input.typedPtr(BF16) orelse return error.InvalidDType;
        const weight_ptr = self.weight.typedPtr(BF16) orelse return error.InvalidDType;
        const bias_ptr = self.bias.typedPtr(BF16) orelse return error.InvalidDType;
        const output_ptr = output.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..n) |i| {
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
            for (0..last_dim) |j| {
                const val = input_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                const b = bias_ptr[j].toFloat32();
                output_ptr[i * last_dim + j] = BF16.fromFloat32((val - mean) * inv_std * w + b);
            }
        }
    }

    pub fn backward(self: *LayerNorm, grad_output: *Tensor, input: *Tensor) !struct { grad_input: *Tensor, grad_weight: *Tensor, grad_bias: *Tensor } {
        std.debug.assert(input.shape.ndim > 0);
        std.debug.assert(input.shape.dims[input.shape.ndim - 1] == self.normalized_shape);
        var grad_input = try Tensor.init(self.allocator, input.shape, input.dtype, self.device, self.device_id);
        errdefer grad_input.deinit();
        const param_shape = Shape.init(&[_]usize{self.normalized_shape});
        var grad_weight = try Tensor.zeros(self.allocator, param_shape, .bf16, self.device, self.device_id);
        errdefer grad_weight.deinit();
        var grad_bias = try Tensor.zeros(self.allocator, param_shape, .bf16, self.device, self.device_id);
        errdefer grad_bias.deinit();
        if (self.device == .cuda) {
            try kernels.layerNormBackwardCuda(grad_output.ptr(), input.ptr(), self.weight.ptr(), grad_input.ptr(), grad_weight.ptr(), grad_bias.ptr(), input.shape.numel(), self.normalized_shape, self.eps);
        } else {
            try self.backwardCpu(grad_output, input, grad_input, grad_weight, grad_bias);
        }
        try storeGrad(self.weight, self.allocator, grad_weight);
        try storeGrad(self.bias, self.allocator, grad_bias);
        return .{ .grad_input = grad_input, .grad_weight = grad_weight, .grad_bias = grad_bias };
    }

    fn backwardCpu(self: *LayerNorm, grad_output: *Tensor, input: *Tensor, grad_input: *Tensor, grad_weight: *Tensor, grad_bias: *Tensor) !void {
        const numel = input.shape.numel();
        const last_dim = self.normalized_shape;
        const n = numel / last_dim;
        const grad_out_ptr = grad_output.typedPtr(BF16) orelse return error.InvalidDType;
        const input_ptr = input.typedPtr(BF16) orelse return error.InvalidDType;
        const weight_ptr = self.weight.typedPtr(BF16) orelse return error.InvalidDType;
        const grad_in_ptr = grad_input.typedPtr(BF16) orelse return error.InvalidDType;
        const grad_w_ptr = grad_weight.typedPtr(BF16) orelse return error.InvalidDType;
        const grad_b_ptr = grad_bias.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..n) |i| {
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
            var sum_dy: f32 = 0.0;
            var sum_dy_xhat: f32 = 0.0;
            for (0..last_dim) |j| {
                const g = grad_out_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                const dy = g * w;
                const x = input_ptr[i * last_dim + j].toFloat32();
                const xhat = (x - mean) * inv_std;
                sum_dy += dy;
                sum_dy_xhat += dy * xhat;
                grad_w_ptr[j] = BF16.fromFloat32(grad_w_ptr[j].toFloat32() + g * xhat);
                grad_b_ptr[j] = BF16.fromFloat32(grad_b_ptr[j].toFloat32() + g);
            }
            const dim_f = @as(f32, @floatFromInt(last_dim));
            for (0..last_dim) |j| {
                const g = grad_out_ptr[i * last_dim + j].toFloat32();
                const w = weight_ptr[j].toFloat32();
                const dy = g * w;
                const x = input_ptr[i * last_dim + j].toFloat32();
                const xhat = (x - mean) * inv_std;
                grad_in_ptr[i * last_dim + j] = BF16.fromFloat32(inv_std * (dy - (sum_dy + xhat * sum_dy_xhat) / dim_f));
            }
        }
    }
};

pub const GELU = struct {
    approximate: bool,
    pub fn init(approximate: bool) GELU { return .{ .approximate = approximate }; }
    pub fn forward(self: GELU, allocator: std.mem.Allocator, input: *Tensor) !*Tensor {
        var output = try Tensor.init(allocator, input.shape, input.dtype, input.device, input.device_id);
        errdefer output.deinit();
        if (input.device == .cuda) {
            try kernels.geluForwardCuda(input.ptr(), output.ptr(), input.shape.numel(), self.approximate);
        } else {
            try self.forwardCpu(input, output);
        }
        return output;
    }
    fn forwardCpu(self: GELU, input: *Tensor, output: *Tensor) !void {
        const input_ptr = input.typedPtr(BF16) orelse return error.InvalidDType;
        const output_ptr = output.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..input.shape.numel()) |i| {
            const x = input_ptr[i].toFloat32();
            output_ptr[i] = BF16.fromFloat32(if (self.approximate) geluApproximate(x) else geluExact(x));
        }
    }
    pub fn backward(self: GELU, allocator: std.mem.Allocator, grad_output: *Tensor, input: *Tensor) !*Tensor {
        var grad_input = try Tensor.init(allocator, input.shape, input.dtype, input.device, input.device_id);
        errdefer grad_input.deinit();
        if (input.device == .cuda) {
            try kernels.geluBackwardCuda(grad_output.ptr(), input.ptr(), grad_input.ptr(), input.shape.numel(), self.approximate);
        } else {
            try self.backwardCpu(grad_output, input, grad_input);
        }
        return grad_input;
    }
    fn backwardCpu(self: GELU, grad_output: *Tensor, input: *Tensor, grad_input: *Tensor) !void {
        const grad_out_ptr = grad_output.typedPtr(BF16) orelse return error.InvalidDType;
        const input_ptr = input.typedPtr(BF16) orelse return error.InvalidDType;
        const grad_in_ptr = grad_input.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..input.shape.numel()) |i| {
            const x = input_ptr[i].toFloat32();
            const g = grad_out_ptr[i].toFloat32();
            grad_in_ptr[i] = BF16.fromFloat32(g * (if (self.approximate) geluApproximateGrad(x) else geluExactGrad(x)));
        }
    }
    fn geluExact(x: f32) f32 { const sqrt2: f32 = @sqrt(2.0); return x * (0.5 * (1.0 + std.math.erf(x / sqrt2))); }
    fn geluApproximate(x: f32) f32 { const sqrt_2_over_pi: f32 = 0.7978845608028654; const x3 = x * x * x; return 0.5 * x * (1.0 + std.math.tanh(sqrt_2_over_pi * (x + 0.044715 * x3))); }
    fn geluExactGrad(x: f32) f32 { const sqrt2: f32 = @sqrt(2.0); const sqrt2pi: f32 = @sqrt(2.0 * std.math.pi); const cdf = 0.5 * (1.0 + std.math.erf(x / sqrt2)); const pdf = @exp(-0.5 * x * x) / sqrt2pi; return cdf + x * pdf; }
    fn geluApproximateGrad(x: f32) f32 { const sqrt_2_over_pi: f32 = 0.7978845608028654; const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x); const tanh_val = std.math.tanh(inner); const sech2 = 1.0 - tanh_val * tanh_val; const inner_grad = sqrt_2_over_pi * (1.0 + 0.134145 * x * x); return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_grad; }
};

pub const Linear = struct {
    in_features: usize,
    out_features: usize,
    weight: *Tensor,
    bias: ?*Tensor,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize, has_bias: bool, device: tensor_mod.Device, device_id: i32, rng: *std.Random) !*Linear {
        std.debug.assert(in_features > 0);
        std.debug.assert(out_features > 0);
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
        self.* = .{ .in_features = in_features, .out_features = out_features, .weight = weight, .bias = bias, .allocator = allocator, .device = device, .device_id = device_id };
        return self;
    }

    pub fn deinit(self: *Linear) void {
        self.weight.deinit();
        if (self.bias) |b| b.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Linear, input: *Tensor) !*Tensor {
        std.debug.assert(input.shape.ndim > 0);
        std.debug.assert(input.shape.dims[input.shape.ndim - 1] == self.in_features);
        if (input.device != self.device or input.device_id != self.device_id) return error.InvalidDevice;
        var out_shape = input.shape;
        out_shape.dims[out_shape.ndim - 1] = self.out_features;
        var output = try Tensor.zeros(self.allocator, out_shape, input.dtype, self.device, self.device_id);
        errdefer output.deinit();
        if (self.device == .cuda) {
            try kernels.gemmForwardCuda(input.ptr(), self.weight.ptr(), if (self.bias) |b| b.ptr() else null, output.ptr(), input.shape.numel() / self.in_features, self.in_features, self.out_features);
        } else {
            try self.forwardCpu(input, output);
        }
        return output;
    }

    fn forwardCpu(self: *Linear, input: *Tensor, output: *Tensor) !void {
        const batch_size = input.shape.numel() / self.in_features;
        const input_ptr = input.typedPtr(BF16) orelse return error.InvalidDType;
        const weight_ptr = self.weight.typedPtr(BF16) orelse return error.InvalidDType;
        const bias_ptr = if (self.bias) |b| b.typedPtr(BF16) else null;
        const output_ptr = output.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..batch_size) |b| for (0..self.out_features) |o| {
            var sum: f32 = 0.0;
            for (0..self.in_features) |i| sum += input_ptr[b * self.in_features + i].toFloat32() * weight_ptr[i * self.out_features + o].toFloat32();
            if (bias_ptr) |bp| sum += bp[o].toFloat32();
            output_ptr[b * self.out_features + o] = BF16.fromFloat32(sum);
        }
    }

    pub fn backward(self: *Linear, grad_output: *Tensor, input: *Tensor) !struct { grad_input: *Tensor, grad_weight: *Tensor, grad_bias: ?*Tensor } {
        std.debug.assert(input.shape.ndim > 0);
        std.debug.assert(input.shape.dims[input.shape.ndim - 1] == self.in_features);
        var grad_input = try Tensor.zeros(self.allocator, input.shape, input.dtype, self.device, self.device_id);
        errdefer grad_input.deinit();
        const weight_shape = Shape.init(&[_]usize{ self.in_features, self.out_features });
        var grad_weight = try Tensor.zeros(self.allocator, weight_shape, .bf16, self.device, self.device_id);
        errdefer grad_weight.deinit();
        var grad_bias: ?*Tensor = null;
        if (self.bias != null) {
            const bias_shape = Shape.init(&[_]usize{self.out_features});
            grad_bias = try Tensor.zeros(self.allocator, bias_shape, .bf16, self.device, self.device_id);
        }
        errdefer if (grad_bias) |gb| gb.deinit();
        if (self.device == .cuda) {
            try kernels.gemmBackwardCuda(grad_output.ptr(), input.ptr(), self.weight.ptr(), grad_input.ptr(), grad_weight.ptr(), if (grad_bias) |gb| gb.ptr() else null, input.shape.numel() / self.in_features, self.in_features, self.out_features);
        } else {
            try self.backwardCpu(grad_output, input, grad_input, grad_weight, grad_bias);
        }
        try storeGrad(self.weight, self.allocator, grad_weight);
        if (self.bias) |b| if (grad_bias) |gb| try storeGrad(b, self.allocator, gb);
        return .{ .grad_input = grad_input, .grad_weight = grad_weight, .grad_bias = grad_bias };
    }

    fn backwardCpu(self: *Linear, grad_output: *Tensor, input: *Tensor, grad_input: *Tensor, grad_weight: *Tensor, grad_bias: ?*Tensor) !void {
        const batch_size = input.shape.numel() / self.in_features;
        const grad_out_ptr = grad_output.typedPtr(BF16) orelse return error.InvalidDType;
        const input_ptr = input.typedPtr(BF16) orelse return error.InvalidDType;
        const weight_ptr = self.weight.typedPtr(BF16) orelse return error.InvalidDType;
        const grad_in_ptr = grad_input.typedPtr(BF16) orelse return error.InvalidDType;
        const grad_w_ptr = grad_weight.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..batch_size) |b| {
            for (0..self.in_features) |i| {
                var sum: f32 = 0.0;
                for (0..self.out_features) |o| sum += grad_out_ptr[b * self.out_features + o].toFloat32() * weight_ptr[i * self.out_features + o].toFloat32();
                grad_in_ptr[b * self.in_features + i] = BF16.fromFloat32(sum);
            }
            for (0..self.in_features) |i| for (0..self.out_features) |o| {
                grad_w_ptr[i * self.out_features + o] = BF16.fromFloat32(grad_w_ptr[i * self.out_features + o].toFloat32() + grad_out_ptr[b * self.out_features + o].toFloat32() * input_ptr[b * self.in_features + i].toFloat32());
            }
            if (grad_bias) |gb| {
                const gb_ptr = gb.typedPtr(BF16) orelse return error.InvalidDType;
                for (0..self.out_features) |o| gb_ptr[o] = BF16.fromFloat32(gb_ptr[o].toFloat32() + grad_out_ptr[b * self.out_features + o].toFloat32());
            }
        }
    }
};

pub const Embedding = struct {
    num_embeddings: usize,
    embedding_dim: usize,
    weight: *Tensor,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, num_embeddings: usize, embedding_dim: usize, device: tensor_mod.Device, device_id: i32, rng: *std.Random) !*Embedding {
        std.debug.assert(num_embeddings > 0);
        std.debug.assert(embedding_dim > 0);
        const self = try allocator.create(Embedding);
        errdefer allocator.destroy(self);
        const scale = @sqrt(1.0 / @as(f64, @floatFromInt(embedding_dim)));
        const weight_shape = Shape.init(&[_]usize{ num_embeddings, embedding_dim });
        const weight = try Tensor.randNormal(allocator, weight_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer weight.deinit();
        self.* = .{ .num_embeddings = num_embeddings, .embedding_dim = embedding_dim, .weight = weight, .allocator = allocator, .device = device, .device_id = device_id };
        return self;
    }

    pub fn deinit(self: *Embedding) void {
        self.weight.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Embedding, input: *Tensor) !*Tensor {
        std.debug.assert(input.shape.ndim > 0);
        std.debug.assert(input.shape.ndim < Shape.MAX_DIMS);
        var out_dims: [Shape.MAX_DIMS]usize = undefined;
        for (0..input.shape.ndim) |d| out_dims[d] = input.shape.dims[d];
        out_dims[input.shape.ndim] = self.embedding_dim;
        const out_shape = Shape.init(out_dims[0 .. input.shape.ndim + 1]);
        var output = try Tensor.zeros(self.allocator, out_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();
        if (self.device == .cuda) {
            try kernels.embeddingForwardCuda(input.ptr(), self.weight.ptr(), output.ptr(), input.shape.numel(), self.embedding_dim, self.num_embeddings);
            return output;
        }
        const indices = input.typedPtr(u32) orelse return error.InvalidDType;
        const weight_ptr = self.weight.typedPtr(BF16) orelse return error.InvalidDType;
        const output_ptr = output.typedPtr(BF16) orelse return error.InvalidDType;
        for (0..input.shape.numel()) |i| {
            const idx = indices[i];
            if (idx >= self.num_embeddings) return error.IndexOutOfRange;
            for (0..self.embedding_dim) |d| output_ptr[i * self.embedding_dim + d] = weight_ptr[idx * self.embedding_dim + d];
        }
        return output;
    }

    pub fn backward(self: *Embedding, grad_output: *Tensor, input: *Tensor) !*Tensor {
        const weight_shape = Shape.init(&[_]usize{ self.num_embeddings, self.embedding_dim });
        var grad_weight = try Tensor.zeros(self.allocator, weight_shape, .bf16, self.device, self.device_id);
        errdefer grad_weight.deinit();
        if (self.device == .cuda) {
            try kernels.embeddingBackwardCuda(grad_output.ptr(), input.ptr(), grad_weight.ptr(), input.shape.numel(), self.embedding_dim, self.num_embeddings);
        } else {
            const indices = input.typedPtr(u32) orelse return error.InvalidDType;
            const grad_out_ptr = grad_output.typedPtr(BF16) orelse return error.InvalidDType;
            const grad_w_ptr = grad_weight.typedPtr(BF16) orelse return error.InvalidDType;
            for (0..input.shape.numel()) |i| {
                const idx = indices[i];
                if (idx >= self.num_embeddings) return error.IndexOutOfRange;
                for (0..self.embedding_dim) |d| grad_w_ptr[idx * self.embedding_dim + d] = BF16.fromFloat32(grad_w_ptr[idx * self.embedding_dim + d].toFloat32() + grad_out_ptr[i * self.embedding_dim + d].toFloat32());
            }
        }
        try storeGrad(self.weight, self.allocator, grad_weight);
        return grad_weight;
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
    const ptr = output.typedPtr(BF16).?;
    try std.testing.expectApproxEqRel(@as(f32, 0.0), ptr[0].toFloat32(), 0.01);
}
