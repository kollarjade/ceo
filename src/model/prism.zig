const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const kernels = @import("../kernels/prism_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;

fn sigmoid(x: f32) f32 {
    if (x >= 0.0) {
        const z = @exp(-x);
        return 1.0 / (1.0 + z);
    }
    const z = @exp(x);
    return z / (1.0 + z);
}

fn geluApprox(x: f32) f32 {
    const c: f32 = 0.7978845608028654;
    const k: f32 = 0.044715;
    const x2 = x * x;
    const inner = c * (x + k * x * x2);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

fn geluApproxDerivative(x: f32) f32 {
    const c: f32 = 0.7978845608028654;
    const k: f32 = 0.044715;
    const x2 = x * x;
    const inner = c * (x + k * x * x2);
    const t = std.math.tanh(inner);
    const sech2 = 1.0 - t * t;
    const inner_prime = c * (1.0 + 3.0 * k * x2);
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * inner_prime;
}

fn zeroSlice(slice: []f32) void {
    for (slice) |*v| {
        v.* = 0.0;
    }
}

fn copySlice(dst: []f32, src: []const f32) void {
    for (dst, src) |*d, s| {
        d.* = s;
    }
}

fn normalizeInPlace(vec: []f32) void {
    var norm_sq: f32 = 0.0;
    for (vec) |v| {
        norm_sq += v * v;
    }
    if (norm_sq <= 1.0e-12) {
        return;
    }
    const inv_norm = 1.0 / @sqrt(norm_sq);
    for (vec) |*v| {
        v.* *= inv_norm;
    }
}

fn projectTokenInto(dst: []f32, weight: *Tensor, token_ptr: anytype, token_offset: usize, hidden_dim: usize, head_dim: usize) void {
    const weight_ptr = weight.typedPtr(dtype_mod.BF16).?;
    for (0..head_dim) |j| {
        var sum: f32 = 0.0;
        for (0..hidden_dim) |i| {
            sum += token_ptr[token_offset + i].toFloat32() * weight_ptr[i * head_dim + j].toFloat32();
        }
        dst[j] = sum;
    }
}

fn accumulateMatVecTransposeToInput(grad_input: []f32, weight: *Tensor, grad_output: []const f32, hidden_dim: usize, head_dim: usize) void {
    const weight_ptr = weight.typedPtr(dtype_mod.BF16).?;
    for (0..hidden_dim) |i| {
        var sum: f32 = 0.0;
        for (0..head_dim) |j| {
            sum += weight_ptr[i * head_dim + j].toFloat32() * grad_output[j];
        }
        grad_input[i] += sum;
    }
}

fn computeStateTimesKey(dst: []f32, state: []const f32, key: []const f32, head_dim: usize) void {
    for (0..head_dim) |row| {
        var sum: f32 = 0.0;
        for (0..head_dim) |col| {
            sum += state[row * head_dim + col] * key[col];
        }
        dst[row] = sum;
    }
}

fn applyForgetAndUpdate(dst: []f32, src: []const f32, state_times_key: []const f32, delta: []const f32, key: []const f32, head_dim: usize, alpha: f32, beta: f32) void {
    for (0..head_dim) |row| {
        for (0..head_dim) |col| {
            dst[row * head_dim + col] = alpha * src[row * head_dim + col] - alpha * beta * state_times_key[row] * key[col] + beta * delta[row] * key[col];
        }
    }
}

fn applyAdditiveUpdate(dst: []f32, src: []const f32, delta: []const f32, key: []const f32, head_dim: usize, beta: f32) void {
    for (0..head_dim) |row| {
        for (0..head_dim) |col| {
            dst[row * head_dim + col] = src[row * head_dim + col] + beta * delta[row] * key[col];
        }
    }
}

pub const PrismLayer = struct {
    config: config_mod.PrismConfig,
    hidden_dim: usize,
    num_iterations: usize,
    head_dim: usize,
    shortconv: *ShortConv,
    w_beta: []*Tensor,
    w_k: []*Tensor,
    w_p: []*Tensor,
    alpha: f32,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(
        allocator: std.mem.Allocator,
        config: config_mod.PrismConfig,
        hidden_dim: usize,
        head_dim: usize,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*PrismLayer {
        if (hidden_dim == 0) return error.InvalidHiddenDimension;
        if (head_dim == 0) return error.InvalidHeadDimension;
        if (head_dim > hidden_dim) return error.InvalidHeadDimension;
        if (config.num_iterations == 0) return error.InvalidNumIterations;
        if (config.shortconv_window == 0) return error.InvalidShortConvWindow;

        const self = try allocator.create(PrismLayer);
        errdefer allocator.destroy(self);

        const num_iterations = config.num_iterations;
        const scale = @sqrt(2.0 / @as(f64, @floatFromInt(hidden_dim)));

        const shortconv = try ShortConv.init(
            allocator,
            hidden_dim,
            config.shortconv_window,
            device,
            device_id,
            rng,
        );
        errdefer shortconv.deinit();

        var w_beta = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_beta);

        var w_k = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_k);

        var w_p = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_p);

        var beta_initialized: usize = 0;
        var k_initialized: usize = 0;
        var p_initialized: usize = 0;

        errdefer {
            var i: usize = 0;
            while (i < beta_initialized) : (i += 1) {
                w_beta[i].deinit();
            }
        }
        errdefer {
            var i: usize = 0;
            while (i < k_initialized) : (i += 1) {
                w_k[i].deinit();
            }
        }
        errdefer {
            var i: usize = 0;
            while (i < p_initialized) : (i += 1) {
                w_p[i].deinit();
            }
        }

        for (0..num_iterations) |l| {
            const beta_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_beta[l] = try Tensor.randNormal(allocator, beta_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
            beta_initialized += 1;

            const k_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_k[l] = try Tensor.randNormal(allocator, k_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
            k_initialized += 1;

            const p_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_p[l] = try Tensor.randNormal(allocator, p_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
            p_initialized += 1;
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

        for (self.w_beta) |w| {
            w.deinit();
        }
        for (self.w_k) |w| {
            w.deinit();
        }
        for (self.w_p) |w| {
            w.deinit();
        }

        self.allocator.free(self.w_beta);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_p);

        self.allocator.destroy(self);
    }

    fn validateForwardInputs(self: *PrismLayer, input: *Tensor, v: *Tensor, state: ?*PrismState) !void {
        if (input.shape.dim(2) != self.hidden_dim) return error.InvalidInputShape;
        if (v.shape.dim(0) != input.shape.dim(0)) return error.InvalidValueShape;
        if (v.shape.dim(1) != input.shape.dim(1)) return error.InvalidValueShape;
        if (v.shape.dim(2) != self.hidden_dim) return error.InvalidValueShape;

        if (state) |s| {
            if (s.state_dim != self.head_dim) return error.InvalidStateShape;
            if (s.value_dim != self.head_dim) return error.InvalidStateShape;
            if (s.state.shape.dim(0) != input.shape.dim(0)) return error.InvalidStateShape;
            if (s.state.shape.dim(1) != self.head_dim) return error.InvalidStateShape;
            if (s.state.shape.dim(2) != self.head_dim) return error.InvalidStateShape;
        }
    }

    pub fn forward(
        self: *PrismLayer,
        input: *Tensor,
        v: *Tensor,
        state: ?*PrismState,
    ) !struct { output: *Tensor, new_state: *PrismState } {
        try self.validateForwardInputs(input, v, state);

        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);

        const u = try self.shortconv.forward(input);
        defer u.deinit();

        var new_state = try PrismState.init(
            self.allocator,
            batch_size,
            self.head_dim,
            self.head_dim,
            self.device,
            self.device_id,
        );
        errdefer new_state.deinit();

        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.hidden_dim });
        var output = try Tensor.zeros(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        if (self.device == .cuda) {
            try self.prismForwardCuda(u, v, state, new_state, output);
        } else {
            try self.prismForwardCpu(u, v, state, new_state, output);
        }

        return .{ .output = output, .new_state = new_state };
    }

    fn prismForwardCuda(
        self: *PrismLayer,
        u: *Tensor,
        v: *Tensor,
        prev_state: ?*PrismState,
        new_state: *PrismState,
        output: *Tensor,
    ) !void {
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
            u.shape.dim(0),
            u.shape.dim(1),
            self.hidden_dim,
            self.head_dim,
            self.num_iterations,
            self.alpha,
        );
    }

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
        const out_ptr = output.typedPtr(dtype_mod.BF16).?;
        const new_state_ptr = new_state.state.typedPtr(dtype_mod.BF16).?;
        const prev_state_ptr = if (prev_state) |s| s.state.typedPtr(dtype_mod.BF16).? else null;

        const total_output_elems = batch_size * seq_len * self.hidden_dim;
        for (0..total_output_elems) |i| {
            out_ptr[i] = u_ptr[i];
        }

        var current_state = try self.allocator.alloc(f32, self.head_dim * self.head_dim);
        defer self.allocator.free(current_state);

        var next_state = try self.allocator.alloc(f32, self.head_dim * self.head_dim);
        defer self.allocator.free(next_state);

        var state_times_key = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(state_times_key);

        var residual = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(residual);

        var refined = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(refined);

        var beta_proj = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(beta_proj);

        var key_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(key_vec);

        var p_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(p_vec);

        var delta = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(delta);

        for (0..batch_size) |b| {
            zeroSlice(current_state);

            if (prev_state_ptr) |ps| {
                for (0..self.head_dim) |row| {
                    for (0..self.head_dim) |col| {
                        const idx = b * self.head_dim * self.head_dim + row * self.head_dim + col;
                        current_state[row * self.head_dim + col] = ps[idx].toFloat32();
                    }
                }
            }

            for (0..seq_len) |t| {
                const token_offset = (b * seq_len + t) * self.hidden_dim;

                zeroSlice(refined);

                for (0..self.head_dim) |d| {
                    residual[d] = v_ptr[token_offset + d].toFloat32() - u_ptr[token_offset + d].toFloat32();
                }

                for (0..self.num_iterations) |l| {
                    projectTokenInto(beta_proj, self.w_beta[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);

                    var beta_sum: f32 = 0.0;
                    for (beta_proj) |v_beta| {
                        beta_sum += v_beta;
                    }
                    const beta_scalar = sigmoid(beta_sum / @as(f32, @floatFromInt(self.head_dim)));

                    projectTokenInto(key_vec, self.w_k[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    normalizeInPlace(key_vec);

                    projectTokenInto(p_vec, self.w_p[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    for (0..self.head_dim) |d| {
                        p_vec[d] = sigmoid(p_vec[d]);
                    }

                    for (0..self.head_dim) |d| {
                        const z = p_vec[d] * residual[d];
                        delta[d] = geluApprox(z);
                    }

                    if (l == 0) {
                        computeStateTimesKey(state_times_key, current_state, key_vec, self.head_dim);
                        applyForgetAndUpdate(next_state, current_state, state_times_key, delta, key_vec, self.head_dim, self.alpha, beta_scalar);
                    } else {
                        applyAdditiveUpdate(next_state, current_state, delta, key_vec, self.head_dim, beta_scalar);
                    }

                    for (0..self.head_dim) |d| {
                        residual[d] -= delta[d];
                        refined[d] += delta[d];
                    }

                    const tmp = current_state;
                    current_state = next_state;
                    next_state = tmp;
                }

                for (0..self.head_dim) |d| {
                    out_ptr[token_offset + d] = dtype_mod.BF16.fromFloat32(u_ptr[token_offset + d].toFloat32() + refined[d]);
                }
            }

            for (0..self.head_dim) |row| {
                for (0..self.head_dim) |col| {
                    const idx = b * self.head_dim * self.head_dim + row * self.head_dim + col;
                    new_state_ptr[idx] = dtype_mod.BF16.fromFloat32(current_state[row * self.head_dim + col]);
                }
            }
        }
    }

    pub fn backward(
        self: *PrismLayer,
        grad_output: *Tensor,
        input: *Tensor,
        v: *Tensor,
        state: *PrismState,
    ) !struct { grad_input: *Tensor, grad_v: *Tensor, grad_state: *PrismState } {
        if (self.device != .cpu) return error.UnsupportedDevice;
        if (grad_output.shape.dim(0) != input.shape.dim(0)) return error.InvalidGradientShape;
        if (grad_output.shape.dim(1) != input.shape.dim(1)) return error.InvalidGradientShape;
        if (grad_output.shape.dim(2) != self.hidden_dim) return error.InvalidGradientShape;
        try self.validateForwardInputs(input, v, state);

        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);
        const total_elems = batch_size * seq_len * self.hidden_dim;

        const u = try self.shortconv.forward(input);
        defer u.deinit();

        const grad_output_ptr = grad_output.typedPtr(dtype_mod.BF16).?;
        const u_ptr = u.typedPtr(dtype_mod.BF16).?;
        const v_ptr = v.typedPtr(dtype_mod.BF16).?;
        const weight_ptr = self.shortconv.weight.typedPtr(dtype_mod.BF16).?;

        const grad_input_shape = input.shape;
        var grad_input = try Tensor.zeros(self.allocator, grad_input_shape, .bf16, self.device, self.device_id);
        errdefer grad_input.deinit();

        var grad_v = try Tensor.zeros(self.allocator, v.shape, .bf16, self.device, self.device_id);
        errdefer grad_v.deinit();

        var grad_state = try PrismState.init(
            self.allocator,
            state.state.shape.dim(0),
            self.head_dim,
            self.head_dim,
            self.device,
            self.device_id,
        );
        errdefer grad_state.deinit();

        const grad_input_ptr = grad_input.typedPtr(dtype_mod.BF16).?;
        const grad_v_ptr = grad_v.typedPtr(dtype_mod.BF16).?;
        const grad_state_ptr = grad_state.state.typedPtr(dtype_mod.BF16).?;

        for (0..state.state.shape.dim(0) * self.head_dim * self.head_dim) |i| {
            grad_state_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        }

        var grad_u = try self.allocator.alloc(f32, total_elems);
        defer self.allocator.free(grad_u);
        zeroSlice(grad_u);

        var residual_before_cache = try self.allocator.alloc(f32, self.num_iterations * self.head_dim);
        defer self.allocator.free(residual_before_cache);

        var p_cache = try self.allocator.alloc(f32, self.num_iterations * self.head_dim);
        defer self.allocator.free(p_cache);

        var gelu_input_cache = try self.allocator.alloc(f32, self.num_iterations * self.head_dim);
        defer self.allocator.free(gelu_input_cache);

        var residual = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(residual);

        var p_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(p_vec);

        var grad_residual_next = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_residual_next);

        var grad_residual_before = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_residual_before);

        var grad_linear = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_linear);

        var token_grad_u = try self.allocator.alloc(f32, self.hidden_dim);
        defer self.allocator.free(token_grad_u);

        for (0..total_elems) |i| {
            grad_v_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        }

        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                const token_offset = (b * seq_len + t) * self.hidden_dim;

                for (0..self.hidden_dim) |d| {
                    token_grad_u[d] = grad_output_ptr[token_offset + d].toFloat32();
                }

                for (0..self.head_dim) |d| {
                    residual[d] = v_ptr[token_offset + d].toFloat32() - u_ptr[token_offset + d].toFloat32();
                }

                for (0..self.num_iterations) |l| {
                    projectTokenInto(p_vec, self.w_p[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);

                    for (0..self.head_dim) |d| {
                        const cache_idx = l * self.head_dim + d;
                        residual_before_cache[cache_idx] = residual[d];
                        p_cache[cache_idx] = sigmoid(p_vec[d]);
                        gelu_input_cache[cache_idx] = p_cache[cache_idx] * residual[d];
                        residual[d] -= geluApprox(gelu_input_cache[cache_idx]);
                    }
                }

                zeroSlice(grad_residual_next);

                var l_rev: usize = self.num_iterations;
                while (l_rev > 0) {
                    l_rev -= 1;

                    for (0..self.head_dim) |d| {
                        const cache_idx = l_rev * self.head_dim + d;
                        const grad_delta = grad_output_ptr[token_offset + d].toFloat32() - grad_residual_next[d];
                        const grad_z = grad_delta * geluApproxDerivative(gelu_input_cache[cache_idx]);
                        const p = p_cache[cache_idx];
                        const r_before = residual_before_cache[cache_idx];
                        grad_linear[d] = (grad_z * r_before) * p * (1.0 - p);
                        grad_residual_before[d] = grad_residual_next[d] + grad_z * p;
                    }

                    accumulateMatVecTransposeToInput(token_grad_u, self.w_p[l_rev], grad_linear, self.hidden_dim, self.head_dim);

                    for (0..self.head_dim) |d| {
                        grad_residual_next[d] = grad_residual_before[d];
                    }
                }

                for (0..self.head_dim) |d| {
                    token_grad_u[d] -= grad_residual_next[d];
                    const current_grad_v = grad_v_ptr[token_offset + d].toFloat32();
                    grad_v_ptr[token_offset + d] = dtype_mod.BF16.fromFloat32(current_grad_v + grad_residual_next[d]);
                }

                for (0..self.hidden_dim) |d| {
                    grad_u[token_offset + d] = token_grad_u[d];
                }
            }
        }

        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                for (0..self.hidden_dim) |d| {
                    var sum: f32 = 0.0;
                    for (0..self.shortconv.window_size) |w| {
                        const out_t = t + w;
                        if (out_t < seq_len) {
                            const grad_u_idx = (b * seq_len + out_t) * self.hidden_dim + d;
                            sum += grad_u[grad_u_idx] * weight_ptr[d * self.shortconv.window_size + w].toFloat32();
                        }
                    }
                    grad_input_ptr[(b * seq_len + t) * self.hidden_dim + d] = dtype_mod.BF16.fromFloat32(sum);
                }
            }
        }

        return .{ .grad_input = grad_input, .grad_v = grad_v, .grad_state = grad_state };
    }
};

pub const ShortConv = struct {
    window_size: usize,
    hidden_dim: usize,
    weight: *Tensor,
    bias: ?*Tensor,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(
        allocator: std.mem.Allocator,
        hidden_dim: usize,
        window_size: usize,
        device: tensor_mod.Device,
        device_id: i32,
        rng: *std.Random,
    ) !*ShortConv {
        if (hidden_dim == 0) return error.InvalidHiddenDimension;
        if (window_size == 0) return error.InvalidWindowSize;

        const self = try allocator.create(ShortConv);
        errdefer allocator.destroy(self);

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
        if (self.bias) |b| {
            b.deinit();
        }
        self.allocator.destroy(self);
    }

    pub fn forward(self: *ShortConv, input: *Tensor) !*Tensor {
        if (input.shape.dim(2) != self.hidden_dim) return error.InvalidInputShape;

        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);

        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.hidden_dim });
        var output = try Tensor.zeros(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        if (self.device == .cuda) {
            if (self.bias != null) return error.UnsupportedBiasConfiguration;
            try kernels.shortConvForwardCuda(
                input.ptr(),
                self.weight.ptr(),
                output.ptr(),
                batch_size,
                seq_len,
                self.hidden_dim,
                self.window_size,
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
        const bias_ptr = if (self.bias) |b| blk: {
            if (b.shape.dim(0) != self.hidden_dim) return error.InvalidBiasShape;
            break :blk b.typedPtr(dtype_mod.BF16).?;
        } else null;

        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                for (0..self.hidden_dim) |d| {
                    var sum: f32 = 0.0;

                    for (0..self.window_size) |w| {
                        const lookback = @as(isize, @intCast(t)) - @as(isize, @intCast(w));
                        if (lookback >= 0) {
                            const t_lookback = @as(usize, @intCast(lookback));
                            const input_val = input_ptr[(b * seq_len + t_lookback) * self.hidden_dim + d].toFloat32();
                            const weight_val = weight_ptr[d * self.window_size + w].toFloat32();
                            sum += input_val * weight_val;
                        }
                    }

                    if (bias_ptr) |bp| {
                        sum += bp[d].toFloat32();
                    }

                    output_ptr[(b * seq_len + t) * self.hidden_dim + d] = dtype_mod.BF16.fromFloat32(sum);
                }
            }
        }
    }
};

pub const PrismState = struct {
    state: *Tensor,
    state_dim: usize,
    value_dim: usize,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        num_states: usize,
        state_dim: usize,
        value_dim: usize,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*PrismState {
        if (num_states == 0) return error.InvalidNumStates;
        if (state_dim == 0) return error.InvalidStateDimension;
        if (value_dim == 0) return error.InvalidValueDimension;

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
