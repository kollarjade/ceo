const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const kernels = @import("../kernels/efla_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
pub const DType = dtype_mod.DType;

pub const EflaState = struct {
    state: *Tensor,
    num_heads: usize,
    state_dim: usize,
    value_dim: usize,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

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

        const dims = try allocator.alloc(usize, 3);
        defer allocator.free(dims);
        dims[0] = num_heads;
        dims[1] = state_dim;
        dims[2] = value_dim;
        const state_shape = Shape.init(dims);

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

    pub fn reset(self: *EflaState) !void {
        try self.state.zero_();
    }

    pub fn clone(self: *EflaState) !*EflaState {
        const new_state = try self.state.to(self.allocator, self.device, self.device_id);
        errdefer new_state.deinit();

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

pub const EflaLayer = struct {
    config: config_mod.EflaConfig,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    w_k: *Tensor,
    w_v: *Tensor,
    w_o: *Tensor,
    beta_param: ?*Tensor,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

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

        const wk_dims = try allocator.alloc(usize, 2);
        defer allocator.free(wk_dims);
        wk_dims[0] = hidden_dim;
        wk_dims[1] = num_heads * head_dim;
        const w_k_shape = Shape.init(wk_dims);
        const w_k = try Tensor.randNormal(allocator, w_k_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_k.deinit();

        const wv_dims = try allocator.alloc(usize, 2);
        defer allocator.free(wv_dims);
        wv_dims[0] = hidden_dim;
        wv_dims[1] = num_heads * head_dim;
        const w_v_shape = Shape.init(wv_dims);
        const w_v = try Tensor.randNormal(allocator, w_v_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_v.deinit();

        const wo_dims = try allocator.alloc(usize, 2);
        defer allocator.free(wo_dims);
        wo_dims[0] = num_heads * head_dim;
        wo_dims[1] = hidden_dim;
        const w_o_shape = Shape.init(wo_dims);
        const w_o = try Tensor.randNormal(allocator, w_o_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_o.deinit();

        var beta_param: ?*Tensor = null;
        if (config.learned_beta) {
            const beta_dims = try allocator.alloc(usize, 1);
            defer allocator.free(beta_dims);
            beta_dims[0] = 1;
            const beta_shape = Shape.init(beta_dims);
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

    pub fn forward(
        self: *EflaLayer,
        input: *Tensor,
        state: ?*EflaState,
    ) !struct { output: *Tensor, new_state: *EflaState } {
        _ = state;

        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);

        const k = try self.matmul(input, self.w_k);
        defer k.deinit();

        const v = try self.matmul(input, self.w_v);
        defer v.deinit();

        const k_dims = try self.allocator.alloc(usize, 4);
        defer self.allocator.free(k_dims);
        k_dims[0] = batch_size;
        k_dims[1] = seq_len;
        k_dims[2] = self.num_heads;
        k_dims[3] = self.head_dim;
        const k_reshaped = try k.reshape(Shape.init(k_dims));
        defer k_reshaped.deinit();

        const v_dims = try self.allocator.alloc(usize, 4);
        defer self.allocator.free(v_dims);
        v_dims[0] = batch_size;
        v_dims[1] = seq_len;
        v_dims[2] = self.num_heads;
        v_dims[3] = self.head_dim;
        const v_reshaped = try v.reshape(Shape.init(v_dims));
        defer v_reshaped.deinit();

        var new_state = try EflaState.init(
            self.allocator,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            self.device,
            self.device_id,
        );
        errdefer new_state.deinit();

        const output = try self.eflaForward(k_reshaped, v_reshaped, new_state);
        defer output.deinit();

        const projected = try self.matmul(output, self.w_o);

        return .{ .output = projected, .new_state = new_state };
    }

    fn eflaForward(
        self: *EflaLayer,
        k: *Tensor,
        v: *Tensor,
        state: *EflaState,
    ) !*Tensor {
        const batch_size = k.shape.dim(0);
        const seq_len = k.shape.dim(1);

        const out_dims = try self.allocator.alloc(usize, 4);
        defer self.allocator.free(out_dims);
        out_dims[0] = batch_size;
        out_dims[1] = seq_len;
        out_dims[2] = self.num_heads;
        out_dims[3] = self.head_dim;
        const output_shape = Shape.init(out_dims);
        const output = try Tensor.init(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        const beta: f32 = if (self.beta_param) |bp| blk: {
            const ptr = bp.typedPtr(dtype_mod.FP32).?;
            break :blk ptr[0];
        } else self.config.initial_beta;

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
            try self.eflaForwardCpu(k, v, state, output, beta);
        }

        const res_dims = try self.allocator.alloc(usize, 3);
        defer self.allocator.free(res_dims);
        res_dims[0] = batch_size;
        res_dims[1] = seq_len;
        res_dims[2] = self.num_heads * self.head_dim;
        const reshaped = try output.reshape(Shape.init(res_dims));

        return reshaped;
    }

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

        const k_ptr = k.typedPtr(dtype_mod.BF16).?;
        const v_ptr = v.typedPtr(dtype_mod.BF16).?;
        const s_ptr = state.state.typedPtr(dtype_mod.BF16).?;
        const o_ptr = output.typedPtr(dtype_mod.BF16).?;

        var s_old_vals = try self.allocator.alloc(f32, self.head_dim * self.head_dim);
        defer self.allocator.free(s_old_vals);

        for (0..batch_size) |b| {
            for (0..self.num_heads) |h| {
                const head_state_offset = h * self.head_dim * self.head_dim;

                for (0..seq_len) |t| {
                    const k_offset = b * seq_len * self.num_heads * self.head_dim + t * self.num_heads * self.head_dim + h * self.head_dim;
                    const v_offset = k_offset;

                    var lambda: f32 = 0.0;
                    for (0..self.head_dim) |d| {
                        const k_val = k_ptr[k_offset + d].toFloat32();
                        lambda += k_val * k_val;
                    }

                    const c_t: f32 = if (lambda < 1e-6) blk: {
                        var c: f32 = beta;
                        const beta_lambda = beta * lambda;
                        c -= 0.5 * beta * beta_lambda;
                        c += (1.0 / 6.0) * beta * beta * beta_lambda * lambda;
                        break :blk c;
                    } else blk: {
                        break :blk -std.math.expm1(-beta * lambda) / lambda;
                    };

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

                    for (0..self.head_dim) |i| {
                        for (0..self.head_dim) |j| {
                            s_old_vals[i * self.head_dim + j] = s_ptr[head_state_offset + i * self.head_dim + j].toFloat32();
                        }
                    }

                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[k_offset + i].toFloat32();
                        const v_i = v_ptr[v_offset + i].toFloat32();

                        for (0..self.head_dim) |j| {
                            var s_old: f32 = s_old_vals[i * self.head_dim + j];

                            var s_k_sum: f32 = 0.0;
                            for (0..self.head_dim) |k_idx| {
                                const s_val = s_old_vals[j * self.head_dim + k_idx];
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

    pub fn backward(
        self: *EflaLayer,
        grad_output: *Tensor,
        input: *Tensor,
        state: *EflaState,
    ) !struct { grad_input: *Tensor, grad_state: *EflaState } {
        _ = state;

        const grad_input = try Tensor.zeros(self.allocator, input.shape, .bf16, self.device, self.device_id);
        errdefer grad_input.deinit();

        const grad_state = try EflaState.init(
            self.allocator,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            self.device,
            self.device_id,
        );
        errdefer grad_state.deinit();

        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);
        const g_in_ptr = grad_input.typedPtr(dtype_mod.BF16).?;
        const g_out_ptr = grad_output.typedPtr(dtype_mod.BF16).?;
        const in_ptr = input.typedPtr(dtype_mod.BF16).?;

        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                for (0..self.hidden_dim) |d| {
                    const idx = b * seq_len * self.hidden_dim + t * self.hidden_dim + d;
                    g_in_ptr[idx] = dtype_mod.BF16.fromFloat32(g_out_ptr[idx].toFloat32() * in_ptr[idx].toFloat32() * 0.01);
                }
            }
        }

        return .{ .grad_input = grad_input, .grad_state = grad_state };
    }

    fn matmul(self: *EflaLayer, a: *Tensor, b: *Tensor) !*Tensor {
        const M = a.shape.dim(a.shape.ndim - 2);
        const K = a.shape.dim(a.shape.ndim - 1);
        const N = b.shape.dim(b.shape.ndim - 1);

        const batch_size = if (a.shape.ndim > 2) a.shape.first() else 1;

        const out_dims = try self.allocator.alloc(usize, if (a.shape.ndim > 2) 3 else 2);
        defer self.allocator.free(out_dims);

        if (a.shape.ndim > 2) {
            out_dims[0] = batch_size;
            out_dims[1] = M;
            out_dims[2] = N;
        } else {
            out_dims[0] = M;
            out_dims[1] = N;
        }

        const output_shape = Shape.init(out_dims);
        const output = try Tensor.init(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        const a_ptr = a.typedPtr(dtype_mod.BF16).?;
        const b_ptr = b.typedPtr(dtype_mod.BF16).?;
        const o_ptr = output.typedPtr(dtype_mod.BF16).?;

        for (0..batch_size) |batch| {
            for (0..M) |i| {
                for (0..N) |j| {
                    var sum: f32 = 0.0;
                    for (0..K) |k| {
                        const a_val = a_ptr[batch * M * K + i * K + k].toFloat32();
                        const b_val = b_ptr[k * N + j].toFloat32();
                        sum += a_val * b_val;
                    }
                    o_ptr[batch * M * N + i * N + j] = dtype_mod.BF16.fromFloat32(sum);
                }
            }
        }

        return output;
    }
};

pub const ChunkedScan = struct {
    chunk_size: usize,
    num_chunks: usize,

    pub fn init(seq_len: usize, chunk_size: usize) ChunkedScan {
        return .{
            .chunk_size = chunk_size,
            .num_chunks = (seq_len + chunk_size - 1) / chunk_size,
        };
    }

    pub fn getChunkRange(self: ChunkedScan, chunk_idx: usize) struct { start: usize, end: usize } {
        const start = chunk_idx * self.chunk_size;
        const end = @min(start + self.chunk_size, self.num_chunks * self.chunk_size);
        return .{ .start = start, .end = end };
    }

    pub fn prefixScan(
        self: ChunkedScan,
        chunk_states: []*EflaState,
        allocator: std.mem.Allocator,
    ) !void {
        _ = allocator;
        if (self.num_chunks <= 1) return;

        var step: usize = 1;
        while (step < self.num_chunks) : (step *= 2) {
            var i: usize = step * 2 - 1;
            while (i < self.num_chunks) : (i += step * 2) {
                const left_state = chunk_states[i - step];
                const right_state = chunk_states[i];

                const l_ptr = left_state.state.typedPtr(dtype_mod.BF16).?;
                const r_ptr = right_state.state.typedPtr(dtype_mod.BF16).?;
                const numel = left_state.state.shape.numel();

                for (0..numel) |idx| {
                    const l_val = l_ptr[idx].toFloat32();
                    const r_val = r_ptr[idx].toFloat32();
                    r_ptr[idx] = dtype_mod.BF16.fromFloat32(l_val + r_val);
                }
            }
        }
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
