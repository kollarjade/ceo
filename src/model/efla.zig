const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const kernels = @import("../kernels/efla_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;

pub const EflaState = struct {
    state: *Tensor,
    batch_size: usize,
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
        return initWithBatch(allocator, 1, num_heads, state_dim, value_dim, device, device_id);
    }

    pub fn initWithBatch(
        allocator: std.mem.Allocator,
        batch_size: usize,
        num_heads: usize,
        state_dim: usize,
        value_dim: usize,
        device: tensor_mod.Device,
        device_id: i32,
    ) !*EflaState {
        if (batch_size == 0 or num_heads == 0 or state_dim == 0 or value_dim == 0) {
            return error.InvalidConfiguration;
        }

        const self = try allocator.create(EflaState);
        errdefer allocator.destroy(self);

        const state_shape = Shape.init(&[_]usize{ batch_size, num_heads, state_dim, value_dim });
        const state_tensor = try Tensor.zeros(allocator, state_shape, .bf16, device, device_id);
        errdefer state_tensor.deinit();

        self.* = .{
            .state = state_tensor,
            .batch_size = batch_size,
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

        const cloned = try self.allocator.create(EflaState);
        errdefer self.allocator.destroy(cloned);

        cloned.* = .{
            .state = new_state,
            .batch_size = self.batch_size,
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
        if (hidden_dim == 0 or num_heads == 0 or head_dim == 0 or hidden_dim != num_heads * head_dim or config.chunk_size == 0) {
            return error.InvalidConfiguration;
        }

        const self = try allocator.create(EflaLayer);
        errdefer allocator.destroy(self);

        const scale = @sqrt(2.0 / @as(f64, @floatFromInt(hidden_dim)));

        const w_k_shape = Shape.init(&[_]usize{ hidden_dim, num_heads * head_dim });
        const w_k = try Tensor.randNormal(allocator, w_k_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_k.deinit();

        const w_v_shape = Shape.init(&[_]usize{ hidden_dim, num_heads * head_dim });
        const w_v = try Tensor.randNormal(allocator, w_v_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_v.deinit();

        const w_o_shape = Shape.init(&[_]usize{ num_heads * head_dim, hidden_dim });
        const w_o = try Tensor.randNormal(allocator, w_o_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_o.deinit();

        var beta_param: ?*Tensor = null;
        if (config.learned_beta) {
            const beta_shape = Shape.init(&[_]usize{1});
            beta_param = try Tensor.full(allocator, beta_shape, .fp32, device, device_id, config.initial_beta);
            errdefer {
                if (beta_param) |bp| {
                    bp.deinit();
                }
            }
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
        if (self.beta_param) |bp| {
            bp.deinit();
        }
        self.allocator.destroy(self);
    }

    pub fn forward(
        self: *EflaLayer,
        input: *Tensor,
        state: ?*EflaState,
    ) !struct { output: *Tensor, new_state: *EflaState } {
        if (input.device != self.device or input.device_id != self.device_id) {
            return error.DeviceMismatch;
        }
        if (input.dtype != .bf16) {
            return error.DTypeMismatch;
        }
        if (input.shape.ndim != 3 or input.shape.dim(2) != self.hidden_dim) {
            return error.ShapeMismatch;
        }

        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);

        const k = try self.matmul(input, self.w_k);
        defer k.deinit();

        const v = try self.matmul(input, self.w_v);
        defer v.deinit();

        const k_reshaped = try k.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim }));
        defer k_reshaped.deinit();

        const v_reshaped = try v.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim }));
        defer v_reshaped.deinit();

        var new_state = if (state) |previous_state| blk: {
            if (previous_state.batch_size != batch_size or previous_state.num_heads != self.num_heads or previous_state.state_dim != self.head_dim or previous_state.value_dim != self.head_dim or previous_state.device != self.device or previous_state.device_id != self.device_id) {
                return error.StateShapeMismatch;
            }
            break :blk try previous_state.clone();
        } else try EflaState.initWithBatch(
            self.allocator,
            batch_size,
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

        return .{
            .output = projected,
            .new_state = new_state,
        };
    }

    fn eflaForward(
        self: *EflaLayer,
        k: *Tensor,
        v: *Tensor,
        state: *EflaState,
    ) !*Tensor {
        const batch_size = k.shape.dim(0);
        const seq_len = k.shape.dim(1);

        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads * self.head_dim });
        const output = try Tensor.init(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        const beta: f32 = self.config.initial_beta;

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
                null,
            );
        } else if (self.device == .cpu) {
            try self.eflaForwardCpu(k, v, state, output, beta);
        } else {
            return error.UnsupportedDevice;
        }

        return output;
    }

    fn eflaForwardCpu(
        self: *EflaLayer,
        k: *Tensor,
        v: *Tensor,
        state: *EflaState,
        output: *Tensor,
        beta: f32,
    ) !void {
        if (state.batch_size != k.shape.dim(0) or state.num_heads != self.num_heads or state.state_dim != self.head_dim or state.value_dim != self.head_dim) {
            return error.StateShapeMismatch;
        }

        const batch_size = k.shape.dim(0);
        const seq_len = k.shape.dim(1);

        const k_ptr = k.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const v_ptr = v.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const s_ptr = state.state.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const o_ptr = output.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;

        const state_dim = state.state_dim;
        const value_dim = state.value_dim;
        const state_matrix_size = state_dim * value_dim;

        var previous_state = try self.allocator.alloc(f32, state_matrix_size);
        defer self.allocator.free(previous_state);

        var projected_state = try self.allocator.alloc(f32, value_dim);
        defer self.allocator.free(projected_state);

        for (0..batch_size) |b| {
            for (0..self.num_heads) |h| {
                const state_offset = (b * self.num_heads + h) * state_matrix_size;

                for (0..seq_len) |t| {
                    const token_offset = ((b * seq_len + t) * self.num_heads + h) * self.head_dim;

                    var lambda: f32 = 0.0;
                    for (0..state_dim) |i| {
                        const k_i = k_ptr[token_offset + i].toFloat32();
                        lambda += k_i * k_i;
                    }

                    const c_t = stableCoefficient(beta, lambda);

                    for (0..state_matrix_size) |idx| {
                        previous_state[idx] = s_ptr[state_offset + idx].toFloat32();
                    }

                    for (0..value_dim) |j| {
                        var sum: f32 = 0.0;
                        for (0..state_dim) |i| {
                            const k_i = k_ptr[token_offset + i].toFloat32();
                            sum += k_i * previous_state[i * value_dim + j];
                        }
                        projected_state[j] = sum;
                    }

                    for (0..state_dim) |i| {
                        const k_i = k_ptr[token_offset + i].toFloat32();
                        for (0..value_dim) |j| {
                            const v_j = v_ptr[token_offset + j].toFloat32();
                            const updated = previous_state[i * value_dim + j] - c_t * k_i * projected_state[j] + c_t * k_i * v_j;
                            s_ptr[state_offset + i * value_dim + j] = dtype_mod.BF16.fromFloat32(updated);
                        }
                    }

                    for (0..value_dim) |j| {
                        var sum: f32 = 0.0;
                        for (0..state_dim) |i| {
                            const s_ij = s_ptr[state_offset + i * value_dim + j].toFloat32();
                            const k_i = k_ptr[token_offset + i].toFloat32();
                            sum += s_ij * k_i;
                        }
                        o_ptr[token_offset + j] = dtype_mod.BF16.fromFloat32(sum);
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
        _ = self;
        _ = grad_output;
        _ = input;
        _ = state;
        return error.UnsupportedOperation;
    }

    fn matmul(self: *EflaLayer, a: *Tensor, b: *Tensor) !*Tensor {
        if (self.device != .cpu) {
            return error.UnsupportedDevice;
        }
        if (a.device != self.device or a.device_id != self.device_id or b.device != self.device or b.device_id != self.device_id) {
            return error.DeviceMismatch;
        }
        if (a.dtype != .bf16 or b.dtype != .bf16) {
            return error.DTypeMismatch;
        }
        if (b.shape.ndim != 2) {
            return error.InvalidInputRank;
        }
        if (a.shape.ndim != 2 and a.shape.ndim != 3) {
            return error.InvalidInputRank;
        }

        const k_dim = a.shape.dim(a.shape.ndim - 1);
        if (k_dim != b.shape.dim(0)) {
            return error.ShapeMismatch;
        }

        const out_shape = switch (a.shape.ndim) {
            2 => Shape.init(&[_]usize{ a.shape.dim(0), b.shape.dim(1) }),
            3 => Shape.init(&[_]usize{ a.shape.dim(0), a.shape.dim(1), b.shape.dim(1) }),
            else => return error.InvalidInputRank,
        };

        const output = try Tensor.init(self.allocator, out_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();

        const a_ptr = a.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const b_ptr = b.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const o_ptr = output.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;

        const n_dim = b.shape.dim(1);

        switch (a.shape.ndim) {
            2 => {
                const m_dim = a.shape.dim(0);
                for (0..m_dim) |m| {
                    for (0..n_dim) |n| {
                        var sum: f32 = 0.0;
                        for (0..k_dim) |k_idx| {
                            const lhs = a_ptr[m * k_dim + k_idx].toFloat32();
                            const rhs = b_ptr[k_idx * n_dim + n].toFloat32();
                            sum += lhs * rhs;
                        }
                        o_ptr[m * n_dim + n] = dtype_mod.BF16.fromFloat32(sum);
                    }
                }
            },
            3 => {
                const batch_size = a.shape.dim(0);
                const m_dim = a.shape.dim(1);
                for (0..batch_size) |batch_idx| {
                    for (0..m_dim) |m| {
                        for (0..n_dim) |n| {
                            var sum: f32 = 0.0;
                            for (0..k_dim) |k_idx| {
                                const lhs = a_ptr[(batch_idx * m_dim + m) * k_dim + k_idx].toFloat32();
                                const rhs = b_ptr[k_idx * n_dim + n].toFloat32();
                                sum += lhs * rhs;
                            }
                            o_ptr[(batch_idx * m_dim + m) * n_dim + n] = dtype_mod.BF16.fromFloat32(sum);
                        }
                    }
                }
            },
            else => return error.InvalidInputRank,
        }

        return output;
    }
};

fn stableCoefficient(beta: f32, lambda: f32) f32 {
    if (lambda < 1e-6) {
        const beta_sq = beta * beta;
        const beta_cu = beta_sq * beta;
        return beta - 0.5 * beta_sq * lambda + (beta_cu * lambda * lambda) / 6.0;
    }
    return (1.0 - @exp(-beta * lambda)) / lambda;
}

pub const ChunkedScan = struct {
    seq_len: usize,
    chunk_size: usize,
    num_chunks: usize,

    pub fn init(seq_len: usize, chunk_size: usize) ChunkedScan {
        return .{
            .seq_len = seq_len,
            .chunk_size = chunk_size,
            .num_chunks = (seq_len + chunk_size - 1) / chunk_size,
        };
    }

    pub fn getChunkRange(self: ChunkedScan, chunk_idx: usize) struct { start: usize, end: usize } {
        const start = chunk_idx * self.chunk_size;
        const end = @min(start + self.chunk_size, self.seq_len);
        return .{ .start = start, .end = end };
    }

    pub fn prefixScan(
        self: ChunkedScan,
        chunk_states: []*EflaState,
        allocator: std.mem.Allocator,
    ) !void {
        _ = allocator;
        if (chunk_states.len != self.num_chunks) {
            return error.InvalidChunkCount;
        }
        if (chunk_states.len <= 1) {
            return;
        }
        return error.UnsupportedOperation;
    }
};

test "EFLA state init" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var state = try EflaState.init(gpa.allocator(), 8, 64, 64, .cpu, 0);
    defer state.deinit();

    try std.testing.expectEqual(@as(usize, 1), state.batch_size);
    try std.testing.expectEqual(@as(usize, 8), state.num_heads);
    try std.testing.expectEqual(@as(usize, 64), state.state_dim);
    try std.testing.expectEqual(@as(usize, 64), state.value_dim);
}

test "ChunkedScan" {
    const scan = ChunkedScan.init(10000, 1024);
    try std.testing.expectEqual(@as(usize, 1024), scan.chunk_size);
    try std.testing.expectEqual(@as(usize, 10), scan.num_chunks);

    const range = scan.getChunkRange(5);
    try std.testing.expectEqual(@as(usize, 5120), range.start);
    try std.testing.expectEqual(@as(usize, 6144), range.end);
}
