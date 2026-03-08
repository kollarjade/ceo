const std = @import("std");
const cuda = @import("../runtime/cuda.zig");
const dtype_mod = @import("dtype.zig");
const layout = @import("layout.zig");
const sharding = @import("sharding.zig");

pub const DType = dtype_mod.DType;
pub const FP8_E4M3 = dtype_mod.FP8_E4M3;
pub const FP8_E5M2 = dtype_mod.FP8_E5M2;
pub const BF16 = dtype_mod.BF16;
pub const FP16 = dtype_mod.FP16;
pub const ScaleFactor = dtype_mod.ScaleFactor;
pub const BlockScale = dtype_mod.BlockScale;
pub const Layout = layout.Layout;
pub const ShardSpec = sharding.ShardSpec;

/// Maximum tensor dimensions
pub const MAX_DIMS = 8;

/// Shape type for tensors
pub const Shape = struct {
    dims: [MAX_DIMS]usize,
    ndim: usize,

    pub fn init(dims: []const usize) Shape {
        var self: Shape = .{
            .dims = [_]usize{0} ** MAX_DIMS,
            .ndim = dims.len,
        };
        @memcpy(self.dims[0..dims.len], dims);
        return self;
    }

    pub fn fromScalar() Shape {
        return .{
            .dims = [_]usize{0} ** MAX_DIMS,
            .ndim = 0,
        };
    }

    pub fn numel(self: Shape) usize {
        if (self.ndim == 0) return 1;
        var n: usize = 1;
        for (self.dims[0..self.ndim]) |d| {
            n *= d;
        }
        return n;
    }

    pub fn sizeBytes(self: Shape, dt: DType) usize {
        return self.numel() * dt.sizeBytes();
    }

    pub fn dim(self: Shape, i: usize) usize {
        std.debug.assert(i < self.ndim);
        return self.dims[i];
    }

    pub fn last(self: Shape) usize {
        return self.dims[self.ndim - 1];
    }

    pub fn first(self: Shape) usize {
        return self.dims[0];
    }

    pub fn equalTo(self: Shape, other: Shape) bool {
        if (self.ndim != other.ndim) return false;
        for (self.dims[0..self.ndim], other.dims[0..self.ndim]) |a, b| {
            if (a != b) return false;
        }
        return true;
    }

    pub fn broadcastable(self: Shape, other: Shape) bool {
        const min_ndim = @min(self.ndim, other.ndim);
        var i: usize = 0;
        while (i < min_ndim) : (i += 1) {
            const a = self.dims[self.ndim - 1 - i];
            const b = other.dims[other.ndim - 1 - i];
            if (a != b and a != 1 and b != 1) return false;
        }
        return true;
    }

    pub fn broadcastShape(self: Shape, other: Shape) Shape {
        const max_ndim = @max(self.ndim, other.ndim);
        var result: Shape = .{
            .dims = [_]usize{0} ** MAX_DIMS,
            .ndim = max_ndim,
        };

        var i: usize = 0;
        while (i < max_ndim) : (i += 1) {
            const a_idx = if (self.ndim > i) self.ndim - 1 - i else null;
            const b_idx = if (other.ndim > i) other.ndim - 1 - i else null;

            const a = if (a_idx) |idx| self.dims[idx] else 1;
            const b = if (b_idx) |idx| other.dims[idx] else 1;

            result.dims[max_ndim - 1 - i] = @max(a, b);
        }

        return result;
    }

    pub fn format(self: Shape, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll("(");
        for (self.dims[0..self.ndim], 0..) |d, i| {
            if (i > 0) try writer.writeAll(", ");
            try std.fmt.format(writer, "{}", .{d});
        }
        try writer.writeAll(")");
    }
};

/// Strides for tensor elements
pub const Strides = struct {
    strides: [MAX_DIMS]usize,
    ndim: usize,

    pub fn fromShape(shape: Shape) Strides {
        var self: Strides = .{
            .strides = [_]usize{0} ** MAX_DIMS,
            .ndim = shape.ndim,
        };

        if (shape.ndim > 0) {
            self.strides[shape.ndim - 1] = 1;
            var i = @as(isize, @intCast(shape.ndim)) - 2;
            while (i >= 0) : (i -= 1) {
                self.strides[@intCast(i)] = self.strides[@intCast(i + 1)] * shape.dims[@intCast(i + 1)];
            }
        }

        return self;
    }

    pub fn isContiguous(self: Strides, shape: Shape) bool {
        if (shape.ndim == 0) return true;

        var expected: usize = 1;
        var i = @as(isize, @intCast(shape.ndim)) - 1;
        while (i >= 0) : (i -= 1) {
            const idx: usize = @intCast(i);
            if (shape.dims[idx] > 1 and self.strides[idx] != expected) return false;
            expected *= shape.dims[idx];
        }
        return true;
    }
};

/// Memory location
pub const Device = enum(u8) {
    cpu = 0,
    cuda = 1,

    pub fn isGpu(self: Device) bool {
        return self == .cuda;
    }
};

/// Tensor flags
pub const TensorFlags = packed struct {
    requires_grad: bool = false,
    is_leaf: bool = true,
    is_view: bool = false,
    is_parameter: bool = false,
    _padding: u4 = 0,
};

/// Main tensor structure
pub const Tensor = struct {
    data: ?*anyopaque,
    shape: Shape,
    strides: Strides,
    dtype: DType,
    device: Device,
    device_id: i32,
    offset: usize,
    flags: TensorFlags,
    scale: ?ScaleFactor,
    block_scale: ?*BlockScale,
    allocator: std.mem.Allocator,
    grad: ?*Tensor,

    /// Create a new tensor with uninitialized data
    pub fn init(
        allocator: std.mem.Allocator,
        shape: Shape,
        dtype: DType,
        device: Device,
        device_id: i32,
    ) !*Tensor {
        const self = try allocator.create(Tensor);
        errdefer allocator.destroy(self);

        const size = shape.sizeBytes(dtype);

        const data: ?*anyopaque = if (size > 0) blk: {
            if (device == .cuda) {
                break :blk try cuda.cudaMalloc(size);
            } else {
                break :blk try allocator.alignedAlloc(u8, 64, size);
            }
        } else null;

        self.* = .{
            .data = data,
            .shape = shape,
            .strides = Strides.fromShape(shape),
            .dtype = dtype,
            .device = device,
            .device_id = device_id,
            .offset = 0,
            .flags = .{ .requires_grad = false, .is_leaf = true },
            .scale = if (dtype.isQuantized()) ScaleFactor.init() else null,
            .block_scale = null,
            .allocator = allocator,
            .grad = null,
        };

        return self;
    }

    /// Create a tensor filled with zeros
    pub fn zeros(
        allocator: std.mem.Allocator,
        shape: Shape,
        dtype: DType,
        device: Device,
        device_id: i32,
    ) !*Tensor {
        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();

        if (self.data) |ptr| {
            if (device == .cuda) {
                try cuda.cudaMemset(ptr, 0, shape.sizeBytes(dtype));
            } else {
                @memset(@as([*]u8, @ptrCast(ptr))[0..shape.sizeBytes(dtype)], 0);
            }
        }

        return self;
    }

    /// Create a tensor filled with a constant value
    pub fn full(
        allocator: std.mem.Allocator,
        shape: Shape,
        dtype: DType,
        device: Device,
        device_id: i32,
        value: f64,
    ) !*Tensor {
        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();

        if (self.data) |ptr| {
            if (device == .cuda) {
                try cuda.cudaFill(ptr, dtype, value, shape.numel());
            } else {
                switch (dtype) {
                    .fp32 => {
                        const slice = @as([*]f32, @ptrCast(ptr))[0..shape.numel()];
                        @memset(slice, @floatCast(value));
                    },
                    .fp16 => {
                        const slice = @as([*]FP16, @ptrCast(ptr))[0..shape.numel()];
                        @memset(slice, FP16.fromFloat32(@floatCast(value)));
                    },
                    .bf16 => {
                        const slice = @as([*]BF16, @ptrCast(ptr))[0..shape.numel()];
                        @memset(slice, BF16.fromFloat32(@floatCast(value)));
                    },
                    else => return error.UnsupportedDType,
                }
            }
        }

        return self;
    }

    /// Create a tensor from a slice of values
    pub fn fromSlice(
        allocator: std.mem.Allocator,
        shape: Shape,
        dtype: DType,
        device: Device,
        device_id: i32,
        values: []const f32,
    ) !*Tensor {
        std.debug.assert(values.len == shape.numel());

        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();

        if (self.data) |ptr| {
            switch (dtype) {
                .fp32 => {
                    const slice = @as([*]f32, @ptrCast(ptr))[0..values.len];
                    @memcpy(slice, values);
                },
                .fp16 => {
                    const slice = @as([*]FP16, @ptrCast(ptr))[0..values.len];
                    for (values, slice) |v, *s| s.* = FP16.fromFloat32(v);
                },
                .bf16 => {
                    const slice = @as([*]BF16, @ptrCast(ptr))[0..values.len];
                    for (values, slice) |v, *s| s.* = BF16.fromFloat32(v);
                },
                else => return error.UnsupportedDType,
            }
        }

        return self;
    }

    /// Create a tensor with random values (uniform distribution)
    pub fn randUniform(
        allocator: std.mem.Allocator,
        shape: Shape,
        dtype: DType,
        device: Device,
        device_id: i32,
        rng: *std.Random,
        low: f32,
        high: f32,
    ) !*Tensor {
        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();

        const numel = shape.numel();
        var values = try allocator.alloc(f32, numel);
        defer allocator.free(values);

        for (values) |*v| {
            v.* = low + (high - low) * rng.float(f32);
        }

        if (self.data) |ptr| {
            if (device == .cuda) {
                try cuda.cudaCopyHostToDevice(ptr, values.ptr, numel * @sizeOf(f32));
            } else {
                switch (dtype) {
                    .fp32 => {
                        const slice = @as([*]f32, @ptrCast(ptr))[0..numel];
                        @memcpy(slice, values);
                    },
                    .fp16 => {
                        const slice = @as([*]FP16, @ptrCast(ptr))[0..numel];
                        for (values, slice) |v, *s| s.* = FP16.fromFloat32(v);
                    },
                    .bf16 => {
                        const slice = @as([*]BF16, @ptrCast(ptr))[0..numel];
                        for (values, slice) |v, *s| s.* = BF16.fromFloat32(v);
                    },
                    else => return error.UnsupportedDType,
                }
            }
        }

        return self;
    }

    /// Create a tensor with random values (normal distribution)
    pub fn randNormal(
        allocator: std.mem.Allocator,
        shape: Shape,
        dtype: DType,
        device: Device,
        device_id: i32,
        rng: *std.Random,
        mean: f32,
        std_dev: f32,
    ) !*Tensor {
        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();

        const numel = shape.numel();
        var values = try allocator.alloc(f32, numel);
        defer allocator.free(values);

        for (values) |*v| {
            v.* = mean + std_dev * rng.floatNorm(f32);
        }

        if (self.data) |ptr| {
            if (device == .cuda) {
                try cuda.cudaCopyHostToDevice(ptr, values.ptr, numel * @sizeOf(f32));
            } else {
                switch (dtype) {
                    .fp32 => {
                        const slice = @as([*]f32, @ptrCast(ptr))[0..numel];
                        @memcpy(slice, values);
                    },
                    .fp16 => {
                        const slice = @as([*]FP16, @ptrCast(ptr))[0..numel];
                        for (values, slice) |v, *s| s.* = FP16.fromFloat32(v);
                    },
                    .bf16 => {
                        const slice = @as([*]BF16, @ptrCast(ptr))[0..numel];
                        for (values, slice) |v, *s| s.* = BF16.fromFloat32(v);
                    },
                    else => return error.UnsupportedDType,
                }
            }
        }

        return self;
    }

    /// Free tensor memory
    pub fn deinit(self: *Tensor) void {
        if (self.data) |ptr| {
            if (self.device == .cuda) {
                cuda.cudaFree(ptr) catch {};
            } else {
                self.allocator.free(@as([*]align(64) u8, @ptrCast(ptr))[0..self.shape.sizeBytes(self.dtype)]);
            }
        }

        if (self.block_scale) |bs| {
            bs.deinit(self.allocator);
            self.allocator.destroy(bs);
        }

        if (self.grad) |g| {
            g.deinit();
            self.allocator.destroy(g);
        }

        self.allocator.destroy(self);
    }

    /// Create a view into this tensor
    pub fn view(self: *Tensor, new_shape: Shape) !*Tensor {
        if (new_shape.numel() != self.shape.numel()) {
            return error.ShapeMismatch;
        }

        const v = try self.allocator.create(Tensor);
        v.* = .{
            .data = self.data,
            .shape = new_shape,
            .strides = Strides.fromShape(new_shape),
            .dtype = self.dtype,
            .device = self.device,
            .device_id = self.device_id,
            .offset = self.offset,
            .flags = .{ .requires_grad = self.flags.requires_grad, .is_leaf = false, .is_view = true },
            .scale = self.scale,
            .block_scale = self.block_scale,
            .allocator = self.allocator,
            .grad = null,
        };

        return v;
    }

    /// Reshape tensor (creates a view if possible)
    pub fn reshape(self: *Tensor, new_shape: Shape) !*Tensor {
        return self.view(new_shape);
    }

    /// Transpose two dimensions
    pub fn transpose(self: *Tensor, dim1: usize, dim2: usize) !*Tensor {
        if (dim1 >= self.shape.ndim or dim2 >= self.shape.ndim) {
            return error.InvalidDimension;
        }

        const v = try self.allocator.create(Tensor);

        var new_shape = self.shape;
        std.mem.swap(usize, &new_shape.dims[dim1], &new_shape.dims[dim2]);

        var new_strides = self.strides;
        std.mem.swap(usize, &new_strides.strides[dim1], &new_strides.strides[dim2]);

        v.* = .{
            .data = self.data,
            .shape = new_shape,
            .strides = new_strides,
            .dtype = self.dtype,
            .device = self.device,
            .device_id = self.device_id,
            .offset = self.offset,
            .flags = .{ .requires_grad = self.flags.requires_grad, .is_leaf = false, .is_view = true },
            .scale = self.scale,
            .block_scale = self.block_scale,
            .allocator = self.allocator,
            .grad = null,
        };

        return v;
    }

    /// Get a slice of the tensor along a dimension
    pub fn slice(self: *Tensor, dim: usize, start: usize, end: usize) !*Tensor {
        if (dim >= self.shape.ndim or start >= end or end > self.shape.dims[dim]) {
            return error.InvalidSlice;
        }

        const v = try self.allocator.create(Tensor);

        var new_shape = self.shape;
        new_shape.dims[dim] = end - start;

        const offset = start * self.strides.strides[dim];

        v.* = .{
            .data = self.data,
            .shape = new_shape,
            .strides = self.strides,
            .dtype = self.dtype,
            .device = self.device,
            .device_id = self.device_id,
            .offset = self.offset + offset,
            .flags = .{ .requires_grad = self.flags.requires_grad, .is_leaf = false, .is_view = true },
            .scale = self.scale,
            .block_scale = self.block_scale,
            .allocator = self.allocator,
            .grad = null,
        };

        return v;
    }

    /// Copy tensor to another device
    pub fn to(self: *const Tensor, allocator: std.mem.Allocator, device: Device, device_id: i32) !*Tensor {
        const dst = try init(allocator, self.shape, self.dtype, device, device_id);
        errdefer dst.deinit();

        if (self.data) |src_ptr| {
            if (dst.data) |dst_ptr| {
                const size = self.shape.sizeBytes(self.dtype);
                if (self.device == .cuda and device == .cpu) {
                    try cuda.cudaCopyDeviceToHost(dst_ptr, src_ptr, size);
                } else if (self.device == .cpu and device == .cuda) {
                    try cuda.cudaCopyHostToDevice(dst_ptr, src_ptr, size);
                } else if (self.device == .cuda and device == .cuda) {
                    try cuda.cudaCopyDeviceToDevice(dst_ptr, src_ptr, size, device_id);
                } else {
                    @memcpy(@as([*]u8, @ptrCast(dst_ptr))[0..size], @as([*]const u8, @ptrCast(src_ptr))[0..size]);
                }
            }
        }

        return dst;
    }

    /// Cast to a different dtype
    pub fn castTo(self: *const Tensor, allocator: std.mem.Allocator, dtype: DType) !*Tensor {
        if (self.dtype == dtype) {
            return self.to(allocator, self.device, self.device_id);
        }

        const dst = try init(allocator, self.shape, dtype, self.device, self.device_id);
        errdefer dst.deinit();

        if (self.device == .cuda) {
            try cuda.cudaCast(self.data, dst.data, self.dtype, dtype, self.shape.numel());
        } else {
            const src_size = self.shape.sizeBytes(self.dtype);
            const dst_size = dst.shape.sizeBytes(dtype);
            var temp_src = try self.allocator.alloc(u8, src_size);
            defer self.allocator.free(temp_src);
            var temp_dst = try self.allocator.alloc(u8, dst_size);
            defer self.allocator.free(temp_dst);

            @memcpy(temp_src, @as([*]const u8, @ptrCast(self.data))[0..src_size]);
            dtype_mod.castSlice(self.dtype, dtype, temp_src, temp_dst);
            @memcpy(@as([*]u8, @ptrCast(dst.data))[0..dst_size], temp_dst);
        }

        return dst;
    }

    /// Get raw pointer to data
    pub fn ptr(self: *const Tensor) ?*anyopaque {
        if (self.data) |p| {
            const base = @as([*]u8, @ptrCast(p)) + self.offset;
            return @ptrCast(base);
        }
        return null;
    }

    /// Get typed pointer to data
    pub fn typedPtr(self: *const Tensor, comptime T: type) ?[*]T {
        if (self.ptr()) |p| {
            return @ptrCast(p);
        }
        return null;
    }

    /// Check if tensor is contiguous
    pub fn isContiguous(self: *const Tensor) bool {
        return self.strides.isContiguous(self.shape);
    }

    /// Make contiguous (copy if needed)
    pub fn contiguous(self: *Tensor) !*Tensor {
        if (self.isContiguous()) {
            return self;
        }

        const dst = try init(self.allocator, self.shape, self.dtype, self.device, self.device_id);
        errdefer dst.deinit();

        // Copy with stride handling
        try self.copyToStrided(dst);

        return dst;
    }

    fn copyToStrided(self: *const Tensor, dst: *Tensor) !void {
        // Implementation would use CUDA kernels for GPU tensors
        _ = self;
        _ = dst;
        return error.NotImplemented;
    }

    /// Fill with a value
    pub fn fill_(self: *Tensor, value: f64) !void {
        if (self.data) |p| {
            if (self.device == .cuda) {
                try cuda.cudaFill(p, self.dtype, value, self.shape.numel());
            } else {
                switch (self.dtype) {
                    .fp32 => {
                        const slice = @as([*]f32, @ptrCast(p))[0..self.shape.numel()];
                        @memset(slice, @floatCast(value));
                    },
                    .fp16 => {
                        const slice = @as([*]FP16, @ptrCast(p))[0..self.shape.numel()];
                        @memset(slice, FP16.fromFloat32(@floatCast(value)));
                    },
                    .bf16 => {
                        const slice = @as([*]BF16, @ptrCast(p))[0..self.shape.numel()];
                        @memset(slice, BF16.fromFloat32(@floatCast(value)));
                    },
                    else => return error.UnsupportedDType,
                }
            }
        }
    }

    /// Zero out the tensor
    pub fn zero_(self: *Tensor) !void {
        if (self.data) |p| {
            if (self.device == .cuda) {
                try cuda.cudaMemset(p, 0, self.shape.sizeBytes(self.dtype));
            } else {
                @memset(@as([*]u8, @ptrCast(p))[0..self.shape.sizeBytes(self.dtype)], 0);
            }
        }
    }

    /// Get element at index (for debugging, not efficient)
    pub fn getItem(self: *const Tensor, indices: []const usize) !f32 {
        _ = self;
        _ = indices;
        return error.NotImplemented;
    }
};

/// Tensor storage for shared memory management
pub const Storage = struct {
    data: ?*anyopaque,
    size: usize,
    device: Device,
    device_id: i32,
    allocator: std.mem.Allocator,
    ref_count: usize,

    pub fn init(allocator: std.mem.Allocator, size: usize, device: Device, device_id: i32) !*Storage {
        const self = try allocator.create(Storage);
        errdefer allocator.destroy(self);

        const data: ?*anyopaque = if (size > 0) blk: {
            if (device == .cuda) {
                break :blk try cuda.cudaMalloc(size);
            } else {
                break :blk try allocator.alignedAlloc(u8, 64, size);
            }
        } else null;

        self.* = .{
            .data = data,
            .size = size,
            .device = device,
            .device_id = device_id,
            .allocator = allocator,
            .ref_count = 1,
        };

        return self;
    }

    pub fn retain(self: *Storage) void {
        self.ref_count += 1;
    }

    pub fn release(self: *Storage) void {
        self.ref_count -= 1;
        if (self.ref_count == 0) {
            if (self.data) |ptr| {
                if (self.device == .cuda) {
                    cuda.cudaFree(ptr) catch {};
                } else {
                    self.allocator.free(@as([*]align(64) u8, @ptrCast(ptr))[0..self.size]);
                }
            }
            self.allocator.destroy(self);
        }
    }
};

test "Shape numel" {
    const s = Shape.init(&[_]usize{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 24), s.numel());
}

test "Shape broadcast" {
    const a = Shape.init(&[_]usize{ 2, 3 });
    const b = Shape.init(&[_]usize{ 1, 3 });
    try std.testing.expect(a.broadcastable(b));

    const c = a.broadcastShape(b);
    try std.testing.expectEqual(@as(usize, 2), c.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), c.dims[1]);
}

test "Tensor zeros" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const shape = Shape.init(&[_]usize{ 2, 3 });
    const t = try Tensor.zeros(gpa.allocator(), shape, .fp32, .cpu, 0);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 6), t.shape.numel());
}

test "Strides contiguous" {
    const shape = Shape.init(&[_]usize{ 2, 3, 4 });
    const strides = Strides.fromShape(shape);

    try std.testing.expect(strides.isContiguous(shape));
    try std.testing.expectEqual(@as(usize, 12), strides.strides[0]);
    try std.testing.expectEqual(@as(usize, 4), strides.strides[1]);
    try std.testing.expectEqual(@as(usize, 1), strides.strides[2]);
}
