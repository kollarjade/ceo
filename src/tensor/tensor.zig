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

pub const MAX_DIMS = 8;

pub const Shape = struct {
    dims: [MAX_DIMS]usize,
    ndim: usize,

    pub fn init(dims:[]const usize) Shape {
        std.debug.assert(dims.len <= MAX_DIMS);
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
            n = std.math.mul(usize, n, d) catch @panic("Shape.numel overflow");
        }
        return n;
    }

    pub fn sizeBytes(self: Shape, dt: DType) usize {
        return std.math.mul(usize, self.numel(), dt.sizeBytes()) catch @panic("Shape.sizeBytes overflow");
    }

    pub fn dim(self: Shape, i: usize) usize {
        std.debug.assert(i < self.ndim);
        return self.dims[i];
    }

    pub fn last(self: Shape) usize {
        if (self.ndim == 0) return 1;
        return self.dims[self.ndim - 1];
    }

    pub fn first(self: Shape) usize {
        if (self.ndim == 0) return 1;
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
        if (min_ndim == 0) return true;
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
        std.debug.assert(max_ndim <= MAX_DIMS);
        var result: Shape = .{
            .dims = [_]usize{0} ** MAX_DIMS,
            .ndim = max_ndim,
        };
        if (max_ndim == 0) return result;

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

    pub fn format(self: Shape, comptime fmt:[]const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
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

pub const Strides = struct {
    strides:[MAX_DIMS]usize,
    ndim: usize,

    pub fn fromShape(shape: Shape) Strides {
        var self: Strides = .{
            .strides = [_]usize{0} ** MAX_DIMS,
            .ndim = shape.ndim,
        };

        if (shape.ndim > 0) {
            self.strides[shape.ndim - 1] = 1;
            if (shape.ndim > 1) {
                var i: usize = shape.ndim - 2;
                while (true) {
                    self.strides[i] = self.strides[i + 1] * shape.dims[i + 1];
                    if (i == 0) break;
                    i -= 1;
                }
            }
        }

        return self;
    }

    pub fn isContiguous(self: Strides, shape: Shape) bool {
        if (shape.ndim == 0) return true;

        var expected: usize = 1;
        var i: usize = shape.ndim - 1;
        while (true) {
            if (shape.dims[i] > 1 and self.strides[i] != expected) return false;
            expected *= shape.dims[i];
            if (i == 0) break;
            i -= 1;
        }
        return true;
    }
};

pub const Device = enum(u8) {
    cpu = 0,
    cuda = 1,

    pub fn isGpu(self: Device) bool {
        return self == .cuda;
    }
};

pub const TensorFlags = packed struct {
    requires_grad: bool = false,
    is_leaf: bool = true,
    is_view: bool = false,
    is_parameter: bool = false,
    _padding: u4 = 0,
};

pub const Storage = struct {
    data: ?*anyopaque,
    size: usize,
    device: Device,
    device_id: i32,
    allocator: std.mem.Allocator,
    ref_count: std.atomic.Value(usize),

    pub fn init(allocator: std.mem.Allocator, size: usize, device: Device, device_id: i32) !*Storage {
        const self = try allocator.create(Storage);
        errdefer allocator.destroy(self);

        const data: ?*anyopaque = if (size > 0) blk: {
            if (device == .cuda) {
                break :blk try cuda.cudaMalloc(size);
            } else {
                const slice = try allocator.alignedAlloc(u8, 64, size);
                break :blk @as(?*anyopaque, @ptrCast(slice.ptr));
            }
        } else null;

        self.* = .{
            .data = data,
            .size = size,
            .device = device,
            .device_id = if (device == .cuda) device_id else -1,
            .allocator = allocator,
            .ref_count = std.atomic.Value(usize).init(1),
        };

        return self;
    }

    pub fn retain(self: *Storage) void {
        _ = self.ref_count.fetchAdd(1, .monotonic);
    }

    pub fn release(self: *Storage) void {
        if (self.ref_count.fetchSub(1, .release) == 1) {
            self.ref_count.fence(.acquire);
            if (self.data) |ptr| {
                if (self.device == .cuda) {
                    cuda.cudaFree(ptr) catch |err| {
                        std.debug.panic("CUDA free failed: {}", .{err});
                    };
                } else {
                    const slice = @as([*]align(64) u8, @ptrCast(ptr))[0..self.size];
                    self.allocator.free(slice);
                }
            }
            self.allocator.destroy(self);
        }
    }
};

pub const Tensor = struct {
    storage: *Storage,
    shape: Shape,
    strides: Strides,
    dtype: DType,
    offset: usize,
    flags: TensorFlags,
    scale: ?ScaleFactor,
    block_scale: ?*BlockScale,
    grad: ?*Tensor,

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
        const storage = try Storage.init(allocator, size, device, device_id);
        errdefer storage.release();

        self.* = .{
            .storage = storage,
            .shape = shape,
            .strides = Strides.fromShape(shape),
            .dtype = dtype,
            .offset = 0,
            .flags = .{
                .requires_grad = false,
                .is_leaf = true,
                .is_view = false,
                .is_parameter = false,
            },
            .scale = if (dtype.isQuantized()) ScaleFactor.init() else null,
            .block_scale = null,
            .grad = null,
        };

        return self;
    }

    pub fn zeros(
        allocator: std.mem.Allocator,
        shape: Shape,
        dtype: DType,
        device: Device,
        device_id: i32,
    ) !*Tensor {
        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();
        try self.zero_();
        return self;
    }

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
        try self.fill_(value);
        return self;
    }

    pub fn fromSlice(
        allocator: std.mem.Allocator,
        shape: Shape,
        dtype: DType,
        device: Device,
        device_id: i32,
        values:[]const f32,
    ) !*Tensor {
        std.debug.assert(values.len == shape.numel());
        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();

        const byte_size = shape.sizeBytes(dtype);
        if (byte_size == 0) return self;

        var host_bytes = try allocator.alignedAlloc(u8, 64, byte_size);
        defer allocator.free(host_bytes);

        for (values, 0..) |v, i| {
            const val = if (std.math.isNan(v)) 0.0 else if (std.math.isInf(v)) (if (v > 0) std.math.floatMax(f32) else -std.math.floatMax(f32)) else v;
            switch (dtype) {
                .fp64 => @as([*]f64, @ptrCast(host_bytes.ptr))[i] = @floatCast(val),
                .fp32 => @as([*]f32, @ptrCast(host_bytes.ptr))[i] = val,
                .fp16 => @as([*]FP16, @ptrCast(host_bytes.ptr))[i] = FP16.fromFloat32(val),
                .bf16 => @as([*]BF16, @ptrCast(host_bytes.ptr))[i] = BF16.fromFloat32(val),
                .int32 => @as([*]i32, @ptrCast(host_bytes.ptr))[i] = @intFromFloat(val),
                else => return error.UnsupportedDType,
            }
        }

        if (device == .cuda) {
            try cuda.cudaCopyHostToDevice(self.ptr().?, host_bytes.ptr, byte_size);
        } else {
            @memcpy(@as([*]u8, @ptrCast(self.ptr().?))[0..byte_size], host_bytes);
        }

        return self;
    }

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
        std.debug.assert(high >= low);
        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();

        const numel = shape.numel();
        const byte_size = shape.sizeBytes(dtype);
        if (byte_size == 0) return self;

        var host_bytes = try allocator.alignedAlloc(u8, 64, byte_size);
        defer allocator.free(host_bytes);

        for (0..numel) |i| {
            const val = low + (high - low) * rng.float(f32);
            switch (dtype) {
                .fp64 => @as([*]f64, @ptrCast(host_bytes.ptr))[i] = @floatCast(val),
                .fp32 => @as([*]f32, @ptrCast(host_bytes.ptr))[i] = val,
                .fp16 => @as([*]FP16, @ptrCast(host_bytes.ptr))[i] = FP16.fromFloat32(val),
                .bf16 => @as([*]BF16, @ptrCast(host_bytes.ptr))[i] = BF16.fromFloat32(val),
                .int32 => @as([*]i32, @ptrCast(host_bytes.ptr))[i] = @intFromFloat(val),
                else => return error.UnsupportedDType,
            }
        }

        if (device == .cuda) {
            try cuda.cudaCopyHostToDevice(self.ptr().?, host_bytes.ptr, byte_size);
        } else {
            @memcpy(@as([*]u8, @ptrCast(self.ptr().?))[0..byte_size], host_bytes);
        }

        return self;
    }

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
        std.debug.assert(std_dev >= 0.0);
        const self = try init(allocator, shape, dtype, device, device_id);
        errdefer self.deinit();

        const numel = shape.numel();
        const byte_size = shape.sizeBytes(dtype);
        if (byte_size == 0) return self;

        var host_bytes = try allocator.alignedAlloc(u8, 64, byte_size);
        defer allocator.free(host_bytes);

        for (0..numel) |i| {
            const val = mean + std_dev * rng.floatNorm(f32);
            switch (dtype) {
                .fp64 => @as([*]f64, @ptrCast(host_bytes.ptr))[i] = @floatCast(val),
                .fp32 => @as([*]f32, @ptrCast(host_bytes.ptr))[i] = val,
                .fp16 => @as([*]FP16, @ptrCast(host_bytes.ptr))[i] = FP16.fromFloat32(val),
                .bf16 => @as([*]BF16, @ptrCast(host_bytes.ptr))[i] = BF16.fromFloat32(val),
                .int32 => @as([*]i32, @ptrCast(host_bytes.ptr))[i] = @intFromFloat(val),
                else => return error.UnsupportedDType,
            }
        }

        if (device == .cuda) {
            try cuda.cudaCopyHostToDevice(self.ptr().?, host_bytes.ptr, byte_size);
        } else {
            @memcpy(@as([*]u8, @ptrCast(self.ptr().?))[0..byte_size], host_bytes);
        }

        return self;
    }

    pub fn deinit(self: *Tensor) void {
        if (!self.flags.is_view) {
            if (self.block_scale) |bs| {
                bs.deinit(self.storage.allocator);
                self.storage.allocator.destroy(bs);
            }
            if (self.grad) |g| {
                g.deinit();
            }
        }
        self.storage.release();
        self.storage.allocator.destroy(self);
    }

    pub fn view(self: *Tensor, new_shape: Shape) !*Tensor {
        if (new_shape.numel() != self.shape.numel()) {
            return error.ShapeMismatch;
        }
        if (!self.isContiguous()) {
            return error.NotContiguous;
        }

        const v = try self.storage.allocator.create(Tensor);
        errdefer self.storage.allocator.destroy(v);

        self.storage.retain();

        v.* = .{
            .storage = self.storage,
            .shape = new_shape,
            .strides = Strides.fromShape(new_shape),
            .dtype = self.dtype,
            .offset = self.offset,
            .flags = .{
                .requires_grad = self.flags.requires_grad,
                .is_leaf = false,
                .is_view = true,
                .is_parameter = self.flags.is_parameter,
            },
            .scale = self.scale,
            .block_scale = self.block_scale,
            .grad = self.grad,
        };

        return v;
    }

    pub fn reshape(self: *Tensor, new_shape: Shape) !*Tensor {
        if (self.isContiguous()) {
            return self.view(new_shape);
        }
        const contig = try self.contiguous();
        errdefer contig.deinit();
        const v = try contig.view(new_shape);
        v.flags.is_view = false;
        contig.flags.is_view = true;
        contig.deinit();
        return v;
    }

    pub fn transpose(self: *Tensor, dim1: usize, dim2: usize) !*Tensor {
        if (dim1 >= self.shape.ndim or dim2 >= self.shape.ndim) {
            return error.InvalidDimension;
        }

        const v = try self.storage.allocator.create(Tensor);
        errdefer self.storage.allocator.destroy(v);

        self.storage.retain();

        var new_shape = self.shape;
        std.mem.swap(usize, &new_shape.dims[dim1], &new_shape.dims[dim2]);

        var new_strides = self.strides;
        std.mem.swap(usize, &new_strides.strides[dim1], &new_strides.strides[dim2]);

        v.* = .{
            .storage = self.storage,
            .shape = new_shape,
            .strides = new_strides,
            .dtype = self.dtype,
            .offset = self.offset,
            .flags = .{
                .requires_grad = self.flags.requires_grad,
                .is_leaf = false,
                .is_view = true,
                .is_parameter = self.flags.is_parameter,
            },
            .scale = self.scale,
            .block_scale = self.block_scale,
            .grad = self.grad,
        };

        return v;
    }

    pub fn slice(self: *Tensor, dim: usize, start: usize, end: usize) !*Tensor {
        if (dim >= self.shape.ndim or start >= end or end > self.shape.dims[dim]) {
            return error.InvalidSlice;
        }

        const v = try self.storage.allocator.create(Tensor);
        errdefer self.storage.allocator.destroy(v);

        self.storage.retain();

        var new_shape = self.shape;
        new_shape.dims[dim] = end - start;

        const byte_offset = start * self.strides.strides[dim] * self.dtype.sizeBytes();

        v.* = .{
            .storage = self.storage,
            .shape = new_shape,
            .strides = self.strides,
            .dtype = self.dtype,
            .offset = self.offset + byte_offset,
            .flags = .{
                .requires_grad = self.flags.requires_grad,
                .is_leaf = false,
                .is_view = true,
                .is_parameter = self.flags.is_parameter,
            },
            .scale = self.scale,
            .block_scale = self.block_scale,
            .grad = self.grad,
        };

        return v;
    }

    pub fn to(self: *const Tensor, allocator: std.mem.Allocator, device: Device, device_id: i32) !*Tensor {
        const dst = try init(allocator, self.shape, self.dtype, device, device_id);
        errdefer dst.deinit();

        dst.flags = self.flags;
        dst.flags.is_view = false;
        dst.flags.is_leaf = true;
        dst.scale = self.scale;

        if (self.ptr()) |src_ptr| {
            if (dst.ptr()) |dst_ptr| {
                if (self.isContiguous()) {
                    const size = self.shape.sizeBytes(self.dtype);
                    if (self.storage.device == .cuda and device == .cpu) {
                        try cuda.cudaCopyDeviceToHost(dst_ptr, src_ptr, size);
                    } else if (self.storage.device == .cpu and device == .cuda) {
                        try cuda.cudaCopyHostToDevice(dst_ptr, src_ptr, size);
                    } else if (self.storage.device == .cuda and device == .cuda) {
                        try cuda.cudaCopyDeviceToDevice(dst_ptr, src_ptr, size, device_id);
                    } else {
                        @memcpy(@as([*]u8, @ptrCast(dst_ptr))[0..size], @as([*]const u8, @ptrCast(src_ptr))[0..size]);
                    }
                } else {
                    try self.copyToStrided(dst);
                }
            }
        }

        return dst;
    }

    pub fn castTo(self: *const Tensor, allocator: std.mem.Allocator, dtype: DType) !*Tensor {
        if (self.dtype == dtype) {
            return self.to(allocator, self.storage.device, self.storage.device_id);
        }

        const dst = try init(allocator, self.shape, dtype, self.storage.device, self.storage.device_id);
        errdefer dst.deinit();

        dst.flags = self.flags;
        dst.flags.is_view = false;
        dst.flags.is_leaf = true;
        dst.scale = self.scale;

        if (!self.isContiguous()) {
            const contig = try self.contiguous();
            defer contig.deinit();
            const casted = try contig.castTo(allocator, dtype);
            dst.deinit();
            return casted;
        }

        if (self.ptr()) |src_ptr| {
            if (dst.ptr()) |dst_ptr| {
                if (self.storage.device == .cuda) {
                    try cuda.cudaCast(src_ptr, dst_ptr, self.dtype, dtype, self.shape.numel());
                } else {
                    const src_size = self.shape.sizeBytes(self.dtype);
                    const dst_size = dst.shape.sizeBytes(dtype);
                    dtype_mod.castSlice(self.dtype, dtype, @as([*]const u8, @ptrCast(src_ptr))[0..src_size], @as([*]u8, @ptrCast(dst_ptr))[0..dst_size], self.shape.numel());
                }
            }
        }

        return dst;
    }

    pub fn ptr(self: *const Tensor) ?*anyopaque {
        if (self.storage.data) |p| {
            const base = @as([*]u8, @ptrCast(p)) + self.offset;
            return @ptrCast(base);
        }
        return null;
    }

    pub fn typedPtr(self: *const Tensor, comptime T: type) ?[*]T {
        if (self.ptr()) |p| {
            return @ptrCast(@alignCast(p));
        }
        return null;
    }

    pub fn isContiguous(self: *const Tensor) bool {
        return self.strides.isContiguous(self.shape);
    }

    pub fn contiguous(self: *Tensor) !*Tensor {
        if (self.isContiguous()) {
            const v = try self.storage.allocator.create(Tensor);
            errdefer self.storage.allocator.destroy(v);
            self.storage.retain();
            v.* = self.*;
            v.flags.is_view = true;
            v.flags.is_leaf = false;
            return v;
        }

        const dst = try init(self.storage.allocator, self.shape, self.dtype, self.storage.device, self.storage.device_id);
        errdefer dst.deinit();

        dst.flags = self.flags;
        dst.flags.is_view = false;
        dst.flags.is_leaf = false;
        dst.scale = self.scale;

        try self.copyToStrided(dst);

        return dst;
    }

    fn copyToStrided(self: *const Tensor, dst: *Tensor) !void {
        if (self.shape.numel() != dst.shape.numel()) return error.ShapeMismatch;
        if (self.shape.numel() == 0) return;

        if (self.storage.device == .cuda or dst.storage.device == .cuda) {
            const cpu_self = try self.to(self.storage.allocator, .cpu, 0);
            defer cpu_self.deinit();
            const cpu_dst = try dst.to(dst.storage.allocator, .cpu, 0);
            defer cpu_dst.deinit();

            try cpu_self.copyToStrided(cpu_dst);

            if (dst.storage.device == .cuda) {
                try cuda.cudaCopyHostToDevice(dst.ptr().?, cpu_dst.ptr().?, dst.shape.sizeBytes(dst.dtype));
            }
            return;
        }

        const element_size = self.dtype.sizeBytes();
        var src_indices = [_]usize{0} ** MAX_DIMS;
        var i: usize = 0;
        const numel = self.shape.numel();
        const src_ptr = @as([*]const u8, @ptrCast(self.ptr().?));
        const dst_ptr = @as([*]u8, @ptrCast(dst.ptr().?));

        while (i < numel) : (i += 1) {
            var src_offset: usize = 0;
            for (self.shape.dims[0..self.shape.ndim], 0..) |d, dim_idx| {
                _ = d;
                src_offset += src_indices[dim_idx] * self.strides.strides[dim_idx];
            }
            src_offset *= element_size;

            const dst_offset = i * element_size;
            @memcpy(dst_ptr[dst_offset .. dst_offset + element_size], src_ptr[src_offset .. src_offset + element_size]);

            var j: isize = @as(isize, @intCast(self.shape.ndim)) - 1;
            while (j >= 0) : (j -= 1) {
                const dim_idx = @as(usize, @intCast(j));
                src_indices[dim_idx] += 1;
                if (src_indices[dim_idx] < self.shape.dims[dim_idx]) break;
                src_indices[dim_idx] = 0;
            }
        }
    }

    pub fn fill_(self: *Tensor, value: f64) !void {
        if (self.flags.requires_grad) return error.RequiresGradInplace;
        if (self.shape.numel() == 0) return;

        if (self.isContiguous()) {
            if (self.storage.device == .cuda) {
                try cuda.cudaFill(self.ptr().?, self.dtype, value, self.shape.numel());
            } else {
                const numel = self.shape.numel();
                const ptr = self.ptr().?;
                switch (self.dtype) {
                    .fp64 => { const s = @as([*]f64, @ptrCast(@alignCast(ptr)))[0..numel]; @memset(s, value); },
                    .fp32 => { const s = @as([*]f32, @ptrCast(@alignCast(ptr)))[0..numel]; @memset(s, @floatCast(value)); },
                    .fp16 => { const s = @as([*]FP16, @ptrCast(@alignCast(ptr)))[0..numel]; @memset(s, FP16.fromFloat32(@floatCast(value))); },
                    .bf16 => { const s = @as([*]BF16, @ptrCast(@alignCast(ptr)))[0..numel]; @memset(s, BF16.fromFloat32(@floatCast(value))); },
                    .int32 => { const s = @as([*]i32, @ptrCast(@alignCast(ptr)))[0..numel]; @memset(s, @intFromFloat(value)); },
                    else => return error.UnsupportedDType,
                }
            }
        } else {
            const cpu_self = if (self.storage.device == .cuda) try self.to(self.storage.allocator, .cpu, 0) else self;
            defer if (self.storage.device == .cuda) cpu_self.deinit();

            const element_size = cpu_self.dtype.sizeBytes();
            var indices = [_]usize{0} ** MAX_DIMS;
            var i: usize = 0;
            const numel = cpu_self.shape.numel();
            const ptr = @as([*]u8, @ptrCast(cpu_self.ptr().?));

            while (i < numel) : (i += 1) {
                var offset: usize = 0;
                for (cpu_self.shape.dims[0..cpu_self.shape.ndim], 0..) |d, dim_idx| {
                    _ = d;
                    offset += indices[dim_idx] * cpu_self.strides.strides[dim_idx];
                }
                offset *= element_size;

                switch (cpu_self.dtype) {
                    .fp64 => @as(*align(1) f64, @ptrCast(&ptr[offset])).* = value,
                    .fp32 => @as(*align(1) f32, @ptrCast(&ptr[offset])).* = @floatCast(value),
                    .fp16 => @as(*align(1) FP16, @ptrCast(&ptr[offset])).* = FP16.fromFloat32(@floatCast(value)),
                    .bf16 => @as(*align(1) BF16, @ptrCast(&ptr[offset])).* = BF16.fromFloat32(@floatCast(value)),
                    .int32 => @as(*align(1) i32, @ptrCast(&ptr[offset])).* = @intFromFloat(value),
                    else => return error.UnsupportedDType,
                }

                var j: isize = @as(isize, @intCast(cpu_self.shape.ndim)) - 1;
                while (j >= 0) : (j -= 1) {
                    const dim_idx = @as(usize, @intCast(j));
                    indices[dim_idx] += 1;
                    if (indices[dim_idx] < cpu_self.shape.dims[dim_idx]) break;
                    indices[dim_idx] = 0;
                }
            }

            if (self.storage.device == .cuda) {
                try cuda.cudaCopyHostToDevice(self.ptr().?, cpu_self.ptr().?, self.shape.sizeBytes(self.dtype));
            }
        }
    }

    pub fn zero_(self: *Tensor) !void {
        if (self.flags.requires_grad) return error.RequiresGradInplace;
        if (self.shape.numel() == 0) return;

        if (self.isContiguous()) {
            if (self.storage.device == .cuda) {
                try cuda.cudaMemset(self.ptr().?, 0, self.shape.sizeBytes(self.dtype));
            } else {
                @memset(@as([*]u8, @ptrCast(self.ptr().?))[0..self.shape.sizeBytes(self.dtype)], 0);
            }
        } else {
            const cpu_self = if (self.storage.device == .cuda) try self.to(self.storage.allocator, .cpu, 0) else self;
            defer if (self.storage.device == .cuda) cpu_self.deinit();

            const element_size = cpu_self.dtype.sizeBytes();
            var indices = [_]usize{0} ** MAX_DIMS;
            var i: usize = 0;
            const numel = cpu_self.shape.numel();
            const ptr = @as([*]u8, @ptrCast(cpu_self.ptr().?));

            while (i < numel) : (i += 1) {
                var offset: usize = 0;
                for (cpu_self.shape.dims[0..cpu_self.shape.ndim], 0..) |d, dim_idx| {
                    _ = d;
                    offset += indices[dim_idx] * cpu_self.strides.strides[dim_idx];
                }
                offset *= element_size;

                @memset(ptr[offset .. offset + element_size], 0);

                var j: isize = @as(isize, @intCast(cpu_self.shape.ndim)) - 1;
                while (j >= 0) : (j -= 1) {
                    const dim_idx = @as(usize, @intCast(j));
                    indices[dim_idx] += 1;
                    if (indices[dim_idx] < cpu_self.shape.dims[dim_idx]) break;
                    indices[dim_idx] = 0;
                }
            }

            if (self.storage.device == .cuda) {
                try cuda.cudaCopyHostToDevice(self.ptr().?, cpu_self.ptr().?, self.shape.sizeBytes(self.dtype));
            }
        }
    }

    pub fn getItem(self: *const Tensor, indices:[]const usize) !f64 {
        if (indices.len != self.shape.ndim) return error.InvalidDimension;
        var offset: usize = 0;
        for (indices, 0..) |idx, i| {
            if (idx >= self.shape.dims[i]) return error.IndexOutOfBounds;
            offset += idx * self.strides.strides[i];
        }
        offset *= self.dtype.sizeBytes();

        var val_bytes: [8]u8 = [_]u8{0} ** 8;
        if (self.storage.device == .cuda) {
            try cuda.cudaCopyDeviceToHost(&val_bytes, @as([*]const u8, @ptrCast(self.ptr().?)) + offset, self.dtype.sizeBytes());
        } else {
            const ptr = @as([*]const u8, @ptrCast(self.ptr().?)) + offset;
            @memcpy(val_bytes[0..self.dtype.sizeBytes()], ptr[0..self.dtype.sizeBytes()]);
        }

        switch (self.dtype) {
            .fp64 => return @as(*align(1) const f64, @ptrCast(&val_bytes)).*,
            .fp32 => return @floatCast(@as(*align(1) const f32, @ptrCast(&val_bytes)).*),
            .fp16 => return @floatCast(@as(*align(1) const FP16, @ptrCast(&val_bytes)).toFloat32()),
            .bf16 => return @floatCast(@as(*align(1) const BF16, @ptrCast(&val_bytes)).toFloat32()),
            .int32 => return @floatFromInt(@as(*align(1) const i32, @ptrCast(&val_bytes)).*),
            else => return error.UnsupportedDType,
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
