const std = @import("std");

/// Memory layout for tensors
pub const Layout = enum(u8) {
    row_major = 0,
    column_major = 1,
    /// Row-major with 32-byte aligned rows
    row_major_aligned = 2,
    /// Column-major with 32-byte aligned columns
    column_major_aligned = 3,
    /// NC/NC layout for convolutions (NCHW/NHWC)
    nchw = 4,
    nhwc = 5,
    /// Tensor memory layout for tensor cores
    tensor_core_32x32 = 6,
    tensor_core_16x16 = 7,
    /// Strided layout (custom strides)
    strided = 8,

    pub fn isAligned(self: Layout) bool {
        return switch (self) {
            .row_major_aligned, .column_major_aligned => true,
            else => false,
        };
    }

    pub fn isRowMajor(self: Layout) bool {
        return switch (self) {
            .row_major, .row_major_aligned, .nchw, .tensor_core_32x32, .tensor_core_16x16 => true,
            else => false,
        };
    }

    pub fn isColumnMajor(self: Layout) bool {
        return switch (self) {
            .column_major, .column_major_aligned, .nhwc => true,
            else => false,
        };
    }
};

/// Tensor operation layout preferences
pub const OpLayout = enum(u8) {
    /// Keep as-is
    preserve,
    /// Prefer row-major for this operation
    prefer_row_major,
    /// Prefer column-major for this operation
    prefer_column_major,
    /// Prefer NHWC for convolution-style ops
    prefer_nhwc,
    /// Prefer tensor-core optimized layout
    prefer_tensor_core,
};

/// Padding strategy for memory alignment
pub const PaddingStrategy = struct {
    alignment: usize,
    pad_to_multiple: usize,

    pub fn init(alignment: usize, multiple: usize) PaddingStrategy {
        return .{
            .alignment = alignment,
            .pad_to_multiple = multiple,
        };
    }

    pub fn paddedSize(self: PaddingStrategy, size: usize) usize {
        const aligned = (size + self.alignment - 1) / self.alignment * self.alignment;
        return (aligned + self.pad_to_multiple - 1) / self.pad_to_multiple * self.pad_to_multiple;
    }

    pub fn default() PaddingStrategy {
        return .{
            .alignment = 32,
            .pad_to_multiple = 128,
        };
    }

    pub fn tensorCore() PaddingStrategy {
        return .{
            .alignment = 128,
            .pad_to_multiple = 128,
        };
    }
};

/// Compute padded dimension for tensor cores
pub fn padDimension(dim: usize, multiple: usize) usize {
    return (dim + multiple - 1) / multiple * multiple;
}

/// Get optimal leading dimension for GEMM
pub fn optimalLeadingDimension(dim: usize, dtype_size: usize) usize {
    // Round up to next multiple of 128 bytes / dtype_size
    const elements_per_128_bytes = 128 / dtype_size;
    return (dim + elements_per_128_bytes - 1) / elements_per_128_bytes * elements_per_128_bytes;
}

/// Tensor core tile sizes for SM100
pub const TensorCoreTiles = struct {
    pub const M: usize = 128;
    pub const N: usize = 128;
    pub const K: usize = 64;

    pub const WGMMA_M: usize = 64;
    pub const WGMMA_N: usize = 64;
    pub const WGMMA_K: usize = 32;

    pub const is_valid_m = (M % WGMMA_M == 0);
    pub const is_valid_n = (N % WGMMA_N == 0);
    pub const is_valid_k = (K % WGMMA_K == 0);
};

/// Memory pool for efficient tensor allocation
pub const MemoryPool = struct {
    const Block = struct {
        ptr: [*]u8,
        size: usize,
        in_use: bool,
    };

    blocks: std.ArrayList(Block),
    total_size: usize,
    used_size: usize,
    device_id: i32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, initial_size: usize, device_id: i32) !MemoryPool {
        _ = initial_size;
        return .{
            .blocks = std.ArrayList(Block).init(allocator),
            .total_size = 0,
            .used_size = 0,
            .device_id = device_id,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MemoryPool) void {
        self.blocks.deinit();
    }

    pub fn allocate(self: *MemoryPool, size: usize) ![*]u8 {
        _ = self;
        _ = size;
        return error.NotImplemented;
    }

    pub fn free(self: *MemoryPool, ptr: [*]u8) void {
        _ = ptr;
        _ = self;
    }

    pub fn defragment(self: *MemoryPool) !void {
        _ = self;
    }
};

test "PaddingStrategy paddedSize" {
    const ps = PaddingStrategy.init(32, 128);
    try std.testing.expectEqual(@as(usize, 128), ps.paddedSize(100));
    try std.testing.expectEqual(@as(usize, 256), ps.paddedSize(200));
}

test "padDimension" {
    try std.testing.expectEqual(@as(usize, 128), padDimension(100, 128));
    try std.testing.expectEqual(@as(usize, 128), padDimension(128, 128));
    try std.testing.expectEqual(@as(usize, 256), padDimension(129, 128));
}
