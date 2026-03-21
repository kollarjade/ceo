const std = @import("std");

pub const Layout = enum(u8) {
    row_major = 0,
    column_major = 1,
    row_major_aligned = 2,
    column_major_aligned = 3,
    nchw = 4,
    nhwc = 5,
    tensor_core_32x32 = 6,
    tensor_core_16x16 = 7,
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

pub const OpLayout = enum(u8) {
    preserve,
    prefer_row_major,
    prefer_column_major,
    prefer_nhwc,
    prefer_tensor_core,
};

fn checkedRoundUp(value: usize, multiple: usize) usize {
    if (multiple == 0) return value;
    const incremented = std.math.add(usize, value, multiple - 1) catch @panic("size overflow");
    return (incremented / multiple) * multiple;
}

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
        const align_val = if (self.alignment == 0) 1 else self.alignment;
        const mult_val = if (self.pad_to_multiple == 0) 1 else self.pad_to_multiple;
        const aligned = checkedRoundUp(size, align_val);
        return checkedRoundUp(aligned, mult_val);
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

pub fn padDimension(dim: usize, multiple: usize) usize {
    return checkedRoundUp(dim, multiple);
}

pub fn optimalLeadingDimension(dim: usize, dtype_size: usize) usize {
    if (dtype_size == 0) return dim;
    const elements_per_128_bytes = if (dtype_size > 128) 1 else 128 / dtype_size;
    return checkedRoundUp(dim, elements_per_128_bytes);
}

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

pub const MemoryPool = struct {
    const Block = struct {
        ptr: [*]u8,
        size: usize,
        in_use: bool,
    };

    blocks: std.ArrayList(Block),
    buffers: std.ArrayList([]align(128) u8),
    total_size: usize,
    used_size: usize,
    device_id: i32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, initial_size: usize, device_id: i32) !MemoryPool {
        var pool = MemoryPool{
            .blocks = std.ArrayList(Block).init(allocator),
            .buffers = std.ArrayList([]align(128) u8).init(allocator),
            .total_size = 0,
            .used_size = 0,
            .device_id = device_id,
            .allocator = allocator,
        };
        errdefer pool.deinit();

        if (initial_size > 0) {
            const aligned_initial = checkedRoundUp(initial_size, 128);
            const buf = try allocator.alignedAlloc(u8, 128, aligned_initial);
            errdefer allocator.free(buf);

            try pool.buffers.append(buf);
            try pool.blocks.append(.{
                .ptr = buf.ptr,
                .size = aligned_initial,
                .in_use = false,
            });
            pool.total_size = aligned_initial;
        }

        return pool;
    }

    pub fn deinit(self: *MemoryPool) void {
        for (self.buffers.items) |buf| {
            self.allocator.free(buf);
        }
        self.buffers.deinit();
        self.blocks.deinit();
    }

    pub fn allocate(self: *MemoryPool, size: usize) ![*]u8 {
        if (size == 0) return error.InvalidSize;
        const aligned_size = checkedRoundUp(size, 128);

        var best_idx: ?usize = null;
        var best_size: usize = std.math.maxInt(usize);

        for (self.blocks.items, 0..) |block, i| {
            if (!block.in_use and block.size >= aligned_size and block.size < best_size) {
                best_size = block.size;
                best_idx = i;
            }
        }

        if (best_idx) |idx| {
            const original = self.blocks.items[idx];
            if (original.size > aligned_size) {
                const remaining_size = original.size - aligned_size;
                const new_ptr = original.ptr + aligned_size;
                try self.blocks.insert(idx + 1, .{
                    .ptr = new_ptr,
                    .size = remaining_size,
                    .in_use = false,
                });
                self.blocks.items[idx].size = aligned_size;
                self.blocks.items[idx].in_use = true;
            } else {
                self.blocks.items[idx].in_use = true;
            }
            self.used_size += aligned_size;
            return self.blocks.items[idx].ptr;
        }

        const min_alloc = 16 * 1024 * 1024;
        const alloc_size = if (aligned_size > min_alloc) aligned_size else min_alloc;
        const buf = try self.allocator.alignedAlloc(u8, 128, alloc_size);
        errdefer self.allocator.free(buf);

        try self.buffers.append(buf);
        errdefer _ = self.buffers.pop();

        try self.blocks.append(.{
            .ptr = buf.ptr,
            .size = aligned_size,
            .in_use = true,
        });
        errdefer _ = self.blocks.pop();

        if (alloc_size > aligned_size) {
            try self.blocks.append(.{
                .ptr = buf.ptr + aligned_size,
                .size = alloc_size - aligned_size,
                .in_use = false,
            });
        }

        self.total_size += alloc_size;
        self.used_size += aligned_size;
        return buf.ptr;
    }

    pub fn free(self: *MemoryPool, ptr: [*]u8) void {
        for (self.blocks.items) |*block| {
            if (block.ptr == ptr) {
                if (block.in_use) {
                    block.in_use = false;
                    self.used_size -= block.size;
                }
                return;
            }
        }
        @panic("invalid pointer");
    }

    pub fn defragment(self: *MemoryPool) !void {
        if (self.blocks.items.len <= 1) return;

        var i: usize = 0;
        while (i < self.blocks.items.len - 1) {
            const current = self.blocks.items[i];
            const next = self.blocks.items[i + 1];

            if (!current.in_use and !next.in_use and @intFromPtr(current.ptr) + current.size == @intFromPtr(next.ptr)) {
                self.blocks.items[i].size += next.size;
                _ = self.blocks.orderedRemove(i + 1);
            } else {
                i += 1;
            }
        }
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

test "MemoryPool basic operations" {
    var pool = try MemoryPool.init(std.testing.allocator, 1024, 0);
    defer pool.deinit();

    const ptr1 = try pool.allocate(100);
    const ptr2 = try pool.allocate(200);

    pool.free(ptr1);
    pool.free(ptr2);

    try pool.defragment();
}
