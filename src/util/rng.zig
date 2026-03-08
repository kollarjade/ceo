const std = @import("std");

/// Deterministic random number generator for reproducibility
pub const DeterministicRng = struct {
    state: u64,
    increment: u64,

    const Self = @This();

    /// Initialize with seed
    pub fn init(seed: u64) Self {
        return .{
            .state = seed,
            .increment = 1,
        };
    }

    /// Generate next random u64
    pub fn next(self: *Self) u64 {
        // PCG-XSH-RR
        const old_state = self.state;
        self.state = old_state *% 6364136223846793005 +% (self.increment | 1);
        const xorshifted = @as(u32, @truncate(((old_state >> 18) ^ old_state) >> 27));
        const rot = @as(u5, @truncate(old_state >> 59));
        return @as(u64, xorshifted >> @intCast(rot)) | @as(u64, xorshifted << @intCast(@as(u5, @truncate((-(rot) & 31))));
    }

    /// Generate random u32
    pub fn nextU32(self: *Self) u32 {
        return @truncate(self.next());
    }

    /// Generate random f32 in [0, 1)
    pub fn float(self: *Self) f32 {
        const bits = self.nextU32();
        return @as(f32, @floatFromInt(bits >> 8)) / 16777216.0;
    }

    /// Generate random f64 in [0, 1)
    pub fn float64(self: *Self) f64 {
        const bits = self.next();
        return @as(f64, @floatFromInt(bits >> 11)) / 9007199254740992.0;
    }

    /// Generate random f32 from normal distribution (Box-Muller)
    pub fn floatNorm(self: *Self) f32 {
        const u1 = self.float();
        const u2 = self.float();

        const r = @sqrt(-2.0 * @log(u1));
        const theta = 2.0 * std.math.pi * u2;

        return r * @cos(theta);
    }

    /// Generate random int in range [min, max]
    pub fn intRange(self: *Self, comptime T: type, min: T, max: T) T {
        std.debug.assert(min <= max);
        const range = max - min + 1;
        return min + @as(T, @intCast(self.next() % @as(u64, range)));
    }

    /// Shuffle a slice
    pub fn shuffle(self: *Self, comptime T: type, items: []T) void {
        var i: usize = items.len - 1;
        while (i > 0) : (i -= 1) {
            const j = self.intRange(usize, 0, i);
            std.mem.swap(T, &items[i], &items[j]);
        }
    }

    /// Fork into multiple independent RNG streams
    pub fn fork(self: *Self, n: usize, allocator: std.mem.Allocator) ![]Self {
        var rngs = try allocator.alloc(Self, n);
        errdefer allocator.free(rngs);

        for (rngs, 0..) |*r, i| {
            r.state = self.next() ^ @as(u64, i);
            r.increment = self.next();
        }

        return rngs;
    }
};

/// Multi-stream RNG for distributed training
pub const DistributedRng = struct {
    base_seed: u64,
    rank: usize,
    world_size: usize,
    streams: []DeterministicRng,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        base_seed: u64,
        rank: usize,
        world_size: usize,
    ) !Self {
        var base_rng = DeterministicRng.init(base_seed);
        var streams = try allocator.alloc(DeterministicRng, 4); // Multiple streams

        for (streams, 0..) |*stream, i| {
            // Derive unique seed for each rank and stream
            const seed = base_rng.next() ^ (@as(u64, rank) * 0x9e3779b97f4a7c15) ^ (@as(u64, i) * 0xbf58476d1ce4e5b9);
            stream.* = DeterministicRng.init(seed);
        }

        return .{
            .base_seed = base_seed,
            .rank = rank,
            .world_size = world_size,
            .streams = streams,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.streams);
    }

    /// Get RNG for a specific purpose (parameter init, dropout, etc.)
    pub fn getStream(self: *Self, stream_id: usize) *DeterministicRng {
        return &self.streams[stream_id % self.streams.len];
    }

    /// Synchronize RNG state across ranks (for validation)
    pub fn sync(self: *Self) void {
        // In distributed setting, would ensure all ranks have same state
        _ = self;
    }
};

/// Philox-based RNG for GPU
pub const PhiloxRng = struct {
    key: u64,
    counter: u64,

    pub fn init(seed: u64) PhiloxRng {
        return .{
            .key = seed,
            .counter = 0,
        };
    }

    pub fn next(self: *PhiloxRng) u64 {
        // Simplified Philox
        var lo = @as(u32, @truncate(self.counter));
        var hi = @as(u32, @truncate(self.counter >> 32));

        // Philox rounds
        inline for (0..10) |_| {
            lo *%= 0xCD9E8D57;
            hi *%= 0xCD9E8D57;
            lo ^= hi;
            hi ^= lo;
            lo +%= @as(u32, @truncate(self.key));
            hi +%= @as(u32, @truncate(self.key >> 32));
        }

        self.counter += 1;
        return @as(u64, lo) | (@as(u64, hi) << 32);
    }
};

test "DeterministicRng reproducibility" {
    var rng1 = DeterministicRng.init(42);
    var rng2 = DeterministicRng.init(42);

    for (0..100) |_| {
        try std.testing.expectEqual(rng1.next(), rng2.next());
    }
}

test "DeterministicRng shuffle" {
    var rng = DeterministicRng.init(42);
    var items = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    rng.shuffle(usize, &items);

    // Should be shuffled (not original order)
    var same_count: usize = 0;
    for (items, 1..11) |item, expected| {
        if (item == expected) same_count += 1;
    }

    // At least some should have moved
    try std.testing.expect(same_count < 10);
}

test "DistributedRng unique streams" {
    const allocator = std.testing.allocator;

    var rng1 = try DistributedRng.init(allocator, 42, 0, 2);
    defer rng1.deinit();

    var rng2 = try DistributedRng.init(allocator, 42, 1, 2);
    defer rng2.deinit();

    // Different ranks should produce different values
    const v1 = rng1.getStream(0).next();
    const v2 = rng2.getStream(0).next();

    try std.testing.expect(v1 != v2);
}
