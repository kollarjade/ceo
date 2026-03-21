const std = @import("std");

pub const DeterministicRng = struct {
    state: u64,
    increment: u64,

    const Self = @This();

    pub fn init(seed: u64) Self {
        var rng = Self{
            .state = 0,
            .increment = (seed << 1) | 1,
        };
        _ = rng.nextU32();
        rng.state +%= seed;
        _ = rng.nextU32();
        return rng;
    }

    pub fn next(self: *Self) u64 {
        const hi = self.nextU32();
        const lo = self.nextU32();
        return (@as(u64, hi) << 32) | @as(u64, lo);
    }

    pub fn nextU32(self: *Self) u32 {
        const old_state = self.state;
        self.state = old_state *% 6364136223846793005 +% self.increment;
        const xorshifted = @as(u32, @truncate(((old_state >> 18) ^ old_state) >> 27));
        const rot = @as(u5, @truncate(old_state >> 59));
        return (xorshifted >> rot) | (xorshifted << (@as(u5, @truncate(@as(u32, 32) -% @as(u32, rot)))));
    }

    pub fn float(self: *Self) f32 {
        const bits = self.nextU32();
        return @as(f32, @floatFromInt(bits >> 8)) / 16777216.0;
    }

    pub fn float64(self: *Self) f64 {
        const bits = self.next();
        return @as(f64, @floatFromInt(bits >> 11)) / 9007199254740992.0;
    }

    pub fn floatNorm(self: *Self) f32 {
        var u1_val = self.float();
        while (u1_val == 0.0) {
            u1_val = self.float();
        }
        const u2_val = self.float();

        const r = @sqrt(-2.0 * @log(u1_val));
        const theta = 2.0 * std.math.pi * u2_val;

        return r * @cos(theta);
    }

    pub fn intRange(self: *Self, comptime T: type, min: T, max: T) T {
        std.debug.assert(min <= max);
        const info = @typeInfo(T);
        if (info != .Int) unreachable;
        comptime std.debug.assert(info.Int.bits <= 64);

        const U = std.meta.Int(.unsigned, info.Int.bits);
        const min_u: U = @bitCast(min);
        const max_u: U = @bitCast(max);
        const span: U = max_u -% min_u +% 1;

        if (span == 0) {
            return @bitCast(@as(U, @truncate(self.next())));
        }

        const max_u_value = std.math.maxInt(U);
        const limit = max_u_value - (max_u_value % span);

        while (true) {
            const raw: U = @as(U, @truncate(self.next()));
            if (raw < limit) {
                const offset = raw % span;
                return @bitCast(min_u +% offset);
            }
        }
    }

    pub fn shuffle(self: *Self, comptime T: type, items: []T) void {
        if (items.len <= 1) return;
        var i: usize = items.len - 1;
        while (i > 0) : (i -= 1) {
            const j = self.intRange(usize, 0, i);
            std.mem.swap(T, &items[i], &items[j]);
        }
    }

    pub fn fork(self: *Self, n: usize, allocator: std.mem.Allocator) ![]Self {
        var rngs = try allocator.alloc(Self, n);
        errdefer allocator.free(rngs);

        for (rngs, 0..) |*r, i| {
            const seed = self.next() ^ @as(u64, @intCast(i));
            r.* = Self.init(seed);
        }

        return rngs;
    }
};

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
        std.debug.assert(rank < world_size);
        std.debug.assert(world_size > 0);

        var base_rng = DeterministicRng.init(base_seed);
        var streams = try allocator.alloc(DeterministicRng, 4);
        errdefer allocator.free(streams);

        for (streams, 0..) |*stream, i| {
            const seed = base_rng.next() ^ (@as(u64, @intCast(rank)) *% 0x9e3779b97f4a7c15) ^ (@as(u64, @intCast(i)) *% 0xbf58476d1ce4e5b9);
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

    pub fn getStream(self: *Self, stream_id: usize) *DeterministicRng {
        return &self.streams[stream_id % self.streams.len];
    }
};

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
        var lo = @as(u32, @truncate(self.counter));
        var hi = @as(u32, @truncate(self.counter >> 32));
        const key_lo = @as(u32, @truncate(self.key));
        const key_hi = @as(u32, @truncate(self.key >> 32));

        inline for (0..10) |_| {
            const prod_lo: u64 = @as(u64, lo) *% 0xD2511F53;
            const prod_hi: u64 = @as(u64, hi) *% 0xCD9E8D57;
            const new_lo = @as(u32, @truncate(prod_hi >> 32)) ^ lo ^ key_lo;
            const new_hi = @as(u32, @truncate(prod_lo >> 32)) ^ hi ^ key_hi;
            lo = new_lo;
            hi = new_hi;
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

    var same_count: usize = 0;
    for (items, 1..11) |item, expected| {
        if (item == expected) same_count += 1;
    }

    try std.testing.expect(same_count < 10);
}

test "DeterministicRng shuffle empty" {
    var rng = DeterministicRng.init(42);
    var items = [_]usize{};
    rng.shuffle(usize, &items);
}

test "DistributedRng unique streams" {
    const allocator = std.testing.allocator;

    var rng1 = try DistributedRng.init(allocator, 42, 0, 2);
    defer rng1.deinit();

    var rng2 = try DistributedRng.init(allocator, 42, 1, 2);
    defer rng2.deinit();

    const v1 = rng1.getStream(0).next();
    const v2 = rng2.getStream(0).next();

    try std.testing.expect(v1 != v2);
}
