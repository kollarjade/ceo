const std = @import("std");

/// SHA-256 hash
pub const Sha256 = std.crypto.hash.sha2.Sha256;

/// Compute SHA-256 hash of data
pub fn hash(data: []const u8) [32]u8 {
    var result: [32]u8 = undefined;
    Sha256.hash(data, &result, .{});
    return result;
}

/// Compute SHA-256 hash of file
pub fn hashFile(path: []const u8) ![32]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var hasher = Sha256.init(.{});

    var buffer: [4096]u8 = undefined;
    while (true) {
        const bytes_read = try file.read(&buffer);
        if (bytes_read == 0) break;
        hasher.update(buffer[0..bytes_read]);
    }

    var result: [32]u8 = undefined;
    hasher.final(&result);
    return result;
}

/// Format hash as hex string
pub fn formatHex(hash_value: [32]u8, allocator: std.mem.Allocator) ![]const u8 {
    var result = try allocator.alloc(u8, 64);
    _ = std.fmt.bufPrint(result, "{s}", .{std.fmt.fmtSliceHexLower(&hash_value)}) catch unreachable;
    return result;
}

/// Verify hash matches expected
pub fn verifyHash(data: []const u8, expected: [32]u8) bool {
    const computed = hash(data);
    return std.mem.eql(u8, &computed, &expected);
}

/// XXHash for fast non-cryptographic hashing
pub const XxHash = struct {
    seed: u64,

    pub fn init(seed: u64) XxHash {
        return .{ .seed = seed };
    }

    pub fn hash(self: XxHash, data: []const u8) u64 {
        return std.hash.xxHash3(data, self.seed);
    }
};

/// Hash a struct deterministically
pub fn hashStruct(comptime T: type, value: T) u64 {
    const bytes = std.mem.asBytes(&value);
    return std.hash.Wyhash.hash(0, bytes);
}

/// Combine two hashes
pub fn combineHash(a: u64, b: u64) u64 {
    return a ^ (b *% 0x9e3779b97f4a7c15);
}

/// Deterministic string hash for use in hash maps
pub fn hashString(str: []const u8) u32 {
    var h: u32 = 0;
    for (str) |c| {
        h = h *% 31 + @as(u32, c);
    }
    return h;
}

test "hash and verify" {
    const data = "Hello, world!";
    const h = hash(data);

    try std.testing.expect(verifyHash(data, h));
    try std.testing.expect(!verifyHash("Different", h));
}

test "formatHex" {
    const h = hash("test");
    const hex = try formatHex(h, std.testing.allocator);
    defer std.testing.allocator.free(hex);

    try std.testing.expectEqual(@as(usize, 64), hex.len);
}

test "XxHash" {
    const hasher = XxHash.init(42);
    const h1 = hasher.hash("test");
    const h2 = hasher.hash("test");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(hasher.hash("different") != h1);
}
