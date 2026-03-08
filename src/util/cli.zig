const std = @import("std");

/// CLI argument parser
pub const ArgParser = struct {
    args: []const []const u8,
    idx: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, args: []const []const u8) Self {
        return .{
            .args = args,
            .idx = 0,
            .allocator = allocator,
        };
    }

    pub fn next(self: *Self) ?[]const u8 {
        if (self.idx >= self.args.len) return null;
        const arg = self.args[self.idx];
        self.idx += 1;
        return arg;
    }

    pub fn peek(self: *Self) ?[]const u8 {
        if (self.idx >= self.args.len) return null;
        return self.args[self.idx];
    }

    pub fn expectString(self: *Self) ![]const u8 {
        return self.next() orelse error.MissingArgument;
    }

    pub fn expectInt(self: *Self, comptime T: type) !T {
        const str = try self.expectString();
        return std.fmt.parseInt(T, str, 10) catch error.InvalidInteger;
    }

    pub fn expectFloat(self: *Self, comptime T: type) !T {
        const str = try self.expectString();
        return std.fmt.parseFloat(T, str) catch error.InvalidFloat;
    }

    pub fn parseFlag(self: *Self, name: []const u8) bool {
        if (self.peek()) |arg| {
            if (std.mem.eql(u8, arg, name)) {
                _ = self.next();
                return true;
            }
        }
        return false;
    }

    pub fn parseOption(self: *Self, name: []const u8) ?[]const u8 {
        if (self.peek()) |arg| {
            if (std.mem.eql(u8, arg, name)) {
                _ = self.next();
                return self.next();
            }
        }
        return null;
    }
};

pub const ParseError = error{
    MissingArgument,
    InvalidInteger,
    InvalidFloat,
    UnknownArgument,
};

/// Parse key=value arguments
pub fn parseKeyValue(allocator: std.mem.Allocator, input: []const u8) !std.StringHashMap([]const u8) {
    var map = std.StringHashMap([]const u8).init(allocator);
    errdefer map.deinit();

    var iter = std.mem.split(u8, input, ",");
    while (iter.next()) |pair| {
        const eq_idx = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
        const key = std.mem.trim(u8, pair[0..eq_idx], " \t");
        const value = std.mem.trim(u8, pair[eq_idx + 1 ..], " \t");
        try map.put(try allocator.dupe(u8, key), try allocator.dupe(u8, value));
    }

    return map;
}

test "ArgParser" {
    const args = [_][]const u8{ "--config", "test.yaml", "--verbose", "42" };

    var parser = ArgParser.init(std.testing.allocator, &args);

    const config = parser.parseOption("--config");
    try std.testing.expect(config != null);
    try std.testing.expectEqualStrings("test.yaml", config.?);

    try std.testing.expect(!parser.parseFlag("--quiet"));
    try std.testing.expect(parser.parseFlag("--verbose"));

    const num = try parser.expectInt(i32);
    try std.testing.expectEqual(@as(i32, 42), num);
}
