const std = @import("std");

/// Simple YAML value
pub const YamlValue = union(enum) {
    null: void,
    bool: bool,
    int: i64,
    float: f64,
    string: []const u8,
    list: YamlList,
    map: YamlMap,

    pub fn deinit(self: *YamlValue, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .string => |s| allocator.free(s),
            .list => |*l| {
                for (l.items) |*item| {
                    item.deinit(allocator);
                }
                l.deinit(allocator);
            },
            .map => |*m| {
                var iter = m.iterator();
                while (iter.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    entry.value_ptr.deinit(allocator);
                }
                m.deinit(allocator);
            },
            else => {},
        }
    }

    pub fn asString(self: YamlValue, allocator: std.mem.Allocator) ![]const u8 {
        return switch (self) {
            .string => |s| allocator.dupe(u8, s),
            else => error.NotAString,
        };
    }

    pub fn asInt(self: YamlValue) !i64 {
        return switch (self) {
            .int => |i| i,
            else => error.NotAnInt,
        };
    }

    pub fn asFloat(self: YamlValue) !f64 {
        return switch (self) {
            .float => |f| f,
            .int => |i| @floatFromInt(i),
            else => error.NotAFloat,
        };
    }

    pub fn asBool(self: YamlValue) !bool {
        return switch (self) {
            .bool => |b| b,
            else => error.NotABool,
        };
    }

    pub fn asList(self: YamlValue) !*YamlList {
        return switch (self) {
            .list => |*l| l,
            else => error.NotAList,
        };
    }

    pub fn asMap(self: YamlValue) !*YamlMap {
        return switch (self) {
            .map => |*m| m,
            else => error.NotAMap,
        };
    }
};

pub const YamlList = std.ArrayList(YamlValue);
pub const YamlMap = std.StringHashMap(YamlValue);

/// YAML parsing result
pub const YamlRoot = struct {
    value: YamlValue,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *YamlRoot) void {
        self.value.deinit(self.allocator);
    }

    pub fn getMap(self: YamlRoot, key: []const u8) ?*YamlMap {
        if (self.value == .map) {
            if (self.value.map.getPtr(key)) |v| {
                if (v.* == .map) {
                    return &v.map;
                }
            }
        }
        return null;
    }

    pub fn getList(self: YamlRoot, key: []const u8) ?*YamlList {
        if (self.value == .map) {
            if (self.value.map.getPtr(key)) |v| {
                if (v.* == .list) {
                    return &v.list;
                }
            }
        }
        return null;
    }

    pub fn getString(self: YamlRoot, key: []const u8) ?[]const u8 {
        if (self.value == .map) {
            if (self.value.map.get(key)) |v| {
                if (v == .string) return v.string;
            }
        }
        return null;
    }
};

/// YAML map helper methods
pub const YamlMapHelper = struct {
    pub fn getString(self: *YamlMap, key: []const u8) ?[]const u8 {
        if (self.getPtr(key)) |v| {
            if (v.* == .string) return v.string;
        }
        return null;
    }

    pub fn getInt(self: *YamlMap, comptime T: type, key: []const u8) ?T {
        if (self.getPtr(key)) |v| {
            if (v.* == .int) {
                return @intCast(v.int);
            }
        }
        return null;
    }

    pub fn getFloat(self: *YamlMap, comptime T: type, key: []const u8) ?T {
        if (self.getPtr(key)) |v| {
            if (v.* == .float) {
                return @floatCast(v.float);
            }
            if (v.* == .int) {
                return @floatFromInt(v.int);
            }
        }
        return null;
    }

    pub fn getBool(self: *YamlMap, key: []const u8) ?bool {
        if (self.getPtr(key)) |v| {
            if (v.* == .bool) return v.bool;
        }
        return null;
    }

    pub fn getMap(self: *YamlMap, key: []const u8) ?*YamlMap {
        if (self.getPtr(key)) |v| {
            if (v.* == .map) return &v.map;
        }
        return null;
    }

    pub fn getList(self: *YamlMap, key: []const u8) ?*YamlList {
        if (self.getPtr(key)) |v| {
            if (v.* == .list) return &v.list;
        }
        return null;
    }
};

/// Extend YamlMap with helper methods
pub fn getMapString(self: *YamlMap, key: []const u8) ?[]const u8 {
    return YamlMapHelper.getString(self, key);
}

pub fn getMapInt(self: *YamlMap, comptime T: type, key: []const u8) ?T {
    return YamlMapHelper.getInt(self, T, key);
}

pub fn getMapFloat(self: *YamlMap, comptime T: type, key: []const u8) ?T {
    return YamlMapHelper.getFloat(self, T, key);
}

pub fn getMapBool(self: *YamlMap, key: []const u8) ?bool {
    return YamlMapHelper.getBool(self, key);
}

pub fn getMapMap(self: *YamlMap, key: []const u8) ?*YamlMap {
    return YamlMapHelper.getMap(self, key);
}

pub fn getMapList(self: *YamlMap, key: []const u8) ?*YamlList {
    return YamlMapHelper.getList(self, key);
}

/// Simple YAML parser
pub const YamlParser = struct {
    content: []const u8,
    pos: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, content: []const u8) Self {
        return .{
            .content = content,
            .pos = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn parse(self: *Self) !YamlRoot {
        var value = try self.parseValue(0);
        return .{
            .value = value,
            .allocator = self.allocator,
        };
    }

    fn parseValue(self: *Self, indent: usize) !YamlValue {
        self.skipWhitespace();

        if (self.pos >= self.content.len) {
            return .{ .null = {} };
        }

        const c = self.content[self.pos];

        // Check for list item
        if (c == '-') {
            return self.parseList(indent);
        }

        // Check for map
        if (self.isMapKey()) {
            return self.parseMap(indent);
        }

        // Check for quoted string
        if (c == '"' or c == '\'') {
            return self.parseQuotedString();
        }

        // Check for number or boolean or plain string
        return self.parseScalar();
    }

    fn parseMap(self: *Self, indent: usize) !YamlValue {
        var map = YamlMap.init(self.allocator);
        errdefer {
            var iter = map.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                entry.value_ptr.deinit(self.allocator);
            }
            map.deinit(self.allocator);
        }

        while (self.pos < self.content.len) {
            self.skipWhitespace();

            if (self.pos >= self.content.len) break;

            // Check current indentation
            const current_indent = self.countIndent();
            if (current_indent < indent) break;
            if (current_indent > indent and indent > 0) {
                // Nested content, should have been parsed already
                break;
            }

            if (!self.isMapKey()) break;

            // Parse key
            const key = try self.parseMapKey();
            errdefer self.allocator.free(key);

            // Skip colon and whitespace
            self.skipChar(':');
            self.skipWhitespace();

            // Parse value
            const value = try self.parseValue(indent + 2);
            errdefer value.deinit(self.allocator);

            try map.put(key, value);
        }

        return .{ .map = map };
    }

    fn parseList(self: *Self, indent: usize) !YamlValue {
        var list = YamlList.init(self.allocator);
        errdefer {
            for (list.items) |*item| {
                item.deinit(self.allocator);
            }
            list.deinit(self.allocator);
        }

        while (self.pos < self.content.len) {
            self.skipWhitespace();

            if (self.pos >= self.content.len) break;

            const current_indent = self.countIndent();
            if (current_indent < indent) break;
            if (current_indent > indent and indent > 0) break;

            if (self.content[self.pos] != '-') break;

            // Skip '-'
            self.pos += 1;
            self.skipWhitespace();

            const value = try self.parseValue(indent + 2);
            errdefer value.deinit(self.allocator);

            try list.append(value);
        }

        return .{ .list = list };
    }

    fn parseMapKey(self: *Self) ![]const u8 {
        const start = self.pos;

        while (self.pos < self.content.len) {
            const c = self.content[self.pos];
            if (c == ':' or std.ascii.isWhitespace(c)) break;
            self.pos += 1;
        }

        return self.allocator.dupe(u8, self.content[start..self.pos]);
    }

    fn parseQuotedString(self: *Self) !YamlValue {
        const quote = self.content[self.pos];
        self.pos += 1;

        const start = self.pos;

        while (self.pos < self.content.len) {
            if (self.content[self.pos] == quote) {
                const str = try self.allocator.dupe(u8, self.content[start..self.pos]);
                self.pos += 1;
                return .{ .string = str };
            }
            if (self.content[self.pos] == '\\') {
                self.pos += 2;
            } else {
                self.pos += 1;
            }
        }

        return error.UnterminatedString;
    }

    fn parseScalar(self: *Self) !YamlValue {
        const start = self.pos;

        while (self.pos < self.content.len) {
            const c = self.content[self.pos];
            if (c == '\n' or c == '#') break;
            self.pos += 1;
        }

        const str = std.mem.trim(u8, self.content[start..self.pos], " \t\r\n");

        // Check for null
        if (std.mem.eql(u8, str, "null") or std.mem.eql(u8, str, "~")) {
            return .{ .null = {} };
        }

        // Check for boolean
        if (std.mem.eql(u8, str, "true") or std.mem.eql(u8, str, "True") or std.mem.eql(u8, str, "yes")) {
            return .{ .bool = true };
        }
        if (std.mem.eql(u8, str, "false") or std.mem.eql(u8, str, "False") or std.mem.eql(u8, str, "no")) {
            return .{ .bool = false };
        }

        // Check for integer
        if (parseInt(str)) |i| {
            return .{ .int = i };
        }

        // Check for float
        if (parseFloat(str)) |f| {
            return .{ .float = f };
        }

        // Plain string
        return .{ .string = try self.allocator.dupe(u8, str) };
    }

    fn parseInt(str: []const u8) ?i64 {
        var result: i64 = 0;
        var sign: i64 = 1;
        var pos: usize = 0;

        if (str.len == 0) return null;

        if (str[0] == '-') {
            sign = -1;
            pos = 1;
        } else if (str[0] == '+') {
            pos = 1;
        }

        if (pos >= str.len) return null;

        // Check for hex
        if (str.len > pos + 2 and str[pos] == '0' and (str[pos + 1] == 'x' or str[pos + 1] == 'X')) {
            pos += 2;
            while (pos < str.len) : (pos += 1) {
                const c = str[pos];
                const digit: i64 = if (c >= '0' and c <= '9')
                    c - '0'
                else if (c >= 'a' and c <= 'f')
                    c - 'a' + 10
                else if (c >= 'A' and c <= 'F')
                    c - 'A' + 10
                else
                    return null;
                result = result * 16 + digit;
            }
            return result * sign;
        }

        // Decimal
        while (pos < str.len) : (pos += 1) {
            const c = str[pos];
            if (c >= '0' and c <= '9') {
                result = result * 10 + (c - '0');
            } else {
                return null;
            }
        }

        return result * sign;
    }

    fn parseFloat(str: []const u8) ?f64 {
        // Simple float parsing
        var has_dot = false;
        var has_exp = false;

        for (str) |c| {
            if (c == '.') {
                if (has_dot) return null;
                has_dot = true;
            } else if (c == 'e' or c == 'E') {
                if (has_exp) return null;
                has_exp = true;
            } else if ((c >= '0' and c <= '9') or c == '-' or c == '+') {
                // OK
            } else {
                return null;
            }
        }

        if (!has_dot and !has_exp) return null;

        return std.fmt.parseFloat(f64, str) catch null;
    }

    fn isMapKey(self: *Self) bool {
        const start = self.pos;

        while (self.pos < self.content.len) {
            const c = self.content[self.pos];
            if (c == ':') {
                self.pos = start;
                return true;
            }
            if (std.ascii.isWhitespace(c)) {
                // Check if next non-whitespace is colon
                var check_pos = self.pos + 1;
                while (check_pos < self.content.len and std.ascii.isWhitespace(self.content[check_pos])) {
                    check_pos += 1;
                }
                if (check_pos < self.content.len and self.content[check_pos] == ':') {
                    self.pos = start;
                    return true;
                }
                break;
            }
            self.pos += 1;
        }

        self.pos = start;
        return false;
    }

    fn skipWhitespace(self: *Self) void {
        while (self.pos < self.content.len) {
            const c = self.content[self.pos];
            if (c == ' ' or c == '\t' or c == '\r') {
                self.pos += 1;
            } else if (c == '\n') {
                self.pos += 1;
            } else if (c == '#') {
                // Skip comment
                while (self.pos < self.content.len and self.content[self.pos] != '\n') {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn skipChar(self: *Self, char: u8) void {
        if (self.pos < self.content.len and self.content[self.pos] == char) {
            self.pos += 1;
        }
    }

    fn countIndent(self: *Self) usize {
        var count: usize = 0;
        var pos = self.pos;

        // Walk back to start of line
        while (pos > 0 and self.content[pos - 1] != '\n') {
            pos -= 1;
        }

        // Count spaces/tabs
        while (pos < self.content.len) {
            if (self.content[pos] == ' ') {
                count += 1;
                pos += 1;
            } else if (self.content[pos] == '\t') {
                count += 2;
                pos += 1;
            } else {
                break;
            }
        }

        return count;
    }
};

test "YAML parse simple map" {
    const yaml = "key: value\nnumber: 42\nfloat: 3.14\nbool: true";

    var parser = YamlParser.init(std.testing.allocator, yaml);
    defer parser.deinit();

    var root = try parser.parse();
    defer root.deinit();

    try std.testing.expect(root.value == .map);
    try std.testing.expectEqualStrings("value", root.value.map.get("key").?.string);
    try std.testing.expectEqual(@as(i64, 42), root.value.map.get("number").?.int);
}

test "YAML parse list" {
    const yaml = "- item1\n- item2\n- item3";

    var parser = YamlParser.init(std.testing.allocator, yaml);
    defer parser.deinit();

    var root = try parser.parse();
    defer root.deinit();

    try std.testing.expect(root.value == .list);
    try std.testing.expectEqual(@as(usize, 3), root.value.list.items.len);
}

test "YAML parse nested map" {
    const yaml =
        \\outer:
        \\  inner: value
        \\  number: 123
    ;

    var parser = YamlParser.init(std.testing.allocator, yaml);
    defer parser.deinit();

    var root = try parser.parse();
    defer root.deinit();

    try std.testing.expect(root.value == .map);
    const outer = root.value.map.get("outer").?;
    try std.testing.expect(outer == .map);
    try std.testing.expectEqualStrings("value", outer.map.get("inner").?.string);
}
