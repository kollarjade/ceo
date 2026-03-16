const std = @import("std");

pub const TokenizerType = enum {
    bpe,
    unigram,
    word,
};

pub const TokenizerConfig = struct {
    vocab_size: usize,
    tokenizer_type: TokenizerType,
    special_tokens: []const []const u8,
    unk_token: []const u8,
    pad_token: []const u8,
    bos_token: []const u8,
    eos_token: []const u8,
};

pub const Tokenizer = struct {
    vocab: std.StringHashMap(u32),
    vocab_inv: std.ArrayList([]const u8),
    merges: std.ArrayList(Merge),
    special_tokens: std.StringHashMap(u32),
    special_token_ids: std.ArrayList(u32),
    allocator: std.mem.Allocator,

    const Self = @This();
    const file_magic: u32 = 0x544B5A31;
    const default_special_tokens = [_][]const u8{ "<pad>", "<unk>", "<bos>", "<eos>" };

    pub const Merge = struct {
        first: u32,
        second: u32,
        result: u32,
    };

    const Pair = struct {
        first: u32,
        second: u32,
    };

    const SpecialMatch = struct {
        id: u32,
        len: usize,
    };

    fn initEmpty(allocator: std.mem.Allocator) Self {
        return .{
            .vocab = std.StringHashMap(u32).init(allocator),
            .vocab_inv = std.ArrayList([]const u8).init(allocator),
            .merges = std.ArrayList(Merge).init(allocator),
            .special_tokens = std.StringHashMap(u32).init(allocator),
            .special_token_ids = std.ArrayList(u32).init(allocator),
            .allocator = allocator,
        };
    }

    fn usizeToU32(value: usize) !u32 {
        if (value > std.math.maxInt(u32)) return error.ValueTooLarge;
        return @intCast(value);
    }

    fn appendToken(self: *Self, token: []const u8, add_to_vocab_map: bool) !u32 {
        const id = try usizeToU32(self.vocab_inv.items.len);
        try self.vocab_inv.append(token);
        if (add_to_vocab_map) {
            const entry = try self.vocab.getOrPut(token);
            if (!entry.found_existing) {
                entry.value_ptr.* = id;
            }
        }
        return id;
    }

    fn addSpecialToken(self: *Self, token_text: []const u8) !void {
        const token = try self.allocator.dupe(u8, token_text);
        const id = try self.appendToken(token, true);
        try self.special_tokens.put(token, id);
        try self.special_token_ids.append(id);
    }

    fn tokenExists(self: *const Self, token: []const u8) ?u32 {
        return self.vocab.get(token);
    }

    fn comparePairLess(a: Pair, b: Pair) bool {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    }

    fn buildMergedToken(self: *Self, first: u32, second: u32) ![]u8 {
        if (first >= self.vocab_inv.items.len or second >= self.vocab_inv.items.len) {
            return error.InvalidToken;
        }
        const first_token = self.vocab_inv.items[first];
        const second_token = self.vocab_inv.items[second];
        var merged = try self.allocator.alloc(u8, first_token.len + second_token.len);
        @memcpy(merged[0..first_token.len], first_token);
        @memcpy(merged[first_token.len..], second_token);
        return merged;
    }

    fn addOrReuseMergedToken(self: *Self, token: []const u8) !u32 {
        if (self.special_tokens.get(token)) |_| {
            const duplicate = try self.allocator.dupe(u8, token);
            return try self.appendToken(duplicate, false);
        }
        if (self.tokenExists(token)) |existing_id| {
            self.allocator.free(token);
            return existing_id;
        }
        return try self.appendToken(token, true);
    }

    fn matchSpecialAt(self: *const Self, text: []const u8, start: usize) ?SpecialMatch {
        var best: ?SpecialMatch = null;
        for (self.special_token_ids.items) |id| {
            const token = self.vocab_inv.items[id];
            if (token.len == 0) continue;
            if (start + token.len > text.len) continue;
            if (!std.mem.eql(u8, text[start .. start + token.len], token)) continue;
            if (best == null or token.len > best.?.len) {
                best = .{ .id = id, .len = token.len };
            }
        }
        return best;
    }

    fn applyMergesInPlace(self: *const Self, tokens: *std.ArrayList(u32)) !void {
        for (self.merges.items) |merge| {
            var i: usize = 0;
            while (i + 1 < tokens.items.len) {
                if (tokens.items[i] == merge.first and tokens.items[i + 1] == merge.second) {
                    _ = tokens.orderedRemove(i + 1);
                    tokens.items[i] = merge.result;
                } else {
                    i += 1;
                }
            }
        }
    }

    fn encodeBytes(self: *const Self, allocator: std.mem.Allocator, bytes: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(allocator);
        errdefer tokens.deinit();
        try tokens.ensureTotalCapacity(bytes.len);
        for (bytes) |byte| {
            tokens.appendAssumeCapacity(@as(u32, byte));
        }
        try self.applyMergesInPlace(&tokens);
        return tokens.toOwnedSlice();
    }

    fn encodeWord(self: *const Self, allocator: std.mem.Allocator, word: []const u8) ![]u32 {
        return self.encodeBytes(allocator, word);
    }

    pub fn train(
        allocator: std.mem.Allocator,
        corpus_path: []const u8,
        vocab_size: usize,
        tokenizer_type: TokenizerType,
    ) !Self {
        switch (tokenizer_type) {
            .bpe => {},
            else => return error.UnsupportedTokenizerType,
        }

        const file = try std.fs.cwd().openFile(corpus_path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
        defer allocator.free(content);

        return trainFromText(allocator, content, vocab_size);
    }

    pub fn trainFromText(
        allocator: std.mem.Allocator,
        text: []const u8,
        vocab_size: usize,
    ) !Self {
        if (vocab_size < 256 + default_special_tokens.len) return error.VocabSizeTooSmall;

        var self = Self.initEmpty(allocator);
        errdefer self.deinit();

        for (0..256) |b| {
            const token = try allocator.alloc(u8, 1);
            token[0] = @intCast(b);
            _ = try self.appendToken(token, true);
        }

        for (default_special_tokens) |token_text| {
            try self.addSpecialToken(token_text);
        }

        var corpus = std.ArrayList(u32).init(allocator);
        defer corpus.deinit();
        errdefer corpus.deinit();

        try corpus.ensureTotalCapacity(text.len);
        for (text) |byte| {
            corpus.appendAssumeCapacity(@as(u32, byte));
        }

        while (self.vocab_inv.items.len < vocab_size and corpus.items.len >= 2) {
            var pair_freqs = std.AutoHashMap(Pair, usize).init(allocator);
            defer pair_freqs.deinit();

            var i: usize = 0;
            while (i + 1 < corpus.items.len) : (i += 1) {
                const pair = Pair{
                    .first = corpus.items[i],
                    .second = corpus.items[i + 1],
                };
                const entry = try pair_freqs.getOrPut(pair);
                if (entry.found_existing) {
                    entry.value_ptr.* += 1;
                } else {
                    entry.value_ptr.* = 1;
                }
            }

            var best_pair: ?Pair = null;
            var best_freq: usize = 0;

            var iter_pairs = pair_freqs.iterator();
            while (iter_pairs.next()) |entry| {
                const pair = entry.key_ptr.*;
                const freq = entry.value_ptr.*;
                if (freq > best_freq) {
                    best_freq = freq;
                    best_pair = pair;
                } else if (freq == best_freq and best_pair != null and comparePairLess(pair, best_pair.?)) {
                    best_pair = pair;
                }
            }

            if (best_pair == null or best_freq == 0) break;

            const merged_token = try self.buildMergedToken(best_pair.?.first, best_pair.?.second);
            const result_id = try self.addOrReuseMergedToken(merged_token);

            try self.merges.append(.{
                .first = best_pair.?.first,
                .second = best_pair.?.second,
                .result = result_id,
            });

            var next_corpus = std.ArrayList(u32).init(allocator);
            errdefer next_corpus.deinit();
            try next_corpus.ensureTotalCapacity(corpus.items.len);

            i = 0;
            while (i < corpus.items.len) {
                if (i + 1 < corpus.items.len and corpus.items[i] == best_pair.?.first and corpus.items[i + 1] == best_pair.?.second) {
                    try next_corpus.append(result_id);
                    i += 2;
                } else {
                    try next_corpus.append(corpus.items[i]);
                    i += 1;
                }
            }

            corpus.deinit();
            corpus = next_corpus;
        }

        return self;
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var self = Self.initEmpty(allocator);
        errdefer self.deinit();

        var buf_reader = std.io.bufferedReader(file.reader());
        var reader = buf_reader.reader();

        const magic = try reader.readInt(u32, .little);
        if (magic != file_magic) return error.InvalidTokenizerFile;

        const vocab_size_u32 = try reader.readInt(u32, .little);
        const special_count_u32 = try reader.readInt(u32, .little);
        const merge_count_u32 = try reader.readInt(u32, .little);

        const vocab_size: usize = @as(usize, vocab_size_u32);
        const special_count: usize = @as(usize, special_count_u32);
        const merge_count: usize = @as(usize, merge_count_u32);

        if (special_count > vocab_size) return error.InvalidTokenizerFile;

        for (0..vocab_size) |_| {
            const token_len_u32 = try reader.readInt(u32, .little);
            const token_len: usize = @as(usize, token_len_u32);
            const token = try allocator.alloc(u8, token_len);
            errdefer allocator.free(token);
            try reader.readNoEof(token);
            _ = try self.appendToken(token, true);
        }

        for (0..special_count) |_| {
            const id = try reader.readInt(u32, .little);
            if (id >= self.vocab_inv.items.len) return error.InvalidTokenizerFile;
            const token = self.vocab_inv.items[id];
            try self.special_tokens.put(token, id);
            try self.special_token_ids.append(id);
        }

        for (0..merge_count) |_| {
            const first = try reader.readInt(u32, .little);
            const second = try reader.readInt(u32, .little);
            const result = try reader.readInt(u32, .little);
            if (first >= self.vocab_inv.items.len or second >= self.vocab_inv.items.len or result >= self.vocab_inv.items.len) {
                return error.InvalidTokenizerFile;
            }
            try self.merges.append(.{
                .first = first,
                .second = second,
                .result = result,
            });
        }

        return self;
    }

    pub fn save(self: *const Self, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();

        var buf_writer = std.io.bufferedWriter(file.writer());
        var writer = buf_writer.writer();

        try writer.writeInt(u32, file_magic, .little);
        try writer.writeInt(u32, try usizeToU32(self.vocab_inv.items.len), .little);
        try writer.writeInt(u32, try usizeToU32(self.special_token_ids.items.len), .little);
        try writer.writeInt(u32, try usizeToU32(self.merges.items.len), .little);

        for (self.vocab_inv.items) |token| {
            try writer.writeInt(u32, try usizeToU32(token.len), .little);
            try writer.writeAll(token);
        }

        for (self.special_token_ids.items) |id| {
            try writer.writeInt(u32, id, .little);
        }

        for (self.merges.items) |merge| {
            try writer.writeInt(u32, merge.first, .little);
            try writer.writeInt(u32, merge.second, .little);
            try writer.writeInt(u32, merge.result, .little);
        }

        try buf_writer.flush();
    }

    pub fn deinit(self: *Self) void {
        for (self.vocab_inv.items) |token| {
            self.allocator.free(token);
        }
        self.vocab.deinit();
        self.vocab_inv.deinit();
        self.merges.deinit();
        self.special_tokens.deinit();
        self.special_token_ids.deinit();
    }

    pub fn encode(self: *const Self, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(allocator);
        errdefer tokens.deinit();

        var segment_start: usize = 0;
        var i: usize = 0;

        while (i < text.len) {
            if (self.matchSpecialAt(text, i)) |special| {
                if (segment_start < i) {
                    const part = try self.encodeBytes(allocator, text[segment_start..i]);
                    defer allocator.free(part);
                    try tokens.appendSlice(part);
                }
                try tokens.append(special.id);
                i += special.len;
                segment_start = i;
            } else {
                i += 1;
            }
        }

        if (segment_start < text.len) {
            const part = try self.encodeBytes(allocator, text[segment_start..]);
            defer allocator.free(part);
            try tokens.appendSlice(part);
        }

        return tokens.toOwnedSlice();
    }

    pub fn decode(self: *const Self, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        var text = std.ArrayList(u8).init(allocator);
        errdefer text.deinit();

        for (tokens) |token| {
            if (token >= self.vocab_inv.items.len) return error.InvalidToken;
            try text.appendSlice(self.vocab_inv.items[token]);
        }

        return text.toOwnedSlice();
    }

    pub fn encodeFile(self: *const Self, input_path: []const u8, output_path: []const u8) !void {
        const input_file = try std.fs.cwd().openFile(input_path, .{});
        defer input_file.close();

        const content = try input_file.readToEndAlloc(self.allocator, std.math.maxInt(usize));
        defer self.allocator.free(content);

        const tokens = try self.encode(self.allocator, content);
        defer self.allocator.free(tokens);

        const output_file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
        defer output_file.close();

        var buf_writer = std.io.bufferedWriter(output_file.writer());
        var writer = buf_writer.writer();

        try writer.writeInt(u64, tokens.len, .little);
        for (tokens) |token| {
            try writer.writeInt(u32, token, .little);
        }

        try buf_writer.flush();
    }

    pub fn vocabSize(self: *const Self) usize {
        return self.vocab_inv.items.len;
    }

    pub fn getTokenId(self: *const Self, token: []const u8) ?u32 {
        return self.vocab.get(token);
    }

    pub fn getToken(self: *const Self, id: u32) ?[]const u8 {
        if (id >= self.vocab_inv.items.len) return null;
        return self.vocab_inv.items[id];
    }
};

test "Tokenizer train and encode decode exact roundtrip" {
    const allocator = std.testing.allocator;
    const text = "hello world hello there\nwith spaces\tand symbols \\ <bos> but literal bytes";
    var tokenizer = try Tokenizer.trainFromText(allocator, text, 320);
    defer tokenizer.deinit();

    const tokens = try tokenizer.encode(allocator, text);
    defer allocator.free(tokens);

    try std.testing.expect(tokens.len > 0);

    const decoded = try tokenizer.decode(allocator, tokens);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(text, decoded);
}

test "Tokenizer special tokens remain atomic" {
    const allocator = std.testing.allocator;
    const text = "abc<bos>def<eos>";
    var tokenizer = try Tokenizer.trainFromText(allocator, "abcdef", 300);
    defer tokenizer.deinit();

    const tokens = try tokenizer.encode(allocator, text);
    defer allocator.free(tokens);

    try std.testing.expect(tokens.len >= 2);

    const bos_id = tokenizer.getTokenId("<bos>") orelse return error.TestUnexpectedResult;
    const eos_id = tokenizer.getTokenId("<eos>") orelse return error.TestUnexpectedResult;

    var saw_bos = false;
    var saw_eos = false;
    for (tokens) |token| {
        if (token == bos_id) saw_bos = true;
        if (token == eos_id) saw_eos = true;
    }

    try std.testing.expect(saw_bos);
    try std.testing.expect(saw_eos);

    const decoded = try tokenizer.decode(allocator, tokens);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(text, decoded);
}
