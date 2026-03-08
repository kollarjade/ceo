const std = @import("std");

pub const TokenizerType = enum {
    bpe,
    unigram,
    word,
};

/// Tokenizer configuration
pub const TokenizerConfig = struct {
    vocab_size: usize,
    tokenizer_type: TokenizerType,
    special_tokens: []const []const u8,
    unk_token: []const u8,
    pad_token: []const u8,
    bos_token: []const u8,
    eos_token: []const u8,
};

/// BPE Tokenizer implementation
pub const Tokenizer = struct {
    vocab: std.StringHashMap(u32),
    vocab_inv: std.ArrayList([]const u8),
    merges: std.ArrayList(Merge),
    special_tokens: std.StringHashMap(u32),
    byte_encoder: [256]u32,
    byte_decoder: std.ArrayList(u8),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub const Merge = struct {
        first: u32,
        second: u32,
        result: u32,
    };

    /// Train a new tokenizer
    pub fn train(
        allocator: std.mem.Allocator,
        corpus_path: []const u8,
        vocab_size: usize,
        tokenizer_type: TokenizerType,
    ) !Self {
        _ = tokenizer_type;

        // Read corpus
        const file = try std.fs.cwd().openFile(corpus_path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
        defer allocator.free(content);

        return trainFromText(allocator, content, vocab_size);
    }

    /// Train tokenizer from text in memory
    pub fn trainFromText(
        allocator: std.mem.Allocator,
        text: []const u8,
        vocab_size: usize,
    ) !Self {
        var self = Self{
            .vocab = std.StringHashMap(u32).init(allocator),
            .vocab_inv = std.ArrayList([]const u8).init(allocator),
            .merges = std.ArrayList(Merge).init(allocator),
            .special_tokens = std.StringHashMap(u32).init(allocator),
            .byte_encoder = undefined,
            .byte_decoder = std.ArrayList(u8).init(allocator),
            .allocator = allocator,
        };
        errdefer self.deinit();

        // Initialize byte encoder/decoder
        // Map bytes to printable characters for BPE
        var n: u32 = 0;
        for (0..256) |b| {
            const ch: u8 = if (b >= 33 and b <= 126 and b != 39 and b != 92)
                @intCast(b)
            else if (b == 39)
                39
            else
                blk: {
                    n += 1;
                    break :blk @intCast(256 + n - 1);
                };

            self.byte_encoder[b] = ch;
        }

        // Initialize with byte-level tokens
        for (0..256) |b| {
            const token = try allocator.dupe(u8, &[_]u8{@intCast(b)});
            const id: u32 = @intCast(self.vocab_inv.items.len);
            try self.vocab.put(token, id);
            try self.vocab_inv.append(token);
        }

        // Add special tokens
        const special = [_][]const u8{ "<pad>", "<unk>", "<bos>", "<eos>" };
        for (special) |tok| {
            const id: u32 = @intCast(self.vocab_inv.items.len);
            const token = try allocator.dupe(u8, tok);
            try self.vocab.put(token, id);
            try self.vocab_inv.append(token);
            try self.special_tokens.put(tok, id);
        }

        // Pre-tokenize text into words
        var words = std.ArrayList([]const u8).init(allocator);
        defer words.deinit();

        var word_frequencies = std.StringHashMap(usize).init(allocator);
        defer word_frequencies.deinit();

        // Split by whitespace
        var iter = std.mem.split(u8, text, " \n\t\r");
        while (iter.next()) |word| {
            if (word.len == 0) continue;

            const owned = try allocator.dupe(u8, word);
            try words.append(owned);

            const entry = try word_frequencies.getOrPut(owned);
            if (entry.found_existing) {
                entry.value_ptr.* += 1;
            } else {
                entry.value_ptr.* = 1;
            }
        }

        // Build initial vocabulary from character frequencies
        var char_freqs = std.AutoHashMap(u8, usize).init(allocator);
        defer char_freqs.deinit();

        for (text) |ch| {
            const entry = try char_freqs.getOrPut(ch);
            if (entry.found_existing) {
                entry.value_ptr.* += 1;
            } else {
                entry.value_ptr.* = 1;
            }
        }

        // BPE training iterations
        var num_merges = vocab_size - 256 - special.len;
        var current_vocab_size = self.vocab_inv.items.len;

        while (current_vocab_size < vocab_size and num_merges > 0) {
            // Find most frequent pair
            var best_pair: ?struct { u32, u32 } = null;
            var best_freq: usize = 0;

            var pair_freqs = std.AutoHashMap(struct { u32, u32 }, usize).init(allocator);
            defer pair_freqs.deinit();

            for (words.items) |word| {
                const tokens = try self.encodeWord(allocator, word);
                defer allocator.free(tokens);

                if (tokens.len < 2) continue;

                for (0..tokens.len - 1) |i| {
                    const pair = .{ tokens[i], tokens[i + 1] };
                    const entry = try pair_freqs.getOrPut(pair);
                    if (entry.found_existing) {
                        entry.value_ptr.* += 1;
                    } else {
                        entry.value_ptr.* = 1;
                    }
                }
            }

            var iter_pairs = pair_freqs.iterator();
            while (iter_pairs.next()) |entry| {
                if (entry.value_ptr.* > best_freq) {
                    best_freq = entry.value_ptr.*;
                    best_pair = entry.key_ptr.*;
                }
            }

            if (best_pair == null) break;

            // Merge the best pair
            const first = best_pair.?.@"0";
            const second = best_pair.?.@"1";

            // Create new token
            const first_token = self.vocab_inv.items[first];
            const second_token = self.vocab_inv.items[second];

            var new_token = std.ArrayList(u8).init(allocator);
            try new_token.appendSlice(first_token);
            try new_token.appendSlice(second_token);

            const new_token_slice = try new_token.toOwnedSlice();
            const new_id: u32 = @intCast(self.vocab_inv.items.len);

            try self.vocab.put(new_token_slice, new_id);
            try self.vocab_inv.append(new_token_slice);

            // Record merge
            try self.merges.append(.{
                .first = first,
                .second = second,
                .result = new_id,
            });

            current_vocab_size += 1;
            num_merges -= 1;
        }

        return self;
    }

    /// Load tokenizer from file
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var self = Self{
            .vocab = std.StringHashMap(u32).init(allocator),
            .vocab_inv = std.ArrayList([]const u8).init(allocator),
            .merges = std.ArrayList(Merge).init(allocator),
            .special_tokens = std.StringHashMap(u32).init(allocator),
            .byte_encoder = undefined,
            .byte_decoder = std.ArrayList(u8).init(allocator),
            .allocator = allocator,
        };
        errdefer self.deinit();

        var buf_reader = std.io.bufferedReader(file.reader());
        var reader = buf_reader.reader();

        // Read header
        const magic = try reader.readInt(u32, .little);
        if (magic != 0xEFLA0001) return error.InvalidTokenizerFile;

        const vocab_size = try reader.readInt(u32, .little);

        // Read vocabulary
        for (0..vocab_size) |_| {
            const token_len = try reader.readInt(u32, .little);
            const token = try allocator.alloc(u8, token_len);
            try reader.readNoEof(token);

            const id: u32 = @intCast(self.vocab_inv.items.len);
            try self.vocab.put(token, id);
            try self.vocab_inv.append(token);
        }

        // Read merges
        const num_merges = try reader.readInt(u32, .little);
        for (0..num_merges) |_| {
            const first = try reader.readInt(u32, .little);
            const second = try reader.readInt(u32, .little);
            const result = try reader.readInt(u32, .little);
            try self.merges.append(.{ .first = first, .second = second, .result = result });
        }

        return self;
    }

    /// Save tokenizer to file
    pub fn save(self: *Self, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        var buf_writer = std.io.bufferedWriter(file.writer());
        var writer = buf_writer.writer();

        // Write header
        try writer.writeInt(u32, 0xEFLA0001, .little);
        try writer.writeInt(u32, @intCast(self.vocab_inv.items.len), .little);

        // Write vocabulary
        for (self.vocab_inv.items) |token| {
            try writer.writeInt(u32, @intCast(token.len), .little);
            try writer.writeAll(token);
        }

        // Write merges
        try writer.writeInt(u32, @intCast(self.merges.items.len), .little);
        for (self.merges.items) |merge| {
            try writer.writeInt(u32, merge.first, .little);
            try writer.writeInt(u32, merge.second, .little);
            try writer.writeInt(u32, merge.result, .little);
        }

        try buf_writer.flush();
    }

    pub fn deinit(self: *Self) void {
        var iter = self.vocab.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit();

        for (self.vocab_inv.items) |token| {
            self.allocator.free(token);
        }
        self.vocab_inv.deinit();
        self.merges.deinit();
        self.special_tokens.deinit();
        self.byte_decoder.deinit();
    }

    /// Encode text to tokens
    pub fn encode(self: *Self, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(allocator);
        defer {
            if (tokens.items.len > 0) {
                // Don't deinit if we're returning the items
            }
        }

        // Check for special tokens first
        if (self.special_tokens.get(text)) |id| {
            try tokens.append(id);
            return tokens.toOwnedSlice();
        }

        // Encode each word
        var iter = std.mem.split(u8, text, " \n\t\r");
        while (iter.next()) |word| {
            if (word.len == 0) continue;

            const word_tokens = try self.encodeWord(allocator, word);
            defer allocator.free(word_tokens);

            try tokens.appendSlice(word_tokens);
        }

        return tokens.toOwnedSlice();
    }

    fn encodeWord(self: *Self, allocator: std.mem.Allocator, word: []const u8) ![]u32 {
        // Convert to byte tokens
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        for (word) |byte| {
            try tokens.append(self.byte_encoder[byte]);
        }

        // Apply merges
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

        return tokens.toOwnedSlice();
    }

    /// Decode tokens to text
    pub fn decode(self: *Self, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        var text = std.ArrayList(u8).init(allocator);
        defer {
            if (text.items.len > 0) {}
        }

        for (tokens) |token| {
            if (token >= self.vocab_inv.items.len) {
                return error.InvalidToken;
            }

            const token_str = self.vocab_inv.items[token];
            try text.appendSlice(token_str);
        }

        return text.toOwnedSlice();
    }

    /// Encode a file
    pub fn encodeFile(self: *Self, input_path: []const u8, output_path: []const u8) !void {
        const input_file = try std.fs.cwd().openFile(input_path, .{});
        defer input_file.close();

        const content = try input_file.readToEndAlloc(self.allocator, std.math.maxInt(usize));
        defer self.allocator.free(content);

        const tokens = try self.encode(self.allocator, content);
        defer self.allocator.free(tokens);

        const output_file = try std.fs.cwd().createFile(output_path, .{});
        defer output_file.close();

        var buf_writer = std.io.bufferedWriter(output_file.writer());
        var writer = buf_writer.writer();

        // Write as binary token sequence
        try writer.writeInt(u64, tokens.len, .little);
        for (tokens) |t| {
            try writer.writeInt(u32, t, .little);
        }

        try buf_writer.flush();
    }

    /// Get vocabulary size
    pub fn vocabSize(self: *Self) usize {
        return self.vocab_inv.items.len;
    }

    /// Get token ID
    pub fn getTokenId(self: *Self, token: []const u8) ?u32 {
        return self.vocab.get(token);
    }

    /// Get token string
    pub fn getToken(self: *Self, id: u32) ?[]const u8 {
        if (id >= self.vocab_inv.items.len) return null;
        return self.vocab_inv.items[id];
    }
};

test "Tokenizer train and encode/decode" {
    const allocator = std.testing.allocator;

    const text = "hello world hello there";

    var tokenizer = try Tokenizer.trainFromText(allocator, text, 300);
    defer tokenizer.deinit();

    const tokens = try tokenizer.encode(allocator, "hello");
    defer allocator.free(tokens);

    try std.testing.expect(tokens.len > 0);

    const decoded = try tokenizer.decode(allocator, tokens);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings("hello", decoded);
}
