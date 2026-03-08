const std = @import("std");
const tokenizer_mod = @import("tokenizer.zig");

pub const Tokenizer = tokenizer_mod.Tokenizer;

/// Dataset shard metadata
pub const ShardMetadata = struct {
    path: []const u8,
    num_tokens: usize,
    checksum: [32]u8, // SHA-256
};

/// Binary token dataset format
pub const BinaryDataset = struct {
    file: std.fs.File,
    mmap: ?[]align(std.mem.page_size) const u8,
    num_tokens: usize,
    shard_metadata: []ShardMetadata,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Open a binary token dataset
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        errdefer file.close();

        // Read header
        var buf_reader = std.io.bufferedReader(file.reader());
        var reader = buf_reader.reader();

        const magic = try reader.readInt(u32, .little);
        if (magic != 0xEFLAD001) return error.InvalidDatasetFile;

        const version = try reader.readInt(u32, .little);
        if (version != 1) return error.UnsupportedVersion;

        const num_tokens = try reader.readInt(u64, .little);
        const num_shards = try reader.readInt(u32, .little);

        var shard_metadata = try allocator.alloc(ShardMetadata, num_shards);
        errdefer allocator.free(shard_metadata);

        for (0..num_shards) |i| {
            const path_len = try reader.readInt(u32, .little);
            const shard_path = try allocator.alloc(u8, path_len);
            try reader.readNoEof(shard_path);

            const shard_num_tokens = try reader.readInt(u64, .little);
            var checksum: [32]u8 = undefined;
            try reader.readNoEof(&checksum);

            shard_metadata[i] = .{
                .path = shard_path,
                .num_tokens = shard_num_tokens,
                .checksum = checksum,
            };
        }

        // Memory map the file
        const file_size = try file.getEndPos();
        const mmap = try std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );

        return .{
            .file = file,
            .mmap = mmap,
            .num_tokens = num_tokens,
            .shard_metadata = shard_metadata,
            .allocator = allocator,
        };
    }

    pub fn close(self: *Self) void {
        if (self.mmap) |m| {
            std.posix.munmap(m);
        }
        self.file.close();

        for (self.shard_metadata) |meta| {
            self.allocator.free(meta.path);
        }
        self.allocator.free(self.shard_metadata);
    }

    /// Get tokens starting at offset
    pub fn getTokens(self: *Self, offset: usize, len: usize) ![]const u32 {
        if (offset + len > self.num_tokens) {
            return error.OutOfRange;
        }

        // Token data starts after header
        const header_size = 4 + 4 + 8 + 4; // magic + version + num_tokens + num_shards
        const shard_data_size = self.shard_metadata.len * (4 + 100 + 8 + 32); // rough estimate

        const token_offset = header_size + shard_data_size + offset * 4;
        const ptr = @as([*]const u32, @ptrCast(self.mmap.?.ptr + token_offset));

        return ptr[0..len];
    }
};

/// Dataset shard writer
pub const ShardWriter = struct {
    file: std.fs.File,
    writer: std.io.BufferedWriter(4096, std.fs.File.Writer),
    num_tokens: usize,
    hasher: std.crypto.hash.sha2.Sha256,

    const Self = @This();

    pub fn init(path: []const u8) !Self {
        const file = try std.fs.cwd().createFile(path, .{});
        errdefer file.close();

        var self = Self{
            .file = file,
            .writer = std.io.bufferedWriter(file.writer()),
            .num_tokens = 0,
            .hasher = std.crypto.hash.sha2.Sha256.init(.{}),
        };

        // Write header placeholder
        try self.writer.writer().writeByteNTimes(0, 64);

        return self;
    }

    pub fn writeTokens(self: *Self, tokens: []const u32) !void {
        for (tokens) |t| {
            try self.writer.writer().writeInt(u32, t, .little);
            self.hasher.update(std.mem.asBytes(&t));
        }
        self.num_tokens += tokens.len;
    }

    pub fn finish(self: *Self) !ShardMetadata {
        try self.writer.flush();

        var checksum: [32]u8 = undefined;
        self.hasher.final(&checksum);

        return .{
            .path = "", // Will be filled by caller
            .num_tokens = self.num_tokens,
            .checksum = checksum,
        };
    }

    pub fn close(self: *Self) void {
        self.file.close();
    }
};

/// Dataset builder for creating sharded binary datasets
pub const DatasetBuilder = struct {
    output_dir: []const u8,
    shard_size: usize, // Tokens per shard
    current_shard: ?ShardWriter,
    current_shard_tokens: usize,
    shard_index: usize,
    shards: std.ArrayList(ShardMetadata),
    total_tokens: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, output_dir: []const u8, shard_size: usize) !Self {
        try std.fs.cwd().makePath(output_dir);

        return .{
            .output_dir = output_dir,
            .shard_size = shard_size,
            .current_shard = null,
            .current_shard_tokens = 0,
            .shard_index = 0,
            .shards = std.ArrayList(ShardMetadata).init(allocator),
            .total_tokens = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.current_shard) |*s| s.close();
        for (self.shards.items) |*s| {
            self.allocator.free(s.path);
        }
        self.shards.deinit();
    }

    /// Add tokens to the dataset
    pub fn addTokens(self: *Self, tokens: []const u32) !void {
        var remaining = tokens;

        while (remaining.len > 0) {
            // Start new shard if needed
            if (self.current_shard == null) {
                const shard_path = try std.fmt.allocPrint(
                    self.allocator,
                    "{s}/shard_{d:0>6}.bin",
                    .{ self.output_dir, self.shard_index },
                );

                self.current_shard = try ShardWriter.init(shard_path);
                self.current_shard_tokens = 0;
                self.shard_index += 1;
            }

            // Calculate how many tokens to write
            const space = self.shard_size - self.current_shard_tokens;
            const to_write = @min(remaining.len, space);

            // Write tokens
            try self.current_shard.?.writeTokens(remaining[0..to_write]);
            self.current_shard_tokens += to_write;
            self.total_tokens += to_write;

            remaining = remaining[to_write..];

            // Finalize shard if full
            if (self.current_shard_tokens >= self.shard_size) {
                var meta = try self.current_shard.?.finish();
                const shard_path = try std.fmt.allocPrint(
                    self.allocator,
                    "{s}/shard_{d:0>6}.bin",
                    .{ self.output_dir, self.shard_index - 1 },
                );
                meta.path = shard_path;
                try self.shards.append(meta);

                self.current_shard.?.close();
                self.current_shard = null;
            }
        }
    }

    /// Finalize the dataset
    pub fn finalize(self: *Self, name: []const u8) !void {
        // Finalize current shard
        if (self.current_shard) |*s| {
            var meta = try s.finish();
            const shard_path = try std.fmt.allocPrint(
                self.allocator,
                "{s}/shard_{d:0>6}.bin",
                .{ self.output_dir, self.shard_index - 1 },
            );
            meta.path = shard_path;
            try self.shards.append(meta);
            s.close();
            self.current_shard = null;
        }

        // Write index file
        const index_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}.index", .{ self.output_dir, name });
        defer self.allocator.free(index_path);

        const file = try std.fs.cwd().createFile(index_path, .{});
        defer file.close();

        var buf_writer = std.io.bufferedWriter(file.writer());
        var writer = buf_writer.writer();

        // Header
        try writer.writeInt(u32, 0xEFLAD001, .little); // Magic
        try writer.writeInt(u32, 1, .little); // Version
        try writer.writeInt(u64, self.total_tokens, .little);
        try writer.writeInt(u32, @intCast(self.shards.items.len), .little);

        // Shard metadata
        for (self.shards.items) |meta| {
            try writer.writeInt(u32, @intCast(meta.path.len), .little);
            try writer.writeAll(meta.path);
            try writer.writeInt(u64, meta.num_tokens, .little);
            try writer.writeAll(&meta.checksum);
        }

        try buf_writer.flush();
    }
};

/// Data loader for training
pub const DataLoader = struct {
    dataset: *BinaryDataset,
    batch_size: usize,
    seq_len: usize,
    drop_last: bool,
    shuffle: bool,
    rng: std.Random,
    position: usize,
    indices: []usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        dataset: *BinaryDataset,
        batch_size: usize,
        seq_len: usize,
        shuffle: bool,
        seed: u64,
    ) !Self {
        // Create index array
        const num_samples = dataset.num_tokens / seq_len;
        var indices = try allocator.alloc(usize, num_samples);
        for (0..num_samples) |i| {
            indices[i] = i * seq_len;
        }

        var prng = std.Random.DefaultPrng.init(seed);
        var rng = prng.random();

        // Shuffle if requested
        if (shuffle) {
            rng.shuffle(usize, indices);
        }

        return .{
            .dataset = dataset,
            .batch_size = batch_size,
            .seq_len = seq_len,
            .drop_last = true,
            .shuffle = shuffle,
            .rng = rng,
            .position = 0,
            .indices = indices,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.indices);
    }

    /// Get next batch
    pub fn next(self: *Self) !?Batch {
        if (self.position + self.batch_size > self.indices.len) {
            return null;
        }

        // Allocate batch tensors
        var input_tokens = try self.allocator.alloc(u32, self.batch_size * self.seq_len);
        errdefer self.allocator.free(input_tokens);

        var target_tokens = try self.allocator.alloc(u32, self.batch_size * self.seq_len);
        errdefer self.allocator.free(target_tokens);

        // Fill batch
        for (0..self.batch_size) |b| {
            const idx = self.indices[self.position + b];
            const tokens = try self.dataset.getTokens(idx, self.seq_len + 1);

            // Input: tokens[0:seq_len]
            // Target: tokens[1:seq_len+1]
            for (0..self.seq_len) |t| {
                input_tokens[b * self.seq_len + t] = tokens[t];
                target_tokens[b * self.seq_len + t] = tokens[t + 1];
            }
        }

        self.position += self.batch_size;

        return .{
            .input = input_tokens,
            .target = target_tokens,
            .batch_size = self.batch_size,
            .seq_len = self.seq_len,
        };
    }

    /// Reset to beginning
    pub fn reset(self: *Self) void {
        self.position = 0;
        if (self.shuffle) {
            self.rng.shuffle(usize, self.indices);
        }
    }

    /// Get number of batches
    pub fn numBatches(self: *Self) usize {
        return self.indices.len / self.batch_size;
    }
};

/// Training batch
pub const Batch = struct {
    input: []u32,
    target: []u32,
    batch_size: usize,
    seq_len: usize,

    pub fn deinit(self: *Batch, allocator: std.mem.Allocator) void {
        allocator.free(self.input);
        allocator.free(self.target);
    }
};

/// Packed sequence for variable-length training
pub const PackedSequence = struct {
    tokens: []u32,
    positions: []usize,
    sequence_ids: []usize,
    cumulative_lengths: []usize,
    num_sequences: usize,

    pub fn deinit(self: *PackedSequence, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.positions);
        allocator.free(self.sequence_ids);
        allocator.free(self.cumulative_lengths);
    }
};

/// Pack multiple sequences into a fixed-length batch
pub fn packSequences(
    allocator: std.mem.Allocator,
    sequences: []const []const u32,
    max_length: usize,
) !PackedSequence {
    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();

    var positions = std.ArrayList(usize).init(allocator);
    defer positions.deinit();

    var sequence_ids = std.ArrayList(usize).init(allocator);
    defer sequence_ids.deinit();

    var cumulative_lengths = std.ArrayList(usize).init(allocator);
    defer cumulative_lengths.deinit();

    var current_length: usize = 0;
    var seq_id: usize = 0;

    for (sequences) |seq| {
        if (current_length + seq.len > max_length) {
            // Can't fit this sequence
            continue;
        }

        for (seq, 0..) |t, pos| {
            try tokens.append(t);
            try positions.append(pos);
            try sequence_ids.append(seq_id);
        }

        current_length += seq.len;
        try cumulative_lengths.append(current_length);
        seq_id += 1;
    }

    return .{
        .tokens = try tokens.toOwnedSlice(),
        .positions = try positions.toOwnedSlice(),
        .sequence_ids = try sequence_ids.toOwnedSlice(),
        .cumulative_lengths = try cumulative_lengths.toOwnedSlice(),
        .num_sequences = seq_id,
    };
}

test "packSequences" {
    const allocator = std.testing.allocator;

    const seq1 = [_]u32{ 1, 2, 3 };
    const seq2 = [_]u32{ 4, 5 };
    const seq3 = [_]u32{ 6, 7, 8, 9 };

    const sequences = [_][]const u32{ &seq1, &seq2, &seq3 };

    var packed = try packSequences(allocator, &sequences, 20);
    defer packed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 9), packed.tokens.len);
    try std.testing.expectEqual(@as(usize, 3), packed.num_sequences);
}
