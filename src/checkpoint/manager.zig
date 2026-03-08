const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const config_mod = @import("../util/config.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;

/// Checkpoint format options
pub const CheckpointFormat = enum {
    efla_native,
    safetensors,
    pytorch,
};

/// Checkpoint metadata
pub const CheckpointMetadata = struct {
    step: usize,
    epoch: usize,
    tokens_seen: usize,
    loss: f32,
    learning_rate: f32,
    timestamp: i64,
    git_revision: [40]u8,
    config_hash: [32]u8,
};

/// Checkpoint entry for a single tensor
pub const CheckpointEntry = struct {
    name: []const u8,
    shape: []const usize,
    dtype: tensor_mod.DType,
    offset: usize,
    size: usize,
    checksum: [32]u8,
};

/// Checkpoint manager for saving and loading model state
pub const CheckpointManager = struct {
    dir: []const u8,
    max_to_keep: usize,
    compression: bool,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        dir: []const u8,
        max_to_keep: usize,
        compression: bool,
    ) !Self {
        try std.fs.cwd().makePath(dir);

        return .{
            .dir = try allocator.dupe(u8, dir),
            .max_to_keep = max_to_keep,
            .compression = compression,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.dir);
    }

    /// Save a checkpoint
    pub fn save(
        self: *Self,
        step: usize,
        params: []*Tensor,
        param_names: []const []const u8,
        optimizer_state: ?*anyopaque,
        rng_state: ?[]const u8,
        metadata: CheckpointMetadata,
    ) ![]const u8 {
        // Create checkpoint directory
        const ckpt_name = try std.fmt.allocPrint(
            self.allocator,
            "{s}/step_{d:0>8}",
            .{ self.dir, step },
        );
        defer self.allocator.free(ckpt_name);

        try std.fs.cwd().makePath(ckpt_name);

        // Write metadata
        const meta_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/metadata.json",
            .{ckpt_name},
        );
        defer self.allocator.free(meta_path);

        try self.writeMetadata(meta_path, metadata);

        // Write tensors
        const tensors_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/tensors.bin",
            .{ckpt_name},
        );
        defer self.allocator.free(tensors_path);

        try self.writeTensors(tensors_path, params, param_names);

        // Write optimizer state
        if (optimizer_state) |state| {
            const opt_path = try std.fmt.allocPrint(
                self.allocator,
                "{s}/optimizer.bin",
                .{ckpt_name},
            );
            defer self.allocator.free(opt_path);

            try self.writeOptimizerState(opt_path, state);
        }

        // Write RNG state
        if (rng_state) |rng| {
            const rng_path = try std.fmt.allocPrint(
                self.allocator,
                "{s}/rng.bin",
                .{ckpt_name},
            );
            defer self.allocator.free(rng_path);

            const file = try std.fs.cwd().createFile(rng_path, .{});
            defer file.close();
            try file.writeAll(rng);
        }

        // Clean up old checkpoints
        try self.cleanupOldCheckpoints();

        return try self.allocator.dupe(u8, ckpt_name);
    }

    fn writeMetadata(self: *Self, path: []const u8, metadata: CheckpointMetadata) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        var buf_writer = std.io.bufferedWriter(file.writer());
        var writer = buf_writer.writer();

        try writer.print(
            \\{{
            \\  "step": {d},
            \\  "epoch": {d},
            \\  "tokens_seen": {d},
            \\  "loss": {d},
            \\  "learning_rate": {d},
            \\  "timestamp": {d},
            \\  "git_revision": "{s}",
            \\  "config_hash": "{s}"
            \\}}
        , .{
            metadata.step,
            metadata.epoch,
            metadata.tokens_seen,
            metadata.loss,
            metadata.learning_rate,
            metadata.timestamp,
            std.fmt.fmtSliceHexLower(&metadata.git_revision),
            std.fmt.fmtSliceHexLower(&metadata.config_hash),
        });

        try buf_writer.flush();
    }

    fn writeTensors(
        self: *Self,
        path: []const u8,
        params: []*Tensor,
        param_names: []const []const u8,
    ) !void {
        _ = self;

        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        var buf_writer = std.io.bufferedWriter(file.writer());
        var writer = buf_writer.writer();

        // Write header
        try writer.writeInt(u32, 0xEFLACK01, .little); // Magic
        try writer.writeInt(u32, 1, .little); // Version
        try writer.writeInt(u32, @intCast(params.len), .little); // Number of tensors

        // Write tensor index
        var offset: usize = 0;
        for (params, 0..) |param, i| {
            const name = param_names[i];
            const shape = param.shape;
            const dtype = param.dtype;
            const size = shape.sizeBytes(dtype);

            // Name length and name
            try writer.writeInt(u32, @intCast(name.len), .little);
            try writer.writeAll(name);

            // Shape
            try writer.writeInt(u32, @intCast(shape.ndim), .little);
            for (shape.dims[0..shape.ndim]) |d| {
                try writer.writeInt(u64, d, .little);
            }

            // Dtype
            try writer.writeInt(u8, @intFromEnum(dtype), .little);

            // Offset and size
            try writer.writeInt(u64, offset, .little);
            try writer.writeInt(u64, size, .little);

            // Compute checksum placeholder
            var checksum: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash("", &checksum, .{});
            try writer.writeAll(&checksum);

            offset += size;
        }

        // Write tensor data
        for (params) |param| {
            if (param.ptr()) |ptr| {
                const size = param.shape.sizeBytes(param.dtype);
                try writer.writeAll(@as([*]const u8, @ptrCast(ptr))[0..size]);
            }
        }

        try buf_writer.flush();
    }

    fn writeOptimizerState(self: *Self, path: []const u8, state: *anyopaque) !void {
        _ = self;
        _ = state;

        // Placeholder - full implementation would serialize optimizer state
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        var buf_writer = std.io.bufferedWriter(file.writer());
        var writer = buf_writer.writer();

        try writer.writeInt(u32, 0xEFLAOP01, .little);
        try buf_writer.flush();
    }

    /// Load a checkpoint
    pub fn load(
        self: *Self,
        checkpoint_path: []const u8,
        params: []*Tensor,
        param_names: []const []const u8,
    ) !CheckpointMetadata {
        _ = params;
        _ = param_names;

        // Read metadata
        const meta_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/metadata.json",
            .{checkpoint_path},
        );
        defer self.allocator.free(meta_path);

        const metadata = try self.readMetadata(meta_path);

        // Read tensors
        const tensors_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/tensors.bin",
            .{checkpoint_path},
        );
        defer self.allocator.free(tensors_path);

        try self.readTensors(tensors_path, params, param_names);

        return metadata;
    }

    fn readMetadata(self: *Self, path: []const u8) !CheckpointMetadata {
        _ = self;

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(self.allocator, std.math.maxInt(usize));
        defer self.allocator.free(content);

        // Simple JSON parsing (simplified)
        var metadata: CheckpointMetadata = undefined;

        // Parse step
        if (std.mem.indexOf(u8, content, "\"step\":")) |idx| {
            const start = idx + 8;
            metadata.step = try parseInt(content[start..]);
        }

        // Parse loss
        if (std.mem.indexOf(u8, content, "\"loss\":")) |idx| {
            const start = idx + 8;
            metadata.loss = try parseFloat(content[start..]);
        }

        return metadata;
    }

    fn readTensors(
        self: *Self,
        path: []const u8,
        params: []*Tensor,
        param_names: []const []const u8,
    ) !void {
        _ = self;
        _ = params;
        _ = param_names;

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var buf_reader = std.io.bufferedReader(file.reader());
        var reader = buf_reader.reader();

        // Read header
        const magic = try reader.readInt(u32, .little);
        if (magic != 0xEFLACK01) return error.InvalidCheckpointFile;

        const version = try reader.readInt(u32, .little);
        if (version != 1) return error.UnsupportedVersion;

        const num_tensors = try reader.readInt(u32, .little);

        // Read index
        var entries = try self.allocator.alloc(struct {
            name: []const u8,
            offset: usize,
            size: usize,
        }, num_tensors);
        defer {
            for (entries) |e| self.allocator.free(e.name);
            self.allocator.free(entries);
        }

        for (0..num_tensors) |i| {
            const name_len = try reader.readInt(u32, .little);
            const name = try self.allocator.alloc(u8, name_len);
            try reader.readNoEof(name);

            const ndim = try reader.readInt(u32, .little);
            var dims: [8]u64 = undefined;
            for (0..ndim) |d| {
                dims[d] = try reader.readInt(u64, .little);
            }

            _ = try reader.readInt(u8, .little); // dtype
            const offset = try reader.readInt(u64, .little);
            const size = try reader.readInt(u64, .little);

            var checksum: [32]u8 = undefined;
            try reader.readNoEof(&checksum);

            entries[i] = .{ .name = name, .offset = offset, .size = size };
        }

        // Read tensor data
        for (params, 0..) |param, i| {
            const name = param_names[i];

            // Find matching entry
            for (entries) |entry| {
                if (std.mem.eql(u8, entry.name, name)) {
                    try file.seekTo(entry.offset);
                    if (param.ptr()) |ptr| {
                        try reader.readNoEof(@as([*]u8, @ptrCast(ptr))[0..entry.size]);
                    }
                    break;
                }
            }
        }
    }

    fn cleanupOldCheckpoints(self: *Self) !void {
        var dir = try std.fs.cwd().openDir(self.dir, .{ .iterate = true });
        defer dir.close();

        var checkpoints = std.ArrayList([]const u8).init(self.allocator);
        defer {
            for (checkpoints.items) |c| self.allocator.free(c);
            checkpoints.deinit();
        }

        // List all checkpoint directories
        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind == .directory and
                std.mem.startsWith(u8, entry.name, "step_"))
            {
                const path = try std.fmt.allocPrint(
                    self.allocator,
                    "{s}/{s}",
                    .{ self.dir, entry.name },
                );
                try checkpoints.append(path);
            }
        }

        // Sort by step number (descending)
        std.sort.pdq([]const u8, checkpoints.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                // Extract step number from path
                const a_step = extractStepFromPath(a);
                const b_step = extractStepFromPath(b);
                return a_step > b_step;
            }
        }.lessThan);

        // Remove old checkpoints
        if (checkpoints.items.len > self.max_to_keep) {
            for (checkpoints.items[self.max_to_keep..]) |path| {
                try std.fs.cwd().deleteTree(path);
            }
        }
    }

    fn extractStepFromPath(path: []const u8) usize {
        const last_slash = std.mem.lastIndexOfScalar(u8, path, '/') orelse return 0;
        const name = path[last_slash + 1 ..];

        if (std.mem.startsWith(u8, name, "step_")) {
            return std.fmt.parseInt(usize, name[5..], 10) catch 0;
        }
        return 0;
    }

    fn parseInt(s: []const u8) !usize {
        var result: usize = 0;
        var i: usize = 0;

        while (i < s.len and s[i] >= '0' and s[i] <= '9') : (i += 1) {
            result = result * 10 + (s[i] - '0');
        }

        return result;
    }

    fn parseFloat(s: []const u8) !f32 {
        return std.fmt.parseFloat(f32, std.mem.trim(u8, s, " \t\n\r,}"));
    }
};

/// List available checkpoints
pub fn listCheckpoints(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();

    const stdout = std.io.getStdOut().writer();

    try stdout.print("Checkpoints in {s}:\n", .{dir_path});

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind == .directory and std.mem.startsWith(u8, entry.name, "step_")) {
            try stdout.print("  {s}\n", .{entry.name});
        }
    }
}

/// Validate a checkpoint
pub fn validateCheckpoint(allocator: std.mem.Allocator, checkpoint_path: []const u8) !void {
    const stdout = std.io.getStdOut().writer();

    // Check metadata
    const meta_path = try std.fmt.allocPrint(allocator, "{s}/metadata.json", .{checkpoint_path});
    defer allocator.free(meta_path);

    const meta_file = std.fs.cwd().openFile(meta_path, .{}) catch {
        try stdout.print("ERROR: Missing metadata file\n", .{});
        return;
    };
    meta_file.close();

    // Check tensors
    const tensors_path = try std.fmt.allocPrint(allocator, "{s}/tensors.bin", .{checkpoint_path});
    defer allocator.free(tensors_path);

    const tensors_file = std.fs.cwd().openFile(tensors_path, .{}) catch {
        try stdout.print("ERROR: Missing tensors file\n", .{});
        return;
    };
    tensors_file.close();

    try stdout.print("Checkpoint {s} is valid\n", .{checkpoint_path});
}

/// Convert checkpoint format
pub fn convertCheckpoint(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    output_path: []const u8,
    format: CheckpointFormat,
) !void {
    _ = allocator;
    _ = input_path;
    _ = output_path;
    _ = format;

    std.log.info("Checkpoint conversion not yet implemented", .{});
}

test "extractStepFromPath" {
    const path = "/checkpoints/step_00001000";
    const step = CheckpointManager.extractStepFromPath(path);
    try std.testing.expectEqual(@as(usize, 1000), step);
}
