const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;

pub const CheckpointFormat = enum {
    efla_native,
    safetensors,
    pytorch,
};

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

pub const CheckpointEntry = struct {
    name: []u8,
    shape: []usize,
    dtype: tensor_mod.DType,
    offset: u64,
    size: u64,
    checksum: [32]u8,
};

pub const LoadedCheckpoint = struct {
    metadata: CheckpointMetadata,
    optimizer_state: ?[]u8,
    rng_state: ?[]u8,
};

pub const CheckpointManager = struct {
    dir: []const u8,
    max_to_keep: usize,
    compression: bool,
    allocator: std.mem.Allocator,

    const Self = @This();
    const tensors_magic = "EFLATNS1";
    const optimizer_magic = "EFLAOPT1";
    const metadata_file_name = "metadata.json";
    const tensors_file_name = "tensors.bin";
    const optimizer_file_name = "optimizer.bin";
    const rng_file_name = "rng.bin";
    const current_version: u32 = 1;

    const MetadataJson = struct {
        step: usize,
        epoch: usize,
        tokens_seen: usize,
        loss: f32,
        learning_rate: f32,
        timestamp: i64,
        git_revision: []const u8,
        config_hash: []const u8,
    };

    const DiskEntry = struct {
        name: []const u8,
        shape: []const usize,
        dtype: tensor_mod.DType,
        offset: u64,
        size: u64,
        checksum: [32]u8,
    };

    pub fn init(allocator: std.mem.Allocator, dir: []const u8, max_to_keep: usize, compression: bool) !Self {
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

    pub fn save(
        self: *Self,
        step: usize,
        params: []*Tensor,
        param_names: []const []const u8,
        optimizer_state: ?[]const u8,
        rng_state: ?[]const u8,
        metadata: CheckpointMetadata,
    ) ![]const u8 {
        if (params.len != param_names.len) return error.ParameterNameCountMismatch;

        var normalized_metadata = metadata;
        normalized_metadata.step = step;

        const final_name = try std.fmt.allocPrint(self.allocator, "step_{d:0>8}", .{step});
        defer self.allocator.free(final_name);

        const final_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.dir, final_name });
        errdefer self.allocator.free(final_path);

        const now_ns = std.time.nanoTimestamp();
        const temp_name = try std.fmt.allocPrint(self.allocator, ".tmp_step_{d:0>8}_{d}", .{ step, now_ns });
        defer self.allocator.free(temp_name);

        const temp_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.dir, temp_name });
        defer self.allocator.free(temp_path);

        const backup_path = try std.fmt.allocPrint(self.allocator, "{s}.bak", .{final_path});
        defer self.allocator.free(backup_path);

        try std.fs.cwd().makePath(self.dir);
        std.fs.cwd().deleteTree(temp_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
        try std.fs.cwd().makePath(temp_path);
        errdefer {
            std.fs.cwd().deleteTree(temp_path) catch {};
        }

        const meta_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ temp_path, metadata_file_name });
        defer self.allocator.free(meta_path);
        try self.writeMetadata(meta_path, normalized_metadata);

        const tensors_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ temp_path, tensors_file_name });
        defer self.allocator.free(tensors_path);
        try self.writeTensors(tensors_path, params, param_names);

        if (optimizer_state) |state| {
            const opt_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ temp_path, optimizer_file_name });
            defer self.allocator.free(opt_path);
            try self.writeOptimizerState(opt_path, state);
        }

        if (rng_state) |rng| {
            const rng_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ temp_path, rng_file_name });
            defer self.allocator.free(rng_path);
            const rng_file = try std.fs.cwd().createFile(rng_path, .{ .truncate = true });
            defer rng_file.close();
            try rng_file.writeAll(rng);
        }

        std.fs.cwd().deleteTree(backup_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };

        var had_existing = false;
        std.fs.cwd().rename(final_path, backup_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
        had_existing = std.fs.cwd().openDir(backup_path, .{}) catch null != null;
        if (had_existing) {
            if (std.fs.cwd().openDir(backup_path, .{})) |dir| dir.close() else |_| had_existing = false;
        }

        std.fs.cwd().rename(temp_path, final_path) catch |err| {
            if (had_existing) {
                std.fs.cwd().rename(backup_path, final_path) catch {};
            }
            return err;
        };

        if (had_existing) {
            std.fs.cwd().deleteTree(backup_path) catch {};
        }

        try self.cleanupOldCheckpoints();

        return final_path;
    }

    pub fn load(self: *Self, checkpoint_path: []const u8, params: []*Tensor, param_names: []const []const u8) !CheckpointMetadata {
        if (params.len != param_names.len) return error.ParameterNameCountMismatch;

        const meta_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ checkpoint_path, metadata_file_name });
        defer self.allocator.free(meta_path);
        const metadata = try self.readMetadata(meta_path);

        const tensors_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ checkpoint_path, tensors_file_name });
        defer self.allocator.free(tensors_path);
        try self.readTensors(tensors_path, params, param_names);

        return metadata;
    }

    pub fn loadFull(self: *Self, checkpoint_path: []const u8, params: []*Tensor, param_names: []const []const u8) !LoadedCheckpoint {
        const metadata = try self.load(checkpoint_path, params, param_names);

        const optimizer_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ checkpoint_path, optimizer_file_name });
        defer self.allocator.free(optimizer_path);

        const rng_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ checkpoint_path, rng_file_name });
        defer self.allocator.free(rng_path);

        const optimizer_state = try self.readOptionalOptimizerState(optimizer_path);
        errdefer if (optimizer_state) |bytes| self.allocator.free(bytes);

        const rng_state = try self.readOptionalBytes(rng_path);
        errdefer if (rng_state) |bytes| self.allocator.free(bytes);

        return .{ .metadata = metadata, .optimizer_state = optimizer_state, .rng_state = rng_state };
    }

    fn trimRightZeroes(bytes: []const u8) []const u8 {
        var end = bytes.len;
        while (end > 0 and bytes[end - 1] == 0) : (end -= 1) {}
        return bytes[0..end];
    }

    fn encodeHexAlloc(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
        const out = try allocator.alloc(u8, bytes.len * 2);
        _ = std.fmt.bufPrint(out, "{s}", .{std.fmt.fmtSliceHexLower(bytes)}) catch unreachable;
        return out;
    }

    fn hexNibble(c: u8) !u8 {
        return switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => 10 + c - 'a',
            'A'...'F' => 10 + c - 'A',
            else => error.InvalidHexCharacter,
        };
    }

    fn decodeHexInto(hex: []const u8, out: []u8) !void {
        if (hex.len != out.len * 2) return error.InvalidHexLength;
        var i: usize = 0;
        while (i < out.len) : (i += 1) {
            const hi = try hexNibble(hex[i * 2]);
            const lo = try hexNibble(hex[i * 2 + 1]);
            out[i] = (hi << 4) | lo;
        }
    }

    fn writeMetadata(self: *Self, path: []const u8, metadata: CheckpointMetadata) !void {
        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();

        const git_revision_slice = trimRightZeroes(&metadata.git_revision);
        const config_hex = try encodeHexAlloc(self.allocator, &metadata.config_hash);
        defer self.allocator.free(config_hex);

        const payload = MetadataJson{
            .step = metadata.step,
            .epoch = metadata.epoch,
            .tokens_seen = metadata.tokens_seen,
            .loss = metadata.loss,
            .learning_rate = metadata.learning_rate,
            .timestamp = metadata.timestamp,
            .git_revision = git_revision_slice,
            .config_hash = config_hex,
        };

        try std.json.stringify(payload, .{}, file.writer());
    }

    fn readMetadata(self: *Self, path: []const u8) !CheckpointMetadata {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        if (stat.size > 1024 * 1024) return error.MetadataTooLarge;

        const content = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(content);

        var parsed = try std.json.parseFromSlice(MetadataJson, self.allocator, content, .{ .ignore_unknown_fields = false, .allocate = .alloc_always });
        defer parsed.deinit();

        var metadata: CheckpointMetadata = .{
            .step = parsed.value.step,
            .epoch = parsed.value.epoch,
            .tokens_seen = parsed.value.tokens_seen,
            .loss = parsed.value.loss,
            .learning_rate = parsed.value.learning_rate,
            .timestamp = parsed.value.timestamp,
            .git_revision = std.mem.zeroes([40]u8),
            .config_hash = std.mem.zeroes([32]u8),
        };

        if (parsed.value.git_revision.len > metadata.git_revision.len) return error.InvalidGitRevisionLength;
        @memcpy(metadata.git_revision[0..parsed.value.git_revision.len], parsed.value.git_revision);
        if (parsed.value.config_hash.len != metadata.config_hash.len * 2) return error.InvalidConfigHashLength;
        try decodeHexInto(parsed.value.config_hash, &metadata.config_hash);
        return metadata;
    }

    fn hostTensorClone(self: *Self, tensor: *Tensor) !*Tensor {
        if (tensor.device == .cpu) {
            return try tensor.to(self.allocator, .cpu, 0);
        }
        return try tensor.to(self.allocator, .cpu, 0);
    }

    fn storeTensorFromHost(self: *Self, dst: *Tensor, host: *Tensor) !void {
        if (dst.device == .cpu) {
            const src_bytes = host.rawBytes() orelse return error.NullTensorPointer;
            const dst_bytes = dst.rawMutBytes() orelse return error.NullTensorPointer;
            if (src_bytes.len != dst_bytes.len) return error.TensorSizeMismatch;
            @memcpy(dst_bytes, src_bytes);
            return;
        }

        if (@hasDecl(Tensor, "copyFrom")) {
            try dst.copyFrom(host);
            return;
        }
        if (@hasDecl(Tensor, "copy_")) {
            try dst.copy_(host);
            return;
        }
        if (@hasDecl(Tensor, "assign")) {
            try dst.assign(host);
            return;
        }
        return error.UnsupportedDevice;
    }

    fn checksumForBytes(bytes: []const u8) [32]u8 {
        var checksum: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &checksum, .{});
        return checksum;
    }

    fn checksumForTensor(self: *Self, param: *Tensor, size: usize) ![32]u8 {
        var host = try self.hostTensorClone(param);
        defer host.deinit();
        const ptr = host.ptr() orelse return error.NullTensorPointer;
        const bytes = @as([*]const u8, @ptrCast(ptr))[0..size];
        return checksumForBytes(bytes);
    }

    fn writeTensors(self: *Self, path: []const u8, params: []*Tensor, param_names: []const []const u8) !void {
        if (params.len != param_names.len) return error.ParameterNameCountMismatch;

        var name_set = std.StringHashMapUnmanaged(void){};
        defer name_set.deinit(self.allocator);

        var entries = try self.allocator.alloc(DiskEntry, params.len);
        defer self.allocator.free(entries);

        var index_size: u64 = 0;
        var data_size: u64 = 0;

        for (params, 0..) |param, i| {
            const name = param_names[i];
            if (name.len == 0) return error.EmptyTensorName;
            const gop = try name_set.getOrPut(self.allocator, name);
            if (gop.found_existing) return error.DuplicateTensorName;

            const shape = param.shape;
            const dtype = param.dtype;
            const byte_size_usize = shape.sizeBytes(dtype);
            const byte_size: u64 = @intCast(byte_size_usize);
            if (byte_size > 0 and param.ptr() == null) return error.NullTensorPointer;

            const checksum = if (byte_size == 0) std.mem.zeroes([32]u8) else try self.checksumForTensor(param, byte_size_usize);

            entries[i] = .{ .name = name, .shape = shape.dims[0..shape.ndim], .dtype = dtype, .offset = 0, .size = byte_size, .checksum = checksum };

            index_size += 4 + @as(u64, @intCast(name.len));
            index_size += 4 + @as(u64, @intCast(shape.ndim)) * 8;
            index_size += 4 + 8 + 8 + 32;
            data_size += byte_size;
        }

        const header_size: u64 = tensors_magic.len + 4 + 4;
        var running_offset: u64 = header_size + index_size;
        for (entries) |*entry| {
            entry.offset = running_offset;
            running_offset += entry.size;
        }

        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        var writer = file.writer();

        try writer.writeAll(tensors_magic);
        try writer.writeInt(u32, current_version, .little);
        try writer.writeInt(u32, @intCast(entries.len), .little);

        for (entries) |entry| {
            try writer.writeInt(u32, @intCast(entry.name.len), .little);
            try writer.writeAll(entry.name);
            try writer.writeInt(u32, @intCast(entry.shape.len), .little);
            for (entry.shape) |dim| try writer.writeInt(u64, @intCast(dim), .little);
            try writer.writeInt(u32, @intFromEnum(entry.dtype), .little);
            try writer.writeInt(u64, entry.offset, .little);
            try writer.writeInt(u64, entry.size, .little);
            try writer.writeAll(&entry.checksum);
        }

        for (params) |param| {
            const size = param.shape.sizeBytes(param.dtype);
            if (size == 0) continue;
            var host = try self.hostTensorClone(param);
            defer host.deinit();
            const ptr = host.ptr() orelse return error.NullTensorPointer;
            const bytes = @as([*]const u8, @ptrCast(ptr))[0..size];
            try writer.writeAll(bytes);
        }

        const final_position = try file.getPos();
        if (final_position != header_size + index_size + data_size) return error.InvalidTensorFileSize;
    }

    fn readTensors(self: *Self, path: []const u8, params: []*Tensor, param_names: []const []const u8) !void {
        if (params.len != param_names.len) return error.ParameterNameCountMismatch;

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const file_size = stat.size;

        var magic_buf: [tensors_magic.len]u8 = undefined;
        try file.reader().readNoEof(&magic_buf);
        if (!std.mem.eql(u8, &magic_buf, tensors_magic)) return error.InvalidCheckpointFileMagic;

        const version = try file.reader().readInt(u32, .little);
        if (version != current_version) return error.UnsupportedVersion;

        const num_tensors: usize = @intCast(try file.reader().readInt(u32, .little));
        var entries = try self.allocator.alloc(CheckpointEntry, num_tensors);
        defer {
            for (entries) |entry| {
                self.allocator.free(entry.name);
                self.allocator.free(entry.shape);
            }
            self.allocator.free(entries);
        }

        var entry_map = std.StringHashMapUnmanaged(usize){};
        defer entry_map.deinit(self.allocator);

        for (entries, 0..) |*entry, i| {
            const name_len: usize = @intCast(try file.reader().readInt(u32, .little));
            if (name_len == 0) return error.InvalidTensorName;
            entry.name = try self.allocator.alloc(u8, name_len);
            try file.reader().readNoEof(entry.name);
            const ndim: usize = @intCast(try file.reader().readInt(u32, .little));
            entry.shape = try self.allocator.alloc(usize, ndim);
            for (entry.shape) |*dim| {
                const raw_dim = try file.reader().readInt(u64, .little);
                dim.* = try std.math.cast(usize, raw_dim) orelse return error.DimensionTooLarge;
            }
            const dtype_raw = try file.reader().readInt(u32, .little);
            entry.dtype = std.meta.intToEnum(tensor_mod.DType, dtype_raw) catch return error.InvalidDType;
            entry.offset = try file.reader().readInt(u64, .little);
            entry.size = try file.reader().readInt(u64, .little);
            try file.reader().readNoEof(&entry.checksum);

            if (entry.offset > file_size) return error.InvalidTensorOffset;
            if (entry.size > file_size) return error.InvalidTensorSize;
            if (entry.offset > file_size - entry.size) return error.InvalidTensorRange;

            const gop = try entry_map.getOrPut(self.allocator, entry.name);
            if (gop.found_existing) return error.DuplicateTensorName;
            gop.value_ptr.* = i;
        }

        for (params, 0..) |param, i| {
            const tensor_name = param_names[i];
            const index = entry_map.get(tensor_name) orelse return error.MissingTensorInCheckpoint;
            const entry = entries[index];

            if (param.dtype != entry.dtype) return error.TensorDTypeMismatch;
            const param_shape = param.shape.dims[0..param.shape.ndim];
            if (param_shape.len != entry.shape.len) return error.TensorShapeMismatch;
            for (param_shape, entry.shape) |expected, actual| {
                if (expected != actual) return error.TensorShapeMismatch;
            }

            const expected_size = param.shape.sizeBytes(param.dtype);
            if (entry.size != expected_size) return error.TensorSizeMismatch;
            if (expected_size == 0) continue;

            var host = try Tensor.init(self.allocator, param.shape, param.dtype, .cpu, 0);
            defer host.deinit();
            const bytes = host.rawMutBytes() orelse return error.NullTensorPointer;
            try file.seekTo(entry.offset);
            try file.reader().readNoEof(bytes);
            const actual_checksum = checksumForBytes(bytes);
            if (!std.mem.eql(u8, &actual_checksum, &entry.checksum)) return error.TensorChecksumMismatch;
            try self.storeTensorFromHost(param, host);
        }
    }

    fn writeOptimizerState(self: *Self, path: []const u8, state: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        var writer = file.writer();
        try writer.writeAll(optimizer_magic);
        try writer.writeInt(u32, current_version, .little);
        try writer.writeInt(u64, @intCast(state.len), .little);
        try writer.writeAll(state);
        _ = self;
    }

    fn readOptionalOptimizerState(self: *Self, path: []const u8) !?[]u8 {
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => return null,
            else => return err,
        };
        defer file.close();

        var magic_buf: [optimizer_magic.len]u8 = undefined;
        try file.reader().readNoEof(&magic_buf);
        if (!std.mem.eql(u8, &magic_buf, optimizer_magic)) return error.InvalidOptimizerStateMagic;
        const version = try file.reader().readInt(u32, .little);
        if (version != current_version) return error.UnsupportedVersion;
        const size = try std.math.cast(usize, try file.reader().readInt(u64, .little)) orelse return error.OptimizerStateTooLarge;
        const bytes = try self.allocator.alloc(u8, size);
        errdefer self.allocator.free(bytes);
        try file.reader().readNoEof(bytes);
        const end_pos = try file.getPos();
        const stat = try file.stat();
        if (end_pos != stat.size) return error.InvalidOptimizerStateSize;
        return bytes;
    }

    fn readOptionalBytes(self: *Self, path: []const u8) !?[]u8 {
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => return null,
            else => return err,
        };
        defer file.close();
        const stat = try file.stat();
        const size = try std.math.cast(usize, stat.size) orelse return error.FileTooLarge;
        const bytes = try self.allocator.alloc(u8, size);
        errdefer self.allocator.free(bytes);
        try file.reader().readNoEof(bytes);
        return bytes;
    }

    fn cleanupOldCheckpoints(self: *Self) !void {
        var dir = try std.fs.cwd().openDir(self.dir, .{ .iterate = true });
        defer dir.close();

        var steps = std.ArrayList(struct { name: []u8, step: usize }).init(self.allocator);
        defer {
            for (steps.items) |item| self.allocator.free(item.name);
            steps.deinit();
        }

        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind != .directory) continue;
            const step = parseStepDirName(entry.name) orelse continue;
            try steps.append(.{ .name = try self.allocator.dupe(u8, entry.name), .step = step });
        }

        if (steps.items.len == 0) return;

        std.sort.pdq(@TypeOf(steps.items[0]), steps.items, {}, struct {
            fn lessThan(_: void, a: @TypeOf(steps.items[0]), b: @TypeOf(steps.items[0])) bool {
                return a.step > b.step;
            }
        }.lessThan);

        const keep = self.max_to_keep;
        if (keep >= steps.items.len) return;

        for (steps.items[keep..]) |item| {
            const full_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.dir, item.name });
            defer self.allocator.free(full_path);
            try std.fs.cwd().deleteTree(full_path);
        }
    }

    fn parseStepDirName(name: []const u8) ?usize {
        if (!std.mem.startsWith(u8, name, "step_")) return null;
        return std.fmt.parseInt(usize, name["step_".len..], 10) catch null;
    }
};

pub fn listCheckpoints(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();

    var items = std.ArrayList(struct { name: []u8, step: usize }).init(allocator);
    defer {
        for (items.items) |item| allocator.free(item.name);
        items.deinit();
    }

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .directory) continue;
        const step = CheckpointManager.parseStepDirName(entry.name) orelse continue;
        try items.append(.{ .name = try allocator.dupe(u8, entry.name), .step = step });
    }

    std.sort.pdq(@TypeOf(items.items[0]), items.items, {}, struct {
        fn lessThan(_: void, a: @TypeOf(items.items[0]), b: @TypeOf(items.items[0])) bool {
            return a.step > b.step;
        }
    }.lessThan);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("Checkpoints in {s}:\n", .{dir_path});
    for (items.items) |item| try stdout.print("  {s}\n", .{item.name});
}

pub fn validateCheckpoint(allocator: std.mem.Allocator, checkpoint_path: []const u8) !void {
    var manager = try CheckpointManager.init(allocator, ".", 1, false);
    defer manager.deinit();

    const meta_path = try std.fmt.allocPrint(allocator, "{s}/metadata.json", .{checkpoint_path});
    defer allocator.free(meta_path);
    _ = try manager.readMetadata(meta_path);

    const tensors_path = try std.fmt.allocPrint(allocator, "{s}/tensors.bin", .{checkpoint_path});
    defer allocator.free(tensors_path);

    const file = try std.fs.cwd().openFile(tensors_path, .{});
    defer file.close();

    var magic_buf: [CheckpointManager.tensors_magic.len]u8 = undefined;
    try file.reader().readNoEof(&magic_buf);
    if (!std.mem.eql(u8, &magic_buf, CheckpointManager.tensors_magic)) return error.InvalidCheckpointFileMagic;

    const version = try file.reader().readInt(u32, .little);
    if (version != CheckpointManager.current_version) return error.UnsupportedVersion;

    const num_tensors: usize = @intCast(try file.reader().readInt(u32, .little));
    const stat = try file.stat();
    const file_size = stat.size;

    var i: usize = 0;
    while (i < num_tensors) : (i += 1) {
        const name_len = try file.reader().readInt(u32, .little);
        if (name_len == 0) return error.InvalidTensorName;

        const name = try allocator.alloc(u8, name_len);
        defer allocator.free(name);
        try file.reader().readNoEof(name);

        const ndim = try file.reader().readInt(u32, .little);
        var dims = try allocator.alloc(u64, ndim);
        defer allocator.free(dims);
        for (dims) |*dim| dim.* = try file.reader().readInt(u64, .little);

        const dtype_raw = try file.reader().readInt(u32, .little);
        _ = std.meta.intToEnum(tensor_mod.DType, dtype_raw) catch return error.InvalidDType;

        const offset = try file.reader().readInt(u64, .little);
        const size = try file.reader().readInt(u64, .little);

        var checksum: [32]u8 = undefined;
        try file.reader().readNoEof(&checksum);

        if (offset > file_size) return error.InvalidTensorOffset;
        if (size > file_size) return error.InvalidTensorSize;
        if (offset > file_size - size) return error.InvalidTensorRange;
    }

    const stdout = std.io.getStdOut().writer();
    try stdout.print("Checkpoint {s} is valid\n", .{checkpoint_path});
}

pub fn convertCheckpoint(allocator: std.mem.Allocator, input_path: []const u8, output_path: []const u8, format: CheckpointFormat) !void {
    switch (format) {
        .efla_native => {
            const cwd = std.fs.cwd();
            const input_dir = try cwd.openDir(input_path, .{ .iterate = true });
            defer input_dir.close();
            cwd.deleteTree(output_path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };
            try cwd.makePath(output_path);
            var iter = input_dir.iterate();
            while (try iter.next()) |entry| {
                const src = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ input_path, entry.name });
                defer allocator.free(src);
                const dst = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ output_path, entry.name });
                defer allocator.free(dst);
                switch (entry.kind) {
                    .file => try copyFile(allocator, src, dst),
                    .directory => try copyDirRecursive(allocator, src, dst),
                    else => return error.UnsupportedEntryType,
                }
            }
        },
        .safetensors, .pytorch => return error.UnsupportedCheckpointFormat,
    }
}

fn copyFile(allocator: std.mem.Allocator, src_path: []const u8, dst_path: []const u8) !void {
    _ = allocator;
    const src = try std.fs.cwd().openFile(src_path, .{});
    defer src.close();
    const dst = try std.fs.cwd().createFile(dst_path, .{ .truncate = true });
    defer dst.close();
    var buffer: [64 * 1024]u8 = undefined;
    while (true) {
        const read_n = try src.read(&buffer);
        if (read_n == 0) break;
        try dst.writeAll(buffer[0..read_n]);
    }
}

fn copyDirRecursive(allocator: std.mem.Allocator, src_path: []const u8, dst_path: []const u8) !void {
    try std.fs.cwd().makePath(dst_path);
    var src_dir = try std.fs.cwd().openDir(src_path, .{ .iterate = true });
    defer src_dir.close();
    var iter = src_dir.iterate();
    while (try iter.next()) |entry| {
        const child_src = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ src_path, entry.name });
        defer allocator.free(child_src);
        const child_dst = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dst_path, entry.name });
        defer allocator.free(child_dst);
        switch (entry.kind) {
            .file => try copyFile(allocator, child_src, child_dst),
            .directory => try copyDirRecursive(allocator, child_src, child_dst),
            else => return error.UnsupportedEntryType,
        }
    }
}

test "parse step dir name" {
    try std.testing.expectEqual(@as(?usize, 1000), CheckpointManager.parseStepDirName("step_00001000"));
}

test "decode config hash hex" {
    var out: [32]u8 = undefined;
    try CheckpointManager.decodeHexInto("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", &out);
    try std.testing.expectEqual(@as(u8, 0), out[0]);
    try std.testing.expectEqual(@as(u8, 31), out[31]);
}
