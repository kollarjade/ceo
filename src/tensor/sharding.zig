const std = @import("std");

pub const ShapeError = error{
    RankTooLarge,
    DimensionOutOfBounds,
};

pub const ParallelError = error{
    InvalidWorldSize,
    InvalidRank,
    InvalidNumShards,
    InvalidShardIndex,
    InvalidDimension,
    DivisionByZero,
    IncompatibleShape,
    InvalidStage,
    InvalidPartitionCount,
    InvalidPartitionIndex,
    Overflow,
    NullPointer,
};

pub const Shape = struct {
    dims: [16]usize,
    strides: [16]usize,
    rank: u8,

    pub fn init(dims_slice: []const usize) ShapeError!Shape {
        if (dims_slice.len > 16) return ShapeError.RankTooLarge;
        var shape = Shape{
            .dims = [_]usize{0} ** 16,
            .strides = [_]usize{0} ** 16,
            .rank = @intCast(dims_slice.len),
        };
        if (shape.rank == 0) return shape;
        for (dims_slice, 0..) |d, i| {
            shape.dims[i] = d;
        }
        shape.strides[shape.rank - 1] = 1;
        if (shape.rank > 1) {
            var i: usize = shape.rank - 1;
            while (i > 0) {
                i -= 1;
                shape.strides[i] = shape.strides[i + 1] * shape.dims[i + 1];
            }
        }
        return shape;
    }
};

pub const DataType = enum {
    f32,
    f16,
    bf16,
    i32,
    i8,
    u8,
};

pub const ShardDim = enum(u8) {
    dim_0 = 0,
    dim_1 = 1,
    dim_2 = 2,
    dim_3 = 3,
    dim_4 = 4,
    dim_5 = 5,
    dim_6 = 6,
    dim_7 = 7,
    dim_8 = 8,
    dim_9 = 9,
    dim_10 = 10,
    dim_11 = 11,
    dim_12 = 12,
    dim_13 = 13,
    dim_14 = 14,
    dim_15 = 15,
    none = 255,

    pub inline fn toIndex(self: ShardDim) ?u8 {
        if (self == .none) return null;
        return @intFromEnum(self);
    }

    pub inline fn fromIndex(idx: usize) ParallelError!ShardDim {
        if (idx > 15) return ParallelError.InvalidDimension;
        return @enumFromInt(@as(u8, @intCast(idx)));
    }
};

pub const ShardSpec = struct {
    shard_dim: ShardDim,
    num_shards: usize,
    shard_idx: usize,
    is_first: bool,
    is_last: bool,

    pub inline fn init(shard_dim: ShardDim, num_shards: usize, shard_idx: usize) ParallelError!ShardSpec {
        if (num_shards == 0) return ParallelError.InvalidNumShards;
        if (shard_idx >= num_shards) return ParallelError.InvalidShardIndex;
        if (shard_dim == .none and num_shards != 1) return ParallelError.InvalidNumShards;

        return ShardSpec{
            .shard_dim = shard_dim,
            .num_shards = num_shards,
            .shard_idx = shard_idx,
            .is_first = (shard_idx == 0),
            .is_last = (shard_idx == num_shards - 1),
        };
    }

    pub inline fn noSharding() ShardSpec {
        return ShardSpec{
            .shard_dim = .none,
            .num_shards = 1,
            .shard_idx = 0,
            .is_first = true,
            .is_last = true,
        };
    }

    pub inline fn localShape(self: ShardSpec, global_shape: Shape) ParallelError!Shape {
        if (self.shard_dim == .none) return global_shape;
        if (global_shape.rank == 0) return ParallelError.IncompatibleShape;

        const dim_idx = self.shard_dim.toIndex() orelse return global_shape;
        if (dim_idx >= global_shape.rank) return ParallelError.InvalidDimension;

        var local = global_shape;
        const dim_size = global_shape.dims[dim_idx];
        const base = dim_size / self.num_shards;
        const remainder = dim_size % self.num_shards;

        if (self.shard_idx < remainder) {
            local.dims[dim_idx] = std.math.add(usize, base, 1) catch return ParallelError.Overflow;
        } else {
            local.dims[dim_idx] = base;
        }

        var i: usize = local.rank - 1;
        local.strides[i] = 1;
        if (local.rank > 1) {
            while (i > 0) {
                i -= 1;
                local.strides[i] = std.math.mul(usize, local.strides[i + 1], local.dims[i + 1]) catch return ParallelError.Overflow;
            }
        }

        return local;
    }

    pub inline fn globalOffset(self: ShardSpec, global_shape: Shape) ParallelError![16]usize {
        var offsets = [_]usize{0} ** 16;
        if (self.shard_dim == .none) return offsets;
        if (global_shape.rank == 0) return ParallelError.IncompatibleShape;

        const dim_idx = self.shard_dim.toIndex() orelse return offsets;
        if (dim_idx >= global_shape.rank) return ParallelError.InvalidDimension;

        const dim_size = global_shape.dims[dim_idx];
        const base = dim_size / self.num_shards;
        const remainder = dim_size % self.num_shards;

        if (self.shard_idx < remainder) {
            const chunk_size = std.math.add(usize, base, 1) catch return ParallelError.Overflow;
            offsets[dim_idx] = std.math.mul(usize, self.shard_idx, chunk_size) catch return ParallelError.Overflow;
        } else {
            const chunk_size = std.math.add(usize, base, 1) catch return ParallelError.Overflow;
            const rem_offset = std.math.mul(usize, remainder, chunk_size) catch return ParallelError.Overflow;
            const idx_diff = std.math.sub(usize, self.shard_idx, remainder) catch return ParallelError.Overflow;
            const base_offset = std.math.mul(usize, idx_diff, base) catch return ParallelError.Overflow;
            offsets[dim_idx] = std.math.add(usize, rem_offset, base_offset) catch return ParallelError.Overflow;
        }

        return offsets;
    }

    pub inline fn isCompatible(self: ShardSpec, other: ShardSpec) bool {
        if (self.shard_dim == .none and other.shard_dim == .none) return true;
        return self.shard_dim == other.shard_dim and self.num_shards == other.num_shards;
    }
};

pub const TensorParallelConfig = struct {
    world_size: usize,
    rank: usize,
    embed_shard_dim: ShardDim,
    attn_qkv_shard_dim: ShardDim,
    attn_out_shard_dim: ShardDim,
    mlp_up_shard_dim: ShardDim,
    mlp_down_shard_dim: ShardDim,
    group_id: usize,

    pub inline fn init(
        world_size: usize,
        rank: usize,
        embed_dim: ShardDim,
        attn_qkv_dim: ShardDim,
        attn_out_dim: ShardDim,
        mlp_up_dim: ShardDim,
        mlp_down_dim: ShardDim,
        group_id: usize,
    ) ParallelError!TensorParallelConfig {
        if (world_size == 0) return ParallelError.InvalidWorldSize;
        if (rank >= world_size) return ParallelError.InvalidRank;

        return TensorParallelConfig{
            .world_size = world_size,
            .rank = rank,
            .embed_shard_dim = embed_dim,
            .attn_qkv_shard_dim = attn_qkv_dim,
            .attn_out_shard_dim = attn_out_dim,
            .mlp_up_shard_dim = mlp_up_dim,
            .mlp_down_shard_dim = mlp_down_dim,
            .group_id = group_id,
        };
    }

    pub inline fn getShardSpec(self: TensorParallelConfig, shard_dim: ShardDim) ParallelError!ShardSpec {
        if (shard_dim == .none or self.world_size == 1) return ShardSpec.noSharding();
        return ShardSpec.init(shard_dim, self.world_size, self.rank);
    }
};

pub const SequenceParallelConfig = struct {
    world_size: usize,
    rank: usize,
    total_seq_len: usize,
    seq_dim: ShardDim,
    use_ring_attention: bool,
    group_id: usize,
    cached_local_len: usize,
    cached_offset: usize,

    pub inline fn init(
        world_size: usize,
        rank: usize,
        total_seq_len: usize,
        seq_dim: ShardDim,
        use_ring_attention: bool,
        group_id: usize,
    ) ParallelError!SequenceParallelConfig {
        if (world_size == 0) return ParallelError.InvalidWorldSize;
        if (rank >= world_size) return ParallelError.InvalidRank;

        const base = total_seq_len / world_size;
        const remainder = total_seq_len % world_size;
        
        var local_len: usize = 0;
        var offset: usize = 0;

        if (rank < remainder) {
            local_len = std.math.add(usize, base, 1) catch return ParallelError.Overflow;
            offset = std.math.mul(usize, rank, local_len) catch return ParallelError.Overflow;
        } else {
            local_len = base;
            const chunk_size = std.math.add(usize, base, 1) catch return ParallelError.Overflow;
            const rem_offset = std.math.mul(usize, remainder, chunk_size) catch return ParallelError.Overflow;
            const idx_diff = std.math.sub(usize, rank, remainder) catch return ParallelError.Overflow;
            const base_offset = std.math.mul(usize, idx_diff, base) catch return ParallelError.Overflow;
            offset = std.math.add(usize, rem_offset, base_offset) catch return ParallelError.Overflow;
        }

        return SequenceParallelConfig{
            .world_size = world_size,
            .rank = rank,
            .total_seq_len = total_seq_len,
            .seq_dim = seq_dim,
            .use_ring_attention = use_ring_attention,
            .group_id = group_id,
            .cached_local_len = local_len,
            .cached_offset = offset,
        };
    }

    pub inline fn localSeqLen(self: SequenceParallelConfig) usize {
        return self.cached_local_len;
    }

    pub inline fn seqOffset(self: SequenceParallelConfig) usize {
        return self.cached_offset;
    }

    pub inline fn getSeqShardSpec(self: SequenceParallelConfig) ParallelError!ShardSpec {
        if (self.world_size == 1 or self.seq_dim == .none) return ShardSpec.noSharding();
        return ShardSpec.init(self.seq_dim, self.world_size, self.rank);
    }
};

pub const ZeroConfig = struct {
    stage: u8,
    offload_optimizer: bool,
    offload_param: bool,
    partition_count: usize,
    partition_idx: usize,
    group_id: usize,

    pub inline fn init(
        stage: u8,
        partition_count: usize,
        partition_idx: usize,
        offload_optimizer: bool,
        offload_param: bool,
        group_id: usize,
    ) ParallelError!ZeroConfig {
        if (partition_count == 0) return ParallelError.InvalidPartitionCount;
        if (partition_idx >= partition_count) return ParallelError.InvalidPartitionIndex;
        if (stage > 3) return ParallelError.InvalidStage;

        return ZeroConfig{
            .stage = stage,
            .offload_optimizer = offload_optimizer,
            .offload_param = offload_param,
            .partition_count = partition_count,
            .partition_idx = partition_idx,
            .group_id = group_id,
        };
    }

    pub inline fn stage1(partition_count: usize, partition_idx: usize, offload_opt: bool, group_id: usize) ParallelError!ZeroConfig {
        return init(1, partition_count, partition_idx, offload_opt, false, group_id);
    }

    pub inline fn stage2(partition_count: usize, partition_idx: usize, offload_opt: bool, offload_par: bool, group_id: usize) ParallelError!ZeroConfig {
        return init(2, partition_count, partition_idx, offload_opt, offload_par, group_id);
    }

    pub inline fn stage3(partition_count: usize, partition_idx: usize, offload_opt: bool, offload_par: bool, group_id: usize) ParallelError!ZeroConfig {
        return init(3, partition_count, partition_idx, offload_opt, offload_par, group_id);
    }

    pub inline fn ownsParameter(self: ZeroConfig, param_offset: usize, param_size: usize) bool {
        _ = param_size;
        const chunk_size = 1024 * 1024;
        const chunk_idx = param_offset / chunk_size;
        return (chunk_idx % self.partition_count) == self.partition_idx;
    }

    pub inline fn getParamPartition(self: ZeroConfig, param_offset: usize) usize {
        const chunk_size = 1024 * 1024;
        const chunk_idx = param_offset / chunk_size;
        return chunk_idx % self.partition_count;
    }

    pub inline fn shouldShardOptimizer(self: ZeroConfig) bool {
        return self.stage >= 1 and self.stage <= 3;
    }

    pub inline fn shouldShardGradients(self: ZeroConfig) bool {
        return self.stage >= 2 and self.stage <= 3;
    }

    pub inline fn shouldShardParams(self: ZeroConfig) bool {
        return self.stage == 3;
    }
};

pub const ParallelConfig = struct {
    dp_world_size: usize,
    dp_rank: usize,
    tp_world_size: usize,
    tp_rank: usize,
    pp_world_size: usize,
    pp_rank: usize,
    tensor_parallel: TensorParallelConfig,
    sequence_parallel: ?SequenceParallelConfig,
    zero: ?ZeroConfig,

    pub inline fn init(
        dp_world_size: usize,
        dp_rank: usize,
        tp_world_size: usize,
        tp_rank: usize,
        pp_world_size: usize,
        pp_rank: usize,
        tp_config: TensorParallelConfig,
    ) ParallelError!ParallelConfig {
        if (dp_world_size == 0 or tp_world_size == 0 or pp_world_size == 0) return ParallelError.InvalidWorldSize;
        if (dp_rank >= dp_world_size or tp_rank >= tp_world_size or pp_rank >= pp_world_size) return ParallelError.InvalidRank;

        return ParallelConfig{
            .dp_world_size = dp_world_size,
            .dp_rank = dp_rank,
            .tp_world_size = tp_world_size,
            .tp_rank = tp_rank,
            .pp_world_size = pp_world_size,
            .pp_rank = pp_rank,
            .tensor_parallel = tp_config,
            .sequence_parallel = null,
            .zero = null,
        };
    }

    pub inline fn setSequenceParallel(self: *ParallelConfig, sp_config: SequenceParallelConfig) void {
        self.sequence_parallel = sp_config;
    }

    pub inline fn setZero(self: *ParallelConfig, zero_config: ZeroConfig) void {
        self.zero = zero_config;
    }

    pub inline fn globalRank(self: ParallelConfig) ParallelError!usize {
        var rank = self.pp_rank;
        rank = std.math.mul(usize, rank, self.dp_world_size) catch return ParallelError.Overflow;
        rank = std.math.add(usize, rank, self.dp_rank) catch return ParallelError.Overflow;
        rank = std.math.mul(usize, rank, self.tp_world_size) catch return ParallelError.Overflow;
        rank = std.math.add(usize, rank, self.tp_rank) catch return ParallelError.Overflow;
        if (self.sequence_parallel) |sp| {
            rank = std.math.mul(usize, rank, sp.world_size) catch return ParallelError.Overflow;
            rank = std.math.add(usize, rank, sp.rank) catch return ParallelError.Overflow;
        }
        return rank;
    }

    pub inline fn globalWorldSize(self: ParallelConfig) ParallelError!usize {
        var size = std.math.mul(usize, self.pp_world_size, self.dp_world_size) catch return ParallelError.Overflow;
        size = std.math.mul(usize, size, self.tp_world_size) catch return ParallelError.Overflow;
        if (self.sequence_parallel) |sp| {
            size = std.math.mul(usize, size, sp.world_size) catch return ParallelError.Overflow;
        }
        return size;
    }
};

pub const ShardedTensor = struct {
    local_tensor: *anyopaque,
    dtype: DataType,
    global_shape: Shape,
    shard_spec: ShardSpec,
    parallel_config: *const ParallelConfig,

    pub inline fn init(
        local_tensor: ?*anyopaque,
        dtype: DataType,
        global_shape: Shape,
        shard_spec: ShardSpec,
        parallel_config: *const ParallelConfig,
    ) ParallelError!ShardedTensor {
        const ptr = local_tensor orelse return ParallelError.NullPointer;
        
        if (shard_spec.shard_dim != .none) {
            const dim_idx = shard_spec.shard_dim.toIndex() orelse return ParallelError.InvalidDimension;
            if (dim_idx >= global_shape.rank) return ParallelError.IncompatibleShape;
        }

        return ShardedTensor{
            .local_tensor = ptr,
            .dtype = dtype,
            .global_shape = global_shape,
            .shard_spec = shard_spec,
            .parallel_config = parallel_config,
        };
    }

    pub inline fn localShape(self: ShardedTensor) ParallelError!Shape {
        return self.shard_spec.localShape(self.global_shape);
    }
};

test "Shape init" {
    const dims = [_]usize{ 128, 64 };
    const shape = try Shape.init(&dims);
    try std.testing.expectEqual(@as(u8, 2), shape.rank);
    try std.testing.expectEqual(@as(usize, 128), shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 64), shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 64), shape.strides[0]);
    try std.testing.expectEqual(@as(usize, 1), shape.strides[1]);
}

test "ShardSpec localShape" {
    const spec = try ShardSpec.init(.dim_0, 4, 0);
    const dims = [_]usize{ 128, 64 };
    const global = try Shape.init(&dims);
    const local = try spec.localShape(global);

    try std.testing.expectEqual(@as(usize, 32), local.dims[0]);
    try std.testing.expectEqual(@as(usize, 64), local.dims[1]);
}

test "SequenceParallelConfig" {
    const cfg = try SequenceParallelConfig.init(4, 0, 1000, .dim_1, false, 0);
    try std.testing.expectEqual(@as(usize, 250), cfg.localSeqLen());
    try std.testing.expectEqual(@as(usize, 0), cfg.seqOffset());

    const cfg2 = try SequenceParallelConfig.init(4, 1, 1000, .dim_1, false, 0);
    try std.testing.expectEqual(@as(usize, 250), cfg2.localSeqLen());
    try std.testing.expectEqual(@as(usize, 250), cfg2.seqOffset());
}

test "ZeroConfig ownsParameter" {
    const zero = try ZeroConfig.stage2(4, 0, false, false, 0);
    try std.testing.expect(zero.ownsParameter(0, 1024));
    try std.testing.expect(!zero.ownsParameter(1024 * 1024, 1024));
    try std.testing.expect(zero.ownsParameter(4 * 1024 * 1024, 1024));
}
