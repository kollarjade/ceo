const std = @import("std");
const Shape = @import("tensor.zig").Shape;

/// Sharding dimension
pub const ShardDim = enum(u8) {
    none = 0,
    dim_0 = 1,
    dim_1 = 2,
    dim_2 = 3,
    dim_3 = 4,
    dim_4 = 5,
    dim_5 = 6,
    dim_6 = 7,
    dim_7 = 8,

    pub fn toIndex(self: ShardDim) ?usize {
        return switch (self) {
            .none => null,
            .dim_0 => 0,
            .dim_1 => 1,
            .dim_2 => 2,
            .dim_3 => 3,
            .dim_4 => 4,
            .dim_5 => 5,
            .dim_6 => 6,
            .dim_7 => 7,
        };
    }

    pub fn fromIndex(idx: usize) ShardDim {
        return switch (idx) {
            0 => .dim_0,
            1 => .dim_1,
            2 => .dim_2,
            3 => .dim_3,
            4 => .dim_4,
            5 => .dim_5,
            6 => .dim_6,
            7 => .dim_7,
            else => .none,
        };
    }
};

/// Sharding specification for a tensor
pub const ShardSpec = struct {
    /// Dimension to shard along
    shard_dim: ShardDim,
    /// Number of shards
    num_shards: usize,
    /// Current shard index
    shard_idx: usize,
    /// Whether this is the first shard
    is_first: bool,
    /// Whether this is the last shard
    is_last: bool,

    pub fn init(shard_dim: ShardDim, num_shards: usize, shard_idx: usize) ShardSpec {
        return .{
            .shard_dim = shard_dim,
            .num_shards = num_shards,
            .shard_idx = shard_idx,
            .is_first = shard_idx == 0,
            .is_last = shard_idx == num_shards - 1,
        };
    }

    pub fn noSharding() ShardSpec {
        return .{
            .shard_dim = .none,
            .num_shards = 1,
            .shard_idx = 0,
            .is_first = true,
            .is_last = true,
        };
    }

    /// Get local shape for this shard
    pub fn localShape(self: ShardSpec, global_shape: Shape) Shape {
        if (self.shard_dim == .none) {
            return global_shape;
        }

        var local = global_shape;
        if (self.shard_dim.toIndex()) |dim| {
            const dim_size = global_shape.dims[dim];
            const base = dim_size / self.num_shards;
            const remainder = dim_size % self.num_shards;

            if (self.shard_idx < remainder) {
                local.dims[dim] = base + 1;
            } else {
                local.dims[dim] = base;
            }
        }

        return local;
    }

    /// Get the offset into the global tensor
    pub fn globalOffset(self: ShardSpec, global_shape: Shape) usize {
        if (self.shard_dim == .none) {
            return 0;
        }

        if (self.shard_dim.toIndex()) |dim| {
            const dim_size = global_shape.dims[dim];
            const base = dim_size / self.num_shards;
            const remainder = dim_size % self.num_shards;

            if (self.shard_idx < remainder) {
                return self.shard_idx * (base + 1);
            } else {
                return remainder * (base + 1) + (self.shard_idx - remainder) * base;
            }
        }

        return 0;
    }

    /// Check if two shard specs are compatible for communication
    pub fn isCompatible(self: ShardSpec, other: ShardSpec) bool {
        return self.shard_dim == other.shard_dim and
            self.num_shards == other.num_shards;
    }
};

/// Tensor parallelism configuration
pub const TensorParallelConfig = struct {
    /// World size for tensor parallelism
    world_size: usize,
    /// Current rank
    rank: usize,
    /// Dimensions to shard for different layer types
    embed_shard_dim: ShardDim = .dim_0, // Shard embedding table
    attn_qkv_shard_dim: ShardDim = .dim_0, // Shard QKV projections
    attn_out_shard_dim: ShardDim = .dim_1, // Shard output projection
    mlp_up_shard_dim: ShardDim = .dim_0, // Shard MLP up projection
    mlp_down_shard_dim: ShardDim = .dim_1, // Shard MLP down projection

    pub fn init(world_size: usize, rank: usize) TensorParallelConfig {
        return .{
            .world_size = world_size,
            .rank = rank,
        };
    }

    pub fn getShardSpec(self: TensorParallelConfig, shard_dim: ShardDim) ShardSpec {
        return ShardSpec.init(shard_dim, self.world_size, self.rank);
    }
};

/// Sequence parallelism configuration
pub const SequenceParallelConfig = struct {
    /// World size for sequence parallelism
    world_size: usize,
    /// Current rank
    rank: usize,
    /// Total sequence length
    total_seq_len: usize,
    /// Whether to use ring attention
    use_ring_attention: bool = false,

    pub fn init(world_size: usize, rank: usize, total_seq_len: usize) SequenceParallelConfig {
        return .{
            .world_size = world_size,
            .rank = rank,
            .total_seq_len = total_seq_len,
        };
    }

    /// Get local sequence length for this rank
    pub fn localSeqLen(self: SequenceParallelConfig) usize {
        const base = self.total_seq_len / self.world_size;
        const remainder = self.total_seq_len % self.world_size;
        return if (self.rank < remainder) base + 1 else base;
    }

    /// Get sequence offset for this rank
    pub fn seqOffset(self: SequenceParallelConfig) usize {
        const base = self.total_seq_len / self.world_size;
        const remainder = self.total_seq_len % self.world_size;
        if (self.rank < remainder) {
            return self.rank * (base + 1);
        } else {
            return remainder * (base + 1) + (self.rank - remainder) * base;
        }
    }

    /// Get shard spec for sequence dimension
    pub fn getSeqShardSpec(self: SequenceParallelConfig) ShardSpec {
        return ShardSpec.init(.dim_0, self.world_size, self.rank);
    }
};

/// ZeRO-style parameter sharding configuration
pub const ZeroConfig = struct {
    /// ZeRO stage (1: optimizer sharding, 2: gradient sharding, 3: parameter sharding)
    stage: u8,
    /// Whether to offload optimizer states to CPU
    offload_optimizer: bool,
    /// Whether to offload parameters to CPU
    offload_param: bool,
    /// Number of partitions for sharding
    partition_count: usize,
    /// Current partition index
    partition_idx: usize,

    pub fn init(stage: u8, partition_count: usize, partition_idx: usize) ZeroConfig {
        return .{
            .stage = stage,
            .offload_optimizer = false,
            .offload_param = false,
            .partition_count = partition_count,
            .partition_idx = partition_idx,
        };
    }

    pub fn stage1(partition_count: usize, partition_idx: usize) ZeroConfig {
        return init(1, partition_count, partition_idx);
    }

    pub fn stage2(partition_count: usize, partition_idx: usize) ZeroConfig {
        return init(2, partition_count, partition_idx);
    }

    pub fn stage3(partition_count: usize, partition_idx: usize) ZeroConfig {
        return init(3, partition_count, partition_idx);
    }

    /// Check if this partition owns a parameter
    pub fn ownsParameter(self: ZeroConfig, param_idx: usize) bool {
        return param_idx % self.partition_count == self.partition_idx;
    }

    /// Get partition for a parameter
    pub fn getParamPartition(self: ZeroConfig, param_idx: usize) usize {
        return param_idx % self.partition_count;
    }

    /// Should shard optimizer states
    pub fn shouldShardOptimizer(self: ZeroConfig) bool {
        return self.stage >= 1;
    }

    /// Should shard gradients
    pub fn shouldShardGradients(self: ZeroConfig) bool {
        return self.stage >= 2;
    }

    /// Should shard parameters
    pub fn shouldShardParams(self: ZeroConfig) bool {
        return self.stage >= 3;
    }
};

/// Combined parallelism configuration
pub const ParallelConfig = struct {
    tensor_parallel: TensorParallelConfig,
    sequence_parallel: ?SequenceParallelConfig,
    zero: ?ZeroConfig,
    pipeline_parallel_size: usize,
    pipeline_parallel_rank: usize,

    pub fn init(
        tp_world_size: usize,
        tp_rank: usize,
        pp_world_size: usize,
        pp_rank: usize,
    ) ParallelConfig {
        return .{
            .tensor_parallel = TensorParallelConfig.init(tp_world_size, tp_rank),
            .sequence_parallel = null,
            .zero = null,
            .pipeline_parallel_size = pp_world_size,
            .pipeline_parallel_rank = pp_rank,
        };
    }

    pub fn withSequenceParallel(self: ParallelConfig, seq_len: usize) ParallelConfig {
        var cfg = self;
        cfg.sequence_parallel = SequenceParallelConfig.init(
            self.tensor_parallel.world_size,
            self.tensor_parallel.rank,
            seq_len,
        );
        return cfg;
    }

    pub fn withZero(self: ParallelConfig, stage: u8) ParallelConfig {
        var cfg = self;
        cfg.zero = ZeroConfig.init(stage, self.tensor_parallel.world_size, self.tensor_parallel.rank);
        return cfg;
    }

    /// Get global rank from parallel config
    pub fn globalRank(self: ParallelConfig) usize {
        return self.pipeline_parallel_rank * self.tensor_parallel.world_size + self.tensor_parallel.rank;
    }

    /// Get global world size
    pub fn globalWorldSize(self: ParallelConfig) usize {
        return self.pipeline_parallel_size * self.tensor_parallel.world_size;
    }
};

/// Sharded tensor view
pub const ShardedTensor = struct {
    local_tensor: *anyopaque, // Pointer to local tensor
    global_shape: Shape,
    shard_spec: ShardSpec,
    parallel_config: ParallelConfig,

    pub fn init(
        local_tensor: *anyopaque,
        global_shape: Shape,
        shard_spec: ShardSpec,
        parallel_config: ParallelConfig,
    ) ShardedTensor {
        return .{
            .local_tensor = local_tensor,
            .global_shape = global_shape,
            .shard_spec = shard_spec,
            .parallel_config = parallel_config,
        };
    }

    pub fn localShape(self: ShardedTensor) Shape {
        return self.shard_spec.localShape(self.global_shape);
    }
};

test "ShardSpec localShape" {
    const spec = ShardSpec.init(.dim_0, 4, 0);
    const global = Shape.init(&[_]usize{ 128, 64 });
    const local = spec.localShape(global);

    try std.testing.expectEqual(@as(usize, 32), local.dims[0]);
    try std.testing.expectEqual(@as(usize, 64), local.dims[1]);
}

test "SequenceParallelConfig" {
    const cfg = SequenceParallelConfig.init(4, 0, 1000);
    try std.testing.expectEqual(@as(usize, 250), cfg.localSeqLen());
    try std.testing.expectEqual(@as(usize, 0), cfg.seqOffset());

    const cfg2 = SequenceParallelConfig.init(4, 1, 1000);
    try std.testing.expectEqual(@as(usize, 250), cfg2.localSeqLen());
    try std.testing.expectEqual(@as(usize, 250), cfg2.seqOffset());
}

test "ZeroConfig ownsParameter" {
    const zero = ZeroConfig.stage2(4, 0);
    try std.testing.expect(zero.ownsParameter(0));
    try std.testing.expect(!zero.ownsParameter(1));
    try std.testing.expect(zero.ownsParameter(4));
}
