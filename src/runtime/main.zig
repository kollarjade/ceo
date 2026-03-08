const std = @import("std");
const cuda_mod = @import("cuda.zig");
const nccl_mod = @import("nccl.zig");
const config = @import("../util/config.zig");

pub const cuda = cuda_mod;
pub const nccl = nccl_mod;

/// Process group for distributed training
pub const ProcessGroup = struct {
    rank: usize,
    world_size: usize,
    device_id: usize,
    nccl_group: ?*nccl_mod.NcclGroup,
    stream: *cuda_mod.CudaStream,
    collectives: nccl_mod.NcclCollectives,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        rank: usize,
        world_size: usize,
        device_id: usize,
        stream: *cuda_mod.CudaStream,
    ) !ProcessGroup {
        var nccl_group: ?*nccl_mod.NcclGroup = null;

        if (world_size > 1) {
            nccl_group = try allocator.create(nccl_mod.NcclGroup);
            nccl_group.?.* = try nccl_mod.NcclGroup.initAllDevices(allocator, world_size);
        }

        const collectives = if (nccl_group) |g|
            nccl_mod.NcclCollectives.init(g, stream.stream)
        else
            undefined;

        return .{
            .rank = rank,
            .world_size = world_size,
            .device_id = device_id,
            .nccl_group = nccl_group,
            .stream = stream,
            .collectives = collectives,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProcessGroup) void {
        if (self.nccl_group) |g| {
            g.deinit();
            self.allocator.destroy(g);
        }
        self.stream.deinit();
    }

    /// All-reduce operation
    pub fn allReduce(
        self: *ProcessGroup,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: nccl_mod.NcclDataType,
        op: nccl_mod.NcclRedOp,
    ) !void {
        if (self.world_size == 1) {
            // Single GPU, just copy
            const size = count * @as(usize, switch (dtype) {
                .fp32 => 4,
                .fp16 => 2,
                .bf16 => 2,
                else => 4,
            });
            @memcpy(
                @as([*]u8, @ptrCast(recvbuf))[0..size],
                @as([*]const u8, @ptrCast(sendbuf))[0..size],
            );
            return;
        }

        try self.collectives.allReduce(sendbuf, recvbuf, count, dtype, op, self.rank);
    }

    /// All-gather operation
    pub fn allGather(
        self: *ProcessGroup,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        sendcount: usize,
        dtype: nccl_mod.NcclDataType,
    ) !void {
        if (self.world_size == 1) {
            return;
        }

        try self.collectives.allGather(sendbuf, recvbuf, sendcount, dtype, self.rank);
    }

    /// Reduce-scatter operation
    pub fn reduceScatter(
        self: *ProcessGroup,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        recvcount: usize,
        dtype: nccl_mod.NcclDataType,
        op: nccl_mod.NcclRedOp,
    ) !void {
        if (self.world_size == 1) {
            return;
        }

        try self.collectives.reduceScatter(sendbuf, recvbuf, recvcount, dtype, op, self.rank);
    }

    /// Broadcast operation
    pub fn broadcast(
        self: *ProcessGroup,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: nccl_mod.NcclDataType,
        root: usize,
    ) !void {
        if (self.world_size == 1) {
            return;
        }

        try self.collectives.broadcast(sendbuf, recvbuf, count, dtype, @intCast(root), self.rank);
    }

    /// Synchronize all processes
    pub fn barrier(self: *ProcessGroup) !void {
        // Use all-reduce of a single value as barrier
        var dummy: f32 = 0;
        try self.allReduce(&dummy, &dummy, 1, .fp32, .sum);
        try self.stream.synchronize();
    }
};

/// Distributed runtime for coordinating multi-GPU training
pub const DistributedRuntime = struct {
    rank: usize,
    world_size: usize,
    device_ids: []usize,
    process_groups: []*ProcessGroup,
    streams: []*cuda_mod.CudaStream,
    devices: []cuda_mod.CudaDevice,
    allocator: std.mem.Allocator,
    initialized: bool,

    /// Initialize distributed runtime for single GPU
    pub fn initSingleGPU(allocator: std.mem.Allocator) !DistributedRuntime {
        var cuda_init = try cuda_mod.CudaInit.init(allocator);
        defer cuda_init.deinit();

        if (cuda_init.device_count == 0) {
            return error.NoGpuAvailable;
        }

        const device_id: usize = 0;
        var stream = try allocator.create(cuda_mod.CudaStream);
        errdefer allocator.destroy(stream);
        stream.* = try cuda_mod.CudaStream.init(@intCast(device_id));

        var pg = try allocator.create(ProcessGroup);
        errdefer allocator.destroy(pg);
        pg.* = try ProcessGroup.init(allocator, 0, 1, device_id, stream);

        var device = try cuda_mod.CudaDevice.init(0);
        var devices = try allocator.alloc(cuda_mod.CudaDevice, 1);
        devices[0] = device;

        var streams = try allocator.alloc(*cuda_mod.CudaStream, 1);
        streams[0] = stream;

        var pgs = try allocator.alloc(*ProcessGroup, 1);
        pgs[0] = pg;

        return .{
            .rank = 0,
            .world_size = 1,
            .device_ids = &.{0},
            .process_groups = pgs,
            .streams = streams,
            .devices = devices,
            .allocator = allocator,
            .initialized = true,
        };
    }

    /// Initialize distributed runtime from config
    pub fn init(allocator: std.mem.Allocator, cfg: config.RuntimeConfig) !DistributedRuntime {
        var cuda_init = try cuda_mod.CudaInit.init(allocator);
        defer cuda_init.deinit();

        const world_size = cfg.world_size;
        if (world_size == 0) {
            return error.InvalidWorldSize;
        }

        if (cuda_init.device_count < world_size) {
            std.log.warn("Requested {d} GPUs but only {d} available", .{ world_size, cuda_init.device_count });
            return error.NotEnoughGpus;
        }

        // Create streams for each GPU
        var streams = try allocator.alloc(*cuda_mod.CudaStream, world_size);
        errdefer allocator.free(streams);

        var devices = try allocator.alloc(cuda_mod.CudaDevice, world_size);
        errdefer allocator.free(devices);

        var pgs = try allocator.alloc(*ProcessGroup, world_size);
        errdefer allocator.free(pgs);

        var device_ids = try allocator.alloc(usize, world_size);
        errdefer allocator.free(device_ids);

        for (0..world_size) |i| {
            device_ids[i] = i;
            devices[i] = cuda_init.devices[i];

            streams[i] = try allocator.create(cuda_mod.CudaStream);
            streams[i].* = try cuda_mod.CudaStream.init(@intCast(i));

            pgs[i] = try allocator.create(ProcessGroup);
            pgs[i].* = try ProcessGroup.init(allocator, i, world_size, i, streams[i]);
        }

        return .{
            .rank = cfg.rank,
            .world_size = world_size,
            .device_ids = device_ids,
            .process_groups = pgs,
            .streams = streams,
            .devices = devices,
            .allocator = allocator,
            .initialized = true,
        };
    }

    pub fn deinit(self: *DistributedRuntime) void {
        if (!self.initialized) return;

        for (self.process_groups) |pg| {
            pg.deinit();
            self.allocator.destroy(pg);
        }
        self.allocator.free(self.process_groups);

        for (self.streams) |stream| {
            self.allocator.destroy(stream);
        }
        self.allocator.free(self.streams);

        self.allocator.free(self.devices);
        self.allocator.free(self.device_ids);
    }

    /// Get process group for a specific rank
    pub fn getProcessGroup(self: *DistributedRuntime, rank: usize) ?*ProcessGroup {
        if (rank >= self.process_groups.len) return null;
        return self.process_groups[rank];
    }

    /// Get current rank's process group
    pub fn currentProcessGroup(self: *DistributedRuntime) *ProcessGroup {
        return self.process_groups[self.rank];
    }

    /// Get stream for a specific rank
    pub fn getStream(self: *DistributedRuntime, rank: usize) ?*cuda_mod.CudaStream {
        if (rank >= self.streams.len) return null;
        return self.streams[rank];
    }

    /// Get current rank's stream
    pub fn currentStream(self: *DistributedRuntime) *cuda_mod.CudaStream {
        return self.streams[self.rank];
    }

    /// Synchronize all streams
    pub fn synchronize(self: *DistributedRuntime) !void {
        for (self.streams) |stream| {
            try stream.synchronize();
        }
    }

    /// Get memory info for all devices
    pub fn getMemoryInfo(self: *DistributedRuntime) ![]struct { free: usize, total: usize } {
        var info = try self.allocator.alloc(struct { free: usize, total: usize }, self.world_size);

        for (0..self.world_size) |i| {
            _ = cuda_mod.cudaSetDevice(@intCast(i));
            info[i] = try cuda_mod.getMemoryInfo();
        }

        return info;
    }

    /// Check if all devices support Blackwell SM100
    pub fn allBlackwell(self: *DistributedRuntime) bool {
        for (self.devices) |device| {
            if (!device.isBlackwell()) return false;
        }
        return true;
    }

    /// Enable peer-to-peer access between all GPUs
    pub fn enablePeerAccess(self: *DistributedRuntime) !void {
        for (0..self.world_size) |i| {
            for (0..self.world_size) |j| {
                if (i != j) {
                    try cuda_mod.enablePeerAccess(@intCast(i), @intCast(j));
                }
            }
        }
    }
};

/// Topology detection for optimal communication
pub const Topology = struct {
    numa_nodes: []NumaNode,
    gpu_affinity: []GpuAffinity,
    allocator: std.mem.Allocator,

    pub const NumaNode = struct {
        id: usize,
        cpus: []usize,
        gpus: []usize,
        memory_size: usize,
    };

    pub const GpuAffinity = struct {
        gpu_id: usize,
        numa_node: usize,
        pci_bus: u32,
        pci_device: u32,
    };

    pub fn detect(allocator: std.mem.Allocator, num_gpus: usize) !Topology {
        // Simplified topology detection
        // In production, this would read from /sys/devices/system/node
        var numa_nodes = try allocator.alloc(NumaNode, 1);
        numa_nodes[0] = .{
            .id = 0,
            .cpus = try allocator.dupe(usize, &[_]usize{0}),
            .gpus = try allocator.alloc(usize, num_gpus),
            .memory_size = 0,
        };

        for (0..num_gpus) |i| {
            numa_nodes[0].gpus[i] = i;
        }

        var gpu_affinity = try allocator.alloc(GpuAffinity, num_gpus);
        for (0..num_gpus) |i| {
            gpu_affinity[i] = .{
                .gpu_id = i,
                .numa_node = 0,
                .pci_bus = @intCast(i),
                .pci_device = 0,
            };
        }

        return .{
            .numa_nodes = numa_nodes,
            .gpu_affinity = gpu_affinity,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Topology) void {
        for (self.numa_nodes) |*node| {
            self.allocator.free(node.cpus);
            self.allocator.free(node.gpus);
        }
        self.allocator.free(self.numa_nodes);
        self.allocator.free(self.gpu_affinity);
    }

    /// Get optimal GPU for a NUMA node
    pub fn getGpuForNuma(self: *Topology, numa_node: usize) ?usize {
        for (self.gpu_affinity) |affinity| {
            if (affinity.numa_node == numa_node) {
                return affinity.gpu_id;
            }
        }
        return null;
    }
};
