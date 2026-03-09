const std = @import("std");
const cuda_mod = @import("cuda.zig");
const nccl_mod = @import("nccl.zig");
const config = @import("../util/config.zig");

pub const cuda = cuda_mod;
pub const nccl = nccl_mod;

pub const ProcessGroup = struct {
    rank: usize,
    world_size: usize,
    device_id: usize,
    nccl_group: ?*nccl_mod.NcclGroup,
    stream: *cuda_mod.CudaStream,
    collectives: ?nccl_mod.NcclCollectives,
    allocator: std.mem.Allocator,
    owns_stream: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        rank: usize,
        world_size: usize,
        device_id: usize,
        stream: *cuda_mod.CudaStream,
    ) !ProcessGroup {
        var nccl_group: ?*nccl_mod.NcclGroup = null;
        var collectives: ?nccl_mod.NcclCollectives = null;

        if (world_size > 1) {
            nccl_group = try allocator.create(nccl_mod.NcclGroup);
            errdefer allocator.destroy(nccl_group.?);
            nccl_group.?.* = try nccl_mod.NcclGroup.initAllDevices(allocator, world_size);
            collectives = nccl_mod.NcclCollectives.init(nccl_group.?, stream.stream);
        }

        return .{
            .rank = rank,
            .world_size = world_size,
            .device_id = device_id,
            .nccl_group = nccl_group,
            .stream = stream,
            .collectives = collectives,
            .allocator = allocator,
            .owns_stream = false,
        };
    }

    pub fn deinit(self: *ProcessGroup) void {
        if (self.nccl_group) |g| {
            g.deinit();
            self.allocator.destroy(g);
        }
    }

    pub fn allReduce(
        self: *ProcessGroup,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: nccl_mod.NcclDataType,
        op: nccl_mod.NcclRedOp,
    ) !void {
        if (self.world_size == 1) {
            const elem_size: usize = switch (dtype) {
                .int8, .uint8 => 1,
                .fp16, .bf16 => 2,
                .int32, .uint32, .fp32 => 4,
                .int64, .uint64, .fp64 => 8,
            };
            const size = count * elem_size;
            const dst_ptr = @as([*]u8, @ptrCast(recvbuf));
            const src_ptr = @as([*]const u8, @ptrCast(sendbuf));
            if (dst_ptr != src_ptr) {
                @memcpy(dst_ptr[0..size], src_ptr[0..size]);
            }
            return;
        }

        if (self.collectives) |*c| {
            try c.allReduce(sendbuf, recvbuf, count, dtype, op, self.rank);
        }
    }

    pub fn allGather(
        self: *ProcessGroup,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        sendcount: usize,
        dtype: nccl_mod.NcclDataType,
    ) !void {
        if (self.world_size == 1) {
            const elem_size: usize = switch (dtype) {
                .int8, .uint8 => 1,
                .fp16, .bf16 => 2,
                .int32, .uint32, .fp32 => 4,
                .int64, .uint64, .fp64 => 8,
            };
            const size = sendcount * elem_size;
            const dst_ptr = @as([*]u8, @ptrCast(recvbuf));
            const src_ptr = @as([*]const u8, @ptrCast(sendbuf));
            if (dst_ptr != src_ptr) {
                @memcpy(dst_ptr[0..size], src_ptr[0..size]);
            }
            return;
        }

        if (self.collectives) |*c| {
            try c.allGather(sendbuf, recvbuf, sendcount, dtype, self.rank);
        }
    }

    pub fn reduceScatter(
        self: *ProcessGroup,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        recvcount: usize,
        dtype: nccl_mod.NcclDataType,
        op: nccl_mod.NcclRedOp,
    ) !void {
        if (self.world_size == 1) {
            const elem_size: usize = switch (dtype) {
                .int8, .uint8 => 1,
                .fp16, .bf16 => 2,
                .int32, .uint32, .fp32 => 4,
                .int64, .uint64, .fp64 => 8,
            };
            const size = recvcount * elem_size;
            const dst_ptr = @as([*]u8, @ptrCast(recvbuf));
            const src_ptr = @as([*]const u8, @ptrCast(sendbuf));
            if (dst_ptr != src_ptr) {
                @memcpy(dst_ptr[0..size], src_ptr[0..size]);
            }
            return;
        }

        if (self.collectives) |*c| {
            try c.reduceScatter(sendbuf, recvbuf, recvcount, dtype, op, self.rank);
        }
    }

    pub fn broadcast(
        self: *ProcessGroup,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: nccl_mod.NcclDataType,
        root: usize,
    ) !void {
        if (self.world_size == 1) {
            const elem_size: usize = switch (dtype) {
                .int8, .uint8 => 1,
                .fp16, .bf16 => 2,
                .int32, .uint32, .fp32 => 4,
                .int64, .uint64, .fp64 => 8,
            };
            const size = count * elem_size;
            const dst_ptr = @as([*]u8, @ptrCast(recvbuf));
            const src_ptr = @as([*]const u8, @ptrCast(sendbuf));
            if (dst_ptr != src_ptr) {
                @memcpy(dst_ptr[0..size], src_ptr[0..size]);
            }
            return;
        }

        if (self.collectives) |*c| {
            try c.broadcast(sendbuf, recvbuf, count, dtype, @intCast(root), self.rank);
        }
    }

    pub fn barrier(self: *ProcessGroup) !void {
        if (self.world_size == 1) return;
        var send_dummy: f32 = 0;
        var recv_dummy: f32 = 0;
        try self.allReduce(@ptrCast(&send_dummy), @ptrCast(&recv_dummy), 1, .fp32, .sum);
        try self.stream.synchronize();
    }
};

pub const DistributedRuntime = struct {
    rank: usize,
    world_size: usize,
    device_ids: []usize,
    process_groups: []*ProcessGroup,
    streams: []*cuda_mod.CudaStream,
    devices: []cuda_mod.CudaDevice,
    allocator: std.mem.Allocator,
    initialized: bool,

    pub fn initSingleGPU(allocator: std.mem.Allocator) !DistributedRuntime {
        var cuda_init = try cuda_mod.CudaInit.init(allocator);
        defer cuda_init.deinit();

        if (cuda_init.device_count == 0) {
            return error.NoGpuAvailable;
        }

        const device_id: usize = 0;

        var device_ids = try allocator.alloc(usize, 1);
        errdefer allocator.free(device_ids);
        device_ids[0] = device_id;

        var devices = try allocator.alloc(cuda_mod.CudaDevice, 1);
        errdefer allocator.free(devices);
        devices[0] = try cuda_mod.CudaDevice.init(0);

        var stream = try allocator.create(cuda_mod.CudaStream);
        errdefer allocator.destroy(stream);
        stream.* = try cuda_mod.CudaStream.init(@intCast(device_id));

        var streams = try allocator.alloc(*cuda_mod.CudaStream, 1);
        errdefer allocator.free(streams);
        streams[0] = stream;

        var pg = try allocator.create(ProcessGroup);
        errdefer allocator.destroy(pg);
        pg.* = try ProcessGroup.init(allocator, 0, 1, device_id, stream);

        var pgs = try allocator.alloc(*ProcessGroup, 1);
        errdefer allocator.free(pgs);
        pgs[0] = pg;

        return .{
            .rank = 0,
            .world_size = 1,
            .device_ids = device_ids,
            .process_groups = pgs,
            .streams = streams,
            .devices = devices,
            .allocator = allocator,
            .initialized = true,
        };
    }

    pub fn init(allocator: std.mem.Allocator, cfg: config.RuntimeConfig) !DistributedRuntime {
        var cuda_init = try cuda_mod.CudaInit.init(allocator);
        defer cuda_init.deinit();

        const world_size = cfg.world_size;
        if (world_size == 0) {
            return error.InvalidWorldSize;
        }

        if (@as(usize, cuda_init.device_count) < world_size) {
            std.log.warn("Requested {d} GPUs but only {d} available", .{ world_size, cuda_init.device_count });
            return error.NotEnoughGpus;
        }

        var device_ids = try allocator.alloc(usize, world_size);
        errdefer allocator.free(device_ids);

        var devices = try allocator.alloc(cuda_mod.CudaDevice, world_size);
        errdefer allocator.free(devices);

        var streams = try allocator.alloc(*cuda_mod.CudaStream, world_size);
        errdefer {
            for (streams[0..world_size]) |s| {
                s.deinit() catch {};
                allocator.destroy(s);
            }
            allocator.free(streams);
        }

        var pgs = try allocator.alloc(*ProcessGroup, world_size);
        errdefer {
            for (pgs[0..world_size]) |p| {
                p.deinit();
                allocator.destroy(p);
            }
            allocator.free(pgs);
        }

        var created_streams: usize = 0;
        var created_pgs: usize = 0;

        errdefer {
            for (0..created_pgs) |i| {
                pgs[i].deinit();
                allocator.destroy(pgs[i]);
            }
            for (0..created_streams) |i| {
                streams[i].deinit() catch {};
                allocator.destroy(streams[i]);
            }
        }

        for (0..world_size) |i| {
            device_ids[i] = i;
            devices[i] = try cuda_mod.CudaDevice.init(@intCast(i));

            const s = try allocator.create(cuda_mod.CudaStream);
            s.* = try cuda_mod.CudaStream.init(@intCast(i));
            streams[i] = s;
            created_streams += 1;

            const p = try allocator.create(ProcessGroup);
            p.* = try ProcessGroup.init(allocator, i, world_size, i, s);
            pgs[i] = p;
            created_pgs += 1;
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
            stream.deinit() catch {};
            self.allocator.destroy(stream);
        }
        self.allocator.free(self.streams);

        self.allocator.free(self.devices);
        self.allocator.free(self.device_ids);

        self.initialized = false;
    }

    pub fn getProcessGroup(self: *DistributedRuntime, rank: usize) ?*ProcessGroup {
        if (rank >= self.process_groups.len) return null;
        return self.process_groups[rank];
    }

    pub fn currentProcessGroup(self: *DistributedRuntime) *ProcessGroup {
        return self.process_groups[self.rank];
    }

    pub fn getStream(self: *DistributedRuntime, rank: usize) ?*cuda_mod.CudaStream {
        if (rank >= self.streams.len) return null;
        return self.streams[rank];
    }

    pub fn currentStream(self: *DistributedRuntime) *cuda_mod.CudaStream {
        return self.streams[self.rank];
    }

    pub fn synchronize(self: *DistributedRuntime) !void {
        for (self.streams) |stream| {
            try stream.synchronize();
        }
    }

    pub fn getMemoryInfo(self: *DistributedRuntime) ![]struct { free: usize, total: usize } {
        var info = try self.allocator.alloc(struct { free: usize, total: usize }, self.world_size);
        errdefer self.allocator.free(info);

        for (0..self.world_size) |i| {
            try cuda_mod.checkCudaError(cuda_mod.cuda_set_device(@intCast(i)));
            info[i] = try cuda_mod.getMemoryInfo();
        }

        return info;
    }

    pub fn freeMemoryInfo(self: *DistributedRuntime, info: []struct { free: usize, total: usize }) void {
        self.allocator.free(info);
    }

    pub fn allBlackwell(self: *DistributedRuntime) bool {
        for (self.devices) |*device| {
            if (!device.isBlackwell()) return false;
        }
        return true;
    }

    pub fn enablePeerAccess(self: *DistributedRuntime) !void {
        for (0..self.world_size) |i| {
            for (i + 1..self.world_size) |j| {
                try cuda_mod.enablePeerAccess(@intCast(i), @intCast(j));
            }
        }
    }
};

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
        var cpus = try allocator.dupe(usize, &[_]usize{0});
        errdefer allocator.free(cpus);

        var gpus = try allocator.alloc(usize, num_gpus);
        errdefer allocator.free(gpus);

        for (0..num_gpus) |i| {
            gpus[i] = i;
        }

        var numa_nodes = try allocator.alloc(NumaNode, 1);
        errdefer allocator.free(numa_nodes);

        numa_nodes[0] = .{
            .id = 0,
            .cpus = cpus,
            .gpus = gpus,
            .memory_size = 0,
        };

        var gpu_affinity = try allocator.alloc(GpuAffinity, num_gpus);
        errdefer allocator.free(gpu_affinity);

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
        for (self.numa_nodes) |node| {
            self.allocator.free(node.cpus);
            self.allocator.free(node.gpus);
        }
        self.allocator.free(self.numa_nodes);
        self.allocator.free(self.gpu_affinity);
    }

    pub fn getGpuForNuma(self: *Topology, numa_node: usize) ?usize {
        for (self.gpu_affinity) |affinity| {
            if (affinity.numa_node == numa_node) {
                return affinity.gpu_id;
            }
        }
        return null;
    }

    pub fn getAllGpusForNuma(self: *Topology, numa_node: usize, allocator: std.mem.Allocator) ![]usize {
        var count: usize = 0;
        for (self.gpu_affinity) |affinity| {
            if (affinity.numa_node == numa_node) {
                count += 1;
            }
        }

        var result = try allocator.alloc(usize, count);
        var idx: usize = 0;
        for (self.gpu_affinity) |affinity| {
            if (affinity.numa_node == numa_node) {
                result[idx] = affinity.gpu_id;
                idx += 1;
            }
        }

        return result;
    }
};
