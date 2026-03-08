const std = @import("std");
const CudaError = @import("cuda.zig").CudaError;

/// NCCL error type
pub const NcclError = error{
    Unknown,
    InvalidArgument,
    UsageError,
    RemoteError,
    InternalError,
    InvalidDevicePointer,
    InvalidRank,
    UnsupportedDeviceCount,
    DeviceLimitReached,
    InvalidDeviceIndex,
    LibNotFoundError,
    CudaError,
    SystemError,
    NumTypes,
};

/// NCCL unique ID
pub const NcclUniqueId = extern struct {
    internal: [128]u8,
};

/// NCCL communicator
pub const NcclComm = extern struct {
    handle: ?*anyopaque,
    rank: i32,
    size: i32,
    device: i32,

    pub fn init(rank: i32, size: i32, device: i32) NcclComm {
        return .{
            .handle = null,
            .rank = rank,
            .size = size,
            .device = device,
        };
    }
};

/// NCCL data types
pub const NcclDataType = enum(u32) {
    int8 = 0,
    uint8 = 1,
    int32 = 2,
    uint32 = 3,
    int64 = 4,
    uint64 = 5,
    fp16 = 6,
    fp32 = 7,
    fp64 = 8,
    bf16 = 9,

    pub fn fromDType(dtype: @import("../tensor/dtype.zig").DType) NcclDataType {
        return switch (dtype) {
            .int8 => .int8,
            .int32 => .int32,
            .int64 => .int64,
            .fp16 => .fp16,
            .fp32 => .fp32,
            .bf16 => .bf16,
            else => .fp32, // Default fallback
        };
    }
};

/// NCCL reduction operations
pub const NcclRedOp = enum(u32) {
    sum = 0,
    prod = 1,
    max = 2,
    min = 3,
    avg = 4,
};

// External NCCL function declarations
pub extern "nccl" fn ncclGetUniqueId(uniqueId: *NcclUniqueId) callconv(.C) u32;
pub extern "nccl" fn ncclCommInitRank(comm: *?*anyopaque, nranks: i32, uniqueId: NcclUniqueId, rank: i32) callconv(.C) u32;
pub extern "nccl" fn ncclCommInitAll(comms: [*]?*anyopaque, ndev: i32, devices: [*]const i32) callconv(.C) u32;
pub extern "nccl" fn ncclCommDestroy(comm: ?*anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclCommAbort(comm: ?*anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclCommCuDevice(comm: ?*anyopaque) callconv(.C) i32;
pub extern "nccl" fn ncclCommUserRank(comm: ?*anyopaque) callconv(.C) i32;
pub extern "nccl" fn ncclCommCount(comm: ?*anyopaque) callconv(.C) i32;
pub extern "nccl" fn ncclAllReduce(sendbuff: *const anyopaque, recvbuff: *anyopaque, count: usize, datatype: NcclDataType, op: NcclRedOp, comm: ?*anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclReduce(sendbuff: *const anyopaque, recvbuff: *anyopaque, count: usize, datatype: NcclDataType, op: NcclRedOp, root: i32, comm: ?*anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclBroadcast(sendbuff: *const anyopaque, recvbuff: *anyopaque, count: usize, datatype: NcclDataType, root: i32, comm: ?*anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclReduceScatter(sendbuff: *const anyopaque, recvbuff: *anyopaque, recvcount: usize, datatype: NcclDataType, op: NcclRedOp, comm: ?*anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclAllGather(sendbuff: *const anyopaque, recvbuff: *anyopaque, sendcount: usize, datatype: NcclDataType, comm: ?*anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclSend(sendbuff: *const anyopaque, count: usize, datatype: NcclDataType, peer: i32, comm: ?*anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclRecv(recvbuff: *anyopaque, count: usize, datatype: NcclDataType, peer: i32, comm: ?*anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclAlltoAll(sendbuff: *const anyopaque, recvbuff: *anyopaque, count: usize, datatype: NcclDataType, comm: ?*anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "nccl" fn ncclGetErrorString(err: u32) callconv(.C) [*:0]const u8;
pub extern "nccl" fn ncclGetVersion(version: *i32) callconv(.C) u32;

/// Convert NCCL error code to NcclError
fn checkNcclError(err: u32) NcclError!void {
    return switch (err) {
        0 => {},
        1 => NcclError.Unknown,
        2 => NcclError.InvalidArgument,
        3 => NcclError.UsageError,
        4 => NcclError.RemoteError,
        5 => NcclError.InternalError,
        6 => NcclError.InvalidDevicePointer,
        7 => NcclError.InvalidRank,
        8 => NcclError.UnsupportedDeviceCount,
        9 => NcclError.DeviceLimitReached,
        10 => NcclError.InvalidDeviceIndex,
        11 => NcclError.LibNotFoundError,
        12 => NcclError.CudaError,
        13 => NcclError.SystemError,
        else => NcclError.Unknown,
    };
}

/// NCCL communicator group for managing multiple GPUs on a node
pub const NcclGroup = struct {
    comms: []?*anyopaque,
    devices: []i32,
    size: i32,
    unique_id: NcclUniqueId,
    allocator: std.mem.Allocator,
    initialized: bool,

    /// Initialize NCCL group for all GPUs on this node
    pub fn initAllDevices(allocator: std.mem.Allocator, num_devices: usize) !NcclGroup {
        var devices = try allocator.alloc(i32, num_devices);
        errdefer allocator.free(devices);

        for (0..num_devices) |i| {
            devices[i] = @intCast(i);
        }

        var comms = try allocator.alloc(?*anyopaque, num_devices);
        errdefer allocator.free(comms);

        @memset(comms, null);

        // Initialize all communicators
        try checkNcclError(ncclCommInitAll(comms.ptr, @intCast(num_devices), devices.ptr));

        var unique_id: NcclUniqueId = undefined;
        try checkNcclError(ncclGetUniqueId(&unique_id));

        return .{
            .comms = comms,
            .devices = devices,
            .size = @intCast(num_devices),
            .unique_id = unique_id,
            .allocator = allocator,
            .initialized = true,
        };
    }

    /// Initialize from unique ID (for multi-node)
    pub fn initFromUniqueId(
        allocator: std.mem.Allocator,
        unique_id: NcclUniqueId,
        rank: i32,
        world_size: i32,
        device: i32,
    ) !NcclGroup {
        var comms = try allocator.alloc(?*anyopaque, 1);
        errdefer allocator.free(comms);

        comms[0] = null;

        try checkNcclError(ncclCommInitRank(&comms[0], world_size, unique_id, rank));

        var devices = try allocator.alloc(i32, 1);
        devices[0] = device;

        return .{
            .comms = comms,
            .devices = devices,
            .size = 1,
            .unique_id = unique_id,
            .allocator = allocator,
            .initialized = true,
        };
    }

    pub fn deinit(self: *NcclGroup) void {
        if (self.initialized) {
            for (self.comms) |comm| {
                if (comm) |c| {
                    _ = ncclCommDestroy(c);
                }
            }
        }
        self.allocator.free(self.comms);
        self.allocator.free(self.devices);
    }

    pub fn getComm(self: *NcclGroup, rank: usize) ?*anyopaque {
        if (rank >= self.comms.len) return null;
        return self.comms[rank];
    }

    pub fn getRank(self: *NcclGroup, comm: ?*anyopaque) i32 {
        return ncclCommUserRank(comm);
    }

    pub fn getSize(self: *NcclGroup, comm: ?*anyopaque) i32 {
        return ncclCommCount(comm);
    }
};

/// NCCL collective operations wrapper
pub const NcclCollectives = struct {
    group: *NcclGroup,
    stream: *anyopaque,

    pub fn init(group: *NcclGroup, stream: *anyopaque) NcclCollectives {
        return .{
            .group = group,
            .stream = stream,
        };
    }

    /// All-reduce: sum across all ranks
    pub fn allReduce(
        self: *NcclCollectives,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: NcclDataType,
        op: NcclRedOp,
        rank: usize,
    ) NcclError!void {
        const comm = self.group.getComm(rank) orelse return NcclError.InvalidRank;
        try checkNcclError(ncclAllReduce(sendbuf, recvbuf, count, dtype, op, comm, self.stream));
    }

    /// All-reduce sum
    pub fn allReduceSum(
        self: *NcclCollectives,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: NcclDataType,
        rank: usize,
    ) NcclError!void {
        return self.allReduce(sendbuf, recvbuf, count, dtype, .sum, rank);
    }

    /// Reduce to root
    pub fn reduce(
        self: *NcclCollectives,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: NcclDataType,
        op: NcclRedOp,
        root: i32,
        rank: usize,
    ) NcclError!void {
        const comm = self.group.getComm(rank) orelse return NcclError.InvalidRank;
        try checkNcclError(ncclReduce(sendbuf, recvbuf, count, dtype, op, root, comm, self.stream));
    }

    /// Broadcast from root
    pub fn broadcast(
        self: *NcclCollectives,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: NcclDataType,
        root: i32,
        rank: usize,
    ) NcclError!void {
        const comm = self.group.getComm(rank) orelse return NcclError.InvalidRank;
        try checkNcclError(ncclBroadcast(sendbuf, recvbuf, count, dtype, root, comm, self.stream));
    }

    /// Reduce-scatter
    pub fn reduceScatter(
        self: *NcclCollectives,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        recvcount: usize,
        dtype: NcclDataType,
        op: NcclRedOp,
        rank: usize,
    ) NcclError!void {
        const comm = self.group.getComm(rank) orelse return NcclError.InvalidRank;
        try checkNcclError(ncclReduceScatter(sendbuf, recvbuf, recvcount, dtype, op, comm, self.stream));
    }

    /// All-gather
    pub fn allGather(
        self: *NcclCollectives,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        sendcount: usize,
        dtype: NcclDataType,
        rank: usize,
    ) NcclError!void {
        const comm = self.group.getComm(rank) orelse return NcclError.InvalidRank;
        try checkNcclError(ncclAllGather(sendbuf, recvbuf, sendcount, dtype, comm, self.stream));
    }

    /// Send to peer
    pub fn send(
        self: *NcclCollectives,
        sendbuf: *const anyopaque,
        count: usize,
        dtype: NcclDataType,
        peer: i32,
        rank: usize,
    ) NcclError!void {
        const comm = self.group.getComm(rank) orelse return NcclError.InvalidRank;
        try checkNcclError(ncclSend(sendbuf, count, dtype, peer, comm, self.stream));
    }

    /// Receive from peer
    pub fn recv(
        self: *NcclCollectives,
        recvbuf: *anyopaque,
        count: usize,
        dtype: NcclDataType,
        peer: i32,
        rank: usize,
    ) NcclError!void {
        const comm = self.group.getComm(rank) orelse return NcclError.InvalidRank;
        try checkNcclError(ncclRecv(recvbuf, count, dtype, peer, comm, self.stream));
    }

    /// All-to-all
    pub fn alltoAll(
        self: *NcclCollectives,
        sendbuf: *const anyopaque,
        recvbuf: *anyopaque,
        count: usize,
        dtype: NcclDataType,
        rank: usize,
    ) NcclError!void {
        const comm = self.group.getComm(rank) orelse return NcclError.InvalidRank;
        try checkNcclError(ncclAlltoAll(sendbuf, recvbuf, count, dtype, comm, self.stream));
    }
};

/// NCCL version info
pub fn getNcclVersion() !struct { major: i32, minor: i32, patch: i32 } {
    var version: i32 = undefined;
    try checkNcclError(ncclGetVersion(&version));

    const major = version / 1000;
    const minor = (version % 1000) / 100;
    const patch = version % 100;

    return .{ .major = major, .minor = minor, .patch = patch };
}

/// Generate unique ID for multi-node setup
pub fn generateUniqueId() !NcclUniqueId {
    var id: NcclUniqueId = undefined;
    try checkNcclError(ncclGetUniqueId(&id));
    return id;
}

test "NCCL types" {
    const dtype = @import("../tensor/dtype.zig").DType;
    try std.testing.expectEqual(NcclDataType.fp32, NcclDataType.fromDType(dtype.fp32));
    try std.testing.expectEqual(NcclDataType.fp16, NcclDataType.fromDType(dtype.fp16));
    try std.testing.expectEqual(NcclDataType.bf16, NcclDataType.fromDType(dtype.bf16));
}
