const std = @import("std");
const dtype_mod = @import("../tensor/dtype.zig");

pub const DType = dtype_mod.DType;

pub const CudaError = error{
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    NoDevice,
    InvalidDevice,
    InvalidImage,
    InvalidContext,
    ContextAlreadyCurrent,
    MapFailed,
    UnmapFailed,
    ArrayIsMapped,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    EccUncorrectable,
    UnsupportedLimit,
    ContextAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPtx,
    InvalidGraphicsContext,
    InvalidSource,
    FileNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidHandle,
    NotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PrimaryContextActive,
    ContextIsDestroyed,
    Assert,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    OperatingSystemCallFailed,
    StreamCaptureUnsupported,
    StreamCaptureInvalidated,
    StreamCaptureMerge,
    StreamCaptureUnmatched,
    StreamCaptureUnjoined,
    StreamCaptureFailure,
    StreamCaptureIsolation,
    StreamCaptureImplicit,
    CapturedEvent,
    StreamCaptureWrongThread,
    Timeout,
    GraphExecUpdateFailure,
    ExternalDevice,
    InvalidClusterSize,
    Unknown,
    CudaNotAvailable,
    NoCudaDevices,
};

fn checkCudaError(err: u32) CudaError!void {
    return switch (err) {
        0 => {},
        1 => CudaError.InvalidValue,
        2 => CudaError.OutOfMemory,
        3 => CudaError.NotInitialized,
        4 => CudaError.Deinitialized,
        5 => CudaError.ProfilerDisabled,
        6 => CudaError.ProfilerNotInitialized,
        7 => CudaError.ProfilerAlreadyStarted,
        8 => CudaError.ProfilerAlreadyStopped,
        100 => CudaError.NoDevice,
        101 => CudaError.InvalidDevice,
        200 => CudaError.InvalidImage,
        201 => CudaError.InvalidContext,
        202 => CudaError.ContextAlreadyCurrent,
        205 => CudaError.MapFailed,
        206 => CudaError.UnmapFailed,
        207 => CudaError.ArrayIsMapped,
        208 => CudaError.AlreadyAcquired,
        209 => CudaError.NotMapped,
        210 => CudaError.NotMappedAsArray,
        211 => CudaError.NotMappedAsPointer,
        214 => CudaError.EccUncorrectable,
        215 => CudaError.UnsupportedLimit,
        216 => CudaError.ContextAlreadyInUse,
        217 => CudaError.PeerAccessUnsupported,
        218 => CudaError.InvalidPtx,
        219 => CudaError.InvalidGraphicsContext,
        220 => CudaError.InvalidSource,
        301 => CudaError.FileNotFound,
        302 => CudaError.SharedObjectInitFailed,
        304 => CudaError.OperatingSystem,
        400 => CudaError.InvalidHandle,
        500 => CudaError.NotFound,
        600 => CudaError.NotReady,
        700 => CudaError.IllegalAddress,
        701 => CudaError.LaunchOutOfResources,
        702 => CudaError.LaunchTimeout,
        703 => CudaError.LaunchIncompatibleTexturing,
        704 => CudaError.PeerAccessAlreadyEnabled,
        705 => CudaError.PeerAccessNotEnabled,
        708 => CudaError.PrimaryContextActive,
        709 => CudaError.ContextIsDestroyed,
        710 => CudaError.Assert,
        711 => CudaError.TooManyPeers,
        712 => CudaError.HostMemoryAlreadyRegistered,
        713 => CudaError.HostMemoryNotRegistered,
        715 => CudaError.OperatingSystemCallFailed,
        900 => CudaError.StreamCaptureUnsupported,
        901 => CudaError.StreamCaptureInvalidated,
        902 => CudaError.StreamCaptureMerge,
        903 => CudaError.StreamCaptureUnmatched,
        904 => CudaError.StreamCaptureUnjoined,
        905 => CudaError.StreamCaptureFailure,
        906 => CudaError.StreamCaptureIsolation,
        907 => CudaError.StreamCaptureImplicit,
        908 => CudaError.CapturedEvent,
        909 => CudaError.StreamCaptureWrongThread,
        910 => CudaError.Timeout,
        911 => CudaError.GraphExecUpdateFailure,
        912 => CudaError.ExternalDevice,
        913 => CudaError.InvalidClusterSize,
        else => CudaError.Unknown,
    };
}

pub extern "cuda" fn cuda_malloc(ptr: **anyopaque, size: usize) callconv(.C) u32;
pub extern "cuda" fn cuda_free(ptr: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_memcpy(dst: *anyopaque, src: *const anyopaque, size: usize, kind: u32) callconv(.C) u32;
pub extern "cuda" fn cuda_memcpy_async(dst: *anyopaque, src: *const anyopaque, size: usize, kind: u32, stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_memset_raw(ptr: *anyopaque, value: u32, size: usize) callconv(.C) u32;
pub extern "cuda" fn cuda_memset_async(ptr: *anyopaque, value: u32, size: usize, stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_get_device_count(count: *u32) callconv(.C) u32;
pub extern "cuda" fn cuda_set_device(device: u32) callconv(.C) u32;
pub extern "cuda" fn cuda_get_device(device: *u32) callconv(.C) u32;
pub extern "cuda" fn cuda_device_synchronize() callconv(.C) u32;
pub extern "cuda" fn cuda_device_reset() callconv(.C) u32;
pub extern "cuda" fn cuda_get_device_properties(props: *DeviceProp, device: u32) callconv(.C) u32;
pub extern "cuda" fn cuda_stream_create(stream: **anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_stream_create_with_flags(stream: **anyopaque, flags: u32) callconv(.C) u32;
pub extern "cuda" fn cuda_stream_destroy(stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_stream_synchronize(stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_event_create(event: **anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_event_create_with_flags(event: **anyopaque, flags: u32) callconv(.C) u32;
pub extern "cuda" fn cuda_event_destroy(event: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_event_record(event: *anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_event_synchronize(event: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_event_elapsed_time(ms: *f32, start: *anyopaque, end: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_malloc_host(ptr: **anyopaque, size: usize) callconv(.C) u32;
pub extern "cuda" fn cuda_free_host(ptr: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_host_register(ptr: *anyopaque, size: usize, flags: u32) callconv(.C) u32;
pub extern "cuda" fn cuda_host_unregister(ptr: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_get_last_error() callconv(.C) u32;
pub extern "cuda" fn cuda_get_error_string(err: u32) callconv(.C) [*:0]const u8;
pub extern "cuda" fn cuda_mem_get_info(free: *usize, total: *usize) callconv(.C) u32;
pub extern "cuda" fn cuda_pointer_get_attributes(attrs: *PointerAttributes, ptr: *const anyopaque) callconv(.C) u32;
pub extern "cuda" fn cuda_device_can_access_peer(can_access: *u32, device: u32, peer_device: u32) callconv(.C) u32;
pub extern "cuda" fn cuda_device_enable_peer_access(peer_device: u32, flags: u32) callconv(.C) u32;
pub extern "cuda" fn cuda_device_disable_peer_access(peer_device: u32) callconv(.C) u32;

pub const MEMCPY_HOST_TO_HOST: u32 = 0;
pub const MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub const MEMCPY_DEVICE_TO_HOST: u32 = 2;
pub const MEMCPY_DEVICE_TO_DEVICE: u32 = 3;
pub const MEMCPY_DEFAULT: u32 = 4;

pub const STREAM_DEFAULT: u32 = 0;
pub const STREAM_NON_BLOCKING: u32 = 1;

pub const EVENT_DEFAULT: u32 = 0;
pub const EVENT_BLOCKING_SYNC: u32 = 1;
pub const EVENT_DISABLE_TIMING: u32 = 2;
pub const EVENT_INTERPROCESS: u32 = 4;

pub const HOST_REGISTER_DEFAULT: u32 = 0;
pub const HOST_REGISTER_PORTABLE: u32 = 1;
pub const HOST_REGISTER_MAPPED: u32 = 2;
pub const HOST_REGISTER_IO_MEMORY: u32 = 4;

pub const DeviceProp = extern struct {
    name: [256]u8,
    uuid: [16]u8,
    luid: [8]u8,
    luidDeviceNodeMask: u32,
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: u32,
    warpSize: u32,
    memPitch: usize,
    maxThreadsPerBlock: u32,
    maxThreadsDim: [3]u32,
    maxGridSize: [3]u32,
    clockRate: u32,
    totalConstMem: usize,
    major: u32,
    minor: u32,
    textureAlignment: usize,
    texturePitchAlignment: usize,
    deviceOverlap: u32,
    multiProcessorCount: u32,
    kernelExecTimeoutEnabled: u32,
    integrated: u32,
    canMapHostMemory: u32,
    computeMode: u32,
    maxTexture1D: u32,
    maxTexture1DMipmap: u32,
    maxTexture1DLinear: u32,
    maxTexture2D: [2]u32,
    maxTexture2DMipmap: [2]u32,
    maxTexture2DLinear: [3]u32,
    maxTexture3D: [3]u32,
    maxTexture3DAlt: [3]u32,
    maxTextureCubemap: u32,
    maxTexture1DLayered: [2]u32,
    maxTexture2DLayered: [3]u32,
    maxTextureCubemapLayered: [2]u32,
    maxSurface1D: u32,
    maxSurface2D: [2]u32,
    maxSurface3D: [3]u32,
    maxSurface1DLayered: [2]u32,
    maxSurface2DLayered: [3]u32,
    maxSurfaceCubemap: u32,
    maxSurfaceCubemapLayered: [2]u32,
    surfaceAlignment: usize,
    concurrentKernels: u32,
    ECCEnabled: u32,
    pciBusID: u32,
    pciDeviceID: u32,
    pciDomainID: u32,
    tccDriver: u32,
    asyncEngineCount: u32,
    unifiedAddressing: u32,
    memoryClockRate: u32,
    memoryBusWidth: u32,
    l2CacheSize: u32,
    persistingL2CacheMaxSize: u32,
    maxThreadsPerMultiProcessor: u32,
    streamPrioritiesSupported: u32,
    globalL1CacheSupported: u32,
    localL1CacheSupported: u32,
    sharedMemPerMultiprocessor: usize,
    regsPerMultiprocessor: u32,
    managedMemory: u32,
    isMultiGpuBoard: u32,
    multiGpuBoardGroupID: u32,
    hostNativeAtomicSupported: u32,
    singleToDoublePrecisionPerfRatio: u32,
    pageableMemoryAccess: u32,
    concurrentManagedAccess: u32,
    computePreemptionSupported: u32,
    canUseHostPointerForRegisteredMem: u32,
    cooperativeLaunch: u32,
    cooperativeMultiDeviceLaunch: u32,
    sharedMemPerBlockOptin: usize,
    pageableMemoryAccessUsesHostPageTables: u32,
    directManagedMemAccessFromHost: u32,
    maxBlocksPerMultiProcessor: u32,
    accessPolicyMaxWindowSize: u32,
    reservedSharedMemPerBlock: usize,
};

pub const PointerAttributes = extern struct {
    type: u32,
    device: u32,
    devicePointer: ?*anyopaque,
    hostPointer: ?*anyopaque,
    allocationSize: usize,
    managed: u32,
};

pub const CudaStream = struct {
    stream: *anyopaque,
    device_id: u32,

    pub fn init(device_id: u32) CudaError!CudaStream {
        try checkCudaError(cuda_set_device(device_id));

        var stream: *anyopaque = undefined;
        try checkCudaError(cuda_stream_create_with_flags(&stream, STREAM_NON_BLOCKING));

        return .{
            .stream = stream,
            .device_id = device_id,
        };
    }

    pub fn deinit(self: *CudaStream) CudaError!void {
        try checkCudaError(cuda_stream_destroy(self.stream));
    }

    pub fn synchronize(self: *CudaStream) CudaError!void {
        try checkCudaError(cuda_stream_synchronize(self.stream));
    }
};

pub const CudaEvent = struct {
    event: *anyopaque,

    pub fn init() CudaError!CudaEvent {
        var event: *anyopaque = undefined;
        try checkCudaError(cuda_event_create_with_flags(&event, EVENT_DISABLE_TIMING));

        return .{ .event = event };
    }

    pub fn initWithTiming() CudaError!CudaEvent {
        var event: *anyopaque = undefined;
        try checkCudaError(cuda_event_create(&event));

        return .{ .event = event };
    }

    pub fn deinit(self: *CudaEvent) CudaError!void {
        try checkCudaError(cuda_event_destroy(self.event));
    }

    pub fn record(self: *CudaEvent, stream: *CudaStream) CudaError!void {
        try checkCudaError(cuda_event_record(self.event, stream.stream));
    }

    pub fn synchronize(self: *CudaEvent) CudaError!void {
        try checkCudaError(cuda_event_synchronize(self.event));
    }

    pub fn elapsedTime(self: *CudaEvent, start: *CudaEvent) CudaError!f32 {
        var ms: f32 = 0;
        try checkCudaError(cuda_event_elapsed_time(&ms, start.event, self.event));
        return ms;
    }
};

pub const CudaDevice = struct {
    device_id: u32,
    props: DeviceProp,

    pub fn init(device_id: u32) CudaError!CudaDevice {
        try checkCudaError(cuda_set_device(device_id));

        var props: DeviceProp = undefined;
        try checkCudaError(cuda_get_device_properties(&props, device_id));

        return .{
            .device_id = device_id,
            .props = props,
        };
    }

    pub fn getName(self: *CudaDevice, allocator: std.mem.Allocator) ![]const u8 {
        const name_len = std.mem.indexOfScalar(u8, &self.props.name, 0) orelse self.props.name.len;
        return allocator.dupe(u8, self.props.name[0..name_len]);
    }

    pub fn totalMemory(self: *CudaDevice) usize {
        return self.props.totalGlobalMem;
    }

    pub fn freeMemory(self: *CudaDevice) CudaError!usize {
        try checkCudaError(cuda_set_device(self.device_id));
        var free_bytes: usize = undefined;
        var total_bytes: usize = undefined;
        try checkCudaError(cuda_mem_get_info(&free_bytes, &total_bytes));
        return free_bytes;
    }

    pub fn computeCapability(self: *CudaDevice) struct { major: u32, minor: u32 } {
        return .{
            .major = self.props.major,
            .minor = self.props.minor,
        };
    }

    pub fn isBlackwell(self: *CudaDevice) bool {
        return self.props.major >= 10;
    }

    pub fn synchronize(self: *CudaDevice) CudaError!void {
        try checkCudaError(cuda_set_device(self.device_id));
        try checkCudaError(cuda_device_synchronize());
    }
};

pub const CudaInit = struct {
    device_count: u32,
    devices: []CudaDevice,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CudaError!CudaInit {
        var device_count: u32 = undefined;
        try checkCudaError(cuda_get_device_count(&device_count));

        if (device_count == 0) {
            return CudaError.NoCudaDevices;
        }

        var devices = allocator.alloc(CudaDevice, device_count) catch return CudaError.OutOfMemory;
        errdefer allocator.free(devices);

        for (0..device_count) |i| {
            devices[i] = try CudaDevice.init(@intCast(i));
        }

        return .{
            .device_count = device_count,
            .devices = devices,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CudaInit) void {
        self.allocator.free(self.devices);
    }

    pub fn selectBestDevice(self: *CudaInit) u32 {
        var best_id: u32 = 0;
        var best_mem: usize = 0;

        for (self.devices, 0..) |*device, i| {
            const mem = device.totalMemory();
            if (mem > best_mem) {
                best_mem = mem;
                best_id = @intCast(i);
            }
        }

        return best_id;
    }
};

pub fn initCUDA(allocator: std.mem.Allocator) CudaError!CudaInit {
    return CudaInit.init(allocator);
}

pub fn allocateDevice(size: usize) CudaError!*anyopaque {
    var ptr: *anyopaque = undefined;
    try checkCudaError(cuda_malloc(&ptr, size));
    return ptr;
}

pub fn freeDevice(ptr: *anyopaque) CudaError!void {
    try checkCudaError(cuda_free(ptr));
}

pub fn memsetDevice(ptr: *anyopaque, value: u32, size: usize) CudaError!void {
    try checkCudaError(cuda_memset_raw(ptr, value, size));
}

pub fn copyHostToDevice(dst: *anyopaque, src: *const anyopaque, size: usize) CudaError!void {
    try checkCudaError(cuda_memcpy(dst, src, size, MEMCPY_HOST_TO_DEVICE));
}

pub fn copyDeviceToHost(dst: *anyopaque, src: *const anyopaque, size: usize) CudaError!void {
    try checkCudaError(cuda_memcpy(dst, src, size, MEMCPY_DEVICE_TO_HOST));
}

pub fn copyDeviceToDevice(dst: *anyopaque, src: *const anyopaque, size: usize) CudaError!void {
    try checkCudaError(cuda_memcpy(dst, src, size, MEMCPY_DEVICE_TO_DEVICE));
}

pub fn allocatePinned(size: usize) CudaError!*anyopaque {
    var ptr: *anyopaque = undefined;
    try checkCudaError(cuda_malloc_host(&ptr, size));
    return ptr;
}

pub fn freePinned(ptr: *anyopaque) CudaError!void {
    try checkCudaError(cuda_free_host(ptr));
}

pub fn enablePeerAccess(device_a: u32, device_b: u32) CudaError!void {
    var can_access: u32 = undefined;
    try checkCudaError(cuda_device_can_access_peer(&can_access, device_a, device_b));

    if (can_access == 1) {
        try checkCudaError(cuda_set_device(device_a));
        try checkCudaError(cuda_device_enable_peer_access(device_b, 0));
        try checkCudaError(cuda_set_device(device_b));
        try checkCudaError(cuda_device_enable_peer_access(device_a, 0));
    }
}

pub fn getMemoryInfo() CudaError!struct { free: usize, total: usize } {
    var free: usize = undefined;
    var total: usize = undefined;
    try checkCudaError(cuda_mem_get_info(&free, &total));
    return .{ .free = free, .total = total };
}

pub fn checkLastError() CudaError!void {
    const err = cuda_get_last_error();
    if (err != 0) {
        const str = cuda_get_error_string(err);
        std.log.err("CUDA error: {s}", .{str});
        try checkCudaError(err);
    }
}
