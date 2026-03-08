const std = @import("std");
const dtype_mod = @import("../tensor/dtype.zig");

pub const DType = dtype_mod.DType;

/// CUDA error type
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
};

/// Convert CUDA error code to CudaError
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

// CUDA driver API function declarations (extern)
pub extern "cuda" fn cudaMalloc(ptr: **anyopaque, size: usize) callconv(.C) u32;
pub extern "cuda" fn cudaFree(ptr: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaMemcpy(dst: *anyopaque, src: *const anyopaque, size: usize, kind: u32) callconv(.C) u32;
pub extern "cuda" fn cudaMemcpyAsync(dst: *anyopaque, src: *const anyopaque, size: usize, kind: u32, stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaMemset(ptr: *anyopaque, value: u32, size: usize) callconv(.C) u32;
pub extern "cuda" fn cudaMemsetAsync(ptr: *anyopaque, value: u32, size: usize, stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaGetDeviceCount(count: *u32) callconv(.C) u32;
pub extern "cuda" fn cudaSetDevice(device: u32) callconv(.C) u32;
pub extern "cuda" fn cudaGetDevice(device: *u32) callconv(.C) u32;
pub extern "cuda" fn cudaDeviceSynchronize() callconv(.C) u32;
pub extern "cuda" fn cudaDeviceReset() callconv(.C) u32;
pub extern "cuda" fn cudaGetDeviceProperties(props: *DeviceProp, device: u32) callconv(.C) u32;
pub extern "cuda" fn cudaStreamCreate(stream: **anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaStreamCreateWithFlags(stream: **anyopaque, flags: u32) callconv(.C) u32;
pub extern "cuda" fn cudaStreamDestroy(stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaStreamSynchronize(stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaEventCreate(event: **anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaEventCreateWithFlags(event: **anyopaque, flags: u32) callconv(.C) u32;
pub extern "cuda" fn cudaEventDestroy(event: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaEventRecord(event: *anyopaque, stream: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaEventSynchronize(event: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaEventElapsedTime(ms: *f32, start: *anyopaque, end: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaMallocHost(ptr: **anyopaque, size: usize) callconv(.C) u32;
pub extern "cuda" fn cudaFreeHost(ptr: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaHostRegister(ptr: *anyopaque, size: usize, flags: u32) callconv(.C) u32;
pub extern "cuda" fn cudaHostUnregister(ptr: *anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaGetLastError() callconv(.C) u32;
pub extern "cuda" fn cudaGetErrorString(err: u32) callconv(.C) [*:0]const u8;
pub extern "cuda" fn cudaMemGetInfo(free: *usize, total: *usize) callconv(.C) u32;
pub extern "cuda" fn cudaPointerGetAttributes(attrs: *PointerAttributes, ptr: *const anyopaque) callconv(.C) u32;
pub extern "cuda" fn cudaDeviceCanAccessPeer(can_access: *u32, device: u32, peer_device: u32) callconv(.C) u32;
pub extern "cuda" fn cudaDeviceEnablePeerAccess(peer_device: u32, flags: u32) callconv(.C) u32;
pub extern "cuda" fn cudaDeviceDisablePeerAccess(peer_device: u32) callconv(.C) u32;

// cudaMemcpy kinds
pub const cudaMemcpyHostToHost: u32 = 0;
pub const cudaMemcpyHostToDevice: u32 = 1;
pub const cudaMemcpyDeviceToHost: u32 = 2;
pub const cudaMemcpyDeviceToDevice: u32 = 3;
pub const cudaMemcpyDefault: u32 = 4;

// Stream flags
pub const cudaStreamDefault: u32 = 0;
pub const cudaStreamNonBlocking: u32 = 1;

// Event flags
pub const cudaEventDefault: u32 = 0;
pub const cudaEventBlockingSync: u32 = 1;
pub const cudaEventDisableTiming: u32 = 2;
pub const cudaEventInterprocess: u32 = 4;

// Host register flags
pub const cudaHostRegisterDefault: u32 = 0;
pub const cudaHostRegisterPortable: u32 = 1;
pub const cudaHostRegisterMapped: u32 = 2;
pub const cudaHostRegisterIoMemory: u32 = 4;

/// CUDA device properties
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

/// Pointer attributes
pub const PointerAttributes = extern struct {
    type: u32,
    device: u32,
    devicePointer: ?*anyopaque,
    hostPointer: ?*anyopaque,
    allocationSize: usize,
    managed: u32,
};

/// CUDA stream wrapper
pub const CudaStream = struct {
    stream: *anyopaque,
    device_id: u32,

    pub fn init(device_id: u32) !CudaStream {
        try checkCudaError(cudaSetDevice(device_id));

        var stream: *anyopaque = undefined;
        try checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        return .{
            .stream = stream,
            .device_id = device_id,
        };
    }

    pub fn deinit(self: *CudaStream) void {
        _ = cudaStreamDestroy(self.stream);
    }

    pub fn synchronize(self: *CudaStream) !void {
        try checkCudaError(cudaStreamSynchronize(self.stream));
    }
};

/// CUDA event wrapper
pub const CudaEvent = struct {
    event: *anyopaque,

    pub fn init() !CudaEvent {
        var event: *anyopaque = undefined;
        try checkCudaError(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

        return .{ .event = event };
    }

    pub fn initWithTiming() !CudaEvent {
        var event: *anyopaque = undefined;
        try checkCudaError(cudaEventCreate(&event));

        return .{ .event = event };
    }

    pub fn deinit(self: *CudaEvent) void {
        _ = cudaEventDestroy(self.event);
    }

    pub fn record(self: *CudaEvent, stream: *CudaStream) !void {
        try checkCudaError(cudaEventRecord(self.event, stream.stream));
    }

    pub fn synchronize(self: *CudaEvent) !void {
        try checkCudaError(cudaEventSynchronize(self.event));
    }

    pub fn elapsedTime(self: *CudaEvent, start: *CudaEvent) !f32 {
        var ms: f32 = 0;
        try checkCudaError(cudaEventElapsedTime(&ms, start.event, self.event));
        return ms;
    }
};

/// CUDA device management
pub const CudaDevice = struct {
    device_id: u32,
    props: DeviceProp,

    pub fn init(device_id: u32) !CudaDevice {
        try checkCudaError(cudaSetDevice(device_id));

        var props: DeviceProp = undefined;
        try checkCudaError(cudaGetDeviceProperties(&props, device_id));

        return .{
            .device_id = device_id,
            .props = props,
        };
    }

    pub fn setName(self: *CudaDevice, allocator: std.mem.Allocator) ![]const u8 {
        const name_len = std.mem.indexOfScalar(u8, &self.props.name, 0) orelse self.props.name.len;
        return allocator.dupe(u8, self.props.name[0..name_len]);
    }

    pub fn totalMemory(self: *CudaDevice) usize {
        return self.props.totalGlobalMem;
    }

    pub fn freeMemory(self: *CudaDevice) !usize {
        var free: usize = undefined;
        var total: usize = undefined;
        try checkCudaError(cudaMemGetInfo(&free, &total));
        return free;
    }

    pub fn computeCapability(self: *CudaDevice) !struct { major: u32, minor: u32 } {
        return .{
            .major = self.props.major,
            .minor = self.props.minor,
        };
    }

    pub fn isBlackwell(self: *CudaDevice) bool {
        // Blackwell SM100 is compute capability 10.x
        return self.props.major >= 10;
    }

    pub fn synchronize(self: *CudaDevice) !void {
        try checkCudaError(cudaSetDevice(self.device_id));
        try checkCudaError(cudaDeviceSynchronize());
    }
};

/// CUDA initialization helper
pub const CudaInit = struct {
    device_count: u32,
    devices: []CudaDevice,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !CudaInit {
        var device_count: u32 = undefined;
        const err = cudaGetDeviceCount(&device_count);

        if (err != 0) {
            return error.CudaNotAvailable;
        }

        if (device_count == 0) {
            return error.NoCudaDevices;
        }

        var devices = try allocator.alloc(CudaDevice, device_count);
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

    pub fn selectBestDevice(self: *CudaInit) !u32 {
        var best_id: u32 = 0;
        var best_mem: usize = 0;

        for (self.devices, 0..) |device, i| {
            const mem = device.totalMemory();
            if (mem > best_mem) {
                best_mem = mem;
                best_id = @intCast(i);
            }
        }

        return best_id;
    }
};

/// High-level CUDA functions
pub fn initCUDA() !CudaInit {
    return CudaInit.init(std.heap.page_allocator);
}

pub fn cudaMalloc(size: usize) CudaError!*anyopaque {
    var ptr: *anyopaque = undefined;
    try checkCudaError(@import("std").mem.zeroInit(u32, .{}));
    const err = @extern(*const fn (**anyopaque, usize) callconv(.C) u32, .{ .name = "cudaMalloc" })(&ptr, size);
    try checkCudaError(err);
    return ptr;
}

pub fn cudaFree(ptr: *anyopaque) CudaError!void {
    try checkCudaError(cudaFree(ptr));
}

pub fn cudaMemset(ptr: *anyopaque, value: u32, size: usize) CudaError!void {
    try checkCudaError(@extern(*const fn (*anyopaque, u32, usize) callconv(.C) u32, .{ .name = "cudaMemset" })(ptr, value, size));
}

pub fn cudaCopyHostToDevice(dst: *anyopaque, src: *const anyopaque, size: usize) CudaError!void {
    try checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

pub fn cudaCopyDeviceToHost(dst: *anyopaque, src: *const anyopaque, size: usize) CudaError!void {
    try checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

pub fn cudaCopyDeviceToDevice(dst: *anyopaque, src: *const anyopaque, size: usize, device_id: i32) CudaError!void {
    _ = device_id;
    try checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

pub fn cudaFill(ptr: *anyopaque, dtype: DType, value: f64, numel: usize) CudaError!void {
    // This would call a CUDA kernel for filling
    _ = ptr;
    _ = dtype;
    _ = value;
    _ = numel;
}

pub fn cudaCast(src: ?*anyopaque, dst: ?*anyopaque, src_dtype: DType, dst_dtype: DType, numel: usize) CudaError!void {
    // This would call a CUDA kernel for type casting
    _ = src;
    _ = dst;
    _ = src_dtype;
    _ = dst_dtype;
    _ = numel;
}

/// Pinned memory allocation
pub fn cudaMallocHost(size: usize) CudaError!*anyopaque {
    var ptr: *anyopaque = undefined;
    try checkCudaError(cudaMallocHost(&ptr, size));
    return ptr;
}

pub fn cudaFreeHost(ptr: *anyopaque) CudaError!void {
    try checkCudaError(cudaFreeHost(ptr));
}

/// Enable peer access between GPUs
pub fn enablePeerAccess(device_a: u32, device_b: u32) CudaError!void {
    var can_access: u32 = undefined;
    try checkCudaError(cudaDeviceCanAccessPeer(&can_access, device_a, device_b));

    if (can_access == 1) {
        try checkCudaError(cudaSetDevice(device_a));
        try checkCudaError(cudaDeviceEnablePeerAccess(device_b, 0));
    }
}

/// Get memory info
pub fn getMemoryInfo() CudaError!struct { free: usize, total: usize } {
    var free: usize = undefined;
    var total: usize = undefined;
    try checkCudaError(cudaMemGetInfo(&free, &total));
    return .{ .free = free, .total = total };
}

/// Check for last CUDA error
pub fn checkLastError() CudaError!void {
    const err = cudaGetLastError();
    if (err != 0) {
        const str = cudaGetErrorString(err);
        std.log.err("CUDA error: {s}", .{str});
        try checkCudaError(err);
    }
}
