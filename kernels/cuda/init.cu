#include "include/kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cmath>

namespace {

constexpr int kDTypeFp32 = 0;
constexpr int kDTypeFp16 = 1;
constexpr int kDTypeBf16 = 2;
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

inline bool checked_mul_size_t(size_t a, size_t b, size_t& out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > static_cast<size_t>(-1) / b) {
        return false;
    }
    out = a * b;
    return true;
}

inline int grid_for_numel(size_t numel) {
    if (numel == 0) {
        return 0;
    }
    size_t blocks = (numel + static_cast<size_t>(kThreadsPerBlock) - 1) / static_cast<size_t>(kThreadsPerBlock);
    if (blocks > static_cast<size_t>(kMaxBlocks)) {
        blocks = static_cast<size_t>(kMaxBlocks);
    }
    return static_cast<int>(blocks);
}

inline bool dtype_is_supported(int dtype) {
    return dtype == kDTypeFp32 || dtype == kDTypeFp16 || dtype == kDTypeBf16;
}

inline size_t element_size_from_dtype(int dtype) {
    switch (dtype) {
        case kDTypeFp32:
            return sizeof(float);
        case kDTypeFp16:
            return sizeof(__half);
        case kDTypeBf16:
            return sizeof(__nv_bfloat16);
        default:
            return 0;
    }
}

template <typename T>
__device__ __forceinline__ float scalar_to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T float_to_scalar(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ float float_to_scalar<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __half float_to_scalar<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename Dst, typename Src>
__device__ __forceinline__ Dst convert_scalar(Src value) {
    if constexpr (std::is_same_v<Dst, Src>) {
        return value;
    } else {
        return float_to_scalar<Dst>(scalar_to_float<Src>(value));
    }
}

template <typename T>
__global__ void fill_kernel(T* ptr, float value, size_t numel) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < numel;
         idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x)) {
        ptr[idx] = float_to_scalar<T>(value);
    }
}

template <typename Src, typename Dst>
__global__ void cast_kernel(const Src* src, Dst* dst, size_t numel) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < numel;
         idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x)) {
        dst[idx] = convert_scalar<Dst>(src[idx]);
    }
}

template <typename T>
__global__ void store_cast_scalar_kernel(const float* src, T* dst) {
    dst[0] = float_to_scalar<T>(src[0]);
}

__global__ void sqrt_scalar_kernel(float* value) {
    value[0] = sqrtf(value[0]);
}

template <typename T, bool Square>
struct TransformToFloat {
    __device__ __forceinline__ float operator()(const T& value) const {
        const float x = scalar_to_float<T>(value);
        if constexpr (Square) {
            return x * x;
        } else {
            return x;
        }
    }
};

inline cudaError_t allocate_workspace(void** ptr, size_t bytes, cudaStream_t stream) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    *ptr = nullptr;
    if (bytes == 0) {
        return cudaSuccess;
    }
#if CUDART_VERSION >= 11020
    return cudaMallocAsync(ptr, bytes, stream);
#else
    (void)stream;
    return cudaMalloc(ptr, bytes);
#endif
}

inline cudaError_t free_workspace(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }
#if CUDART_VERSION >= 11020
    return cudaFreeAsync(ptr, stream);
#else
    (void)stream;
    return cudaFree(ptr);
#endif
}

template <typename T, bool Square>
cudaError_t reduce_to_float_cuda_impl(const T* input, float* output, size_t numel, cudaStream_t stream) {
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (numel == 0) {
        return cudaMemsetAsync(output, 0, sizeof(float), stream);
    }
    if (input == nullptr) {
        return cudaErrorInvalidValue;
    }

    using Iterator = cub::TransformInputIterator<float, TransformToFloat<T, Square>, const T*>;
    Iterator iterator(input, TransformToFloat<T, Square>());

    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    cudaError_t err = cub::DeviceReduce::Sum(nullptr, workspace_bytes, iterator, output, numel, stream);
    if (err != cudaSuccess) {
        return err;
    }

    err = allocate_workspace(&workspace, workspace_bytes, stream);
    if (err != cudaSuccess) {
        return err;
    }

    err = cub::DeviceReduce::Sum(workspace, workspace_bytes, iterator, output, numel, stream);
    cudaError_t free_err = free_workspace(workspace, stream);

    if (err != cudaSuccess) {
        return err;
    }
    if (free_err != cudaSuccess) {
        return free_err;
    }
    return cudaSuccess;
}

template <typename T>
cudaError_t reduce_to_typed_output_cuda_impl(const T* input, T* output, size_t numel, cudaStream_t stream) {
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (numel == 0) {
        return cudaMemsetAsync(output, 0, sizeof(T), stream);
    }
    if (input == nullptr) {
        return cudaErrorInvalidValue;
    }

    float* temp_output = nullptr;
    cudaError_t err = allocate_workspace(reinterpret_cast<void**>(&temp_output), sizeof(float), stream);
    if (err != cudaSuccess) {
        return err;
    }

    err = reduce_to_float_cuda_impl<T, false>(input, temp_output, numel, stream);
    if (err != cudaSuccess) {
        cudaError_t free_err = free_workspace(temp_output, stream);
        if (free_err != cudaSuccess) {
            return free_err;
        }
        return err;
    }

    store_cast_scalar_kernel<<<1, 1, 0, stream>>>(temp_output, output);
    err = cudaPeekAtLastError();
    cudaError_t free_err = free_workspace(temp_output, stream);

    if (err != cudaSuccess) {
        return err;
    }
    if (free_err != cudaSuccess) {
        return free_err;
    }
    return cudaSuccess;
}

template <typename T>
cudaError_t launch_fill_impl(void* ptr, float value, size_t numel, cudaStream_t stream) {
    if (numel == 0) {
        return cudaSuccess;
    }
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    const int blocks = grid_for_numel(numel);
    fill_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(static_cast<T*>(ptr), value, numel);
    return cudaPeekAtLastError();
}

template <typename Src, typename Dst>
cudaError_t launch_cast_impl(const void* src, void* dst, size_t numel, cudaStream_t stream) {
    if (numel == 0) {
        return cudaSuccess;
    }
    if (src == nullptr || dst == nullptr) {
        return cudaErrorInvalidValue;
    }
    const int blocks = grid_for_numel(numel);
    cast_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(static_cast<const Src*>(src), static_cast<Dst*>(dst), numel);
    return cudaPeekAtLastError();
}

template <typename T>
cudaError_t launch_sum_impl(const void* input, void* output, size_t numel, cudaStream_t stream) {
    return reduce_to_typed_output_cuda_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), numel, stream);
}

template <>
cudaError_t launch_sum_impl<float>(const void* input, void* output, size_t numel, cudaStream_t stream) {
    return reduce_to_float_cuda_impl<float, false>(static_cast<const float*>(input), static_cast<float*>(output), numel, stream);
}

template <typename T>
cudaError_t launch_norm_impl(const void* input, float* output, size_t numel, cudaStream_t stream) {
    cudaError_t err = reduce_to_float_cuda_impl<T, true>(static_cast<const T*>(input), output, numel, stream);
    if (err != cudaSuccess) {
        return err;
    }
    if (numel == 0) {
        return cudaSuccess;
    }
    sqrt_scalar_kernel<<<1, 1, 0, stream>>>(output);
    return cudaPeekAtLastError();
}

}

extern "C" {

cudaError_t fill_cuda(
    void* ptr,
    float value,
    size_t numel,
    int dtype,
    cudaStream_t stream
) {
    switch (dtype) {
        case kDTypeFp32:
            return launch_fill_impl<float>(ptr, value, numel, stream);
        case kDTypeFp16:
            return launch_fill_impl<__half>(ptr, value, numel, stream);
        case kDTypeBf16:
            return launch_fill_impl<__nv_bfloat16>(ptr, value, numel, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t copy_cuda(
    const void* src,
    void* dst,
    size_t numel,
    size_t element_size,
    cudaStream_t stream
) {
    if (element_size == 0) {
        return cudaErrorInvalidValue;
    }
    if (numel == 0) {
        return cudaSuccess;
    }
    if (src == nullptr || dst == nullptr) {
        return cudaErrorInvalidValue;
    }
    size_t bytes = 0;
    if (!checked_mul_size_t(numel, element_size, bytes)) {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}

cudaError_t cast_cuda(
    const void* src,
    void* dst,
    size_t numel,
    int src_dtype,
    int dst_dtype,
    cudaStream_t stream
) {
    if (!dtype_is_supported(src_dtype) || !dtype_is_supported(dst_dtype)) {
        return cudaErrorInvalidValue;
    }
    if (numel == 0) {
        return cudaSuccess;
    }
    if (src == nullptr || dst == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (src_dtype == dst_dtype) {
        if (src == dst) {
            return cudaSuccess;
        }
        return copy_cuda(src, dst, numel, element_size_from_dtype(src_dtype), stream);
    }

    if (src_dtype == kDTypeFp32 && dst_dtype == kDTypeFp16) {
        return launch_cast_impl<float, __half>(src, dst, numel, stream);
    }
    if (src_dtype == kDTypeFp32 && dst_dtype == kDTypeBf16) {
        return launch_cast_impl<float, __nv_bfloat16>(src, dst, numel, stream);
    }
    if (src_dtype == kDTypeFp16 && dst_dtype == kDTypeFp32) {
        return launch_cast_impl<__half, float>(src, dst, numel, stream);
    }
    if (src_dtype == kDTypeFp16 && dst_dtype == kDTypeBf16) {
        return launch_cast_impl<__half, __nv_bfloat16>(src, dst, numel, stream);
    }
    if (src_dtype == kDTypeBf16 && dst_dtype == kDTypeFp32) {
        return launch_cast_impl<__nv_bfloat16, float>(src, dst, numel, stream);
    }
    if (src_dtype == kDTypeBf16 && dst_dtype == kDTypeFp16) {
        return launch_cast_impl<__nv_bfloat16, __half>(src, dst, numel, stream);
    }

    return cudaErrorInvalidValue;
}

cudaError_t sum_reduce_cuda(
    const void* input,
    void* output,
    size_t numel,
    int dtype,
    cudaStream_t stream
) {
    if (!dtype_is_supported(dtype)) {
        return cudaErrorInvalidValue;
    }
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    switch (dtype) {
        case kDTypeFp32:
            return launch_sum_impl<float>(input, output, numel, stream);
        case kDTypeFp16:
            return launch_sum_impl<__half>(input, output, numel, stream);
        case kDTypeBf16:
            return launch_sum_impl<__nv_bfloat16>(input, output, numel, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t norm_cuda(
    const void* input,
    float* output,
    size_t numel,
    int dtype,
    cudaStream_t stream
) {
    if (!dtype_is_supported(dtype)) {
        return cudaErrorInvalidValue;
    }
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    switch (dtype) {
        case kDTypeFp32:
            return launch_norm_impl<float>(input, output, numel, stream);
        case kDTypeFp16:
            return launch_norm_impl<__half>(input, output, numel, stream);
        case kDTypeBf16:
            return launch_norm_impl<__nv_bfloat16>(input, output, numel, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}
