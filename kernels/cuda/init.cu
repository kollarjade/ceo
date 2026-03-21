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

inline bool dtype_is_supported(efla_dtype_t dtype) {
    switch (dtype) {
        case EFLA_DTYPE_FLOAT16:
        case EFLA_DTYPE_BFLOAT16:
        case EFLA_DTYPE_FLOAT32:
        case EFLA_DTYPE_FLOAT64:
        case EFLA_DTYPE_INT32:
        case EFLA_DTYPE_INT64:
        case EFLA_DTYPE_UINT8:
        case EFLA_DTYPE_INT8:
            return true;
        default:
            return false;
    }
}

inline size_t element_size_from_dtype(efla_dtype_t dtype) {
    switch (dtype) {
        case EFLA_DTYPE_FLOAT16:
            return sizeof(__half);
        case EFLA_DTYPE_BFLOAT16:
            return sizeof(__nv_bfloat16);
        case EFLA_DTYPE_FLOAT32:
            return sizeof(float);
        case EFLA_DTYPE_FLOAT64:
            return sizeof(double);
        case EFLA_DTYPE_INT32:
            return sizeof(int32_t);
        case EFLA_DTYPE_INT64:
            return sizeof(int64_t);
        case EFLA_DTYPE_UINT8:
            return sizeof(uint8_t);
        case EFLA_DTYPE_INT8:
            return sizeof(int8_t);
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
__device__ __forceinline__ float scalar_to_float<double>(double value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename Dst, typename Src>
__device__ __forceinline__ Dst cast_value(Src value) {
    if constexpr (std::is_same_v<Dst, __half>) {
        return __float2half_rn(scalar_to_float<Src>(value));
    } else if constexpr (std::is_same_v<Dst, __nv_bfloat16>) {
        return __float2bfloat16(scalar_to_float<Src>(value));
    } else if constexpr (std::is_same_v<Src, __half>) {
        return static_cast<Dst>(__half2float(value));
    } else if constexpr (std::is_same_v<Src, __nv_bfloat16>) {
        return static_cast<Dst>(__bfloat162float(value));
    } else {
        return static_cast<Dst>(value);
    }
}

template <typename T>
__global__ void fill_kernel(T* ptr, float value, size_t numel) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < numel;
         idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x)) {
        ptr[idx] = cast_value<T>(value);
    }
}

template <typename Src, typename Dst>
__global__ void cast_kernel(const Src* src, Dst* dst, size_t numel) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < numel;
         idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x)) {
        dst[idx] = cast_value<Dst>(src[idx]);
    }
}

template <typename T>
__global__ void store_cast_scalar_kernel(const float* src, T* dst) {
    dst[0] = cast_value<T>(src[0]);
}

__global__ void sqrt_scalar_kernel(float* value) {
    value[0] = sqrtf(value[0]);
}

template <typename T, bool Square, bool AbsValue>
struct TransformToFloat {
    __device__ __forceinline__ float operator()(const T& value) const {
        float x = scalar_to_float<T>(value);
        if constexpr (AbsValue) {
            x = fabsf(x);
        }
        if constexpr (Square) {
            x = x * x;
        }
        return x;
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

template <typename T, bool Square, bool AbsValue>
cudaError_t reduce_sum_to_float_cuda_impl(const T* input, float* output, size_t numel, cudaStream_t stream) {
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (numel == 0) {
        return cudaMemsetAsync(output, 0, sizeof(float), stream);
    }
    if (input == nullptr) {
        return cudaErrorInvalidValue;
    }

    using Iterator = cub::TransformInputIterator<float, TransformToFloat<T, Square, AbsValue>, const T*>;
    Iterator iterator(input, TransformToFloat<T, Square, AbsValue>());

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
cudaError_t reduce_sum_typed_cuda_impl(const T* input, T* output, size_t numel, cudaStream_t stream) {
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (numel == 0) {
        return cudaMemsetAsync(output, 0, sizeof(T), stream);
    }
    if (input == nullptr) {
        return cudaErrorInvalidValue;
    }

    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaError_t err = cub::DeviceReduce::Sum(nullptr, workspace_bytes, input, output, numel, stream);
    if (err != cudaSuccess) {
        return err;
    }

    err = allocate_workspace(&workspace, workspace_bytes, stream);
    if (err != cudaSuccess) {
        return err;
    }

    err = cub::DeviceReduce::Sum(workspace, workspace_bytes, input, output, numel, stream);
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
cudaError_t reduce_sum_half_like_cuda_impl(const T* input, T* output, size_t numel, cudaStream_t stream) {
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    float* temp_output = nullptr;
    cudaError_t err = allocate_workspace(reinterpret_cast<void**>(&temp_output), sizeof(float), stream);
    if (err != cudaSuccess) {
        return err;
    }

    err = reduce_sum_to_float_cuda_impl<T, false, false>(input, temp_output, numel, stream);
    if (err != cudaSuccess) {
        free_workspace(temp_output, stream);
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
cudaError_t reduce_max_abs_to_float_cuda_impl(const T* input, float* output, size_t numel, cudaStream_t stream) {
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (numel == 0) {
        return cudaMemsetAsync(output, 0, sizeof(float), stream);
    }
    if (input == nullptr) {
        return cudaErrorInvalidValue;
    }

    using Iterator = cub::TransformInputIterator<float, TransformToFloat<T, false, true>, const T*>;
    Iterator iterator(input, TransformToFloat<T, false, true>());

    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    cudaError_t err = cub::DeviceReduce::Max(nullptr, workspace_bytes, iterator, output, numel, stream);
    if (err != cudaSuccess) {
        return err;
    }

    err = allocate_workspace(&workspace, workspace_bytes, stream);
    if (err != cudaSuccess) {
        return err;
    }

    err = cub::DeviceReduce::Max(workspace, workspace_bytes, iterator, output, numel, stream);
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
    return reduce_sum_typed_cuda_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), numel, stream);
}

template <>
cudaError_t launch_sum_impl<__half>(const void* input, void* output, size_t numel, cudaStream_t stream) {
    return reduce_sum_half_like_cuda_impl<__half>(static_cast<const __half*>(input), static_cast<__half*>(output), numel, stream);
}

template <>
cudaError_t launch_sum_impl<__nv_bfloat16>(const void* input, void* output, size_t numel, cudaStream_t stream) {
    return reduce_sum_half_like_cuda_impl<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(input), static_cast<__nv_bfloat16*>(output), numel, stream);
}

template <typename T>
cudaError_t launch_norm_impl(const void* input, float* output, size_t numel, efla_norm_kind_t norm_kind, cudaStream_t stream) {
    switch (norm_kind) {
        case EFLA_NORM_KIND_L1:
            return reduce_sum_to_float_cuda_impl<T, false, true>(static_cast<const T*>(input), output, numel, stream);
        case EFLA_NORM_KIND_L2: {
            cudaError_t err = reduce_sum_to_float_cuda_impl<T, true, false>(static_cast<const T*>(input), output, numel, stream);
            if (err != cudaSuccess || numel == 0) {
                return err;
            }
            sqrt_scalar_kernel<<<1, 1, 0, stream>>>(output);
            return cudaPeekAtLastError();
        }
        case EFLA_NORM_KIND_INF:
            return reduce_max_abs_to_float_cuda_impl<T>(static_cast<const T*>(input), output, numel, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

template <typename T>
cudaError_t dispatch_output_copy(float* device_value, float* output, efla_memory_kind_t output_memory_kind, cudaStream_t stream) {
    (void)sizeof(T);
    switch (output_memory_kind) {
        case EFLA_MEMORY_KIND_DEVICE:
        case EFLA_MEMORY_KIND_MANAGED:
            return cudaSuccess;
        case EFLA_MEMORY_KIND_HOST: {
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                return err;
            }
            return cudaMemcpy(output, device_value, sizeof(float), cudaMemcpyDeviceToHost);
        }
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t launch_fill_by_dtype(void* ptr, float value, size_t numel, efla_dtype_t dtype, cudaStream_t stream) {
    switch (dtype) {
        case EFLA_DTYPE_FLOAT16:
            return launch_fill_impl<__half>(ptr, value, numel, stream);
        case EFLA_DTYPE_BFLOAT16:
            return launch_fill_impl<__nv_bfloat16>(ptr, value, numel, stream);
        case EFLA_DTYPE_FLOAT32:
            return launch_fill_impl<float>(ptr, value, numel, stream);
        case EFLA_DTYPE_FLOAT64:
            return launch_fill_impl<double>(ptr, value, numel, stream);
        case EFLA_DTYPE_INT32:
            return launch_fill_impl<int32_t>(ptr, value, numel, stream);
        case EFLA_DTYPE_INT64:
            return launch_fill_impl<int64_t>(ptr, value, numel, stream);
        case EFLA_DTYPE_UINT8:
            return launch_fill_impl<uint8_t>(ptr, value, numel, stream);
        case EFLA_DTYPE_INT8:
            return launch_fill_impl<int8_t>(ptr, value, numel, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

template <typename Src>
cudaError_t launch_cast_dst_dispatch(const void* src, void* dst, size_t numel, efla_dtype_t dst_dtype, cudaStream_t stream) {
    switch (dst_dtype) {
        case EFLA_DTYPE_FLOAT16:
            return launch_cast_impl<Src, __half>(src, dst, numel, stream);
        case EFLA_DTYPE_BFLOAT16:
            return launch_cast_impl<Src, __nv_bfloat16>(src, dst, numel, stream);
        case EFLA_DTYPE_FLOAT32:
            return launch_cast_impl<Src, float>(src, dst, numel, stream);
        case EFLA_DTYPE_FLOAT64:
            return launch_cast_impl<Src, double>(src, dst, numel, stream);
        case EFLA_DTYPE_INT32:
            return launch_cast_impl<Src, int32_t>(src, dst, numel, stream);
        case EFLA_DTYPE_INT64:
            return launch_cast_impl<Src, int64_t>(src, dst, numel, stream);
        case EFLA_DTYPE_UINT8:
            return launch_cast_impl<Src, uint8_t>(src, dst, numel, stream);
        case EFLA_DTYPE_INT8:
            return launch_cast_impl<Src, int8_t>(src, dst, numel, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t launch_cast_by_dtype(const void* src, void* dst, size_t numel, efla_dtype_t src_dtype, efla_dtype_t dst_dtype, cudaStream_t stream) {
    switch (src_dtype) {
        case EFLA_DTYPE_FLOAT16:
            return launch_cast_dst_dispatch<__half>(src, dst, numel, dst_dtype, stream);
        case EFLA_DTYPE_BFLOAT16:
            return launch_cast_dst_dispatch<__nv_bfloat16>(src, dst, numel, dst_dtype, stream);
        case EFLA_DTYPE_FLOAT32:
            return launch_cast_dst_dispatch<float>(src, dst, numel, dst_dtype, stream);
        case EFLA_DTYPE_FLOAT64:
            return launch_cast_dst_dispatch<double>(src, dst, numel, dst_dtype, stream);
        case EFLA_DTYPE_INT32:
            return launch_cast_dst_dispatch<int32_t>(src, dst, numel, dst_dtype, stream);
        case EFLA_DTYPE_INT64:
            return launch_cast_dst_dispatch<int64_t>(src, dst, numel, dst_dtype, stream);
        case EFLA_DTYPE_UINT8:
            return launch_cast_dst_dispatch<uint8_t>(src, dst, numel, dst_dtype, stream);
        case EFLA_DTYPE_INT8:
            return launch_cast_dst_dispatch<int8_t>(src, dst, numel, dst_dtype, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t launch_sum_by_dtype(const void* input, void* output, size_t numel, efla_dtype_t dtype, cudaStream_t stream) {
    switch (dtype) {
        case EFLA_DTYPE_FLOAT16:
            return launch_sum_impl<__half>(input, output, numel, stream);
        case EFLA_DTYPE_BFLOAT16:
            return launch_sum_impl<__nv_bfloat16>(input, output, numel, stream);
        case EFLA_DTYPE_FLOAT32:
            return launch_sum_impl<float>(input, output, numel, stream);
        case EFLA_DTYPE_FLOAT64:
            return launch_sum_impl<double>(input, output, numel, stream);
        case EFLA_DTYPE_INT32:
            return launch_sum_impl<int32_t>(input, output, numel, stream);
        case EFLA_DTYPE_INT64:
            return launch_sum_impl<int64_t>(input, output, numel, stream);
        case EFLA_DTYPE_UINT8:
            return launch_sum_impl<uint8_t>(input, output, numel, stream);
        case EFLA_DTYPE_INT8:
            return launch_sum_impl<int8_t>(input, output, numel, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t launch_norm_by_dtype(const void* input, float* output, size_t numel, efla_dtype_t dtype, efla_norm_kind_t norm_kind, cudaStream_t stream) {
    switch (dtype) {
        case EFLA_DTYPE_FLOAT16:
            return launch_norm_impl<__half>(input, output, numel, norm_kind, stream);
        case EFLA_DTYPE_BFLOAT16:
            return launch_norm_impl<__nv_bfloat16>(input, output, numel, norm_kind, stream);
        case EFLA_DTYPE_FLOAT32:
            return launch_norm_impl<float>(input, output, numel, norm_kind, stream);
        case EFLA_DTYPE_FLOAT64:
            return launch_norm_impl<double>(input, output, numel, norm_kind, stream);
        case EFLA_DTYPE_INT32:
            return launch_norm_impl<int32_t>(input, output, numel, norm_kind, stream);
        case EFLA_DTYPE_INT64:
            return launch_norm_impl<int64_t>(input, output, numel, norm_kind, stream);
        case EFLA_DTYPE_UINT8:
            return launch_norm_impl<uint8_t>(input, output, numel, norm_kind, stream);
        case EFLA_DTYPE_INT8:
            return launch_norm_impl<int8_t>(input, output, numel, norm_kind, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}

extern "C" {

cudaError_t fill_cuda(
    void* ptr,
    float value,
    size_t numel,
    efla_dtype_t dtype,
    cudaStream_t stream
) {
    if (!dtype_is_supported(dtype)) {
        return cudaErrorInvalidValue;
    }
    return launch_fill_by_dtype(ptr, value, numel, dtype, stream);
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
    efla_dtype_t src_dtype,
    efla_dtype_t dst_dtype,
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

    return launch_cast_by_dtype(src, dst, numel, src_dtype, dst_dtype, stream);
}

cudaError_t sum_reduce_cuda(
    const void* input,
    void* output,
    size_t numel,
    efla_dtype_t dtype,
    cudaStream_t stream
) {
    if (!dtype_is_supported(dtype)) {
        return cudaErrorInvalidValue;
    }
    if (output == nullptr) {
        return cudaErrorInvalidValue;
    }
    return launch_sum_by_dtype(input, output, numel, dtype, stream);
}

cudaError_t norm_cuda(
    const void* input,
    float* output,
    size_t numel,
    efla_dtype_t dtype,
    efla_norm_kind_t norm_kind,
    efla_memory_kind_t output_memory_kind,
    cudaStream_t stream
) {
    if (!dtype_is_supported(dtype) || output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (norm_kind != EFLA_NORM_KIND_L1 &&
        norm_kind != EFLA_NORM_KIND_L2 &&
        norm_kind != EFLA_NORM_KIND_INF) {
        return cudaErrorInvalidValue;
    }
    if (output_memory_kind != EFLA_MEMORY_KIND_HOST &&
        output_memory_kind != EFLA_MEMORY_KIND_DEVICE &&
        output_memory_kind != EFLA_MEMORY_KIND_MANAGED) {
        return cudaErrorInvalidValue;
    }

    float* device_output = output;
    bool need_temporary = output_memory_kind == EFLA_MEMORY_KIND_HOST;
    cudaError_t err = cudaSuccess;

    if (need_temporary) {
        err = allocate_workspace(reinterpret_cast<void**>(&device_output), sizeof(float), stream);
        if (err != cudaSuccess) {
            return err;
        }
    }

    err = launch_norm_by_dtype(input, device_output, numel, dtype, norm_kind, stream);
    if (err == cudaSuccess && need_temporary) {
        err = dispatch_output_copy<float>(device_output, output, output_memory_kind, stream);
    }

    cudaError_t free_err = need_temporary ? free_workspace(device_output, stream) : cudaSuccess;
    if (err != cudaSuccess) {
        return err;
    }
    if (free_err != cudaSuccess) {
        return free_err;
    }
    return cudaSuccess;
}

}
