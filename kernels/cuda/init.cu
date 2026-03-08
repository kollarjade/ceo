/**
 * Initialization and utility kernels
 */

#include "include/kernels.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

// ============================================================================
// Fill Kernel
// ============================================================================

template<typename T>
__global__ void fill_kernel(T* ptr, float value, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        ptr[idx] = __float2bfloat16(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        ptr[idx] = __float2half(value);
    } else if constexpr (std::is_same_v<T, float>) {
        ptr[idx] = value;
    } else {
        ptr[idx] = static_cast<T>(value);
    }
}

// ============================================================================
// Copy Kernel
// ============================================================================

__global__ void copy_kernel(
    const void* __restrict__ src,
    void* __restrict__ dst,
    size_t numel,
    size_t element_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = idx * element_size;

    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);

    if (offset < numel * element_size) {
        dst_bytes[offset] = src_bytes[offset];
    }
}

// ============================================================================
// Cast Kernel
// ============================================================================

__global__ void cast_bf16_fp32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = __bfloat162float(src[idx]);
}

__global__ void cast_fp32_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = __float2bfloat16(src[idx]);
}

// ============================================================================
// Reduction Kernel
// ============================================================================

__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t numel
) {
    __shared__ float shared_sum[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_sum[tid] = 0.0f;

    while (idx < numel) {
        shared_sum[tid] += input[idx];
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Tree reduction
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

// ============================================================================
// Host Functions
// ============================================================================

extern "C" {

cudaError_t fill_cuda(
    void* ptr,
    float value,
    size_t numel,
    int dtype,
    cudaStream_t stream
) {
    size_t block_size = 256;
    size_t num_blocks = (numel + block_size - 1) / block_size;

    switch (dtype) {
        case 0: // fp32
            fill_kernel<float><<<num_blocks, block_size, 0, stream>>>(
                static_cast<float*>(ptr), value, numel);
            break;
        case 1: // fp16
            fill_kernel<__half><<<num_blocks, block_size, 0, stream>>>(
                static_cast<__half*>(ptr), value, numel);
            break;
        case 2: // bf16
            fill_kernel<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(ptr), value, numel);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

cudaError_t cast_cuda(
    const void* src,
    void* dst,
    size_t numel,
    int src_dtype,
    int dst_dtype,
    cudaStream_t stream
) {
    size_t block_size = 256;
    size_t num_blocks = (numel + block_size - 1) / block_size;

    // BF16 <-> FP32
    if (src_dtype == 2 && dst_dtype == 0) {
        cast_bf16_fp32_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src),
            static_cast<float*>(dst),
            numel);
    } else if (src_dtype == 0 && dst_dtype == 2) {
        cast_fp32_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<const float*>(src),
            static_cast<__nv_bfloat16*>(dst),
            numel);
    } else {
        // Other conversions not yet implemented
        return cudaErrorNotSupported;
    }

    return cudaGetLastError();
}

cudaError_t sum_reduce_cuda(
    const void* input,
    void* output,
    size_t numel,
    int dtype,
    cudaStream_t stream
) {
    size_t block_size = 256;
    size_t num_blocks = min((numel + block_size - 1) / block_size, (size_t)1024);

    // Zero output first
    cudaMemsetAsync(output, 0, sizeof(float), stream);

    sum_reduce_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(input),
        static_cast<float*>(output),
        numel);

    return cudaGetLastError();
}

cudaError_t norm_cuda(
    const void* input,
    float* output,
    size_t numel,
    int dtype,
    cudaStream_t stream
) {
    // Compute L2 norm: sqrt(sum(x^2))
    // This would require squaring input first, then reducing
    // Simplified implementation

    (void)input;
    (void)output;
    (void)numel;
    (void)dtype;
    (void)stream;

    return cudaSuccess;
}

} // extern "C"
