/**
 * Normalization and Activation Kernels
 * SM100-optimized implementations
 */

#include "include/kernels.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// RMSNorm Forward Kernel
// ============================================================================

__global__ void rmsnorm_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    size_t n_rows,
    size_t normalized_shape,
    float eps
) {
    size_t row_idx = blockIdx.x;

    if (row_idx >= n_rows) return;

    const __nv_bfloat16* row_in = input + row_idx * normalized_shape;
    __nv_bfloat16* row_out = output + row_idx * normalized_shape;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (size_t i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        sum_sq += val * val;
    }

    // Reduce within block
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();

    atomicAdd(&shared_sum, sum_sq);
    __syncthreads();

    float mean_sq = shared_sum / normalized_shape;
    float rsqrt = rsqrtf(mean_sq + eps);

    // Normalize and scale
    for (size_t i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        float w = __bfloat162float(weight[i]);
        row_out[i] = __float2bfloat16(val * rsqrt * w);
    }
}

// ============================================================================
// RMSNorm Backward Kernel
// ============================================================================

__global__ void rmsnorm_backward_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ grad_input,
    __nv_bfloat16* __restrict__ grad_weight,
    size_t n_rows,
    size_t normalized_shape,
    float eps
) {
    size_t row_idx = blockIdx.x;

    if (row_idx >= n_rows) return;

    const __nv_bfloat16* row_in = input + row_idx * normalized_shape;
    const __nv_bfloat16* row_grad = grad_output + row_idx * normalized_shape;
    __nv_bfloat16* row_grad_in = grad_input + row_idx * normalized_shape;

    // Recompute normalization stats
    float sum_sq = 0.0f;
    for (size_t i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        sum_sq += val * val;
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum_sq);
    __syncthreads();

    float mean_sq = shared_sum / normalized_shape;
    float rsqrt = rsqrtf(mean_sq + eps);

    // Compute sum of weighted gradients
    float sum_grad_weighted = 0.0f;
    for (size_t i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
        float g = __bfloat162float(row_grad[i]);
        float w = __bfloat162float(weight[i]);
        float x = __bfloat162float(row_in[i]);
        sum_grad_weighted += g * w * x;
    }

    __shared__ float shared_grad_sum;
    if (threadIdx.x == 0) shared_grad_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_grad_sum, sum_grad_weighted);
    __syncthreads();

    float norm_factor = rsqrt / normalized_shape;

    // Backprop
    for (size_t i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
        float g = __bfloat162float(row_grad[i]);
        float w = __bfloat162float(weight[i]);
        float x = __bfloat162float(row_in[i]);
        float grad = w * rsqrt * g - x * norm_factor * shared_grad_sum;
        row_grad_in[i] = __float2bfloat16(grad);

        // Accumulate weight gradient (if provided)
        if (grad_weight) {
            atomicAdd(
                reinterpret_cast<float*>(&grad_weight[i]),
                g * x * rsqrt
            );
        }
    }
}

// ============================================================================
// LayerNorm Forward Kernel
// ============================================================================

__global__ void layernorm_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    size_t n_rows,
    size_t normalized_shape,
    float eps
) {
    size_t row_idx = blockIdx.x;

    if (row_idx >= n_rows) return;

    const __nv_bfloat16* row_in = input + row_idx * normalized_shape;
    __nv_bfloat16* row_out = output + row_idx * normalized_shape;

    // Compute mean and variance
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (size_t i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        sum += val;
        sum_sq += val * val;
    }

    __shared__ float shared_sum;
    __shared__ float shared_sum_sq;
    if (threadIdx.x == 0) {
        shared_sum = 0.0f;
        shared_sum_sq = 0.0f;
    }
    __syncthreads();

    atomicAdd(&shared_sum, sum);
    atomicAdd(&shared_sum_sq, sum_sq);
    __syncthreads();

    float mean = shared_sum / normalized_shape;
    float variance = shared_sum_sq / normalized_shape - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    // Normalize
    for (size_t i = threadIdx.x; i < normalized_shape; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        float w = __bfloat162float(weight[i]);
        float b = bias ? __bfloat162float(bias[i]) : 0.0f;
        float normalized = (val - mean) * inv_std;
        row_out[i] = __float2bfloat16(normalized * w + b);
    }
}

// ============================================================================
// GELU Forward Kernel
// ============================================================================

__device__ __forceinline__ float gelu_approximate(float x) {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x3)));
}

__device__ __forceinline__ float gelu_exact(float x) {
    // x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    constexpr float sqrt2 = 1.41421356237f;
    return 0.5f * x * (1.0f + erff(x / sqrt2));
}

__global__ void gelu_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t numel,
    bool approximate
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numel) return;

    float x = __bfloat162float(input[idx]);
    float y = approximate ? gelu_approximate(x) : gelu_exact(x);
    output[idx] = __float2bfloat16(y);
}

__global__ void gelu_backward_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ grad_input,
    size_t numel,
    bool approximate
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numel) return;

    float x = __bfloat162float(input[idx]);
    float g = __bfloat162float(grad_output[idx]);

    float grad;
    if (approximate) {
        // Derivative of approximate GELU
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x3);
        float tanh_val = tanhf(tanh_arg);
        float sech_sq = 1.0f - tanh_val * tanh_val;
        float inner_deriv = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
        grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech_sq * inner_deriv;
    } else {
        // Derivative of exact GELU
        constexpr float sqrt2 = 1.41421356237f;
        float cdf = 0.5f * (1.0f + erff(x / sqrt2));
        float pdf = expf(-0.5f * x * x) * 0.3989422804f; // 1/sqrt(2*pi)
        grad = cdf + x * pdf;
    }

    grad_input[idx] = __float2bfloat16(g * grad);
}

// ============================================================================
// Softmax Forward Kernel
// ============================================================================

__global__ void softmax_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t outer_size,
    size_t dim_size,
    size_t inner_size
) {
    size_t outer_idx = blockIdx.x / inner_size;
    size_t inner_idx = blockIdx.x % inner_size;

    if (outer_idx >= outer_size) return;

    size_t offset = outer_idx * dim_size * inner_size + inner_idx;

    // Find max for numerical stability
    __shared__ float max_val;
    if (threadIdx.x == 0) max_val = -INFINITY;
    __syncthreads();

    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = __bfloat162float(input[offset + i * inner_size]);
        atomicMax(reinterpret_cast<int*>(&max_val), __float_as_int(val));
    }
    __syncthreads();

    // Compute exp and sum
    __shared__ float sum_exp;
    if (threadIdx.x == 0) sum_exp = 0.0f;
    __syncthreads();

    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = __bfloat162float(input[offset + i * inner_size]);
        float exp_val = expf(val - max_val);
        output[offset + i * inner_size] = __float2bfloat16(exp_val);
        atomicAdd(&sum_exp, exp_val);
    }
    __syncthreads();

    // Normalize
    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = __bfloat162float(output[offset + i * inner_size]);
        output[offset + i * inner_size] = __float2bfloat16(val / sum_exp);
    }
}

// ============================================================================
// Host Functions
// ============================================================================

extern "C" {

cudaError_t rmsnorm_forward_cuda(
    const void* input,
    const void* weight,
    void* output,
    size_t numel,
    size_t normalized_shape,
    float eps,
    cudaStream_t stream
) {
    size_t n_rows = numel / normalized_shape;
    size_t block_size = min(normalized_shape, (size_t)256);

    rmsnorm_forward_kernel<<<n_rows, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output),
        n_rows,
        normalized_shape,
        eps
    );

    return cudaGetLastError();
}

cudaError_t rmsnorm_backward_cuda(
    const void* grad_output,
    const void* input,
    const void* weight,
    void* grad_input,
    void* grad_weight,
    size_t numel,
    size_t normalized_shape,
    float eps,
    cudaStream_t stream
) {
    size_t n_rows = numel / normalized_shape;
    size_t block_size = min(normalized_shape, (size_t)256);

    rmsnorm_backward_kernel<<<n_rows, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(grad_output),
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(grad_input),
        static_cast<__nv_bfloat16*>(grad_weight),
        n_rows,
        normalized_shape,
        eps
    );

    return cudaGetLastError();
}

cudaError_t layernorm_forward_cuda(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    size_t numel,
    size_t normalized_shape,
    float eps,
    cudaStream_t stream
) {
    size_t n_rows = numel / normalized_shape;
    size_t block_size = min(normalized_shape, (size_t)256);

    layernorm_forward_kernel<<<n_rows, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(output),
        n_rows,
        normalized_shape,
        eps
    );

    return cudaGetLastError();
}

cudaError_t gelu_forward_cuda(
    const void* input,
    void* output,
    size_t numel,
    bool approximate,
    cudaStream_t stream
) {
    size_t block_size = 256;
    size_t num_blocks = (numel + block_size - 1) / block_size;

    gelu_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        numel,
        approximate
    );

    return cudaGetLastError();
}

cudaError_t gelu_backward_cuda(
    const void* grad_output,
    const void* input,
    void* grad_input,
    size_t numel,
    bool approximate,
    cudaStream_t stream
) {
    size_t block_size = 256;
    size_t num_blocks = (numel + block_size - 1) / block_size;

    gelu_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(grad_output),
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(grad_input),
        numel,
        approximate
    );

    return cudaGetLastError();
}

cudaError_t softmax_forward_cuda(
    const void* input,
    void* output,
    size_t outer_size,
    size_t dim_size,
    size_t inner_size,
    cudaStream_t stream
) {
    size_t num_blocks = outer_size * inner_size;
    size_t block_size = min(dim_size, (size_t)256);

    softmax_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        outer_size,
        dim_size,
        inner_size
    );

    return cudaGetLastError();
}

} // extern "C"
