#include "include/kernels.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int kBlockSize = 256;
constexpr unsigned int kMaxGridX = 65535u;

inline bool is_finite_scalar(float x) {
    return std::isfinite(static_cast<double>(x));
}

inline bool safe_mul_size(size_t a, size_t b, size_t* out) {
    if (out == nullptr) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    *out = a * b;
    return true;
}

inline size_t ceil_div_size(size_t a, size_t b) {
    return (a + b - 1) / b;
}

inline int launch_block_size(size_t work_items) {
    size_t block = std::min<size_t>(work_items, static_cast<size_t>(kBlockSize));
    if (block == 0) {
        block = 1;
    }
    return static_cast<int>(block);
}

inline int launch_grid_size_for_rows(size_t rows) {
    if (rows == 0) {
        return 0;
    }
    return static_cast<int>(std::min<size_t>(rows, static_cast<size_t>(kMaxGridX)));
}

inline int launch_grid_size_for_numel(size_t numel, int block_size) {
    if (numel == 0) {
        return 0;
    }
    size_t blocks = ceil_div_size(numel, static_cast<size_t>(block_size));
    blocks = std::min<size_t>(blocks, static_cast<size_t>(kMaxGridX));
    return static_cast<int>(blocks);
}

inline cudaError_t free_if_needed(void* ptr) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }
    return cudaFree(ptr);
}

__device__ __forceinline__ float gelu_approximate_device(float x) {
    constexpr float k = 0.7978845608028654f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x3)));
}

__device__ __forceinline__ float gelu_exact_device(float x) {
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__global__ void rmsnorm_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    size_t n_rows,
    size_t normalized_shape,
    float eps
) {
    __shared__ float shared[kBlockSize];
    for (size_t row_idx = static_cast<size_t>(blockIdx.x); row_idx < n_rows; row_idx += static_cast<size_t>(gridDim.x)) {
        const __nv_bfloat16* row_in = input + row_idx * normalized_shape;
        __nv_bfloat16* row_out = output + row_idx * normalized_shape;

        float local_sum_sq = 0.0f;
        for (size_t i = static_cast<size_t>(threadIdx.x); i < normalized_shape; i += static_cast<size_t>(blockDim.x)) {
            float x = __bfloat162float(row_in[i]);
            local_sum_sq += x * x;
        }

        shared[threadIdx.x] = local_sum_sq;
        __syncthreads();

        for (unsigned int stride = static_cast<unsigned int>(blockDim.x) >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float inv_rms = rsqrtf(shared[0] / static_cast<float>(normalized_shape) + eps);

        for (size_t i = static_cast<size_t>(threadIdx.x); i < normalized_shape; i += static_cast<size_t>(blockDim.x)) {
            float x = __bfloat162float(row_in[i]);
            float w = __bfloat162float(weight[i]);
            row_out[i] = __float2bfloat16_rn(x * inv_rms * w);
        }

        __syncthreads();
    }
}

__global__ void rmsnorm_backward_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ grad_input,
    float* __restrict__ grad_weight_accum,
    size_t n_rows,
    size_t normalized_shape,
    float eps
) {
    __shared__ float shared_sum_sq[kBlockSize];
    __shared__ float shared_dot[kBlockSize];

    for (size_t row_idx = static_cast<size_t>(blockIdx.x); row_idx < n_rows; row_idx += static_cast<size_t>(gridDim.x)) {
        const __nv_bfloat16* row_grad_out = grad_output + row_idx * normalized_shape;
        const __nv_bfloat16* row_in = input + row_idx * normalized_shape;
        __nv_bfloat16* row_grad_in = grad_input + row_idx * normalized_shape;

        float local_sum_sq = 0.0f;
        for (size_t i = static_cast<size_t>(threadIdx.x); i < normalized_shape; i += static_cast<size_t>(blockDim.x)) {
            float x = __bfloat162float(row_in[i]);
            local_sum_sq += x * x;
        }

        shared_sum_sq[threadIdx.x] = local_sum_sq;
        __syncthreads();

        for (unsigned int stride = static_cast<unsigned int>(blockDim.x) >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float inv_rms = rsqrtf(shared_sum_sq[0] / static_cast<float>(normalized_shape) + eps);

        float local_dot = 0.0f;
        for (size_t i = static_cast<size_t>(threadIdx.x); i < normalized_shape; i += static_cast<size_t>(blockDim.x)) {
            float g = __bfloat162float(row_grad_out[i]);
            float x = __bfloat162float(row_in[i]);
            float w = __bfloat162float(weight[i]);
            local_dot += g * w * x;
        }

        shared_dot[threadIdx.x] = local_dot;
        __syncthreads();

        for (unsigned int stride = static_cast<unsigned int>(blockDim.x) >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_dot[threadIdx.x] += shared_dot[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float coeff = shared_dot[0] * inv_rms * inv_rms * inv_rms / static_cast<float>(normalized_shape);

        for (size_t i = static_cast<size_t>(threadIdx.x); i < normalized_shape; i += static_cast<size_t>(blockDim.x)) {
            float g = __bfloat162float(row_grad_out[i]);
            float x = __bfloat162float(row_in[i]);
            float w = __bfloat162float(weight[i]);
            float dx = g * w * inv_rms - x * coeff;
            row_grad_in[i] = __float2bfloat16_rn(dx);
            if (grad_weight_accum != nullptr) {
                atomicAdd(&grad_weight_accum[i], g * x * inv_rms);
            }
        }

        __syncthreads();
    }
}

__global__ void layernorm_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    size_t n_rows,
    size_t normalized_shape,
    float eps
) {
    __shared__ float shared_sum[kBlockSize];
    __shared__ float shared_sum_sq[kBlockSize];

    for (size_t row_idx = static_cast<size_t>(blockIdx.x); row_idx < n_rows; row_idx += static_cast<size_t>(gridDim.x)) {
        const __nv_bfloat16* row_in = input + row_idx * normalized_shape;
        __nv_bfloat16* row_out = output + row_idx * normalized_shape;

        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        for (size_t i = static_cast<size_t>(threadIdx.x); i < normalized_shape; i += static_cast<size_t>(blockDim.x)) {
            float x = __bfloat162float(row_in[i]);
            local_sum += x;
            local_sum_sq += x * x;
        }

        shared_sum[threadIdx.x] = local_sum;
        shared_sum_sq[threadIdx.x] = local_sum_sq;
        __syncthreads();

        for (unsigned int stride = static_cast<unsigned int>(blockDim.x) >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
                shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float mean = shared_sum[0] / static_cast<float>(normalized_shape);
        float variance = shared_sum_sq[0] / static_cast<float>(normalized_shape) - mean * mean;
        variance = fmaxf(variance, 0.0f);
        float inv_std = rsqrtf(variance + eps);

        for (size_t i = static_cast<size_t>(threadIdx.x); i < normalized_shape; i += static_cast<size_t>(blockDim.x)) {
            float x = __bfloat162float(row_in[i]);
            float w = __bfloat162float(weight[i]);
            float b = bias != nullptr ? __bfloat162float(bias[i]) : 0.0f;
            float y = (x - mean) * inv_std;
            row_out[i] = __float2bfloat16_rn(y * w + b);
        }

        __syncthreads();
    }
}

__global__ void gelu_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t numel,
    bool approximate
) {
    size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < numel;
         idx += stride) {
        float x = __bfloat162float(input[idx]);
        float y = approximate ? gelu_approximate_device(x) : gelu_exact_device(x);
        output[idx] = __float2bfloat16_rn(y);
    }
}

__global__ void gelu_backward_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ grad_input,
    size_t numel,
    bool approximate
) {
    size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < numel;
         idx += stride) {
        float x = __bfloat162float(input[idx]);
        float g = __bfloat162float(grad_output[idx]);
        float grad = 0.0f;

        if (approximate) {
            constexpr float k = 0.7978845608028654f;
            float x2 = x * x;
            float x3 = x2 * x;
            float t = k * (x + 0.044715f * x3);
            float th = tanhf(t);
            float sech2 = 1.0f - th * th;
            float dt = k * (1.0f + 3.0f * 0.044715f * x2);
            grad = 0.5f * (1.0f + th) + 0.5f * x * sech2 * dt;
        } else {
            constexpr float inv_sqrt2 = 0.7071067811865475f;
            constexpr float inv_sqrt_2pi = 0.3989422804014327f;
            float cdf = 0.5f * (1.0f + erff(x * inv_sqrt2));
            float pdf = expf(-0.5f * x * x) * inv_sqrt_2pi;
            grad = cdf + x * pdf;
        }

        grad_input[idx] = __float2bfloat16_rn(g * grad);
    }
}

__global__ void softmax_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t outer_size,
    size_t dim_size,
    size_t inner_size
) {
    __shared__ float shared_max[kBlockSize];
    __shared__ float shared_sum[kBlockSize];

    size_t rows = outer_size * inner_size;

    for (size_t row = static_cast<size_t>(blockIdx.x); row < rows; row += static_cast<size_t>(gridDim.x)) {
        size_t outer_idx = row / inner_size;
        size_t inner_idx = row % inner_size;
        size_t base = outer_idx * dim_size * inner_size + inner_idx;

        float local_max = -CUDART_INF_F;
        for (size_t i = static_cast<size_t>(threadIdx.x); i < dim_size; i += static_cast<size_t>(blockDim.x)) {
            float x = __bfloat162float(input[base + i * inner_size]);
            local_max = fmaxf(local_max, x);
        }

        shared_max[threadIdx.x] = local_max;
        __syncthreads();

        for (unsigned int stride = static_cast<unsigned int>(blockDim.x) >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        float row_max = shared_max[0];

        float local_sum = 0.0f;
        for (size_t i = static_cast<size_t>(threadIdx.x); i < dim_size; i += static_cast<size_t>(blockDim.x)) {
            float x = __bfloat162float(input[base + i * inner_size]);
            local_sum += expf(x - row_max);
        }

        shared_sum[threadIdx.x] = local_sum;
        __syncthreads();

        for (unsigned int stride = static_cast<unsigned int>(blockDim.x) >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float denom = shared_sum[0];

        for (size_t i = static_cast<size_t>(threadIdx.x); i < dim_size; i += static_cast<size_t>(blockDim.x)) {
            float x = __bfloat162float(input[base + i * inner_size]);
            float y = expf(x - row_max) / denom;
            output[base + i * inner_size] = __float2bfloat16_rn(y);
        }

        __syncthreads();
    }
}

__global__ void float_to_bf16_kernel(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t numel
) {
    size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < numel;
         idx += stride) {
        output[idx] = __float2bfloat16_rn(input[idx]);
    }
}

}

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
    if (numel == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || weight == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (normalized_shape == 0 || numel % normalized_shape != 0) {
        return cudaErrorInvalidValue;
    }
    if (!is_finite_scalar(eps) || eps <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    size_t n_rows = numel / normalized_shape;
    int block_size = launch_block_size(normalized_shape);
    int grid_size = launch_grid_size_for_rows(n_rows);
    if (grid_size == 0) {
        return cudaSuccess;
    }

    rmsnorm_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output),
        n_rows,
        normalized_shape,
        eps
    );

    return cudaPeekAtLastError();
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
    if (numel == 0) {
        return cudaSuccess;
    }
    if (grad_output == nullptr || input == nullptr || weight == nullptr || grad_input == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (normalized_shape == 0 || numel % normalized_shape != 0) {
        return cudaErrorInvalidValue;
    }
    if (!is_finite_scalar(eps) || eps <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    size_t n_rows = numel / normalized_shape;
    int block_size = launch_block_size(normalized_shape);
    int grid_size = launch_grid_size_for_rows(n_rows);
    if (grid_size == 0) {
        return cudaSuccess;
    }

    float* grad_weight_accum = nullptr;
    cudaError_t err = cudaSuccess;

    if (grad_weight != nullptr) {
        size_t grad_weight_bytes = 0;
        if (!safe_mul_size(normalized_shape, sizeof(float), &grad_weight_bytes)) {
            return cudaErrorInvalidValue;
        }
        err = cudaMalloc(&grad_weight_accum, grad_weight_bytes);
        if (err != cudaSuccess) {
            return err;
        }
        err = cudaMemsetAsync(grad_weight_accum, 0, grad_weight_bytes, stream);
        if (err != cudaSuccess) {
            cudaError_t free_err = free_if_needed(grad_weight_accum);
            return free_err != cudaSuccess ? free_err : err;
        }
    }

    rmsnorm_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(grad_output),
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(grad_input),
        grad_weight_accum,
        n_rows,
        normalized_shape,
        eps
    );

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        cudaError_t free_err = free_if_needed(grad_weight_accum);
        return free_err != cudaSuccess ? free_err : err;
    }

    if (grad_weight != nullptr) {
        int gw_block_size = launch_block_size(normalized_shape);
        int gw_grid_size = launch_grid_size_for_numel(normalized_shape, gw_block_size);
        float_to_bf16_kernel<<<gw_grid_size, gw_block_size, 0, stream>>>(
            grad_weight_accum,
            static_cast<__nv_bfloat16*>(grad_weight),
            normalized_shape
        );
        err = cudaPeekAtLastError();
        cudaError_t free_err = free_if_needed(grad_weight_accum);
        if (err != cudaSuccess) {
            return err;
        }
        if (free_err != cudaSuccess) {
            return free_err;
        }
    }

    return cudaSuccess;
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
    if (numel == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || weight == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (normalized_shape == 0 || numel % normalized_shape != 0) {
        return cudaErrorInvalidValue;
    }
    if (!is_finite_scalar(eps) || eps <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    size_t n_rows = numel / normalized_shape;
    int block_size = launch_block_size(normalized_shape);
    int grid_size = launch_grid_size_for_rows(n_rows);
    if (grid_size == 0) {
        return cudaSuccess;
    }

    layernorm_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(output),
        n_rows,
        normalized_shape,
        eps
    );

    return cudaPeekAtLastError();
}

cudaError_t gelu_forward_cuda(
    const void* input,
    void* output,
    size_t numel,
    bool approximate,
    cudaStream_t stream
) {
    if (numel == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    int block_size = kBlockSize;
    int grid_size = launch_grid_size_for_numel(numel, block_size);
    if (grid_size == 0) {
        return cudaSuccess;
    }

    gelu_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        numel,
        approximate
    );

    return cudaPeekAtLastError();
}

cudaError_t gelu_backward_cuda(
    const void* grad_output,
    const void* input,
    void* grad_input,
    size_t numel,
    bool approximate,
    cudaStream_t stream
) {
    if (numel == 0) {
        return cudaSuccess;
    }
    if (grad_output == nullptr || input == nullptr || grad_input == nullptr) {
        return cudaErrorInvalidValue;
    }

    int block_size = kBlockSize;
    int grid_size = launch_grid_size_for_numel(numel, block_size);
    if (grid_size == 0) {
        return cudaSuccess;
    }

    gelu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(grad_output),
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(grad_input),
        numel,
        approximate
    );

    return cudaPeekAtLastError();
}

cudaError_t softmax_forward_cuda(
    const void* input,
    void* output,
    size_t outer_size,
    size_t dim_size,
    size_t inner_size,
    cudaStream_t stream
) {
    if (outer_size == 0 || dim_size == 0 || inner_size == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    size_t rows = 0;
    size_t tmp = 0;
    if (!safe_mul_size(outer_size, inner_size, &rows)) {
        return cudaErrorInvalidValue;
    }
    if (!safe_mul_size(dim_size, inner_size, &tmp)) {
        return cudaErrorInvalidValue;
    }
    if (!safe_mul_size(outer_size, tmp, &tmp)) {
        return cudaErrorInvalidValue;
    }

    int block_size = launch_block_size(dim_size);
    int grid_size = launch_grid_size_for_rows(rows);
    if (grid_size == 0) {
        return cudaSuccess;
    }

    softmax_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        outer_size,
        dim_size,
        inner_size
    );

    return cudaPeekAtLastError();
}

}
