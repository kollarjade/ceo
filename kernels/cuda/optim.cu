/**
 * Optimizer Kernels
 * Lion, Muon, and AdamW implementations
 */

#include "include/kernels.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// Lion Optimizer Kernel
// ============================================================================

__global__ void lion_step_kernel(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ momentum,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float weight_decay
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numel) return;

    float p = __bfloat162float(param[idx]);
    float g = __bfloat162float(grad[idx]);
    float m = __bfloat162float(momentum[idx]);

    // v_t = β1 * m_{t-1} + (1-β1) * g_t
    float v = beta1 * m + (1.0f - beta1) * g;

    // m_t = β2 * m_{t-1} + (1-β2) * g_t
    float m_new = beta2 * m + (1.0f - beta2) * g;

    // sign(v)
    float sign_v = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);

    // Update: x = x - lr * sign(v) - lr * weight_decay * x
    float p_new = p - lr * sign_v - lr * weight_decay * p;

    param[idx] = __float2bfloat16(p_new);
    momentum[idx] = __float2bfloat16(m_new);
}

// ============================================================================
// Muon Optimizer Kernel (Newton-Schulz Orthogonalization)
// ============================================================================

__global__ void muon_step_kernel(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ momentum,
    size_t m,
    size_t n,
    float lr,
    float beta,
    size_t ns_iterations
) {
    // Each block processes one matrix parameter
    size_t matrix_idx = blockIdx.x;

    // Update momentum: B_t = β * B_{t-1} + G_t
    for (size_t i = threadIdx.x; i < m * n; i += blockDim.x) {
        float mom = __bfloat162float(momentum[i]);
        float g = __bfloat162float(grad[i]);
        momentum[i] = __float2bfloat16(beta * mom + g);
    }
    __syncthreads();

    // Compute Frobenius norm
    __shared__ float norm_sq;
    if (threadIdx.x == 0) norm_sq = 0.0f;
    __syncthreads();

    for (size_t i = threadIdx.x; i < m * n; i += blockDim.x) {
        float val = __bfloat162float(momentum[i]);
        atomicAdd(&norm_sq, val * val);
    }
    __syncthreads();

    float norm = sqrtf(norm_sq);
    float scale = 1.0f / (norm + 1e-8f);

    // Scale Y_0 = B / ||B||_F
    extern __shared__ float Y[];
    for (size_t i = threadIdx.x; i < m * n; i += blockDim.x) {
        Y[i] = __bfloat162float(momentum[i]) * scale;
    }
    __syncthreads();

    // Newton-Schulz iterations
    float* YtY = Y + m * n;

    for (size_t iter = 0; iter < ns_iterations; ++iter) {
        // Compute Y^T Y (n x n)
        if (threadIdx.x < n * n) {
            size_t row = threadIdx.x / n;
            size_t col = threadIdx.x % n;
            float sum = 0.0f;
            for (size_t k = 0; k < m; ++k) {
                sum += Y[k * n + row] * Y[k * n + col];
            }
            YtY[threadIdx.x] = sum;
        }
        __syncthreads();

        // Compute 3I - Y^T Y
        if (threadIdx.x < n * n) {
            size_t row = threadIdx.x / n;
            size_t col = threadIdx.x % n;
            float identity = (row == col) ? 3.0f : 0.0f;
            YtY[threadIdx.x] = identity - YtY[threadIdx.x];
        }
        __syncthreads();

        // Compute Y_new = 0.5 * Y * (3I - Y^T Y)
        for (size_t i = threadIdx.x; i < m * n; i += blockDim.x) {
            size_t row = i / n;
            size_t col = i % n;
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                sum += Y[row * n + k] * YtY[k * n + col];
            }
            Y[i] = 0.5f * sum;
        }
        __syncthreads();
    }

    // Update: X = X - lr * O_t
    for (size_t i = threadIdx.x; i < m * n; i += blockDim.x) {
        float p = __bfloat162float(param[i]);
        param[i] = __float2bfloat16(p - lr * Y[i]);
    }
}

// ============================================================================
// AdamW Optimizer Kernel
// ============================================================================

__global__ void adamw_step_kernel(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ exp_avg,
    __nv_bfloat16* __restrict__ exp_avg_sq,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    size_t step
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numel) return;

    float p = __bfloat162float(param[idx]);
    float g = __bfloat162float(grad[idx]);
    float ea = __bfloat162float(exp_avg[idx]);
    float eas = __bfloat162float(exp_avg_sq[idx]);

    // Bias correction
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);

    // Update moments
    float ea_new = beta1 * ea + (1.0f - beta1) * g;
    float eas_new = beta2 * eas + (1.0f - beta2) * g * g;

    // Bias-corrected estimates
    float ea_hat = ea_new / bc1;
    float eas_hat = eas_new / bc2;

    // Update
    float denom = sqrtf(eas_hat) + eps;
    float p_new = p - lr * (ea_hat / denom + weight_decay * p);

    param[idx] = __float2bfloat16(p_new);
    exp_avg[idx] = __float2bfloat16(ea_new);
    exp_avg_sq[idx] = __float2bfloat16(eas_new);
}

// ============================================================================
// Gradient Clipping Kernel
// ============================================================================

__global__ void compute_norm_kernel(
    const __nv_bfloat16* __restrict__ grad,
    size_t numel,
    float* __restrict__ norm_out
) {
    __shared__ float block_sum;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;

    if (idx < numel) {
        float g = __bfloat162float(grad[idx]);
        local_sum = g * g;
    }

    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();

    atomicAdd(&block_sum, local_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(norm_out, block_sum);
    }
}

__global__ void clip_grad_kernel(
    __nv_bfloat16* __restrict__ grad,
    size_t numel,
    float scale
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numel) return;

    float g = __bfloat162float(grad[idx]);
    grad[idx] = __float2bfloat16(g * scale);
}

// ============================================================================
// Host Functions
// ============================================================================

extern "C" {

cudaError_t lion_step_cuda(
    void* param,
    const void* grad,
    void* momentum,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    cudaStream_t stream
) {
    size_t block_size = 256;
    size_t num_blocks = (numel + block_size - 1) / block_size;

    lion_step_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<__nv_bfloat16*>(param),
        static_cast<const __nv_bfloat16*>(grad),
        static_cast<__nv_bfloat16*>(momentum),
        numel,
        lr,
        beta1,
        beta2,
        weight_decay
    );

    return cudaGetLastError();
}

cudaError_t muon_step_cuda(
    void* param,
    const void* grad,
    void* momentum,
    size_t m,
    size_t n,
    float lr,
    float beta,
    size_t ns_iterations,
    cudaStream_t stream
) {
    size_t block_size = 256;
    size_t shared_mem = 2 * m * n * sizeof(float) + n * n * sizeof(float);

    muon_step_kernel<<<1, block_size, shared_mem, stream>>>(
        static_cast<__nv_bfloat16*>(param),
        static_cast<const __nv_bfloat16*>(grad),
        static_cast<__nv_bfloat16*>(momentum),
        m,
        n,
        lr,
        beta,
        ns_iterations
    );

    return cudaGetLastError();
}

cudaError_t adamw_step_cuda(
    void* param,
    const void* grad,
    void* exp_avg,
    void* exp_avg_sq,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    size_t step,
    cudaStream_t stream
) {
    size_t block_size = 256;
    size_t num_blocks = (numel + block_size - 1) / block_size;

    adamw_step_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<__nv_bfloat16*>(param),
        static_cast<const __nv_bfloat16*>(grad),
        static_cast<__nv_bfloat16*>(exp_avg),
        static_cast<__nv_bfloat16*>(exp_avg_sq),
        numel,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step
    );

    return cudaGetLastError();
}

cudaError_t clip_grad_norm_cuda(
    void** grads,
    const size_t* numels,
    size_t num_params,
    float max_norm,
    float* global_norm,
    cudaStream_t stream
) {
    // Compute global norm
    float* d_norm;
    cudaMalloc(&d_norm, sizeof(float));
    cudaMemset(d_norm, 0, sizeof(float));

    for (size_t i = 0; i < num_params; ++i) {
        size_t block_size = 256;
        size_t num_blocks = (numels[i] + block_size - 1) / block_size;

        compute_norm_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(grads[i]),
            numels[i],
            d_norm
        );
    }

    // Copy norm to host
    float h_norm;
    cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
    h_norm = sqrtf(h_norm);
    cudaMemcpy(global_norm, &h_norm, sizeof(float), cudaMemcpyHostToDevice);

    // Clip if necessary
    if (h_norm > max_norm) {
        float scale = max_norm / h_norm;
        for (size_t i = 0; i < num_params; ++i) {
            size_t block_size = 256;
            size_t num_blocks = (numels[i] + block_size - 1) / block_size;

            clip_grad_kernel<<<num_blocks, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(grads[i]),
                numels[i],
                scale
            );
        }
    }

    cudaFree(d_norm);
    return cudaGetLastError();
}

} // extern "C"
