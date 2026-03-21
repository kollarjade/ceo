#include "titan_kernels.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int hidden_dim, float eps
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ float shared_data[];

    const float* x = input + token_idx * hidden_dim;
    float* o = output + token_idx * hidden_dim;

    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += stride) {
        float val = x[i];
        local_sum_sq += val * val;
    }

    shared_data[tid] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(shared_data[0] / (float)hidden_dim + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < hidden_dim; i += stride) {
        o[i] = x[i] * inv_rms * weight[i];
    }
}

__global__ void rmsnorm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    int hidden_dim, float eps
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ float shared[];

    const float* go = grad_output + token_idx * hidden_dim;
    const float* x = input + token_idx * hidden_dim;
    float* gi = grad_input + token_idx * hidden_dim;

    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += stride) {
        local_sum_sq += x[i] * x[i];
    }

    shared[tid] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float variance = shared[0] / (float)hidden_dim;
    float rms = sqrtf(variance + eps);
    float inv_rms = 1.0f / rms;

    float local_dot = 0.0f;
    for (int i = tid; i < hidden_dim; i += stride) {
        local_dot += go[i] * weight[i] * x[i];
    }

    shared[tid] = local_dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float dot_product = shared[0];
    float coeff = dot_product / (rms * rms * rms * (float)hidden_dim);

    for (int i = tid; i < hidden_dim; i += stride) {
        gi[i] = (go[i] * weight[i] * inv_rms) - (x[i] * coeff);
        atomicAdd(&grad_weight[i], go[i] * x[i] * inv_rms);
    }
}

extern "C" titan_status_t titan_rmsnorm_forward(
    const float* input, const float* weight, float* output,
    int batch_size, int hidden_dim, float eps,
    titan_stream_t stream
) {
    if (!input || !weight || !output || batch_size <= 0 || hidden_dim <= 0 || eps < 0.0f) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    int threads = hidden_dim < 1024 ? hidden_dim : 1024;
    if (threads < 32) {
        threads = 32;
    }
    threads = (threads + 31) & ~31;

    size_t shared_mem = static_cast<size_t>(threads) * sizeof(float);

    rmsnorm_forward_kernel<<<batch_size, threads, shared_mem, (cudaStream_t)stream>>>(
        input, weight, output, hidden_dim, eps
    );

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_rmsnorm_backward(
    const float* grad_output, const float* input, const float* weight,
    float* grad_input, float* grad_weight,
    int batch_size, int hidden_dim, float eps,
    titan_stream_t stream
) {
    if (!grad_output || !input || !weight || !grad_input || !grad_weight ||
        batch_size <= 0 || hidden_dim <= 0 || eps < 0.0f) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = (cudaStream_t)stream;
    cudaError_t err = cudaMemsetAsync(grad_weight, 0, static_cast<size_t>(hidden_dim) * sizeof(float), cuda_stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }

    int threads = hidden_dim < 1024 ? hidden_dim : 1024;
    if (threads < 32) {
        threads = 32;
    }
    threads = (threads + 31) & ~31;

    size_t shared_mem = static_cast<size_t>(threads) * sizeof(float);

    rmsnorm_backward_kernel<<<batch_size, threads, shared_mem, cuda_stream>>>(
        grad_output, input, weight, grad_input, grad_weight,
        hidden_dim, eps
    );

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}
