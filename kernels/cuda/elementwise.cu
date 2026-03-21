#include "titan_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <math.h>

namespace {

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

struct AbsTransform {
    __device__ __forceinline__ float operator()(const float& value) const {
        return fabsf(value);
    }
};

__global__ void outer_product_accumulate_kernel(
    const float* __restrict__ delta,
    const float* __restrict__ k,
    float* __restrict__ state,
    float beta_scale,
    int dim
) {
    int batch_idx = blockIdx.z;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= dim || j >= dim) {
        return;
    }

    int state_idx = batch_idx * dim * dim + i * dim + j;
    int delta_idx = batch_idx * dim + i;
    int k_idx = batch_idx * dim + j;

    state[state_idx] += beta_scale * delta[delta_idx] * k[k_idx];
}

__global__ void rank1_update_kernel(
    float* __restrict__ matrix,
    const float* __restrict__ u,
    const float* __restrict__ v,
    float alpha,
    int rows, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= cols) {
        return;
    }

    matrix[i * cols + j] += alpha * u[i] * v[j];
}

__global__ void elementwise_mul_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ float shared[];
    const float* in = input + row * dim;
    float* out = output + row * dim;

    float local_max = -CUDART_INF_F;
    for (int i = tid; i < dim; i += stride) {
        local_max = fmaxf(local_max, in[i]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        float value = expf(in[i] - row_max);
        out[i] = value;
        local_sum += value;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / shared[0];

    for (int i = tid; i < dim; i += stride) {
        out[i] *= inv_sum;
    }
}

__global__ void cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int vocab_size,
    float label_smoothing,
    int* __restrict__ error_flag
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (token_idx >= batch_size) {
        return;
    }

    extern __shared__ float shared[];
    const float* x = logits + token_idx * vocab_size;
    int target = targets[token_idx];

    if (target < 0 || target >= vocab_size) {
        if (tid == 0) {
            atomicExch(error_flag, 1);
        }
        return;
    }

    float local_max = -CUDART_INF_F;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, x[i]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared[0];

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        local_sum += expf(x[i] - max_val);
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float log_sum_exp = logf(shared[0]) + max_val;
        float target_log_prob = x[target] - log_sum_exp;
        float loss = -(1.0f - label_smoothing) * target_log_prob;

        if (label_smoothing > 0.0f) {
            float sum_log_probs = 0.0f;
            for (int i = 0; i < vocab_size; ++i) {
                sum_log_probs += x[i] - log_sum_exp;
            }
            loss -= (label_smoothing / static_cast<float>(vocab_size)) * sum_log_probs;
        }

        losses[token_idx] = loss;
    }
}

__global__ void cross_entropy_backward_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ grad_logits,
    int batch_size,
    int vocab_size,
    float label_smoothing,
    int* __restrict__ error_flag
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (token_idx >= batch_size) {
        return;
    }

    extern __shared__ float shared[];
    const float* x = logits + token_idx * vocab_size;
    float* g = grad_logits + token_idx * vocab_size;
    int target = targets[token_idx];

    if (target < 0 || target >= vocab_size) {
        if (tid == 0) {
            atomicExch(error_flag, 1);
        }
        return;
    }

    float local_max = -CUDART_INF_F;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, x[i]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared[0];

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        local_sum += expf(x[i] - max_val);
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float denom = shared[0];
    float uniform = label_smoothing / static_cast<float>(vocab_size);

    for (int i = tid; i < vocab_size; i += stride) {
        float prob = expf(x[i] - max_val) / denom;
        float target_prob = uniform;
        if (i == target) {
            target_prob += 1.0f - label_smoothing;
        }
        g[i] = prob - target_prob;
    }
}

__global__ void quantize_fp8_kernel(
    const float* input, __nv_fp8_e4m3* output, float scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<__nv_fp8_e4m3>(input[idx] * scale);
    }
}

__global__ void dequantize_fp8_kernel(
    const __nv_fp8_e4m3* input, float* output, float inv_scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<float>(input[idx]) * inv_scale;
    }
}

titan_status_t launch_checked_ce_kernel(
    const float* logits,
    const int* targets,
    float* output,
    int batch_size,
    int vocab_size,
    float label_smoothing,
    cudaStream_t stream,
    bool backward
) {
    if (!logits || !targets || !output || batch_size <= 0 || vocab_size <= 0 ||
        label_smoothing < 0.0f || label_smoothing > 1.0f) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    int threads = vocab_size < 1024 ? vocab_size : 1024;
    if (threads < 32) {
        threads = 32;
    }
    threads = (threads + 31) & ~31;

    int* error_flag = nullptr;
    cudaError_t err = allocate_workspace(reinterpret_cast<void**>(&error_flag), sizeof(int), stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_OUT_OF_MEMORY;
    }

    err = cudaMemsetAsync(error_flag, 0, sizeof(int), stream);
    if (err != cudaSuccess) {
        free_workspace(error_flag, stream);
        return TITAN_ERROR_CUDA;
    }

    if (backward) {
        cross_entropy_backward_kernel<<<batch_size, threads, threads * sizeof(float), stream>>>(
            logits, targets, output, batch_size, vocab_size, label_smoothing, error_flag
        );
    } else {
        cross_entropy_forward_kernel<<<batch_size, threads, threads * sizeof(float), stream>>>(
            logits, targets, output, batch_size, vocab_size, label_smoothing, error_flag
        );
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        free_workspace(error_flag, stream);
        return TITAN_ERROR_KERNEL_LAUNCH;
    }

    int host_error = 0;
    err = cudaMemcpyAsync(&host_error, error_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        free_workspace(error_flag, stream);
        return TITAN_ERROR_CUDA;
    }
    err = cudaStreamSynchronize(stream);
    cudaError_t free_err = free_workspace(error_flag, stream);

    if (err != cudaSuccess || free_err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }
    if (host_error != 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }
    return TITAN_SUCCESS;
}

}

extern "C" titan_status_t titan_outer_product_accumulate(
    const float* delta, const float* k, float* state,
    float beta_scale, int batch_size, int dim,
    titan_stream_t stream
) {
    if (!delta || !k || !state || batch_size <= 0 || dim <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((dim + 15) / 16),
        static_cast<unsigned int>((dim + 15) / 16),
        static_cast<unsigned int>(batch_size)
    );

    outer_product_accumulate_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        delta, k, state, beta_scale, dim
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_rank1_update(
    float* matrix, const float* u, const float* v,
    float alpha, int rows, int cols,
    titan_stream_t stream
) {
    if (!matrix || !u || !v || rows <= 0 || cols <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((rows + 15) / 16),
        static_cast<unsigned int>((cols + 15) / 16)
    );

    rank1_update_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        matrix, u, v, alpha, rows, cols
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_elementwise_mul(
    const float* a, const float* b, float* c, int n,
    titan_stream_t stream
) {
    if (!a || !b || !c || n <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    elementwise_mul_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(a, b, c, n);
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_compute_amax(
    const float* data, float* amax, int n,
    titan_stream_t stream
) {
    if (!data || !amax || n <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = (cudaStream_t)stream;
    using Iterator = cub::TransformInputIterator<float, AbsTransform, const float*>;
    Iterator iterator(data, AbsTransform());

    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaError_t err = cub::DeviceReduce::Max(nullptr, workspace_bytes, iterator, amax, n, cuda_stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }

    err = allocate_workspace(&workspace, workspace_bytes, cuda_stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_OUT_OF_MEMORY;
    }

    err = cub::DeviceReduce::Max(workspace, workspace_bytes, iterator, amax, n, cuda_stream);
    cudaError_t free_err = free_workspace(workspace, cuda_stream);

    if (err != cudaSuccess || free_err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }
    return TITAN_SUCCESS;
}

extern "C" titan_status_t titan_prefix_scan_f32(
    const float* input, float* output, int n,
    titan_stream_t stream
) {
    if (!input || !output || n <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = (cudaStream_t)stream;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaError_t err = cub::DeviceScan::ExclusiveSum(nullptr, workspace_bytes, input, output, n, cuda_stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }

    err = allocate_workspace(&workspace, workspace_bytes, cuda_stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_OUT_OF_MEMORY;
    }

    err = cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes, input, output, n, cuda_stream);
    cudaError_t free_err = free_workspace(workspace, cuda_stream);

    if (err != cudaSuccess || free_err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }
    return TITAN_SUCCESS;
}

extern "C" titan_status_t titan_cross_entropy_forward(
    const float* logits, const int* targets, float* losses,
    int batch_size, int vocab_size, float label_smoothing,
    titan_stream_t stream
) {
    return launch_checked_ce_kernel(
        logits,
        targets,
        losses,
        batch_size,
        vocab_size,
        label_smoothing,
        (cudaStream_t)stream,
        false
    );
}

extern "C" titan_status_t titan_cross_entropy_backward(
    const float* logits, const int* targets, float* grad_logits,
    int batch_size, int vocab_size, float label_smoothing,
    titan_stream_t stream
) {
    return launch_checked_ce_kernel(
        logits,
        targets,
        grad_logits,
        batch_size,
        vocab_size,
        label_smoothing,
        (cudaStream_t)stream,
        true
    );
}

extern "C" titan_status_t titan_softmax_forward(
    const float* input, float* output, int batch_size, int dim,
    titan_stream_t stream
) {
    if (!input || !output || batch_size <= 0 || dim <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    int threads = dim < 1024 ? dim : 1024;
    if (threads < 32) {
        threads = 32;
    }
    threads = (threads + 31) & ~31;

    softmax_forward_kernel<<<batch_size, threads, threads * sizeof(float), (cudaStream_t)stream>>>(
        input, output, dim
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_quantize_fp8(
    const float* input, void* output, float scale, int n,
    titan_stream_t stream
) {
    if (!input || !output || n <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    quantize_fp8_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        input, static_cast<__nv_fp8_e4m3*>(output), scale, n
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_dequantize_fp8(
    const void* input, float* output, float inv_scale, int n,
    titan_stream_t stream
) {
    if (!input || !output || n <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    dequantize_fp8_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        static_cast<const __nv_fp8_e4m3*>(input), output, inv_scale, n
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" int titan_get_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

extern "C" titan_status_t titan_set_device(int device_id) {
    return (cudaSetDevice(device_id) == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_CUDA;
}

extern "C" titan_status_t titan_device_synchronize(void) {
    return (cudaDeviceSynchronize() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_CUDA;
}

extern "C" titan_status_t titan_get_device_memory(int device_id, size_t* free_mem, size_t* total_mem) {
    if (free_mem == nullptr || total_mem == nullptr) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device_id) != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }
    return (cudaMemGetInfo(free_mem, total_mem) == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_CUDA;
}
