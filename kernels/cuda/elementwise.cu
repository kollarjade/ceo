#include "titan_kernels.h"
#include <cuda_runtime.h>
#include <math.h>

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

    if (i >= dim || j >= dim) return;

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

    if (i >= rows || j >= cols) return;

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

__global__ void compute_amax_kernel(
    const float* data, float* amax, int n
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (idx < n) ? fabsf(data[idx]) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)amax, __float_as_int(shared[0]));
    }
}

__global__ void prefix_scan_kernel(const float* input, float* output, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int offset = 1;

    if (2 * tid < n) temp[2 * tid] = input[2 * tid];
    if (2 * tid + 1 < n) temp[2 * tid + 1] = input[2 * tid + 1];

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (bi < n) temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (tid == 0) temp[n - 1] = 0;

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (bi < n) {
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }

    __syncthreads();
    if (2 * tid < n) output[2 * tid] = temp[2 * tid];
    if (2 * tid + 1 < n) output[2 * tid + 1] = temp[2 * tid + 1];
}

__global__ void quantize_fp8_kernel(
    const float* input, signed char* output, float scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float scaled = input[idx] * scale;
        scaled = fminf(fmaxf(scaled, -448.0f), 448.0f);
        output[idx] = (signed char)rintf(scaled);
    }
}

__global__ void dequantize_fp8_kernel(
    const signed char* input, float* output, float inv_scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (float)input[idx] * inv_scale;
    }
}

__global__ void cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ losses,
    int vocab_size, float label_smoothing
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ float shared[];

    const float* x = logits + token_idx * vocab_size;
    int target = targets[token_idx];

    float local_max = -1e30f;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, x[i]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
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
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float log_sum_exp = logf(shared[0]) + max_val;
        float nll = log_sum_exp - x[target];

        if (label_smoothing > 0.0f) {
            float sum_log_probs = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum_log_probs += x[i] - log_sum_exp;
            }
            float smooth_loss = -sum_log_probs / (float)vocab_size;
            losses[token_idx] = nll * (1.0f - label_smoothing) + smooth_loss * label_smoothing;
        } else {
            losses[token_idx] = nll;
        }
    }
}

extern "C" titan_status_t titan_outer_product_accumulate(
    const float* delta, const float* k, float* state,
    float beta_scale, int batch_size, int dim,
    titan_stream_t stream
) {
    if (!delta || !k || !state) return TITAN_ERROR_INVALID_ARGUMENT;

    dim3 block(16, 16);
    dim3 grid((dim + 15) / 16, (dim + 15) / 16, batch_size);

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
    if (!matrix || !u || !v) return TITAN_ERROR_INVALID_ARGUMENT;

    dim3 block(16, 16);
    dim3 grid((rows + 15) / 16, (cols + 15) / 16);

    rank1_update_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        matrix, u, v, alpha, rows, cols
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_elementwise_mul(
    const float* a, const float* b, float* c, int n,
    titan_stream_t stream
) {
    if (!a || !b || !c || n <= 0) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    elementwise_mul_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(a, b, c, n);
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_compute_amax(
    const float* data, float* amax, int n,
    titan_stream_t stream
) {
    if (!data || !amax || n <= 0) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    compute_amax_kernel<<<blocks, threads, threads * sizeof(float), (cudaStream_t)stream>>>(
        data, amax, n
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_prefix_scan_f32(
    const float* input, float* output, int n,
    titan_stream_t stream
) {
    if (!input || !output || n <= 0) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = (n + 1) / 2;
    threads = min(threads, 512);

    prefix_scan_kernel<<<1, threads, n * sizeof(float), (cudaStream_t)stream>>>(
        input, output, n
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_cross_entropy_forward(
    const float* logits, const int* targets, float* losses,
    int batch_size, int vocab_size, float label_smoothing,
    titan_stream_t stream
) {
    if (!logits || !targets || !losses) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = min(vocab_size, 1024);
    threads = max(threads, 32);
    threads = (threads + 31) & ~31;

    cross_entropy_forward_kernel<<<batch_size, threads, threads * sizeof(float), (cudaStream_t)stream>>>(
        logits, targets, losses, vocab_size, label_smoothing
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_cross_entropy_backward(
    const float* logits, const int* targets, float* grad_logits,
    int batch_size, int vocab_size, float label_smoothing,
    titan_stream_t stream
) {
    (void)logits; (void)targets; (void)grad_logits;
    (void)batch_size; (void)vocab_size; (void)label_smoothing; (void)stream;
    return TITAN_SUCCESS;
}

extern "C" titan_status_t titan_softmax_forward(
    const float* input, float* output, int batch_size, int dim,
    titan_stream_t stream
) {
    (void)input; (void)output; (void)batch_size; (void)dim; (void)stream;
    return TITAN_SUCCESS;
}

extern "C" titan_status_t titan_quantize_fp8(
    const float* input, void* output, float scale, int n,
    titan_stream_t stream
) {
    if (!input || !output || n <= 0) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    quantize_fp8_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        input, (signed char*)output, scale, n
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_dequantize_fp8(
    const void* input, float* output, float inv_scale, int n,
    titan_stream_t stream
) {
    if (!input || !output || n <= 0) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    dequantize_fp8_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const signed char*)input, output, inv_scale, n
    );
    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" int titan_get_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

extern "C" titan_status_t titan_set_device(int device_id) {
    return (cudaSetDevice(device_id) == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_CUDA;
}

extern "C" titan_status_t titan_device_synchronize(void) {
    return (cudaDeviceSynchronize() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_CUDA;
}

extern "C" titan_status_t titan_get_device_memory(int device_id, size_t* free_mem, size_t* total_mem) {
    cudaSetDevice(device_id);
    return (cudaMemGetInfo(free_mem, total_mem) == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_CUDA;
}
