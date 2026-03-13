#include "include/kernels.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

constexpr unsigned int kBlockSize = 256u;

__host__ __device__ inline size_t min_size_t(size_t a, size_t b) {
    return a < b ? a : b;
}

inline bool checked_mul_size_t(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

inline bool checked_mul4_size_t(size_t a, size_t b, size_t c, size_t d, size_t* out) {
    size_t t0 = 0;
    size_t t1 = 0;
    size_t t2 = 0;
    return checked_mul_size_t(a, b, &t0) &&
           checked_mul_size_t(t0, c, &t1) &&
           checked_mul_size_t(t1, d, &t2) &&
           ((*out = t2), true);
}

inline bool checked_mul5_size_t(size_t a, size_t b, size_t c, size_t d, size_t e, size_t* out) {
    size_t t0 = 0;
    size_t t1 = 0;
    size_t t2 = 0;
    size_t t3 = 0;
    return checked_mul_size_t(a, b, &t0) &&
           checked_mul_size_t(t0, c, &t1) &&
           checked_mul_size_t(t1, d, &t2) &&
           checked_mul_size_t(t2, e, &t3) &&
           ((*out = t3), true);
}

inline unsigned int blocks_for_elements(size_t n) {
    size_t blocks = (n + static_cast<size_t>(kBlockSize) - 1) / static_cast<size_t>(kBlockSize);
    const size_t max_blocks = static_cast<size_t>(std::numeric_limits<unsigned int>::max());
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    return static_cast<unsigned int>(blocks);
}

template <typename KernelT>
inline cudaError_t configure_dynamic_shared(KernelT kernel, size_t bytes) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return err;
    }
    int default_limit = 0;
    int optin_limit = 0;
    err = cudaDeviceGetAttribute(&default_limit, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (err != cudaSuccess) {
        return err;
    }
    int max_limit = default_limit > optin_limit ? default_limit : optin_limit;
    if (bytes > static_cast<size_t>(max_limit)) {
        return cudaErrorInvalidConfiguration;
    }
    if (bytes > static_cast<size_t>(default_limit)) {
        err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
        if (err != cudaSuccess) {
            return err;
        }
    }
    return cudaSuccess;
}

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ void compute_coefficient_and_grad(float lambda, float beta, float* c, float* dc_dlambda) {
    if (lambda <= 1.0e-6f) {
        const float beta2 = beta * beta;
        const float beta3 = beta2 * beta;
        const float beta4 = beta3 * beta;
        const float beta5 = beta4 * beta;
        const float lambda2 = lambda * lambda;
        const float lambda3 = lambda2 * lambda;
        *c = beta
           - 0.5f * beta2 * lambda
           + (1.0f / 6.0f) * beta3 * lambda2
           - (1.0f / 24.0f) * beta4 * lambda3;
        *dc_dlambda = -0.5f * beta2
                    + (1.0f / 3.0f) * beta3 * lambda
                    - (1.0f / 8.0f) * beta4 * lambda2
                    + (1.0f / 30.0f) * beta5 * lambda3;
        return;
    }
    const float x = beta * lambda;
    const float e = expf(-x);
    *c = (1.0f - e) / lambda;
    *dc_dlambda = (beta * lambda * e - 1.0f + e) / (lambda * lambda);
}

__device__ __forceinline__ float block_reduce_sum(float v, float* reduce_buf) {
    const unsigned int tid = threadIdx.x;
    reduce_buf[tid] = v;
    __syncthreads();
    for (unsigned int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }
    return reduce_buf[0];
}

__global__ void efla_forward_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float beta
) {
    const size_t bh = static_cast<size_t>(blockIdx.x);
    const size_t total_bh = batch_size * num_heads;
    if (bh >= total_bh) {
        return;
    }

    const size_t batch_idx = bh / num_heads;
    const size_t head_idx = bh % num_heads;
    const size_t state_base = bh * head_dim * head_dim;

    float* shared = reinterpret_cast<float*>(extern_shared);
}

__global__ void efla_forward_kernel_impl(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float beta
) {
    const size_t bh = static_cast<size_t>(blockIdx.x);
    const size_t total_bh = batch_size * num_heads;
    if (bh >= total_bh) {
        return;
    }

    const size_t batch_idx = bh / num_heads;
    const size_t head_idx = bh % num_heads;
    const size_t state_base = bh * head_dim * head_dim;

    extern __shared__ float shared_mem[];
    float* k_vec = shared_mem;
    float* v_vec = shared_mem + head_dim;
    float* u_vec = shared_mem + 2 * head_dim;
    float* reduce_buf = shared_mem + 3 * head_dim;

    for (size_t t = 0; t < seq_len; ++t) {
        const size_t tok_base = ((batch_idx * seq_len + t) * num_heads + head_idx) * head_dim;

        for (size_t d = threadIdx.x; d < head_dim; d += kBlockSize) {
            k_vec[d] = bf16_to_float(k[tok_base + d]);
            v_vec[d] = bf16_to_float(v[tok_base + d]);
        }
        __syncthreads();

        float lambda_partial = 0.0f;
        for (size_t d = threadIdx.x; d < head_dim; d += kBlockSize) {
            const float kv = k_vec[d];
            lambda_partial += kv * kv;
        }
        const float lambda = block_reduce_sum(lambda_partial, reduce_buf);
        if (threadIdx.x == 0) {
            float c = 0.0f;
            float dc = 0.0f;
            compute_coefficient_and_grad(lambda, beta, &c, &dc);
            reduce_buf[0] = c;
        }
        __syncthreads();
        const float c = reduce_buf[0];

        for (size_t col = threadIdx.x; col < head_dim; col += kBlockSize) {
            float sum = 0.0f;
            for (size_t row = 0; row < head_dim; ++row) {
                sum += k_vec[row] * bf16_to_float(state[state_base + row * head_dim + col]);
            }
            u_vec[col] = sum;
        }
        __syncthreads();

        const size_t matrix_elems = head_dim * head_dim;
        for (size_t idx = threadIdx.x; idx < matrix_elems; idx += kBlockSize) {
            const size_t row = idx / head_dim;
            const size_t col = idx - row * head_dim;
            const float s_old = bf16_to_float(state[state_base + idx]);
            const float s_new = s_old + c * k_vec[row] * (v_vec[col] - u_vec[col]);
            state[state_base + idx] = float_to_bf16(s_new);
        }
        __syncthreads();

        for (size_t row = threadIdx.x; row < head_dim; row += kBlockSize) {
            float sum = 0.0f;
            for (size_t col = 0; col < head_dim; ++col) {
                sum += bf16_to_float(state[state_base + row * head_dim + col]) * k_vec[col];
            }
            output[tok_base + row] = float_to_bf16(sum);
        }
        __syncthreads();
    }
}

__global__ void build_state_history_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ state0,
    __nv_bfloat16* __restrict__ history,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float beta
) {
    const size_t bh = static_cast<size_t>(blockIdx.x);
    const size_t total_bh = batch_size * num_heads;
    if (bh >= total_bh) {
        return;
    }

    const size_t batch_idx = bh / num_heads;
    const size_t head_idx = bh % num_heads;
    const size_t state0_base = bh * head_dim * head_dim;
    const size_t history_stride = head_dim * head_dim;
    const size_t history_base0 = (bh * (seq_len + 1)) * history_stride;

    extern __shared__ float shared_mem[];
    float* k_vec = shared_mem;
    float* v_vec = shared_mem + head_dim;
    float* u_vec = shared_mem + 2 * head_dim;
    float* reduce_buf = shared_mem + 3 * head_dim;

    for (size_t idx = threadIdx.x; idx < history_stride; idx += kBlockSize) {
        history[history_base0 + idx] = state0[state0_base + idx];
    }
    __syncthreads();

    for (size_t t = 0; t < seq_len; ++t) {
        const size_t tok_base = ((batch_idx * seq_len + t) * num_heads + head_idx) * head_dim;
        const size_t prev_base = history_base0 + t * history_stride;
        const size_t next_base = prev_base + history_stride;

        for (size_t d = threadIdx.x; d < head_dim; d += kBlockSize) {
            k_vec[d] = bf16_to_float(k[tok_base + d]);
            v_vec[d] = bf16_to_float(v[tok_base + d]);
        }
        __syncthreads();

        float lambda_partial = 0.0f;
        for (size_t d = threadIdx.x; d < head_dim; d += kBlockSize) {
            const float kv = k_vec[d];
            lambda_partial += kv * kv;
        }
        const float lambda = block_reduce_sum(lambda_partial, reduce_buf);
        if (threadIdx.x == 0) {
            float c = 0.0f;
            float dc = 0.0f;
            compute_coefficient_and_grad(lambda, beta, &c, &dc);
            reduce_buf[0] = c;
        }
        __syncthreads();
        const float c = reduce_buf[0];

        for (size_t col = threadIdx.x; col < head_dim; col += kBlockSize) {
            float sum = 0.0f;
            for (size_t row = 0; row < head_dim; ++row) {
                sum += k_vec[row] * bf16_to_float(history[prev_base + row * head_dim + col]);
            }
            u_vec[col] = sum;
        }
        __syncthreads();

        for (size_t idx = threadIdx.x; idx < history_stride; idx += kBlockSize) {
            const size_t row = idx / head_dim;
            const size_t col = idx - row * head_dim;
            const float s_prev = bf16_to_float(history[prev_base + idx]);
            const float s_next = s_prev + c * k_vec[row] * (v_vec[col] - u_vec[col]);
            history[next_base + idx] = float_to_bf16(s_next);
        }
        __syncthreads();
    }
}

__global__ void efla_backward_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ history,
    float* __restrict__ grad_state_work,
    __nv_bfloat16* __restrict__ grad_k,
    __nv_bfloat16* __restrict__ grad_v,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float beta
) {
    const size_t bh = static_cast<size_t>(blockIdx.x);
    const size_t total_bh = batch_size * num_heads;
    if (bh >= total_bh) {
        return;
    }

    const size_t batch_idx = bh / num_heads;
    const size_t head_idx = bh % num_heads;
    const size_t state_stride = head_dim * head_dim;
    const size_t state_base = bh * state_stride;
    const size_t history_base0 = (bh * (seq_len + 1)) * state_stride;

    extern __shared__ float shared_mem[];
    float* k_vec = shared_mem;
    float* v_vec = shared_mem + head_dim;
    float* go_vec = shared_mem + 2 * head_dim;
    float* work1 = shared_mem + 3 * head_dim;
    float* work2 = shared_mem + 4 * head_dim;
    float* gk_vec = shared_mem + 5 * head_dim;
    float* reduce_buf = shared_mem + 6 * head_dim;

    for (size_t idx = threadIdx.x; idx < state_stride; idx += kBlockSize) {
        grad_state_work[state_base + idx] = 0.0f;
    }
    __syncthreads();

    for (size_t t = seq_len; t-- > 0;) {
        const size_t tok_base = ((batch_idx * seq_len + t) * num_heads + head_idx) * head_dim;
        const size_t prev_base = history_base0 + t * state_stride;
        const size_t next_base = prev_base + state_stride;

        for (size_t d = threadIdx.x; d < head_dim; d += kBlockSize) {
            k_vec[d] = bf16_to_float(k[tok_base + d]);
            v_vec[d] = bf16_to_float(v[tok_base + d]);
            go_vec[d] = bf16_to_float(grad_output[tok_base + d]);
        }
        __syncthreads();

        float lambda_partial = 0.0f;
        for (size_t d = threadIdx.x; d < head_dim; d += kBlockSize) {
            const float kv = k_vec[d];
            lambda_partial += kv * kv;
        }
        const float lambda = block_reduce_sum(lambda_partial, reduce_buf);
        if (threadIdx.x == 0) {
            float c = 0.0f;
            float dc = 0.0f;
            compute_coefficient_and_grad(lambda, beta, &c, &dc);
            reduce_buf[0] = c;
            reduce_buf[1] = dc;
        }
        __syncthreads();
        const float c = reduce_buf[0];
        const float dc = reduce_buf[1];

        for (size_t col = threadIdx.x; col < head_dim; col += kBlockSize) {
            float u = 0.0f;
            for (size_t row = 0; row < head_dim; ++row) {
                u += k_vec[row] * bf16_to_float(history[prev_base + row * head_dim + col]);
            }
            work1[col] = u;
            work2[col] = v_vec[col] - u;
        }
        __syncthreads();

        for (size_t col = threadIdx.x; col < head_dim; col += kBlockSize) {
            float sum = 0.0f;
            for (size_t row = 0; row < head_dim; ++row) {
                sum += bf16_to_float(history[next_base + row * head_dim + col]) * go_vec[row];
            }
            gk_vec[col] = sum;
        }
        __syncthreads();

        for (size_t col = threadIdx.x; col < head_dim; col += kBlockSize) {
            float sum = 0.0f;
            for (size_t row = 0; row < head_dim; ++row) {
                const float g = grad_state_work[state_base + row * head_dim + col] + go_vec[row] * k_vec[col];
                sum += g * k_vec[row];
            }
            work1[col] = c * sum;
        }
        __syncthreads();

        float grad_c_partial = 0.0f;
        for (size_t idx = threadIdx.x; idx < state_stride; idx += kBlockSize) {
            const size_t row = idx / head_dim;
            const size_t col = idx - row * head_dim;
            const float g = grad_state_work[state_base + idx] + go_vec[row] * k_vec[col];
            grad_c_partial += g * k_vec[row] * work2[col];
        }
        const float grad_c = block_reduce_sum(grad_c_partial, reduce_buf);
        if (threadIdx.x == 0) {
            reduce_buf[0] = grad_c * dc;
        }
        __syncthreads();
        const float grad_lambda = reduce_buf[0];

        for (size_t row = threadIdx.x; row < head_dim; row += kBlockSize) {
            float explicit_term = 0.0f;
            float upath_term = 0.0f;
            for (size_t col = 0; col < head_dim; ++col) {
                const float g = grad_state_work[state_base + row * head_dim + col] + go_vec[row] * k_vec[col];
                explicit_term += g * work2[col];
                upath_term += bf16_to_float(history[prev_base + row * head_dim + col]) * work1[col];
            }
            const float gk = gk_vec[row] + c * explicit_term - upath_term + 2.0f * grad_lambda * k_vec[row];
            gk_vec[row] = gk;
        }
        __syncthreads();

        for (size_t d = threadIdx.x; d < head_dim; d += kBlockSize) {
            grad_v[tok_base + d] = float_to_bf16(work1[d]);
            grad_k[tok_base + d] = float_to_bf16(gk_vec[d]);
        }
        __syncthreads();

        for (size_t idx = threadIdx.x; idx < state_stride; idx += kBlockSize) {
            const size_t row = idx / head_dim;
            const size_t col = idx - row * head_dim;
            const float g = grad_state_work[state_base + idx] + go_vec[row] * k_vec[col];
            grad_state_work[state_base + idx] = g - k_vec[row] * work1[col];
        }
        __syncthreads();
    }
}

__global__ void convert_float_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t n
) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        dst[idx] = float_to_bf16(src[idx]);
    }
}

__global__ void add_state_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t n
) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float a = bf16_to_float(src[idx]);
        const float b = bf16_to_float(dst[idx]);
        dst[idx] = float_to_bf16(a + b);
    }
}

} 

extern "C" {

cudaError_t efla_forward_cuda(
    const void* k,
    const void* v,
    void* state,
    void* output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float beta,
    size_t chunk_size,
    cudaStream_t stream
) {
    (void)chunk_size;

    if (batch_size == 0 || num_heads == 0 || head_dim == 0) {
        return cudaErrorInvalidValue;
    }
    if (seq_len == 0) {
        return cudaSuccess;
    }
    if (k == nullptr || v == nullptr || state == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    size_t num_blocks_size_t = 0;
    if (!checked_mul_size_t(batch_size, num_heads, &num_blocks_size_t)) {
        return cudaErrorInvalidConfiguration;
    }
    if (num_blocks_size_t > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        return cudaErrorInvalidConfiguration;
    }

    size_t shared_floats = 0;
    if (!checked_mul_size_t(3, head_dim, &shared_floats)) {
        return cudaErrorInvalidConfiguration;
    }
    if (!checked_mul_size_t(shared_floats + kBlockSize, sizeof(float), &shared_floats)) {
        return cudaErrorInvalidConfiguration;
    }

    cudaError_t err = configure_dynamic_shared(efla_forward_kernel_impl, shared_floats);
    if (err != cudaSuccess) {
        return err;
    }

    efla_forward_kernel_impl<<<static_cast<unsigned int>(num_blocks_size_t), kBlockSize, shared_floats, stream>>>(
        static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v),
        static_cast<__nv_bfloat16*>(state),
        static_cast<__nv_bfloat16*>(output),
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        beta
    );

    return cudaPeekAtLastError();
}

cudaError_t efla_backward_cuda(
    const void* grad_output,
    const void* k,
    const void* v,
    const void* state,
    void* grad_k,
    void* grad_v,
    void* grad_state,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float beta,
    cudaStream_t stream
) {
    if (batch_size == 0 || num_heads == 0 || head_dim == 0) {
        return cudaErrorInvalidValue;
    }
    if (state == nullptr || grad_state == nullptr) {
        return cudaErrorInvalidValue;
    }

    size_t num_blocks_size_t = 0;
    if (!checked_mul_size_t(batch_size, num_heads, &num_blocks_size_t)) {
        return cudaErrorInvalidConfiguration;
    }
    if (num_blocks_size_t > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        return cudaErrorInvalidConfiguration;
    }

    size_t grad_state_elems = 0;
    if (!checked_mul4_size_t(batch_size, num_heads, head_dim, head_dim, &grad_state_elems)) {
        return cudaErrorInvalidConfiguration;
    }
    size_t grad_state_bytes_bf16 = 0;
    if (!checked_mul_size_t(grad_state_elems, sizeof(__nv_bfloat16), &grad_state_bytes_bf16)) {
        return cudaErrorInvalidConfiguration;
    }

    if (seq_len == 0) {
        return cudaMemsetAsync(grad_state, 0, grad_state_bytes_bf16, stream);
    }

    if (grad_output == nullptr || k == nullptr || v == nullptr || grad_k == nullptr || grad_v == nullptr) {
        return cudaErrorInvalidValue;
    }

    size_t history_elems = 0;
    if (!checked_mul5_size_t(batch_size, num_heads, seq_len + 1, head_dim, head_dim, &history_elems)) {
        return cudaErrorInvalidConfiguration;
    }
    size_t history_bytes = 0;
    if (!checked_mul_size_t(history_elems, sizeof(__nv_bfloat16), &history_bytes)) {
        return cudaErrorInvalidConfiguration;
    }
    size_t grad_state_work_bytes = 0;
    if (!checked_mul_size_t(grad_state_elems, sizeof(float), &grad_state_work_bytes)) {
        return cudaErrorInvalidConfiguration;
    }

    size_t shared_floats_history = 0;
    if (!checked_mul_size_t(3, head_dim, &shared_floats_history)) {
        return cudaErrorInvalidConfiguration;
    }
    if (!checked_mul_size_t(shared_floats_history + kBlockSize, sizeof(float), &shared_floats_history)) {
        return cudaErrorInvalidConfiguration;
    }

    size_t shared_floats_backward = 0;
    if (!checked_mul_size_t(6, head_dim, &shared_floats_backward)) {
        return cudaErrorInvalidConfiguration;
    }
    if (!checked_mul_size_t(shared_floats_backward + kBlockSize, sizeof(float), &shared_floats_backward)) {
        return cudaErrorInvalidConfiguration;
    }

    cudaError_t err = configure_dynamic_shared(build_state_history_kernel, shared_floats_history);
    if (err != cudaSuccess) {
        return err;
    }
    err = configure_dynamic_shared(efla_backward_kernel, shared_floats_backward);
    if (err != cudaSuccess) {
        return err;
    }

    __nv_bfloat16* history = nullptr;
    float* grad_state_work = nullptr;

    err = cudaMalloc(&history, history_bytes);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMalloc(&grad_state_work, grad_state_work_bytes);
    if (err != cudaSuccess) {
        cudaFree(history);
        return err;
    }

    err = cudaMemsetAsync(grad_state_work, 0, grad_state_work_bytes, stream);
    if (err != cudaSuccess) {
        cudaFree(grad_state_work);
        cudaFree(history);
        return err;
    }

    build_state_history_kernel<<<static_cast<unsigned int>(num_blocks_size_t), kBlockSize, shared_floats_history, stream>>>(
        static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v),
        static_cast<const __nv_bfloat16*>(state),
        history,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        beta
    );
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        cudaFree(grad_state_work);
        cudaFree(history);
        return err;
    }

    efla_backward_kernel<<<static_cast<unsigned int>(num_blocks_size_t), kBlockSize, shared_floats_backward, stream>>>(
        static_cast<const __nv_bfloat16*>(grad_output),
        static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v),
        history,
        grad_state_work,
        static_cast<__nv_bfloat16*>(grad_k),
        static_cast<__nv_bfloat16*>(grad_v),
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        beta
    );
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        cudaFree(grad_state_work);
        cudaFree(history);
        return err;
    }

    convert_float_to_bf16_kernel<<<blocks_for_elements(grad_state_elems), kBlockSize, 0, stream>>>(
        grad_state_work,
        static_cast<__nv_bfloat16*>(grad_state),
        grad_state_elems
    );
    err = cudaPeekAtLastError();

    cudaError_t free_err_0 = cudaFree(grad_state_work);
    cudaError_t free_err_1 = cudaFree(history);

    if (err != cudaSuccess) {
        return err;
    }
    if (free_err_0 != cudaSuccess) {
        return free_err_0;
    }
    if (free_err_1 != cudaSuccess) {
        return free_err_1;
    }

    return cudaSuccess;
}

cudaError_t efla_chunked_scan_cuda(
    void** chunk_states,
    size_t num_chunks,
    size_t num_heads,
    size_t head_dim,
    cudaStream_t stream
) {
    if (num_chunks == 0 || num_heads == 0 || head_dim == 0) {
        return cudaErrorInvalidValue;
    }
    if (chunk_states == nullptr) {
        return cudaErrorInvalidValue;
    }

    size_t elems_per_chunk = 0;
    if (!checked_mul4_size_t(1, num_heads, head_dim, head_dim, &elems_per_chunk)) {
        return cudaErrorInvalidConfiguration;
    }

    for (size_t i = 0; i < num_chunks; ++i) {
        if (chunk_states[i] == nullptr) {
            return cudaErrorInvalidValue;
        }
    }

    const unsigned int grid = blocks_for_elements(elems_per_chunk);
    for (size_t i = 1; i < num_chunks; ++i) {
        const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(chunk_states[i - 1]);
        __nv_bfloat16* dst = static_cast<__nv_bfloat16*>(chunk_states[i]);
        add_state_kernel<<<grid, kBlockSize, 0, stream>>>(src, dst, elems_per_chunk);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}

}
