#include "include/kernels.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMaxGridX = 65535;

inline bool is_finite_float(float x) {
    return std::isfinite(static_cast<double>(x));
}

inline size_t ceil_div_size_t(size_t a, size_t b) {
    return (a + b - 1) / b;
}

inline int bounded_num_blocks(size_t numel) {
    if (numel == 0) return 0;
    size_t blocks = ceil_div_size_t(numel, static_cast<size_t>(kBlockSize));
    if (blocks > kMaxGridX) blocks = kMaxGridX;
    return static_cast<int>(blocks);
}

inline bool safe_mul_size(size_t a, size_t b, size_t* out) {
    if (out == nullptr) return false;
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > static_cast<size_t>(-1) / b) return false;
    *out = a * b;
    return true;
}

inline cudaError_t get_pointer_kind(const void* ptr, bool* is_device_like) {
    if (is_device_like == nullptr || ptr == nullptr) return cudaErrorInvalidValue;
    *is_device_like = false;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err == cudaSuccess) {
#if CUDART_VERSION >= 10000
        cudaMemoryType type = attr.type;
        *is_device_like = (type == cudaMemoryTypeDevice || type == cudaMemoryTypeManaged);
#else
        cudaMemoryType type = attr.memoryType;
        *is_device_like = (type == cudaMemoryTypeDevice);
#endif
        return cudaSuccess;
    }
    if (err == cudaErrorInvalidValue) {
        cudaGetLastError();
        *is_device_like = false;
        return cudaSuccess;
    }
    return err;
}

template <typename T>
inline cudaError_t release_cuda_ptr(T** ptr) {
    if (ptr != nullptr && *ptr != nullptr) {
        cudaError_t err = cudaFree(*ptr);
        *ptr = nullptr;
        return err;
    }
    return cudaSuccess;
}

__device__ __forceinline__ double atomic_add_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void write_scalar_kernel(float* dst, float value) {
    dst[0] = value;
}

__global__ void compute_inv_norm_kernel(const double* norm_sq, float* inv_norm) {
    const double v = norm_sq[0];
    if (!(v > 0.0) || !isfinite(v)) {
        inv_norm[0] = 0.0f;
    } else {
        inv_norm[0] = static_cast<float>(1.0 / sqrt(v));
    }
}

__global__ void compute_norm_kernel(const double* norm_sq, float* norm) {
    const double v = norm_sq[0];
    if (!(v >= 0.0) || !isfinite(v)) {
        norm[0] = 0.0f;
    } else {
        norm[0] = static_cast<float>(sqrt(v));
    }
}

__global__ void compute_clip_scale_kernel(const double* norm_sq, float* scale, float max_norm) {
    const double v = norm_sq[0];
    if (!(v > 0.0) || !isfinite(v) || !(max_norm >= 0.0f)) {
        scale[0] = 1.0f;
        return;
    }
    const double norm = sqrt(v);
    if (norm > static_cast<double>(max_norm) && max_norm >= 0.0f) {
        scale[0] = static_cast<float>(static_cast<double>(max_norm) / norm);
    } else {
        scale[0] = 1.0f;
    }
}

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
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float p = __bfloat162float(param[idx]);
        float g = __bfloat162float(grad[idx]);
        float m = __bfloat162float(momentum[idx]);
        float v = beta1 * m + (1.0f - beta1) * g;
        float m_new = beta2 * m + (1.0f - beta2) * g;
        float sign_v = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
        float p_new = p * (1.0f - lr * weight_decay) - lr * sign_v;
        param[idx] = __float2bfloat16_rn(p_new);
        momentum[idx] = __float2bfloat16_rn(m_new);
    }
}

__global__ void adamw_step_kernel(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ exp_avg,
    __nv_bfloat16* __restrict__ exp_avg_sq,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float bc1,
    float bc2,
    float eps,
    float weight_decay
) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float p = __bfloat162float(param[idx]);
        float g = __bfloat162float(grad[idx]);
        float ea = __bfloat162float(exp_avg[idx]);
        float eas = __bfloat162float(exp_avg_sq[idx]);
        float ea_new = beta1 * ea + (1.0f - beta1) * g;
        float eas_new = beta2 * eas + (1.0f - beta2) * g * g;
        float ea_hat = ea_new / bc1;
        float eas_hat = eas_new / bc2;
        float denom = sqrtf(fmaxf(eas_hat, 0.0f)) + eps;
        float p_new = p * (1.0f - lr * weight_decay) - lr * (ea_hat / denom);
        param[idx] = __float2bfloat16_rn(p_new);
        exp_avg[idx] = __float2bfloat16_rn(ea_new);
        exp_avg_sq[idx] = __float2bfloat16_rn(eas_new);
    }
}

__global__ void update_momentum_bf16_kernel(__nv_bfloat16* __restrict__ momentum, const __nv_bfloat16* __restrict__ grad, size_t numel, float beta) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float mom = __bfloat162float(momentum[idx]);
        float g = __bfloat162float(grad[idx]);
        momentum[idx] = __float2bfloat16_rn(beta * mom + g);
    }
}

__global__ void reduce_sum_squares_bf16_kernel(const __nv_bfloat16* __restrict__ data, size_t numel, double* __restrict__ out) {
    __shared__ double shared[kBlockSize];
    double local_sum = 0.0;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float v = __bfloat162float(data[idx]);
        local_sum += static_cast<double>(v) * static_cast<double>(v);
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomic_add_double(out, shared[0]);
}

__global__ void clip_grad_kernel(__nv_bfloat16* __restrict__ grad, size_t numel, const float* __restrict__ scale) {
    const float s = scale[0];
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float g = __bfloat162float(grad[idx]);
        grad[idx] = __float2bfloat16_rn(g * s);
    }
}

__global__ void bf16_to_float_scaled_kernel(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t numel, const float* __restrict__ scale) {
    const float s = scale[0];
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        dst[idx] = __bfloat162float(src[idx]) * s;
    }
}

__global__ void gram_right_kernel(const float* __restrict__ y, float* __restrict__ gram, size_t m, size_t n) {
    size_t total = n * n;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / n;
        size_t col = idx % n;
        double sum = 0.0;
        for (size_t k = 0; k < m; ++k) sum += static_cast<double>(y[k * n + row]) * static_cast<double>(y[k * n + col]);
        gram[idx] = static_cast<float>(sum);
    }
}

__global__ void gram_left_kernel(const float* __restrict__ y, float* __restrict__ gram, size_t m, size_t n) {
    size_t total = m * m;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / m;
        size_t col = idx % m;
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) sum += static_cast<double>(y[row * n + k]) * static_cast<double>(y[col * n + k]);
        gram[idx] = static_cast<float>(sum);
    }
}

__global__ void three_i_minus_kernel(float* __restrict__ gram, size_t dim) {
    size_t total = dim * dim;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / dim;
        size_t col = idx % dim;
        float identity = (row == col) ? 3.0f : 0.0f;
        gram[idx] = identity - gram[idx];
    }
}

__global__ void right_update_kernel(const float* __restrict__ y, const float* __restrict__ transform, float* __restrict__ y_next, size_t m, size_t n) {
    size_t total = m * n;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / n;
        size_t col = idx % n;
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) sum += static_cast<double>(y[row * n + k]) * static_cast<double>(transform[k * n + col]);
        y_next[idx] = 0.5f * static_cast<float>(sum);
    }
}

__global__ void left_update_kernel(const float* __restrict__ transform, const float* __restrict__ y, float* __restrict__ y_next, size_t m, size_t n) {
    size_t total = m * n;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / n;
        size_t col = idx % n;
        double sum = 0.0;
        for (size_t k = 0; k < m; ++k) sum += static_cast<double>(transform[row * m + k]) * static_cast<double>(y[k * n + col]);
        y_next[idx] = 0.5f * static_cast<float>(sum);
    }
}

__global__ void apply_muon_update_kernel(__nv_bfloat16* __restrict__ param, const float* __restrict__ update, size_t numel, float lr) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float p = __bfloat162float(param[idx]);
        param[idx] = __float2bfloat16_rn(p - lr * update[idx]);
    }
}

inline cudaError_t write_scalar_output(float* dst, float value, cudaStream_t stream) {
    if (dst == nullptr) return cudaErrorInvalidValue;
    bool is_device_like = false;
    cudaError_t err = get_pointer_kind(dst, &is_device_like);
    if (err != cudaSuccess) return err;
    if (is_device_like) {
        write_scalar_kernel<<<1, 1, 0, stream>>>(dst, value);
        return cudaPeekAtLastError();
    }
    *dst = value;
    return cudaSuccess;
}

} 

extern "C" {

cudaError_t lion_step_cuda(void* param, const void* grad, void* momentum, size_t numel, float lr, float beta1, float beta2, float weight_decay, cudaStream_t stream) {
    if (numel == 0) return cudaSuccess;
    if (param == nullptr || grad == nullptr || momentum == nullptr) return cudaErrorInvalidValue;
    if (!is_finite_float(lr) || !is_finite_float(beta1) || !is_finite_float(beta2) || !is_finite_float(weight_decay)) return cudaErrorInvalidValue;
    if (beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f) return cudaErrorInvalidValue;
    int num_blocks = bounded_num_blocks(numel);
    lion_step_kernel<<<num_blocks, kBlockSize, 0, stream>>>(static_cast<__nv_bfloat16*>(param), static_cast<const __nv_bfloat16*>(grad), static_cast<__nv_bfloat16*>(momentum), numel, lr, beta1, beta2, weight_decay);
    return cudaPeekAtLastError();
}

cudaError_t muon_step_cuda(void* param, const void* grad, void* momentum, size_t m, size_t n, float lr, float beta, size_t ns_iterations, cudaStream_t stream) {
    if (m == 0 || n == 0) return cudaSuccess;
    if (param == nullptr || grad == nullptr || momentum == nullptr) return cudaErrorInvalidValue;
    if (!is_finite_float(lr) || !is_finite_float(beta)) return cudaErrorInvalidValue;
    if (beta < 0.0f || beta >= 1.0f) return cudaErrorInvalidValue;

    size_t numel = 0;
    if (!safe_mul_size(m, n, &numel)) return cudaErrorInvalidValue;

    size_t gram_dim = (m >= n) ? n : m;
    size_t y_bytes = 0;
    size_t gram_elems = 0;
    size_t gram_bytes = 0;
    if (!safe_mul_size(numel, sizeof(float), &y_bytes) || !safe_mul_size(gram_dim, gram_dim, &gram_elems) || !safe_mul_size(gram_elems, sizeof(float), &gram_bytes)) return cudaErrorInvalidValue;

    float* y = nullptr;
    float* y_next = nullptr;
    float* gram = nullptr;
    double* d_norm = nullptr;
    float* d_inv_norm = nullptr;

    cudaError_t err = cudaMalloc(&y, y_bytes);
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&y_next, y_bytes);
    if (err != cudaSuccess) {
        release_cuda_ptr(&y);
        return err;
    }
    err = cudaMalloc(&gram, gram_bytes);
    if (err != cudaSuccess) {
        release_cuda_ptr(&y_next);
        release_cuda_ptr(&y);
        return err;
    }
    err = cudaMalloc(&d_norm, sizeof(double));
    if (err != cudaSuccess) {
        release_cuda_ptr(&gram);
        release_cuda_ptr(&y_next);
        release_cuda_ptr(&y);
        return err;
    }
    err = cudaMalloc(&d_inv_norm, sizeof(float));
    if (err != cudaSuccess) {
        release_cuda_ptr(&d_norm);
        release_cuda_ptr(&gram);
        release_cuda_ptr(&y_next);
        release_cuda_ptr(&y);
        return err;
    }

    int num_blocks = bounded_num_blocks(numel);
    update_momentum_bf16_kernel<<<num_blocks, kBlockSize, 0, stream>>>(static_cast<__nv_bfloat16*>(momentum), static_cast<const __nv_bfloat16*>(grad), numel, beta);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemsetAsync(d_norm, 0, sizeof(double), stream);
    if (err != cudaSuccess) goto cleanup;

    reduce_sum_squares_bf16_kernel<<<num_blocks, kBlockSize, 0, stream>>>(static_cast<const __nv_bfloat16*>(momentum), numel, d_norm);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) goto cleanup;

    compute_inv_norm_kernel<<<1, 1, 0, stream>>>(d_norm, d_inv_norm);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) goto cleanup;

    bf16_to_float_scaled_kernel<<<num_blocks, kBlockSize, 0, stream>>>(static_cast<const __nv_bfloat16*>(momentum), y, numel, d_inv_norm);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) goto cleanup;

    int gram_blocks = bounded_num_blocks(gram_elems);
    for (size_t iter = 0; iter < ns_iterations; ++iter) {
        if (m >= n) {
            gram_right_kernel<<<gram_blocks, kBlockSize, 0, stream>>>(y, gram, m, n);
            err = cudaPeekAtLastError();
            if (err != cudaSuccess) goto cleanup;
            three_i_minus_kernel<<<gram_blocks, kBlockSize, 0, stream>>>(gram, n);
            err = cudaPeekAtLastError();
            if (err != cudaSuccess) goto cleanup;
            right_update_kernel<<<num_blocks, kBlockSize, 0, stream>>>(y, gram, y_next, m, n);
            err = cudaPeekAtLastError();
            if (err != cudaSuccess) goto cleanup;
        } else {
            gram_left_kernel<<<gram_blocks, kBlockSize, 0, stream>>>(y, gram, m, n);
            err = cudaPeekAtLastError();
            if (err != cudaSuccess) goto cleanup;
            three_i_minus_kernel<<<gram_blocks, kBlockSize, 0, stream>>>(gram, m);
            err = cudaPeekAtLastError();
            if (err != cudaSuccess) goto cleanup;
            left_update_kernel<<<num_blocks, kBlockSize, 0, stream>>>(gram, y, y_next, m, n);
            err = cudaPeekAtLastError();
            if (err != cudaSuccess) goto cleanup;
        }
        float* tmp = y;
        y = y_next;
        y_next = tmp;
    }

    apply_muon_update_kernel<<<num_blocks, kBlockSize, 0, stream>>>(static_cast<__nv_bfloat16*>(param), y, numel, lr);
    err = cudaPeekAtLastError();

cleanup:
    {
        cudaError_t free_err = cudaSuccess;
        cudaError_t tmp_err = release_cuda_ptr(&d_inv_norm);
        if (free_err == cudaSuccess && tmp_err != cudaSuccess) free_err = tmp_err;
        tmp_err = release_cuda_ptr(&d_norm);
        if (free_err == cudaSuccess && tmp_err != cudaSuccess) free_err = tmp_err;
        tmp_err = release_cuda_ptr(&gram);
        if (free_err == cudaSuccess && tmp_err != cudaSuccess) free_err = tmp_err;
        tmp_err = release_cuda_ptr(&y_next);
        if (free_err == cudaSuccess && tmp_err != cudaSuccess) free_err = tmp_err;
        tmp_err = release_cuda_ptr(&y);
        if (free_err == cudaSuccess && tmp_err != cudaSuccess) free_err = tmp_err;
        if (err != cudaSuccess) return err;
        return free_err;
    }
}

cudaError_t adamw_step_cuda(void* param, const void* grad, void* exp_avg, void* exp_avg_sq, size_t numel, float lr, float beta1, float beta2, float eps, float weight_decay, size_t step, cudaStream_t stream) {
    if (numel == 0) return cudaSuccess;
    if (param == nullptr || grad == nullptr || exp_avg == nullptr || exp_avg_sq == nullptr) return cudaErrorInvalidValue;
    if (!is_finite_float(lr) || !is_finite_float(beta1) || !is_finite_float(beta2) || !is_finite_float(eps) || !is_finite_float(weight_decay)) return cudaErrorInvalidValue;
    if (beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || eps <= 0.0f || step == 0) return cudaErrorInvalidValue;

    double bc1_d = 1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(step));
    double bc2_d = 1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(step));
    if (!(bc1_d > 0.0) || !(bc2_d > 0.0) || !std::isfinite(bc1_d) || !std::isfinite(bc2_d)) return cudaErrorInvalidValue;

    int num_blocks = bounded_num_blocks(numel);
    adamw_step_kernel<<<num_blocks, kBlockSize, 0, stream>>>(static_cast<__nv_bfloat16*>(param), static_cast<const __nv_bfloat16*>(grad), static_cast<__nv_bfloat16*>(exp_avg), static_cast<__nv_bfloat16*>(exp_avg_sq), numel, lr, beta1, beta2, static_cast<float>(bc1_d), static_cast<float>(bc2_d), eps, weight_decay);
    return cudaPeekAtLastError();
}

cudaError_t clip_grad_norm_cuda(void** grads, const size_t* numels, size_t num_params, float max_norm, float* global_norm, cudaStream_t stream) {
    if (!is_finite_float(max_norm) || max_norm < 0.0f) return cudaErrorInvalidValue;
    if (global_norm == nullptr) return cudaErrorInvalidValue;
    if (num_params == 0) return write_scalar_output(global_norm, 0.0f, stream);
    if (grads == nullptr || numels == nullptr) return cudaErrorInvalidValue;

    double* d_norm = nullptr;
    float* d_scale = nullptr;
    cudaError_t err = cudaMalloc(&d_norm, sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_scale, sizeof(float));
    if (err != cudaSuccess) {
        release_cuda_ptr(&d_norm);
        return err;
    }

    err = cudaMemsetAsync(d_norm, 0, sizeof(double), stream);
    if (err != cudaSuccess) goto cleanup;

    for (size_t i = 0; i < num_params; ++i) {
        if (numels[i] == 0) continue;
        if (grads[i] == nullptr) {
            err = cudaErrorInvalidValue;
            goto cleanup;
        }
        int num_blocks = bounded_num_blocks(numels[i]);
        reduce_sum_squares_bf16_kernel<<<num_blocks, kBlockSize, 0, stream>>>(static_cast<const __nv_bfloat16*>(grads[i]), numels[i], d_norm);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    compute_clip_scale_kernel<<<1, 1, 0, stream>>>(d_norm, d_scale, max_norm);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) goto cleanup;

    bool global_norm_is_device = false;
    err = get_pointer_kind(global_norm, &global_norm_is_device);
    if (err != cudaSuccess) goto cleanup;
    if (global_norm_is_device) {
        compute_norm_kernel<<<1, 1, 0, stream>>>(d_norm, global_norm);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) goto cleanup;
    } else {
        double h_norm_sq = 0.0;
        err = cudaMemcpy(&h_norm_sq, d_norm, sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto cleanup;
        float h_norm = (!(h_norm_sq >= 0.0) || !std::isfinite(h_norm_sq)) ? 0.0f : static_cast<float>(std::sqrt(h_norm_sq));
        *global_norm = h_norm;
    }

    for (size_t i = 0; i < num_params; ++i) {
        if (numels[i] == 0) continue;
        int num_blocks = bounded_num_blocks(numels[i]);
        clip_grad_kernel<<<num_blocks, kBlockSize, 0, stream>>>(static_cast<__nv_bfloat16*>(grads[i]), numels[i], d_scale);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) goto cleanup;
    }

cleanup:
    {
        cudaError_t free_err = release_cuda_ptr(&d_scale);
        cudaError_t tmp_err = release_cuda_ptr(&d_norm);
        if (free_err == cudaSuccess && tmp_err != cudaSuccess) free_err = tmp_err;
        if (err != cudaSuccess) return err;
        return free_err;
    }
}

}
