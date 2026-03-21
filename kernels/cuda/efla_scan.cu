#include "titan_kernels.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

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

__device__ float compute_coefficient_device(float lambda, float beta, float threshold) {
    if (lambda < threshold) {
        float beta2 = beta * beta;
        float beta3 = beta2 * beta;
        float beta4 = beta3 * beta;
        float lambda2 = lambda * lambda;
        float lambda3 = lambda2 * lambda;
        return beta
            - 0.5f * beta2 * lambda
            + beta3 * lambda2 / 6.0f
            - beta4 * lambda3 / 24.0f;
    }
    return (1.0f - expf(-beta * lambda)) / lambda;
}

__device__ float compute_dcdlambda_device(float lambda, float beta, float threshold) {
    if (lambda < threshold) {
        float beta2 = beta * beta;
        float beta3 = beta2 * beta;
        float beta4 = beta3 * beta;
        return -0.5f * beta2
            + beta3 * lambda / 3.0f
            - beta4 * lambda * lambda / 8.0f;
    }
    float exp_term = expf(-beta * lambda);
    return (beta * lambda * exp_term - (1.0f - exp_term)) / (lambda * lambda);
}

__device__ float compute_dcdbeta_device(float lambda, float beta, float threshold) {
    if (lambda < threshold) {
        float beta2 = beta * beta;
        float beta3 = beta2 * beta;
        float lambda2 = lambda * lambda;
        float lambda3 = lambda2 * lambda;
        return 1.0f
            - beta * lambda
            + 0.5f * beta2 * lambda2
            - beta3 * lambda3 / 6.0f;
    }
    return expf(-beta * lambda);
}

__global__ void efla_scan_forward_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ values,
    float* __restrict__ output,
    float* __restrict__ state,
    int seq_len, int state_dim, int value_dim,
    float beta, float lambda_threshold
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    int state_size = state_dim * value_dim;
    float* batch_state = state + batch_idx * state_size;

    for (int t = 0; t < seq_len; ++t) {
        int k_offset = (batch_idx * seq_len + t) * state_dim;
        int v_offset = (batch_idx * seq_len + t) * value_dim;
        int o_offset = (batch_idx * seq_len + t) * value_dim;

        float lambda = 0.0f;
        for (int i = 0; i < state_dim; ++i) {
            float ki = keys[k_offset + i];
            lambda += ki * ki;
        }

        float c = compute_coefficient_device(lambda, beta, lambda_threshold);

        for (int j = tid; j < value_dim; j += blockDim.x) {
            float dot = 0.0f;
            for (int i = 0; i < state_dim; ++i) {
                dot += keys[k_offset + i] * batch_state[i * value_dim + j];
            }

            for (int i = 0; i < state_dim; ++i) {
                batch_state[i * value_dim + j] =
                    batch_state[i * value_dim + j]
                    - c * keys[k_offset + i] * dot
                    + c * keys[k_offset + i] * values[v_offset + j];
            }
        }

        __syncthreads();

        for (int j = tid; j < value_dim; j += blockDim.x) {
            float out = 0.0f;
            for (int i = 0; i < state_dim; ++i) {
                out += keys[k_offset + i] * batch_state[i * value_dim + j];
            }
            output[o_offset + j] = out;
        }

        __syncthreads();
    }
}

__global__ void efla_scan_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const float* __restrict__ saved_states,
    float* __restrict__ grad_keys,
    float* __restrict__ grad_values,
    float* __restrict__ grad_beta,
    float* __restrict__ workspace,
    int seq_len,
    int state_dim,
    int value_dim,
    float beta,
    float lambda_threshold
) {
    if (threadIdx.x != 0) {
        return;
    }

    int batch_idx = blockIdx.x;
    int state_size = state_dim * value_dim;
    int stride = state_size + 3 * value_dim;

    float* grad_state = workspace + batch_idx * stride;
    float* q = grad_state + state_size;
    float* h = q + value_dim;
    float* dldp = h + value_dim;

    for (int t = seq_len - 1; t >= 0; --t) {
        const float* key = keys + (batch_idx * seq_len + t) * state_dim;
        const float* value = values + (batch_idx * seq_len + t) * value_dim;
        const float* state = saved_states + ((batch_idx * seq_len + t) * state_size);
        const float* go = grad_output + (batch_idx * seq_len + t) * value_dim;
        float* gk = grad_keys + (batch_idx * seq_len + t) * state_dim;
        float* gv = grad_values + (batch_idx * seq_len + t) * value_dim;

        float lambda = 0.0f;
        for (int i = 0; i < state_dim; ++i) {
            lambda += key[i] * key[i];
        }

        float c = compute_coefficient_device(lambda, beta, lambda_threshold);
        float dc_dlambda = compute_dcdlambda_device(lambda, beta, lambda_threshold);
        float dc_dbeta = compute_dcdbeta_device(lambda, beta, lambda_threshold);

        for (int j = 0; j < value_dim; ++j) {
            float proj = 0.0f;
            for (int i = 0; i < state_dim; ++i) {
                proj += key[i] * state[i * value_dim + j];
            }
            q[j] = value[j] - proj;
        }

        for (int i = 0; i < state_dim; ++i) {
            float grad_from_output = 0.0f;
            for (int j = 0; j < value_dim; ++j) {
                float updated = state[i * value_dim + j] + c * key[i] * q[j];
                grad_from_output += updated * go[j];
                grad_state[i * value_dim + j] += key[i] * go[j];
            }
            gk[i] += grad_from_output;
        }

        for (int j = 0; j < value_dim; ++j) {
            float accum = 0.0f;
            for (int i = 0; i < state_dim; ++i) {
                accum += grad_state[i * value_dim + j] * key[i];
            }
            h[j] = accum;
        }

        float dldc = 0.0f;
        for (int j = 0; j < value_dim; ++j) {
            dldc += h[j] * q[j];
            gv[j] += c * h[j];
            dldp[j] = -c * h[j];
        }

        if (grad_beta != nullptr) {
            atomicAdd(grad_beta, dc_dbeta * dldc);
        }

        for (int i = 0; i < state_dim; ++i) {
            float sum_gq = 0.0f;
            float sum_sdp = 0.0f;
            for (int j = 0; j < value_dim; ++j) {
                sum_gq += grad_state[i * value_dim + j] * q[j];
                sum_sdp += state[i * value_dim + j] * dldp[j];
            }
            gk[i] += c * sum_gq + sum_sdp + 2.0f * dc_dlambda * dldc * key[i];
        }

        for (int i = 0; i < state_dim; ++i) {
            for (int j = 0; j < value_dim; ++j) {
                grad_state[i * value_dim + j] += key[i] * dldp[j];
            }
        }
    }
}

}

extern "C" titan_status_t titan_efla_scan_forward(
    const float* keys, const float* values, float* output, float* state,
    int batch_size, int seq_len, int state_dim, int value_dim,
    float beta, float lambda_threshold,
    titan_stream_t stream
) {
    if (!keys || !values || !output || !state ||
        batch_size <= 0 || seq_len <= 0 || state_dim <= 0 || value_dim <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    int threads = value_dim < 256 ? value_dim : 256;
    if (threads < 32) {
        threads = 32;
    }

    efla_scan_forward_kernel<<<batch_size, threads, 0, (cudaStream_t)stream>>>(
        keys, values, output, state,
        seq_len, state_dim, value_dim,
        beta, lambda_threshold
    );

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_efla_scan_backward(
    const float* grad_output, const float* keys, const float* values,
    const float* saved_states,
    float* grad_keys, float* grad_values, float* grad_beta,
    int batch_size, int seq_len, int state_dim, int value_dim,
    float beta, float lambda_threshold,
    titan_stream_t stream
) {
    if (!grad_output || !keys || !values || !saved_states || !grad_keys || !grad_values ||
        batch_size <= 0 || seq_len <= 0 || state_dim <= 0 || value_dim <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = (cudaStream_t)stream;
    size_t key_bytes = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * static_cast<size_t>(state_dim) * sizeof(float);
    size_t value_bytes = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * static_cast<size_t>(value_dim) * sizeof(float);
    cudaError_t err = cudaMemsetAsync(grad_keys, 0, key_bytes, cuda_stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }
    err = cudaMemsetAsync(grad_values, 0, value_bytes, cuda_stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }
    if (grad_beta != nullptr) {
        err = cudaMemsetAsync(grad_beta, 0, sizeof(float), cuda_stream);
        if (err != cudaSuccess) {
            return TITAN_ERROR_CUDA;
        }
    }

    size_t state_size = static_cast<size_t>(state_dim) * static_cast<size_t>(value_dim);
    size_t stride = state_size + 3 * static_cast<size_t>(value_dim);
    size_t workspace_elems = static_cast<size_t>(batch_size) * stride;
    float* workspace = nullptr;
    err = allocate_workspace(reinterpret_cast<void**>(&workspace), workspace_elems * sizeof(float), cuda_stream);
    if (err != cudaSuccess) {
        return TITAN_ERROR_OUT_OF_MEMORY;
    }

    err = cudaMemsetAsync(workspace, 0, workspace_elems * sizeof(float), cuda_stream);
    if (err != cudaSuccess) {
        free_workspace(workspace, cuda_stream);
        return TITAN_ERROR_CUDA;
    }

    efla_scan_backward_kernel<<<batch_size, 1, 0, cuda_stream>>>(
        grad_output,
        keys,
        values,
        saved_states,
        grad_keys,
        grad_values,
        grad_beta,
        workspace,
        seq_len,
        state_dim,
        value_dim,
        beta,
        lambda_threshold
    );

    cudaError_t launch_err = cudaGetLastError();
    cudaError_t free_err = free_workspace(workspace, cuda_stream);

    if (launch_err != cudaSuccess) {
        return TITAN_ERROR_KERNEL_LAUNCH;
    }
    if (free_err != cudaSuccess) {
        return TITAN_ERROR_CUDA;
    }
    return TITAN_SUCCESS;
}
