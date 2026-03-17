#include "titan_kernels.h"
#include <cuda_runtime.h>
#include <math.h>

__device__ float compute_coefficient_device(float lambda, float beta, float threshold) {
    if (lambda < threshold) {
        float c = beta;
        c += -0.5f * beta * beta * lambda;
        c += beta * beta * beta * lambda * lambda / 6.0f;
        c += -beta * beta * beta * beta * lambda * lambda * lambda / 24.0f;
        return c;
    }
    return (1.0f - expf(-beta * lambda)) / lambda;
}

__global__ void efla_scan_forward_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ values,
    float* __restrict__ output,
    float* __restrict__ state,
    int seq_len, int state_dim, int value_dim,
    float beta, float lambda_threshold
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    int state_size = state_dim * value_dim;
    float* head_state = state + head_idx * state_size;

    for (int t = 0; t < seq_len; t++) {
        int k_offset = (head_idx * seq_len + t) * state_dim;
        int v_offset = (head_idx * seq_len + t) * value_dim;
        int o_offset = (head_idx * seq_len + t) * value_dim;

        float lambda = 0.0f;
        for (int i = 0; i < state_dim; i++) {
            float ki = keys[k_offset + i];
            lambda += ki * ki;
        }

        float c = compute_coefficient_device(lambda, beta, lambda_threshold);

        for (int j = tid; j < value_dim; j += blockDim.x) {
            float dot = 0.0f;
            for (int i = 0; i < state_dim; i++) {
                dot += keys[k_offset + i] * head_state[i * value_dim + j];
            }

            for (int i = 0; i < state_dim; i++) {
                head_state[i * value_dim + j] =
                    head_state[i * value_dim + j]
                    - c * keys[k_offset + i] * dot
                    + c * keys[k_offset + i] * values[v_offset + j];
            }
        }

        __syncthreads();

        for (int j = tid; j < value_dim; j += blockDim.x) {
            float o = 0.0f;
            for (int i = 0; i < state_dim; i++) {
                o += keys[k_offset + i] * head_state[i * value_dim + j];
            }
            output[o_offset + j] = o;
        }

        __syncthreads();
    }
}

extern "C" titan_status_t titan_efla_scan_forward(
    const float* keys, const float* values, float* output, float* state,
    int batch_size, int seq_len, int state_dim, int value_dim,
    float beta, float lambda_threshold,
    titan_stream_t stream
) {
    if (!keys || !values || !output || !state) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = min(value_dim, 256);
    threads = max(threads, 32);

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
    (void)grad_output; (void)keys; (void)values; (void)saved_states;
    (void)grad_keys; (void)grad_values; (void)grad_beta;
    (void)batch_size; (void)seq_len; (void)state_dim; (void)value_dim;
    (void)beta; (void)lambda_threshold; (void)stream;

    return TITAN_SUCCESS;
}
