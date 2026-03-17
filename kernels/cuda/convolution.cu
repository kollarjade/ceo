#include "titan_kernels.h"
#include <cuda_runtime.h>

__global__ void short_conv_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int seq_len, int channels, int window_size
) {
    int batch_idx = blockIdx.z;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= seq_len || c >= channels) return;

    float sum = 0.0f;
    for (int j = 0; j < window_size; j++) {
        int src_t = t - j;
        if (src_t >= 0) {
            int input_idx = (batch_idx * seq_len + src_t) * channels + c;
            sum += input[input_idx] * weights[c * window_size + j];
        }
    }

    int output_idx = (batch_idx * seq_len + t) * channels + c;
    output[output_idx] = sum;
}

__global__ void short_conv_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weights,
    int seq_len, int channels, int window_size
) {
    int batch_idx = blockIdx.z;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= seq_len || c >= channels) return;

    int go_idx = (batch_idx * seq_len + t) * channels + c;
    float go_val = grad_output[go_idx];

    for (int j = 0; j < window_size; j++) {
        int src_t = t - j;
        if (src_t >= 0) {
            int input_idx = (batch_idx * seq_len + src_t) * channels + c;
            atomicAdd(&grad_input[input_idx], go_val * weights[c * window_size + j]);
            atomicAdd(&grad_weights[c * window_size + j], go_val * input[input_idx]);
        }
    }
}

extern "C" titan_status_t titan_short_conv_forward(
    const float* input, const float* weights, float* output,
    int batch_size, int seq_len, int channels, int window_size,
    titan_stream_t stream
) {
    if (!input || !weights || !output) return TITAN_ERROR_INVALID_ARGUMENT;

    dim3 block(16, 16);
    dim3 grid(
        (seq_len + block.x - 1) / block.x,
        (channels + block.y - 1) / block.y,
        batch_size
    );

    short_conv_forward_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        input, weights, output, seq_len, channels, window_size
    );

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_short_conv_backward(
    const float* grad_output, const float* input, const float* weights,
    float* grad_input, float* grad_weights,
    int batch_size, int seq_len, int channels, int window_size,
    titan_stream_t stream
) {
    if (!grad_output || !input || !weights || !grad_input || !grad_weights) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    dim3 block(16, 16);
    dim3 grid(
        (seq_len + block.x - 1) / block.x,
        (channels + block.y - 1) / block.y,
        batch_size
    );

    short_conv_backward_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        grad_output, input, weights, grad_input, grad_weights,
        seq_len, channels, window_size
    );

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}
