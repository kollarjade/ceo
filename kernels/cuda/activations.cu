#include "titan_kernels.h"
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu_device(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ float gelu_backward_device(float x, float grad_out) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    float tanh_inner = tanhf(inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
    return grad_out * (0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner);
}

__global__ void gelu_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu_device(input[idx]);
    }
}

__global__ void gelu_backward_kernel(
    const float* input, const float* grad_output,
    float* grad_input, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = gelu_backward_device(input[idx], grad_output[idx]);
    }
}

__global__ void silu_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

extern "C" titan_status_t titan_gelu_forward(
    const float* input, float* output, int n,
    titan_stream_t stream
) {
    if (!input || !output || n <= 0) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    gelu_forward_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(input, output, n);

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_gelu_backward(
    const float* input, const float* grad_output, float* grad_input, int n,
    titan_stream_t stream
) {
    if (!input || !grad_output || !grad_input || n <= 0) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    gelu_backward_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        input, grad_output, grad_input, n
    );

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_silu_forward(
    const float* input, float* output, int n,
    titan_stream_t stream
) {
    if (!input || !output || n <= 0) return TITAN_ERROR_INVALID_ARGUMENT;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    silu_forward_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(input, output, n);

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}
