#include "titan_kernels.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        float a_val = __bfloat162float(A[row * K + kk]);
        float b_val = __bfloat162float(B[kk * N + col]);
        acc += a_val * b_val;
    }

    float result = alpha * acc;
    if (beta != 0.0f) {
        result += beta * __bfloat162float(C[row * N + col]);
    }
    C[row * N + col] = __float2bfloat16(result);
}

__global__ void gemm_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    __nv_fp8_e4m3* __restrict__ C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_out
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        float a_val = static_cast<float>(A[row * K + kk]) * scale_a;
        float b_val = static_cast<float>(B[kk * N + col]) * scale_b;
        acc += a_val * b_val;
    }

    C[row * N + col] = static_cast<__nv_fp8_e4m3>(acc * scale_out);
}

extern "C" titan_status_t titan_gemm_bf16(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    titan_stream_t stream
) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((N + block.x - 1) / block.x),
        static_cast<unsigned int>((M + block.y - 1) / block.y)
    );

    gemm_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        static_cast<const __nv_bfloat16*>(A),
        static_cast<const __nv_bfloat16*>(B),
        static_cast<__nv_bfloat16*>(C),
        M, N, K, alpha, beta
    );

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_gemm_fp8(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_out,
    titan_stream_t stream
) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((N + block.x - 1) / block.x),
        static_cast<unsigned int>((M + block.y - 1) / block.y)
    );

    gemm_fp8_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        static_cast<const __nv_fp8_e4m3*>(A),
        static_cast<const __nv_fp8_e4m3*>(B),
        static_cast<__nv_fp8_e4m3*>(C),
        M, N, K, scale_a, scale_b, scale_out
    );

    return (cudaGetLastError() == cudaSuccess) ? TITAN_SUCCESS : TITAN_ERROR_KERNEL_LAUNCH;
}

extern "C" titan_status_t titan_gemm_autotune(
    int M, int N, int K,
    titan_gemm_config_t* best_config
) {
    if (!best_config || M <= 0 || N <= 0 || K <= 0) {
        return TITAN_ERROR_INVALID_ARGUMENT;
    }

    if (M >= 4096 && N >= 4096) {
        best_config->block_m = 256;
        best_config->block_n = 128;
        best_config->block_k = 64;
        best_config->stages = 4;
        best_config->split_k = 1;
    } else if (M >= 1024 || N >= 1024) {
        best_config->block_m = 128;
        best_config->block_n = 128;
        best_config->block_k = 32;
        best_config->stages = 3;
        best_config->split_k = 1;
    } else {
        best_config->block_m = 64;
        best_config->block_n = 64;
        best_config->block_k = 32;
        best_config->stages = 2;
        best_config->split_k = 1;
    }

    return TITAN_SUCCESS;
}
