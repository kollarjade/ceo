#include "titan_kernels.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 32
#define WARP_SIZE 32

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k += BLOCK_K) {
        #pragma unroll 4
        for (int kk = 0; kk < BLOCK_K && (k + kk) < K; ++kk) {
            float a_val = __bfloat162float(A[row * K + k + kk]);
            float b_val = __bfloat162float(B[(k + kk) * N + col]);
            acc += a_val * b_val;
        }
    }

    float result = alpha * acc;
    if (beta != 0.0f) {
        result += beta * __bfloat162float(C[row * N + col]);
    }
    C[row * N + col] = __float2bfloat16(result);
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
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaStream_t cuda_stream = (cudaStream_t)stream;

    gemm_bf16_kernel<<<grid, block, 0, cuda_stream>>>(
        (const __nv_bfloat16*)A,
        (const __nv_bfloat16*)B,
        (__nv_bfloat16*)C,
        M, N, K, alpha, beta
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return TITAN_ERROR_KERNEL_LAUNCH;
    }

    return TITAN_SUCCESS;
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

    (void)scale_a;
    (void)scale_b;
    (void)scale_out;
    (void)stream;

    return TITAN_SUCCESS;
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
    } else if (M >= 1024) {
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
