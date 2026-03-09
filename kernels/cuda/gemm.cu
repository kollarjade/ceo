#include "include/kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

template <typename T>
__device__ __forceinline__ float scalar_to_float(T v);

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float scalar_to_float<float>(float v) {
    return v;
}

template <typename T>
__device__ __forceinline__ T float_to_scalar(float v);

template <>
__device__ __forceinline__ __half float_to_scalar<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float float_to_scalar<float>(float v) {
    return v;
}

template <typename T>
__global__ void add_bias_kernel(T* c, const T* bias, size_t m, size_t n) {
    const size_t total = m * n;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t col = idx % n;
        const float value = scalar_to_float<T>(c[idx]) + scalar_to_float<T>(bias[col]);
        c[idx] = float_to_scalar<T>(value);
    }
}

template <typename T>
__global__ void column_sum_kernel(const T* x, T* y, size_t m, size_t n) {
    for (size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         col < n;
         col += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float sum = 0.0f;
        for (size_t row = 0; row < m; ++row) {
            sum += scalar_to_float<T>(x[row * n + col]);
        }
        y[col] = float_to_scalar<T>(sum);
    }
}

bool safe_mul_size(size_t a, size_t b, size_t* out) {
    if (out == nullptr) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    *out = a * b;
    return true;
}

bool normalize_dtype(int dtype, cudaDataType_t* out) {
    if (out == nullptr) {
        return false;
    }
    switch (static_cast<cudaDataType_t>(dtype)) {
        case CUDA_R_16F:
            *out = CUDA_R_16F;
            return true;
        case CUDA_R_16BF:
            *out = CUDA_R_16BF;
            return true;
        case CUDA_R_32F:
            *out = CUDA_R_32F;
            return true;
        default:
            return false;
    }
}

size_t element_size(cudaDataType_t type) {
    switch (type) {
        case CUDA_R_16F:
            return sizeof(__half);
        case CUDA_R_16BF:
            return sizeof(__nv_bfloat16);
        case CUDA_R_32F:
            return sizeof(float);
        default:
            return 0;
    }
}

cudaError_t cublas_status_to_cuda_error(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return cudaSuccess;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return cudaErrorInitializationError;
        case CUBLAS_STATUS_ALLOC_FAILED:
            return cudaErrorMemoryAllocation;
        case CUBLAS_STATUS_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return cudaErrorInvalidDeviceFunction;
        case CUBLAS_STATUS_MAPPING_ERROR:
            return cudaErrorMapBufferObjectFailed;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return cudaErrorLaunchFailure;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return cudaErrorUnknown;
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return cudaErrorNotSupported;
        default:
            return cudaErrorUnknown;
    }
}

cudaError_t create_cublas_handle(cudaStream_t stream, cublasHandle_t* handle) {
    if (handle == nullptr) {
        return cudaErrorInvalidValue;
    }
    *handle = nullptr;
    cublasStatus_t status = cublasCreate(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return cublas_status_to_cuda_error(status);
    }
    status = cublasSetStream(*handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(*handle);
        *handle = nullptr;
        return cublas_status_to_cuda_error(status);
    }
    return cudaSuccess;
}

cudaError_t destroy_cublas_handle(cublasHandle_t handle) {
    if (handle == nullptr) {
        return cudaSuccess;
    }
    return cublas_status_to_cuda_error(cublasDestroy(handle));
}

cudaError_t zero_buffer_async(void* ptr, size_t elements, size_t elem_size, cudaStream_t stream) {
    if (ptr == nullptr || elements == 0) {
        return cudaSuccess;
    }
    size_t bytes = 0;
    if (!safe_mul_size(elements, elem_size, &bytes)) {
        return cudaErrorInvalidValue;
    }
    return cudaMemsetAsync(ptr, 0, bytes, stream);
}

cudaError_t gemm_row_major_ex(
    cublasHandle_t handle,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    size_t m,
    size_t n,
    size_t k,
    const void* a,
    cudaDataType_t type_a,
    size_t lda,
    const void* b,
    cudaDataType_t type_b,
    size_t ldb,
    void* c,
    cudaDataType_t type_c,
    size_t ldc
) {
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        lda > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ldb > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ldc > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidValue;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        handle,
        op_b,
        op_a,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        b,
        type_b,
        static_cast<int>(ldb),
        a,
        type_a,
        static_cast<int>(lda),
        &beta,
        c,
        type_c,
        static_cast<int>(ldc),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    return cublas_status_to_cuda_error(status);
}

cudaError_t launch_add_bias(
    void* c,
    const void* bias,
    size_t m,
    size_t n,
    cudaDataType_t type,
    cudaStream_t stream
) {
    if (c == nullptr || bias == nullptr || m == 0 || n == 0) {
        return cudaSuccess;
    }
    size_t total = 0;
    if (!safe_mul_size(m, n, &total)) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((total + threads - 1) / threads, 4096));
    switch (type) {
        case CUDA_R_16F:
            add_bias_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<__half*>(c),
                static_cast<const __half*>(bias),
                m,
                n
            );
            return cudaPeekAtLastError();
        case CUDA_R_16BF:
            add_bias_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<__nv_bfloat16*>(c),
                static_cast<const __nv_bfloat16*>(bias),
                m,
                n
            );
            return cudaPeekAtLastError();
        case CUDA_R_32F:
            add_bias_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<float*>(c),
                static_cast<const float*>(bias),
                m,
                n
            );
            return cudaPeekAtLastError();
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t launch_column_sum_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    size_t m,
    size_t n,
    cudaStream_t stream
) {
    if (x == nullptr || y == nullptr || n == 0) {
        return cudaSuccess;
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((n + threads - 1) / threads, 4096));
    column_sum_kernel<<<blocks, threads, 0, stream>>>(x, y, m, n);
    return cudaPeekAtLastError();
}

} 

extern "C" {

cudaError_t gemm_forward_cuda(
    const void* a,
    const void* b,
    const void* bias,
    void* c,
    size_t m,
    size_t k,
    size_t n,
    int dtype_a,
    int dtype_b,
    int dtype_c,
    cudaStream_t stream
) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaDataType_t type_a;
    cudaDataType_t type_b;
    cudaDataType_t type_c;

    if (!normalize_dtype(dtype_a, &type_a) ||
        !normalize_dtype(dtype_b, &type_b) ||
        !normalize_dtype(dtype_c, &type_c)) {
        return cudaErrorInvalidValue;
    }

    const size_t c_elem_size = element_size(type_c);
    if (c_elem_size == 0) {
        return cudaErrorInvalidValue;
    }

    size_t c_elements = 0;
    if (!safe_mul_size(m, n, &c_elements)) {
        return cudaErrorInvalidValue;
    }

    if (m == 0 || n == 0) {
        return cudaSuccess;
    }

    if (k == 0) {
        cudaError_t err = zero_buffer_async(c, c_elements, c_elem_size, stream);
        if (err != cudaSuccess) {
            return err;
        }
        if (bias != nullptr) {
            err = launch_add_bias(c, bias, m, n, type_c, stream);
            if (err != cudaSuccess) {
                return err;
            }
        }
        return cudaSuccess;
    }

    cublasHandle_t handle = nullptr;
    cudaError_t err = create_cublas_handle(stream, &handle);
    if (err != cudaSuccess) {
        return err;
    }

    err = gemm_row_major_ex(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        a,
        type_a,
        k,
        b,
        type_b,
        n,
        c,
        type_c,
        n
    );

    cudaError_t destroy_err = destroy_cublas_handle(handle);
    if (err != cudaSuccess) {
        return err;
    }
    if (destroy_err != cudaSuccess) {
        return destroy_err;
    }

    if (bias != nullptr) {
        err = launch_add_bias(c, bias, m, n, type_c, stream);
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}

cudaError_t gemm_backward_cuda(
    const void* grad_c,
    const void* a,
    const void* b,
    void* grad_a,
    void* grad_b,
    void* grad_bias,
    size_t m,
    size_t k,
    size_t n,
    cudaStream_t stream
) {
    const size_t bf16_size = sizeof(__nv_bfloat16);

    size_t grad_a_elements = 0;
    size_t grad_b_elements = 0;

    if (!safe_mul_size(m, k, &grad_a_elements) ||
        !safe_mul_size(k, n, &grad_b_elements)) {
        return cudaErrorInvalidValue;
    }

    const bool need_grad_a = grad_a != nullptr;
    const bool need_grad_b = grad_b != nullptr;
    const bool need_grad_bias = grad_bias != nullptr;

    if (!need_grad_a && !need_grad_b && !need_grad_bias) {
        return cudaSuccess;
    }

    if (need_grad_a) {
        if (m == 0 || k == 0 || n == 0) {
            cudaError_t err = zero_buffer_async(grad_a, grad_a_elements, bf16_size, stream);
            if (err != cudaSuccess) {
                return err;
            }
        } else if (grad_c == nullptr || b == nullptr) {
            return cudaErrorInvalidValue;
        }
    }

    if (need_grad_b) {
        if (k == 0 || n == 0 || m == 0) {
            cudaError_t err = zero_buffer_async(grad_b, grad_b_elements, bf16_size, stream);
            if (err != cudaSuccess) {
                return err;
            }
        } else if (a == nullptr || grad_c == nullptr) {
            return cudaErrorInvalidValue;
        }
    }

    if (need_grad_bias) {
        if (n == 0) {
            return cudaSuccess;
        }
        if (m == 0) {
            cudaError_t err = zero_buffer_async(grad_bias, n, bf16_size, stream);
            if (err != cudaSuccess) {
                return err;
            }
        } else if (grad_c == nullptr) {
            return cudaErrorInvalidValue;
        }
    }

    cublasHandle_t handle = nullptr;
    if ((need_grad_a && m != 0 && k != 0 && n != 0) ||
        (need_grad_b && k != 0 && n != 0 && m != 0)) {
        cudaError_t err = create_cublas_handle(stream, &handle);
        if (err != cudaSuccess) {
            return err;
        }
    }

    if (need_grad_a && m != 0 && k != 0 && n != 0) {
        cudaError_t err = gemm_row_major_ex(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            m,
            k,
            n,
            grad_c,
            CUDA_R_16BF,
            n,
            b,
            CUDA_R_16BF,
            n,
            grad_a,
            CUDA_R_16BF,
            k
        );
        if (err != cudaSuccess) {
            cudaError_t destroy_err = destroy_cublas_handle(handle);
            if (destroy_err != cudaSuccess) {
                return destroy_err;
            }
            return err;
        }
    }

    if (need_grad_b && k != 0 && n != 0 && m != 0) {
        cudaError_t err = gemm_row_major_ex(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            k,
            n,
            m,
            a,
            CUDA_R_16BF,
            k,
            grad_c,
            CUDA_R_16BF,
            n,
            grad_b,
            CUDA_R_16BF,
            n
        );
        if (err != cudaSuccess) {
            cudaError_t destroy_err = destroy_cublas_handle(handle);
            if (destroy_err != cudaSuccess) {
                return destroy_err;
            }
            return err;
        }
    }

    cudaError_t destroy_err = destroy_cublas_handle(handle);
    if (destroy_err != cudaSuccess) {
        return destroy_err;
    }

    if (need_grad_bias && m != 0 && n != 0) {
        cudaError_t err = launch_column_sum_bf16(
            static_cast<const __nv_bfloat16*>(grad_c),
            static_cast<__nv_bfloat16*>(grad_bias),
            m,
            n,
            stream
        );
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}

}
