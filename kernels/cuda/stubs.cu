#include "include/kernels.h"

extern "C" {

cudaError_t prism_forward_cuda(
    const void*, const void*, const void*, void*, void*,
    const void* const*, const void* const*, const void* const*,
    size_t, size_t, size_t, size_t, size_t, efla_dtype_t, float, cudaStream_t
) { return cudaErrorNotSupported; }

cudaError_t shortconv_forward_cuda(
    const void*, const void*, void*, size_t, size_t, size_t, size_t, efla_dtype_t, cudaStream_t
) { return cudaErrorNotSupported; }

cudaError_t cross_entropy_forward_cuda(
    const void*, const int32_t*, void*, size_t, size_t, efla_dtype_t, efla_dtype_t, float, efla_reduction_t, cudaStream_t
) { return cudaErrorNotSupported; }

cudaError_t cross_entropy_backward_cuda(
    const void*, const void*, const int32_t*, void*, size_t, size_t, efla_dtype_t, efla_dtype_t, efla_dtype_t, float, efla_reduction_t, cudaStream_t
) { return cudaErrorNotSupported; }

cudaError_t embedding_forward_cuda(
    const int32_t*, const void*, void*, size_t, size_t, efla_dtype_t, efla_dtype_t, cudaStream_t
) { return cudaErrorNotSupported; }

cudaError_t quantize_fp8_cuda(
    const void*, void*, float*, size_t, efla_dtype_t, efla_fp8_format_t, efla_memory_kind_t, cudaStream_t
) { return cudaErrorNotSupported; }

cudaError_t dequantize_fp8_cuda(
    const void*, void*, float, size_t, efla_fp8_format_t, efla_dtype_t, cudaStream_t
) { return cudaErrorNotSupported; }

}
