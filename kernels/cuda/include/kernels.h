#ifndef EFLA_KERNELS_H
#define EFLA_KERNELS_H
#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum efla_dtype_t {
    EFLA_DTYPE_INVALID = 0,
    EFLA_DTYPE_FLOAT16 = 1,
    EFLA_DTYPE_BFLOAT16 = 2,
    EFLA_DTYPE_FLOAT32 = 3,
    EFLA_DTYPE_FLOAT64 = 4,
    EFLA_DTYPE_INT32 = 5,
    EFLA_DTYPE_INT64 = 6,
    EFLA_DTYPE_UINT8 = 7,
    EFLA_DTYPE_INT8 = 8
} efla_dtype_t;
typedef enum efla_fp8_format_t {
    EFLA_FP8_FORMAT_INVALID = 0,
    EFLA_FP8_FORMAT_E4M3 = 1,
    EFLA_FP8_FORMAT_E5M2 = 2
} efla_fp8_format_t;
typedef enum efla_reduction_t {
    EFLA_REDUCTION_INVALID = 0,
    EFLA_REDUCTION_NONE = 1,
    EFLA_REDUCTION_SUM = 2,
    EFLA_REDUCTION_MEAN = 3
} efla_reduction_t;
typedef enum efla_norm_kind_t {
    EFLA_NORM_KIND_INVALID = 0,
    EFLA_NORM_KIND_L1 = 1,
    EFLA_NORM_KIND_L2 = 2,
    EFLA_NORM_KIND_INF = 3
} efla_norm_kind_t;
typedef enum efla_memory_kind_t {
    EFLA_MEMORY_KIND_INVALID = 0,
    EFLA_MEMORY_KIND_HOST = 1,
    EFLA_MEMORY_KIND_DEVICE = 2,
    EFLA_MEMORY_KIND_MANAGED = 3
} efla_memory_kind_t;
#define EFLA_CUDA_RETURN_IF_ERROR(call) \
    do { \
        cudaError_t efla_cuda_status__ = (call); \
        if (efla_cuda_status__ != cudaSuccess) { \
            return efla_cuda_status__; \
        } \
    } while (0)
#define EFLA_CUDA_GOTO_IF_ERROR(call, status_var, label) \
    do { \
        cudaError_t efla_cuda_status__ = (call); \
        if (efla_cuda_status__ != cudaSuccess) { \
            (status_var) = efla_cuda_status__; \
            goto label; \
        } \
    } while (0)
cudaError_t efla_forward_cuda(
    const void* k,
    const void* v,
    const void* initial_state,
    void* final_state,
    void* output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    efla_dtype_t tensor_dtype,
    float beta,
    float lambda,
    size_t chunk_size,
    cudaStream_t stream
);
cudaError_t efla_backward_cuda(
    const void* grad_output,
    const void* k,
    const void* v,
    const void* initial_state,
    const void* final_state,
    void* grad_k,
    void* grad_v,
    void* grad_initial_state,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    efla_dtype_t tensor_dtype,
    float beta,
    float lambda,
    size_t chunk_size,
    cudaStream_t stream
);
cudaError_t efla_chunked_scan_cuda(
    void* const* chunk_states,
    size_t num_chunks,
    size_t batch_size,
    size_t num_heads,
    size_t head_dim,
    efla_dtype_t state_dtype,
    cudaStream_t stream
);
cudaError_t prism_forward_cuda(
    const void* u,
    const void* v,
    const void* prev_state,
    void* new_state,
    void* output,
    const void* const* w_beta,
    const void* const* w_k,
    const void* const* w_p,
    size_t batch_size,
    size_t seq_len,
    size_t hidden_dim,
    size_t head_dim,
    size_t num_iterations,
    efla_dtype_t tensor_dtype,
    float alpha,
    cudaStream_t stream
);
cudaError_t shortconv_forward_cuda(
    const void* input,
    const void* weight,
    void* output,
    size_t batch_size,
    size_t seq_len,
    size_t hidden_dim,
    size_t window_size,
    efla_dtype_t tensor_dtype,
    cudaStream_t stream
);
cudaError_t rmsnorm_forward_cuda(
    const void* input,
    const void* weight,
    void* output,
    size_t numel,
    size_t normalized_shape,
    efla_dtype_t tensor_dtype,
    float eps,
    cudaStream_t stream
);
cudaError_t rmsnorm_backward_cuda(
    const void* grad_output,
    const void* input,
    const void* weight,
    void* grad_input,
    void* grad_weight,
    size_t numel,
    size_t normalized_shape,
    efla_dtype_t tensor_dtype,
    efla_dtype_t grad_weight_dtype,
    float eps,
    cudaStream_t stream
);
cudaError_t layernorm_forward_cuda(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    size_t numel,
    size_t normalized_shape,
    efla_dtype_t tensor_dtype,
    float eps,
    cudaStream_t stream
);
cudaError_t gelu_forward_cuda(
    const void* input,
    void* output,
    size_t numel,
    efla_dtype_t tensor_dtype,
    bool approximate,
    cudaStream_t stream
);
cudaError_t gelu_backward_cuda(
    const void* grad_output,
    const void* input,
    void* grad_input,
    size_t numel,
    efla_dtype_t tensor_dtype,
    bool approximate,
    cudaStream_t stream
);
cudaError_t softmax_forward_cuda(
    const void* input,
    void* output,
    size_t outer_size,
    size_t dim_size,
    size_t inner_size,
    efla_dtype_t tensor_dtype,
    cudaStream_t stream
);
cudaError_t gemm_forward_cuda(
    const void* a,
    const void* b,
    const void* bias,
    void* c,
    size_t m,
    size_t k,
    size_t n,
    efla_dtype_t dtype_a,
    efla_dtype_t dtype_b,
    efla_dtype_t dtype_c,
    cudaStream_t stream
);
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
    efla_dtype_t grad_c_dtype,
    efla_dtype_t a_dtype,
    efla_dtype_t b_dtype,
    efla_dtype_t grad_a_dtype,
    efla_dtype_t grad_b_dtype,
    efla_dtype_t grad_bias_dtype,
    cudaStream_t stream
);
cudaError_t cross_entropy_forward_cuda(
    const void* logits,
    const int32_t* targets,
    void* loss,
    size_t batch_size,
    size_t vocab_size,
    efla_dtype_t logits_dtype,
    efla_dtype_t loss_dtype,
    float label_smoothing,
    efla_reduction_t reduction,
    cudaStream_t stream
);
cudaError_t cross_entropy_backward_cuda(
    const void* grad_loss,
    const void* logits,
    const int32_t* targets,
    void* grad_logits,
    size_t batch_size,
    size_t vocab_size,
    efla_dtype_t grad_loss_dtype,
    efla_dtype_t logits_dtype,
    efla_dtype_t grad_logits_dtype,
    float label_smoothing,
    efla_reduction_t reduction,
    cudaStream_t stream
);
cudaError_t lion_step_cuda(
    void* param,
    const void* grad,
    void* momentum,
    size_t numel,
    efla_dtype_t tensor_dtype,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    cudaStream_t stream
);
cudaError_t muon_step_cuda(
    void* param,
    const void* grad,
    void* momentum,
    size_t m,
    size_t n,
    efla_dtype_t tensor_dtype,
    float lr,
    float beta,
    size_t ns_iterations,
    cudaStream_t stream
);
cudaError_t adamw_step_cuda(
    void* param,
    const void* grad,
    void* exp_avg,
    void* exp_avg_sq,
    size_t numel,
    efla_dtype_t tensor_dtype,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    size_t step,
    cudaStream_t stream
);
cudaError_t clip_grad_norm_cuda(
    void** grads,
    const size_t* numels,
    size_t num_params,
    efla_dtype_t tensor_dtype,
    float max_norm,
    float* global_norm,
    efla_memory_kind_t global_norm_memory_kind,
    cudaStream_t stream
);
cudaError_t fill_cuda(
    void* ptr,
    float value,
    size_t numel,
    efla_dtype_t dtype,
    cudaStream_t stream
);
cudaError_t copy_cuda(
    const void* src,
    void* dst,
    size_t numel,
    size_t element_size,
    cudaStream_t stream
);
cudaError_t cast_cuda(
    const void* src,
    void* dst,
    size_t numel,
    efla_dtype_t src_dtype,
    efla_dtype_t dst_dtype,
    cudaStream_t stream
);
cudaError_t embedding_forward_cuda(
    const int32_t* indices,
    const void* weight,
    void* output,
    size_t num_indices,
    size_t embedding_dim,
    efla_dtype_t weight_dtype,
    efla_dtype_t output_dtype,
    cudaStream_t stream
);
cudaError_t quantize_fp8_cuda(
    const void* input,
    void* output,
    float* scale,
    size_t numel,
    efla_dtype_t input_dtype,
    efla_fp8_format_t output_format,
    efla_memory_kind_t scale_memory_kind,
    cudaStream_t stream
);
cudaError_t dequantize_fp8_cuda(
    const void* input,
    void* output,
    float scale,
    size_t numel,
    efla_fp8_format_t input_format,
    efla_dtype_t output_dtype,
    cudaStream_t stream
);
cudaError_t sum_reduce_cuda(
    const void* input,
    void* output,
    size_t numel,
    efla_dtype_t dtype,
    cudaStream_t stream
);
cudaError_t norm_cuda(
    const void* input,
    float* output,
    size_t numel,
    efla_dtype_t dtype,
    efla_norm_kind_t norm_kind,
    efla_memory_kind_t output_memory_kind,
    cudaStream_t stream
);
#ifdef __cplusplus
}
#endif
#endif
