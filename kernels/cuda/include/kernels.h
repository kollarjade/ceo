/**
 * EFLA Trainer - CUDA Kernel Header
 * SM100-optimized kernels for ultra-long context training
 */

#ifndef EFLA_KERNELS_H
#define EFLA_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Error handling
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return err; \
        } \
    } while(0)

// ============================================================================
// EFLA Kernels
// ============================================================================

/**
 * EFLA forward pass - Exact closed-form state update
 *
 * Implements: S_t = (I - c_t * k_t * k_t^T) * S_{t-1} + c_t * k_t * v_t^T
 * With coefficient: c_t = (1 - exp(-beta * lambda)) / lambda
 */
cudaError_t efla_forward_cuda(
    const void* k,          // Keys: (batch, seq_len, num_heads, head_dim)
    const void* v,          // Values: (batch, seq_len, num_heads, head_dim)
    void* state,            // State: (num_heads, head_dim, head_dim)
    void* output,           // Output: (batch, seq_len, num_heads, head_dim)
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float beta,
    size_t chunk_size,
    cudaStream_t stream
);

/**
 * EFLA backward pass
 */
cudaError_t efla_backward_cuda(
    const void* grad_output,
    const void* k,
    const void* v,
    const void* state,
    void* grad_k,
    void* grad_v,
    void* grad_state,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float beta,
    cudaStream_t stream
);

/**
 * Chunked scan for parallel EFLA processing
 */
cudaError_t efla_chunked_scan_cuda(
    void** chunk_states,
    size_t num_chunks,
    size_t num_heads,
    size_t head_dim,
    cudaStream_t stream
);

// ============================================================================
// PRISM Kernels
// ============================================================================

/**
 * PRISM forward pass - Iterative rank accumulation with write-forget decoupling
 */
cudaError_t prism_forward_cuda(
    const void* u,          // Proxy: (batch, seq_len, hidden_dim)
    const void* v,          // Values: (batch, seq_len, hidden_dim)
    const void* prev_state,
    void* new_state,
    void* output,
    const void** w_beta,    // Weights per iteration
    const void** w_k,
    const void** w_p,
    size_t batch_size,
    size_t seq_len,
    size_t hidden_dim,
    size_t head_dim,
    size_t num_iterations,
    float alpha,
    cudaStream_t stream
);

/**
 * ShortConv forward - Causal convolution
 */
cudaError_t shortconv_forward_cuda(
    const void* input,
    const void* weight,
    void* output,
    size_t batch_size,
    size_t seq_len,
    size_t hidden_dim,
    size_t window_size,
    cudaStream_t stream
);

// ============================================================================
// Normalization Kernels
// ============================================================================

/**
 * RMSNorm forward
 */
cudaError_t rmsnorm_forward_cuda(
    const void* input,
    const void* weight,
    void* output,
    size_t numel,
    size_t normalized_shape,
    float eps,
    cudaStream_t stream
);

/**
 * RMSNorm backward
 */
cudaError_t rmsnorm_backward_cuda(
    const void* grad_output,
    const void* input,
    const void* weight,
    void* grad_input,
    void* grad_weight,
    size_t numel,
    size_t normalized_shape,
    float eps,
    cudaStream_t stream
);

/**
 * LayerNorm forward
 */
cudaError_t layernorm_forward_cuda(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    size_t numel,
    size_t normalized_shape,
    float eps,
    cudaStream_t stream
);

// ============================================================================
// Activation Kernels
// ============================================================================

/**
 * GELU forward
 */
cudaError_t gelu_forward_cuda(
    const void* input,
    void* output,
    size_t numel,
    bool approximate,
    cudaStream_t stream
);

/**
 * GELU backward
 */
cudaError_t gelu_backward_cuda(
    const void* grad_output,
    const void* input,
    void* grad_input,
    size_t numel,
    bool approximate,
    cudaStream_t stream
);

/**
 * Softmax forward
 */
cudaError_t softmax_forward_cuda(
    const void* input,
    void* output,
    size_t outer_size,
    size_t dim_size,
    size_t inner_size,
    cudaStream_t stream
);

// ============================================================================
// GEMM Kernels (SM100 Tensor Core Optimized)
// ============================================================================

/**
 * GEMM forward - General matrix multiply with optional bias
 * C = A @ B + bias
 */
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
);

/**
 * GEMM backward
 */
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
);

// ============================================================================
// Loss Kernels
// ============================================================================

/**
 * Cross entropy loss forward
 */
cudaError_t cross_entropy_forward_cuda(
    const void* logits,     // (batch, vocab_size)
    const int32_t* targets, // (batch,)
    void* loss,             // scalar or (batch,)
    size_t batch_size,
    size_t vocab_size,
    float label_smoothing,
    cudaStream_t stream
);

/**
 * Cross entropy loss backward
 */
cudaError_t cross_entropy_backward_cuda(
    const void* grad_loss,
    const void* logits,
    const int32_t* targets,
    void* grad_logits,
    size_t batch_size,
    size_t vocab_size,
    float label_smoothing,
    cudaStream_t stream
);

// ============================================================================
// Optimizer Kernels
// ============================================================================

/**
 * Lion optimizer step
 */
cudaError_t lion_step_cuda(
    void* param,
    const void* grad,
    void* momentum,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    cudaStream_t stream
);

/**
 * Muon optimizer step - Newton-Schulz orthogonalization
 */
cudaError_t muon_step_cuda(
    void* param,
    const void* grad,
    void* momentum,
    size_t m,
    size_t n,
    float lr,
    float beta,
    size_t ns_iterations,
    cudaStream_t stream
);

/**
 * AdamW optimizer step
 */
cudaError_t adamw_step_cuda(
    void* param,
    const void* grad,
    void* exp_avg,
    void* exp_avg_sq,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    size_t step,
    cudaStream_t stream
);

/**
 * Gradient clipping by norm
 */
cudaError_t clip_grad_norm_cuda(
    void** grads,
    const size_t* numels,
    size_t num_params,
    float max_norm,
    float* global_norm,
    cudaStream_t stream
);

// ============================================================================
// Memory and Utility Kernels
// ============================================================================

/**
 * Fill tensor with value
 */
cudaError_t fill_cuda(
    void* ptr,
    float value,
    size_t numel,
    int dtype,
    cudaStream_t stream
);

/**
 * Copy and cast between dtypes
 */
cudaError_t cast_cuda(
    const void* src,
    void* dst,
    size_t numel,
    int src_dtype,
    int dst_dtype,
    cudaStream_t stream
);

/**
 * Embedding lookup
 */
cudaError_t embedding_forward_cuda(
    const int32_t* indices,
    const void* weight,
    void* output,
    size_t num_indices,
    size_t embedding_dim,
    cudaStream_t stream
);

/**
 * FP8 quantization
 */
cudaError_t quantize_fp8_cuda(
    const void* input,
    void* output,
    float* scale,
    size_t numel,
    cudaStream_t stream
);

/**
 * FP8 dequantization
 */
cudaError_t dequantize_fp8_cuda(
    const void* input,
    void* output,
    float scale,
    size_t numel,
    cudaStream_t stream
);

// ============================================================================
// Reduction Kernels
// ============================================================================

/**
 * Sum reduction
 */
cudaError_t sum_reduce_cuda(
    const void* input,
    void* output,
    size_t numel,
    int dtype,
    cudaStream_t stream
);

/**
 * Norm computation
 */
cudaError_t norm_cuda(
    const void* input,
    float* output,
    size_t numel,
    int dtype,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // EFLA_KERNELS_H
