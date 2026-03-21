#ifndef TITAN_KERNELS_H
#define TITAN_KERNELS_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    TITAN_SUCCESS = 0,
    TITAN_ERROR_INVALID_ARGUMENT = 1,
    TITAN_ERROR_CUDA = 2,
    TITAN_ERROR_OUT_OF_MEMORY = 3,
    TITAN_ERROR_KERNEL_LAUNCH = 4,
} titan_status_t;
typedef void* titan_stream_t;
titan_status_t titan_gemm_bf16(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    titan_stream_t stream
);
titan_status_t titan_gemm_fp8(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float scale_a, float scale_b, float scale_out,
    titan_stream_t stream
);
titan_status_t titan_rmsnorm_forward(
    const float* input, const float* weight, float* output,
    int batch_size, int hidden_dim, float eps,
    titan_stream_t stream
);
titan_status_t titan_rmsnorm_backward(
    const float* grad_output, const float* input, const float* weight,
    float* grad_input, float* grad_weight,
    int batch_size, int hidden_dim, float eps,
    titan_stream_t stream
);
titan_status_t titan_gelu_forward(
    const float* input, float* output, int n,
    titan_stream_t stream
);
titan_status_t titan_gelu_backward(
    const float* input, const float* grad_output, float* grad_input, int n,
    titan_stream_t stream
);
titan_status_t titan_silu_forward(
    const float* input, float* output, int n,
    titan_stream_t stream
);
titan_status_t titan_short_conv_forward(
    const float* input, const float* weights, float* output,
    int batch_size, int seq_len, int channels, int window_size,
    titan_stream_t stream
);
titan_status_t titan_short_conv_backward(
    const float* grad_output, const float* input, const float* weights,
    float* grad_input, float* grad_weights,
    int batch_size, int seq_len, int channels, int window_size,
    titan_stream_t stream
);
titan_status_t titan_efla_scan_forward(
    const float* keys, const float* values, float* output, float* state,
    int batch_size, int seq_len, int state_dim, int value_dim,
    float beta, float lambda_threshold,
    titan_stream_t stream
);
titan_status_t titan_efla_scan_backward(
    const float* grad_output, const float* keys, const float* values,
    const float* saved_states,
    float* grad_keys, float* grad_values, float* grad_beta,
    int batch_size, int seq_len, int state_dim, int value_dim,
    float beta, float lambda_threshold,
    titan_stream_t stream
);
titan_status_t titan_outer_product_accumulate(
    const float* delta, const float* k, float* state,
    float beta_scale,
    int batch_size, int dim,
    titan_stream_t stream
);
titan_status_t titan_rank1_update(
    float* matrix, const float* u, const float* v,
    float alpha, int rows, int cols,
    titan_stream_t stream
);
titan_status_t titan_prefix_scan_f32(
    const float* input, float* output, int n,
    titan_stream_t stream
);
titan_status_t titan_cross_entropy_forward(
    const float* logits, const int* targets, float* losses,
    int batch_size, int vocab_size, float label_smoothing,
    titan_stream_t stream
);
titan_status_t titan_cross_entropy_backward(
    const float* logits, const int* targets, float* grad_logits,
    int batch_size, int vocab_size, float label_smoothing,
    titan_stream_t stream
);
titan_status_t titan_softmax_forward(
    const float* input, float* output, int batch_size, int dim,
    titan_stream_t stream
);
titan_status_t titan_elementwise_mul(
    const float* a, const float* b, float* c, int n,
    titan_stream_t stream
);
titan_status_t titan_compute_amax(
    const float* data, float* amax, int n,
    titan_stream_t stream
);
titan_status_t titan_quantize_fp8(
    const float* input, void* output, float scale, int n,
    titan_stream_t stream
);
titan_status_t titan_dequantize_fp8(
    const void* input, float* output, float inv_scale, int n,
    titan_stream_t stream
);
typedef struct {
    int block_m;
    int block_n;
    int block_k;
    int stages;
    int split_k;
} titan_gemm_config_t;
titan_status_t titan_gemm_autotune(
    int M, int N, int K,
    titan_gemm_config_t* best_config
);
int titan_get_device_count(void);
titan_status_t titan_set_device(int device_id);
titan_status_t titan_device_synchronize(void);
titan_status_t titan_get_device_memory(int device_id, size_t* free, size_t* total);
#ifdef __cplusplus
}
#endif
#endif
