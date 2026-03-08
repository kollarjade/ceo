# EFLA Trainer Architecture

## Overview

EFLA Trainer is designed from the ground up for training trillion-parameter language models with ultra-long context (50M+ tokens). The architecture combines novel attention mechanisms (EFLA, PRISM) with production-grade distributed training infrastructure.

## Core Components

### 1. EFLA (Error-Free Linear Attention)

EFLA implements a state-space-like linear attention with an **exact** closed-form update, not an approximation.

#### Mathematical Foundation

At each timestep $t$, given:
- Key vector $k_t \in \mathbb{R}^d$
- Value vector $v_t \in \mathbb{R}^{d_v}$
- State matrix $S_t \in \mathbb{R}^{d \times d_v}$
- Step size $\beta_t$

The coefficient $c_t$ is computed as:

$$c_t = \frac{1 - \exp(-\beta_t \lambda_t)}{\lambda_t}$$

where $\lambda_t = k_t^T k_t$.

**Numerical Stability**: For $\lambda_t$ near zero, we use Taylor series expansion:

$$c_t \approx \beta_t - \frac{1}{2}\beta_t^2 \lambda_t + \frac{1}{6}\beta_t^3 \lambda_t^2 - \frac{1}{24}\beta_t^4 \lambda_t^3$$

**State Update**:

$$S_t = (I - c_t k_t k_t^T) S_{t-1} + c_t k_t v_t^T$$

This can be expanded as:
1. Compute $k^T S_{t-1}$ (row vector)
2. Subtract $c_t k (k^T S_{t-1})$ from each row
3. Add $c_t k v^T$ (outer product)

#### Chunked Processing

For 50M token contexts, we process in chunks:

1. **Intra-chunk**: Parallel processing within each chunk
2. **Inter-chunk**: Prefix scan to combine chunk states

```
Chunk 0: [S_0 → S_1 → ... → S_{c-1}] → state_0
Chunk 1: [state_0 → ... → S_{2c-1}] → state_1
...
Combine via parallel prefix scan
```

### 2. PRISM (Parallel Residual Iterative Sequence Model)

PRISM handles short/mid-range dependencies through iterative rank accumulation with write-forget decoupling.

#### Components

**Input-Anchored Proxy**:
$$u_t = \text{ShortConv}(X_{\leq t}) \approx S_{t-1} k_t$$

**Gates and Projections**:
$$\beta_t^{(l)} = W_\beta^{(l)} u_t$$
$$k_t^{(l)} = W_k^{(l)} u_t$$
$$p_t^{(l)} = W_p^{(l)} u_t \approx \sigma'(S_{t-1} k_t)$$

**Iterative Refinement**:
$$r_t^{(1)} = v_t - u_t$$
$$\delta_t^{(l)} = \text{GELU}(p_t^{(l)} \odot r_t^{(l)})$$
$$r_t^{(l+1)} = r_t^{(l)} - \delta_t^{(l)}$$

**State Update**:
$$S_t = \alpha_t S_{t-1} (I - \beta_t^{(1)} k_t^{(1)} \otimes k_t^{(1)}) + \sum_{l=1}^{L} \beta_t^{(l)} (\delta_t^{(l)} \otimes k_t^{(l)})$$

### 3. Model Architecture

```
┌─────────────────────────────────────┐
│            Input Tokens              │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│         Token Embedding              │
│         (vocab_size, hidden_dim)     │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│         ┌─────────────────┐          │
│         │   RMSNorm       │          │
│         └────────┬────────┘          │
│                  │                   │
│         ┌────────▼────────┐          │
│         │   EFLA Layer    │          │
│         │   (Linear Attn) │          │
│         └────────┬────────┘          │
│                  │                   │
│         ┌────────▼────────┐          │
│         │   PRISM Layer   │          │
│         │   (Short-range) │          │
│         └────────┬────────┘          │
│                  │                   │
│         ┌────────▼────────┐          │
│         │   RMSNorm       │          │
│         └────────┬────────┘          │
│                  │                   │
│         ┌────────▼────────┐          │
│         │      MLP        │          │
│         │  (SwiGLU/GELU)  │          │
│         └────────┬────────┘          │
│                  │                   │
│         Residual Connection          │
│                  │                   │
└──────────────────┴──────────────────┘
                   │
                   │ × num_layers
                   │
┌──────────────────▼──────────────────┐
│         Final RMSNorm               │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│         LM Head                     │
│         (hidden_dim, vocab_size)    │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│         Output Logits               │
└─────────────────────────────────────┘
```

### 4. Distributed Runtime

#### Process Topology

```
┌─────────────────────────────────────────────────────┐
│                    Node (8 GPUs)                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │   │
│  │ Rank 0  │ │ Rank 1  │ │ Rank 2  │ │ Rank 3  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │        │
│       └───────────┴─────┬─────┴───────────┘        │
│                         │                          │
│                    NCCL Group                       │
│                         │                          │
│       ┌───────────┬─────┴─────┬───────────┐        │
│       │           │           │           │        │
│  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐   │
│  │ GPU 4   │ │ GPU 5   │ │ GPU 6   │ │ GPU 7   │   │
│  │ Rank 4  │ │ Rank 5  │ │ Rank 6  │ │ Rank 7  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────┘
```

#### Tensor Parallelism

For GEMM-heavy layers, we shard along specific dimensions:

| Layer Type | Shard Dim | Explanation |
|------------|-----------|-------------|
| Embedding | 0 | Shard vocabulary |
| QKV Projection | 0 | Shard heads |
| Output Projection | 1 | Shard output features |
| MLP Up | 0 | Shard intermediate dim |
| MLP Down | 1 | Shard hidden dim |

#### ZeRO Sharding

ZeRO Stage 2 implementation:
- **Stage 1**: Shard optimizer states
- **Stage 2**: Shard gradients + optimizer states
- **Stage 3**: Shard parameters + gradients + optimizer states

### 5. Precision Strategy

| Component | Dtype | Reasoning |
|-----------|-------|-----------|
| Weights | BF16 | Native tensor core support |
| Activations | BF16 | Stability with large values |
| Accumulation | FP32 | Precision for GEMM |
| Optimizer State | FP32 | Numerical stability |
| Checkpoints | BF16 | Storage efficiency |

**FP8 Training**: Optional FP8 for forward activations with dynamic scaling.

### 6. Memory Management

**Activation Checkpointing**:
- Store subset of activations
- Recompute during backward pass
- Trade compute for memory

**CPU Offload**:
- Offload optimizer states to pinned CPU memory
- Async transfer during compute

**NVMe Offload**:
- For models exceeding CPU memory
- Sequential access pattern optimized

### 7. Data Pipeline

```
┌────────────────────────────────────────────────────┐
│                   Data Pipeline                     │
│                                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │ Raw Text │ -> │ Tokenize │ -> │   Pack   │    │
│  └──────────┘    └──────────┘    └──────────┘    │
│       │                               │           │
│       │                               ▼           │
│       │                        ┌──────────┐      │
│       │                        │  Shuffle │      │
│       │                        └──────────┘      │
│       │                               │           │
│       ▼                               ▼           │
│  ┌──────────────────────────────────────────┐    │
│  │           Sharded Binary Format           │    │
│  │  - Memory-mapped access                   │    │
│  │  - Random access for shuffling            │    │
│  │  - Checksums for integrity                │    │
│  └──────────────────────────────────────────┘    │
│                         │                         │
│                         ▼                         │
│  ┌──────────────────────────────────────────┐    │
│  │           DataLoader                      │    │
│  │  - Prefetch buffer                        │    │
│  │  - Multi-worker loading                   │    │
│  │  - Deterministic shuffling                │    │
│  └──────────────────────────────────────────┘    │
└────────────────────────────────────────────────────┘
```

## Performance Considerations

### Kernel Optimization

- **Tensor Core Utilization**: All GEMMs use tcgen05-based approaches
- **Shared Memory**: Maximize shared memory usage for reduction operations
- **Warp-Level Primitives**: Use warp shuffle for fast reductions
- **Persistent Kernels**: For long-reduction operations

### Communication Overlap

```
┌─────────────────────────────────────────────────────┐
│                    Timeline                          │
│                                                     │
│  Compute:  ████████████░░░░░░░░████████████        │
│                     │               │               │
│  Comm:     ░░░░░░░░░████████████░░░░░░░░░░░        │
│                     │               │               │
│            Forward │   All-Reduce  │  Backward     │
│            (local) │   (overlapped)│  (local)      │
└─────────────────────────────────────────────────────┘
```

### Numerical Stability

- **EFLA coefficient**: Taylor expansion for small λ
- **Loss scaling**: Dynamic loss scaling for FP8/FP16
- **Gradient clipping**: By global norm before optimizer step
- **NaN detection**: Check before state updates

## Extensibility

### Adding New Layers

1. Implement in `src/nn/` or `src/model/`
2. Add forward/backward methods
3. Register with parameter collection
4. Add CUDA kernel if performance-critical

### Adding New Optimizers

1. Implement in `src/optim/`
2. Add kernel to `kernels/cuda/optim.cu`
3. Add C ABI declaration
4. Register with optimizer factory

### Adding New Kernels

1. Implement in `kernels/cuda/`
2. Add C ABI function in `kernels/cuda/include/kernels.h`
3. Add Zig extern declaration in `src/kernels/`
4. Build with `scripts/build_kernels.sh`
