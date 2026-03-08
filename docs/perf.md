# Performance Guide

## Profiling Workflow

### 1. Nsight Systems (Timeline Analysis)

```bash
# Profile a training run
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --output=profile \
    zig-out/bin/efla-train train --config configs/train.yaml

# View results
nsys-ui profile.nsys-rep
```

**Key Metrics to Check**:
- GPU utilization percentage
- Memory bandwidth utilization
- Kernel execution overlap
- CPU-GPU synchronization points

### 2. Nsight Compute (Kernel Analysis)

```bash
# Profile specific kernel
ncu --set=full \
    --kernel-name="efla_forward" \
    --output=kernel_profile \
    zig-out/bin/efla-train train --config configs/train.yaml

# View results
ncu-ui kernel_profile.ncu-rep
```

**Key Metrics**:
- Warp execution efficiency
- Memory throughput
- Tensor core utilization
- Shared memory bank conflicts

### 3. Memory Profiling

```bash
# Check for memory errors
compute-sanitizer --tool memcheck \
    zig-out/bin/efla-train smoke-test --config configs/smoke.yaml

# Check for race conditions
compute-sanitizer --tool racecheck \
    zig-out/bin/efla-train smoke-test --config configs/smoke.yaml
```

## Kernel Tuning

### GEMM Tuning

For SM100, GEMM performance depends on:
- Tile sizes (M, N, K tiles)
- Pipeline depth
- Shared memory staging
- Register allocation

**Autotuning**:
```bash
# Run GEMM autotuner
zig-out/bin/efla-train tune-gemm \
    --m 16384 --n 16384 --k 16384 \
    --output gemm_config.yaml
```

### EFLA Tuning

Key parameters:
- `chunk_size`: Balance between parallelism and state synchronization
- `head_dim`: Affects state matrix size
- `num_heads`: Affects parallelization

```yaml
efla:
  chunk_size: 4096  # Larger for more parallelism
  num_heads: 128    # Must divide hidden_dim
```

### PRISM Tuning

Key parameters:
- `num_iterations`: More iterations = better approximation
- `shortconv_window`: Larger = more context

```yaml
prism:
  num_iterations: 3
  shortconv_window: 64
```

## Communication Overlap

### Strategy

1. **Forward pass**: Compute locally, overlap gradient all-reduce
2. **Backward pass**: Compute gradients, overlap parameter all-reduce

```
Timeline:
─────────────────────────────────────────────────────
Forward:  [Compute] [Wait] [Compute] [Wait]
                   ↓
Comm:              [All-Reduce gradients]
                           ↓
Backward:         [Compute] [All-Reduce params]
─────────────────────────────────────────────────────
```

### NCCL Tuning

```bash
# Environment variables for NCCL tuning
export NCCL_ALGO=Ring      # Ring algorithm for small messages
export NCCL_PROTO=Simple   # Simple protocol
export NCCL_IB_DISABLE=0   # Enable InfiniBand if available
export NCCL_DEBUG=INFO     # Debug NCCL operations
```

## Memory Optimization

### Activation Checkpointing

Enable to reduce memory:
```yaml
training:
  gradient_checkpointing: true
  checkpoint_layers: [0, 2, 4, 6]  # Checkpoint specific layers
```

### CPU Offloading

For ZeRO Stage 3:
```yaml
runtime:
  zero_stage: 3
  cpu_offload: true
```

### NVMe Offloading

For extreme memory constraints:
```yaml
runtime:
  nvme_offload: true
  nvme_path: /mnt/nvme/offload
```

## Throughput Benchmarks

### Expected Performance (8×B200)

| Model Size | Context | Throughput | Memory/GPU |
|------------|---------|------------|------------|
| 7B | 4K | ~1M tok/s | 20 GB |
| 70B | 8K | ~300K tok/s | 80 GB |
| 1T | 8K | ~50K tok/s | 180 GB |
| 1T | 50M* | ~10K tok/s | 192 GB |

*With activation checkpointing

### Measuring Throughput

```bash
# Training automatically logs throughput
zig-out/bin/efla-train train --config configs/train.yaml

# Output includes:
# step=1000 loss=2.5 lr=1.0e-04 tokens=5120000 throughput=51200.0 tok/s
```

## Scaling Analysis

### Strong Scaling (Fixed Model, Varying GPUs)

```
GPUs  | Throughput | Speedup | Efficiency
------|------------|---------|------------
1     | 6.4K tok/s | 1.0x    | 100%
2     | 12.5K tok/s| 2.0x    | 98%
4     | 24.2K tok/s| 3.8x    | 95%
8     | 46.0K tok/s| 7.2x    | 90%
```

### Weak Scaling (Proportional Model/GPUs)

```
GPUs  | Model  | Throughput | Efficiency
------|--------|------------|------------
1     | 125B   | 50K tok/s  | 100%
2     | 250B   | 50K tok/s  | 100%
4     | 500B   | 49K tok/s  | 98%
8     | 1T     | 48K tok/s  | 96%
```

## Performance Debugging

### Low GPU Utilization

1. Check for CPU bottlenecks:
```bash
nsys profile --sample=cpu ...
```

2. Check for synchronization:
```bash
nsys profile --trace=osrt ...
```

3. Verify batch size is sufficient:
```yaml
training:
  global_batch_size: 512  # Increase if utilization low
```

### Low Memory Bandwidth

1. Check kernel efficiency with Nsight Compute
2. Verify coalesced memory access patterns
3. Check for shared memory bank conflicts

### High Communication Overhead

1. Increase gradient accumulation:
```yaml
training:
  gradient_accumulation_steps: 64  # More steps = fewer syncs
```

2. Use tensor parallelism efficiently:
```yaml
runtime:
  tensor_parallel_size: 8  # Maximize for communication reduction
```

### Training Instability

1. Reduce learning rate:
```yaml
training:
  learning_rate: 5.0e-5  # Half the default
```

2. Enable gradient clipping:
```yaml
training:
  gradient_clip: 0.5  # Stricter clipping
```

3. Use dynamic loss scaling:
```yaml
training:
  dynamic_loss_scale: true
  loss_scale: 65536.0
```
