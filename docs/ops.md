# Operations Guide

## Training Operations

### Starting Training

```bash
# Fresh training run
zig-out/bin/efla-train train --config configs/train.yaml

# Resume from checkpoint
zig-out/bin/efla-train train --config configs/train.yaml \
    --resume checkpoints/step_00010000

# Dry run (validate config without training)
zig-out/bin/efla-train train --config configs/train.yaml --dry-run
```

### Monitoring Training

#### Log Files

Training produces structured JSONL logs:
```
{"step":1000,"tokens":5120000,"loss":2.5,"lr":0.0001,"timestamp":1234567890}
{"step":1001,"tokens":5125120,"loss":2.48,"lr":0.0001,"timestamp":1234567891}
```

Monitor in real-time:
```bash
tail -f training.log.jsonl | jq .
```

#### Metrics Dashboard

If Prometheus enabled:
```bash
# Start Prometheus
prometheus --config.file=prometheus.yml

# Metrics endpoint
curl http://localhost:9090/metrics
```

#### Health Checks

```bash
# Check training status
zig-out/bin/efla-train status --checkpoint checkpoints/latest

# Check GPU health
nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE
```

### Handling Failures

#### Automatic Recovery

Training automatically saves checkpoints at intervals:
```yaml
checkpoint:
  save_interval: 1000  # Save every 1000 steps
  keep_last_n: 5       # Keep last 5 checkpoints
```

Resume from last checkpoint:
```bash
zig-out/bin/efla-train train --config configs/train.yaml \
    --resume checkpoints/latest
```

#### Manual Recovery

If training crashes:
1. Identify last good checkpoint
2. Verify checkpoint integrity
3. Resume training

```bash
# List checkpoints
zig-out/bin/efla-train checkpoint list --dir checkpoints

# Validate checkpoint
zig-out/bin/efla-train checkpoint validate \
    --path checkpoints/step_00005000

# Resume
zig-out/bin/efla-train train --config configs/train.yaml \
    --resume checkpoints/step_00005000
```

#### NaN/Inf Recovery

If NaN detected:
1. Training automatically skips step and adjusts loss scale
2. If persistent, check data integrity
3. May need to reduce learning rate

```yaml
training:
  nan_handling: skip_step  # Options: skip_step, stop, reduce_lr
  loss_scale_reduction: 0.5
```

### Checkpoint Management

#### Backup Checkpoints

```bash
# Backup to external storage
rsync -av checkpoints/ s3://bucket/checkpoints/

# Or to local backup
tar -czf checkpoint_backup_$(date +%Y%m%d).tar.gz checkpoints/
```

#### Prune Old Checkpoints

```bash
# Keep only last N checkpoints
zig-out/bin/efla-train checkpoint prune --dir checkpoints --keep 10
```

#### Convert Formats

```bash
# Convert to safetensors
zig-out/bin/efla-train checkpoint convert \
    --input checkpoints/step_00010000 \
    --output exported/model.safetensors \
    --format safetensors
```

## Evaluation

### Perplexity Evaluation

```bash
# Evaluate on validation set
zig-out/bin/efla-train evaluate \
    --checkpoint checkpoints/best \
    --data data/validation.bin \
    --max-tokens 10000000

# Output:
# Perplexity: 3.42
# Loss: 1.23
# Tokens: 10000000
```

### Long Context Evaluation

```bash
# Test at various context lengths
zig-out/bin/efla-train evaluate \
    --checkpoint checkpoints/latest \
    --data data/long_context.bin \
    --context-lengths 4096,16384,65536,262144
```

### Benchmarking

```bash
# Benchmark throughput
zig-out/bin/efla-train benchmark \
    --config configs/train.yaml \
    --batch-sizes 1,2,4,8 \
    --seq-lengths 1024,4096,16384
```

## Data Management

### Data Preparation

```bash
# Train tokenizer
zig-out/bin/efla-train tokenizer train \
    --corpus data/raw_text.txt \
    --output tokenizer.bin \
    --vocab-size 131072

# Encode corpus
zig-out/bin/efla-train tokenizer encode \
    --tokenizer tokenizer.bin \
    --input data/raw_text.txt \
    --output data/encoded.bin

# Create sharded dataset
zig-out/bin/efla-train dataset create \
    --input data/encoded.bin \
    --output data/train \
    --shard-size 100000000
```

### Data Validation

```bash
# Validate dataset integrity
zig-out/bin/efla-train dataset validate \
    --path data/train.bin

# Check token distribution
zig-out/bin/efla-train dataset stats \
    --path data/train.bin
```

## Distributed Training

### Multi-GPU Setup

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run with 8 GPUs
zig-out/bin/efla-train train --config configs/train.yaml
```

### NCCL Debugging

```bash
# Enable NCCL debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Run training
zig-out/bin/efla-train train --config configs/train.yaml 2>&1 | tee nccl_debug.log
```

### GPU Affinity

```bash
# Set GPU-CPU affinity for optimal performance
numactl --cpunodebind=0 --membind=0 \
    zig-out/bin/efla-train train --config configs/train.yaml
```

## Cloud Operations

### Modal Deployment

```bash
# Deploy to Modal
modal deploy cloud/modal/train.py

# Run training
modal run cloud/modal/train.py --config configs/train.yaml
```

### Cost Management

```bash
# Estimate training cost
zig-out/bin/efla-train estimate-cost \
    --config configs/train.yaml \
    --provider modal \
    --gpu-type b200
```

## Maintenance

### Log Rotation

```bash
# Rotate logs daily
logrotate /etc/logrotate.d/efla-trainer
```

Example logrotate config:
```
/var/log/efla-trainer/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
}
```

### Disk Cleanup

```bash
# Clean old checkpoints
find checkpoints/ -type d -mtime +30 -exec rm -rf {} \;

# Clean temporary files
rm -rf /tmp/efla-*
```

### Health Monitoring Script

```bash
#!/bin/bash
# health_check.sh

# Check GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -gt 180000 ]; then
    echo "WARNING: GPU memory usage high: ${GPU_MEM}MB"
fi

# Check training progress
LAST_STEP=$(tail -1 training.log.jsonl | jq -r '.step')
EXPECTED=$(( $(date +%s) - $(stat -c %Y training.log.jsonl) ))
if [ "$EXPECTED" -gt 300 ]; then
    echo "WARNING: No training progress for ${EXPECTED} seconds"
fi

# Check for NaN
NAN_COUNT=$(grep -c "nan\|inf" training.log.jsonl || true)
if [ "$NAN_COUNT" -gt 0 ]; then
    echo "WARNING: NaN/Inf detected in training log"
fi
```

### Alerting

Set up alerts for:
- Training stalled (no progress for > 5 minutes)
- GPU temperature > 85°C
- Disk space < 10%
- NaN/Inf in training
- Checkpoint save failures

## Troubleshooting

### Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing
3. Use ZeRO Stage 2/3
4. Enable CPU offload

```yaml
training:
  micro_batch_size: 1
  gradient_checkpointing: true

runtime:
  zero_stage: 2
  cpu_offload: true
```

### Slow Training

1. Check GPU utilization
2. Verify NCCL bandwidth
3. Check data loading bottleneck
4. Profile with Nsight

### Training Instability

1. Reduce learning rate
2. Increase warmup steps
3. Check data quality
4. Verify gradient clipping

```yaml
training:
  learning_rate: 5.0e-5
  warmup_steps: 5000
  gradient_clip: 0.5
```

### Checkpoint Corruption

1. Restore from backup
2. Use previous checkpoint
3. Verify data integrity

```bash
# Verify checksum
sha256sum checkpoints/step_*/tensors.bin
```
