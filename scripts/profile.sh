#!/bin/bash
# Profile training with Nsight Systems/Compute

set -e

PROFILE_TYPE="${1:-nsys}"
CONFIG="${2:-configs/train.yaml}"
OUTPUT="${3:-profile}"

echo "Profiling EFLA training..."
echo "Profile type: ${PROFILE_TYPE}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT}"

case "${PROFILE_TYPE}" in
    nsys|nsight-systems)
        echo "Using Nsight Systems..."
        nsys profile \
            --trace=cuda,nvtx,osrt,cudnn,cublas \
            --sample=cpu \
            --cpuctxsw=none \
            --cudabacktrace=true \
            --output="${OUTPUT}" \
            zig-out/bin/efla-train train --config "${CONFIG}"
        echo "Profile saved to ${OUTPUT}.nsys-rep"
        echo "View with: nsys-ui ${OUTPUT}.nsys-rep"
        ;;

    ncu|nsight-compute)
        echo "Using Nsight Compute..."
        ncu \
            --set=full \
            --export="${OUTPUT}" \
            --force-overwrite \
            zig-out/bin/efla-train train --config "${CONFIG}"
        echo "Profile saved to ${OUTPUT}.ncu-rep"
        echo "View with: ncu-ui ${OUTPUT}.ncu-rep"
        ;;

    ncu-kernel)
        echo "Profiling specific kernel with Nsight Compute..."
        KERNEL_NAME="${4:-efla_forward}"
        ncu \
            --kernel-name="${KERNEL_NAME}" \
            --set=full \
            --export="${OUTPUT}" \
            --force-overwrite \
            zig-out/bin/efla-train train --config "${CONFIG}"
        ;;

    memory)
        echo "Memory profiling..."
        compute-sanitizer \
            --tool memcheck \
            --leak-check full \
            --output "${OUTPUT}" \
            zig-out/bin/efla-train smoke-test --config configs/smoke.yaml
        ;;

    racecheck)
        echo "Race condition profiling..."
        compute-sanitizer \
            --tool racecheck \
            --output "${OUTPUT}" \
            zig-out/bin/efla-train smoke-test --config configs/smoke.yaml
        ;;

    *)
        echo "Unknown profile type: ${PROFILE_TYPE}"
        echo "Options: nsys, ncu, ncu-kernel, memory, racecheck"
        exit 1
        ;;
esac

echo "Profiling complete!"
