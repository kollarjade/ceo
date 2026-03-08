#!/bin/bash
# Build CUDA kernels for SM100 (Blackwell)

set -e

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
CUDA_ARCH="${CUDA_ARCH:-sm_100}"

echo "Building CUDA kernels for ${CUDA_ARCH}..."
echo "CUDA_PATH: ${CUDA_PATH}"

# Create build directory
mkdir -p build/cuda

# Compiler flags
NVCC_FLAGS=(
    -arch=${CUDA_ARCH}
    -O3
    --use_fast_math
    -DCUDA_ARCH_SM100
    --ptxas-options=-v
    -lineinfo
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -std=c++20
    -I kernels/cuda/include
    -I ${CUDA_PATH}/include
)

# Debug flags (uncomment for debugging)
# NVCC_FLAGS=(
#     -arch=${CUDA_ARCH}
#     -O0
#     -g
#     -G
#     -DCUDA_ARCH_SM100
#     --expt-relaxed-constexpr
#     --expt-extended-lambda
#     -std=c++20
#     -I kernels/cuda/include
#     -I ${CUDA_PATH}/include
# )

# Build each kernel file
KERNELS=(
    "efla"
    "prism"
    "layernorm"
    "gelu"
    "softmax"
    "gemm"
    "embedding"
    "cross_entropy"
    "optim"
    "memory"
    "init"
)

for kernel in "${KERNELS[@]}"; do
    echo "Building ${kernel}.cu..."
    ${CUDA_PATH}/bin/nvcc "${NVCC_FLAGS[@]}" \
        -dlink \
        -o build/cuda/${kernel}_link.o \
        kernels/cuda/${kernel}.cu

    ${CUDA_PATH}/bin/nvcc "${NVCC_FLAGS[@]}" \
        -lib \
        -o build/cuda/libcuda_${kernel}.a \
        kernels/cuda/${kernel}.cu

    ${CUDA_PATH}/bin/nvcc "${NVCC_FLAGS[@]}" \
        -shared \
        -o build/cuda/libcuda_${kernel}.so \
        kernels/cuda/${kernel}.cu \
        -L ${CUDA_PATH}/lib64 \
        -lcudart -lcublas -lcublasLt
done

echo "CUDA kernels built successfully!"
echo "Output: build/cuda/"
ls -la build/cuda/
