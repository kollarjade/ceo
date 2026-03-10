#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
CUDA_ARCH="${CUDA_ARCH:-sm_100}"

NVCC_BIN="${CUDA_PATH}/bin/nvcc"
CUDA_INCLUDE="${CUDA_PATH}/include"
CUDA_LIB="${CUDA_PATH}/lib64"

if [ ! -x "${NVCC_BIN}" ]; then
    echo "nvcc not found at ${NVCC_BIN}"
    exit 1
fi

if [ ! -d "${CUDA_INCLUDE}" ]; then
    echo "CUDA include directory not found at ${CUDA_INCLUDE}"
    exit 1
fi

if [ ! -d "${CUDA_LIB}" ]; then
    echo "CUDA lib directory not found at ${CUDA_LIB}"
    exit 1
fi

KERNEL_SRC_DIR="${PROJECT_ROOT}/kernels/cuda"
KERNEL_INCLUDE_DIR="${KERNEL_SRC_DIR}/include"
BUILD_DIR="${PROJECT_ROOT}/build/cuda"
OBJ_DIR="${BUILD_DIR}/obj"
LIB_DIR="${BUILD_DIR}/lib"

mkdir -p "${OBJ_DIR}"
mkdir -p "${LIB_DIR}"

NVCC_FLAGS=(
    "-arch=${CUDA_ARCH}"
    "-O3"
    "--use_fast_math"
    "-DCUDA_ARCH_SM100"
    "--ptxas-options=-v"
    "-lineinfo"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "-std=c++20"
    "-rdc=true"
    "-I${KERNEL_INCLUDE_DIR}"
    "-I${CUDA_INCLUDE}"
)

HOST_FLAGS=(
    "-Xcompiler"
    "-fPIC"
)

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
    SRC_FILE="${KERNEL_SRC_DIR}/${kernel}.cu"
    OBJ_FILE="${OBJ_DIR}/${kernel}.o"
    LINK_OBJ="${OBJ_DIR}/${kernel}_dlink.o"
    STATIC_LIB="${LIB_DIR}/libcuda_${kernel}.a"
    SHARED_LIB="${LIB_DIR}/libcuda_${kernel}.so"

    if [ ! -f "${SRC_FILE}" ]; then
        echo "Missing kernel source: ${SRC_FILE}"
        exit 1
    fi

    echo "Compiling ${SRC_FILE}"
    "${NVCC_BIN}" "${NVCC_FLAGS[@]}" "${HOST_FLAGS[@]}" -dc -c "${SRC_FILE}" -o "${OBJ_FILE}"

    echo "Device linking ${kernel}"
    "${NVCC_BIN}" "${NVCC_FLAGS[@]}" -dlink "${OBJ_FILE}" -o "${LINK_OBJ}"

    echo "Creating static library ${STATIC_LIB}"
    "${NVCC_BIN}" -lib -o "${STATIC_LIB}" "${OBJ_FILE}"

    echo "Creating shared library ${SHARED_LIB}"
    "${NVCC_BIN}" "${NVCC_FLAGS[@]}" "${HOST_FLAGS[@]}" -shared -o "${SHARED_LIB}" "${OBJ_FILE}" "${LINK_OBJ}" -L"${CUDA_LIB}" -lcudart -lcublas -lcublasLt
done

echo "Build finished"
ls -la "${BUILD_DIR}"
