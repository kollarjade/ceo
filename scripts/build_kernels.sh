set -Eeuo pipefail

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

resolve_dir() {
    local path="$1"
    [[ -d "$path" ]] || fail "Directory not found: $path"
    cd -- "$path" >/dev/null 2>&1 && pwd -P
}

find_cuda_lib_dir() {
    local candidates=(
        "$CUDA_PATH/lib64"
        "$CUDA_PATH/lib/x64"
        "$CUDA_PATH/lib"
        "$CUDA_PATH/targets/x86_64-linux/lib"
        "$CUDA_PATH/targets/aarch64-linux/lib"
    )
    local dir
    for dir in "${candidates[@]}"; do
        if [[ -d "$dir" ]]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done
    return 1
}

require_cuda_library() {
    local stem="$1"
    local matches=()
    shopt -s nullglob
    matches=(
        "$CUDA_LIB_DIR"/lib"$stem".so
        "$CUDA_LIB_DIR"/lib"$stem".so.*
        "$CUDA_LIB_DIR"/lib"$stem".a
    )
    shopt -u nullglob
    ((${#matches[@]} > 0)) || fail "Required CUDA library not found in $CUDA_LIB_DIR: lib${stem}"
}

detect_link_flags() {
    local src="$1"
    LINK_FLAGS=("-lcudart")
    if grep -Eiq '\bcublasLt\b' "$src"; then
        require_cuda_library "cublasLt"
        require_cuda_library "cublas"
        LINK_FLAGS+=("-lcublasLt" "-lcublas")
    elif grep -Eiq '\bcublas\b' "$src"; then
        require_cuda_library "cublas"
        LINK_FLAGS+=("-lcublas")
    fi
}

build_kernel() {
    local src="$1"
    local kernel
    local obj
    local static_lib
    local shared_lib
    local tmp_obj
    local tmp_static
    local tmp_shared

    kernel="$(basename -- "$src" .cu)"
    obj="$BUILD_DIR/${kernel}.o"
    static_lib="$BUILD_DIR/libcuda_${kernel}.a"
    shared_lib="$BUILD_DIR/libcuda_${kernel}.so"
    tmp_obj="$TEMP_DIR/${kernel}.o"
    tmp_static="$TEMP_DIR/libcuda_${kernel}.a"
    tmp_shared="$TEMP_DIR/libcuda_${kernel}.so"

    printf 'Building %s.cu...\n' "$kernel"
    detect_link_flags "$src"

    "$NVCC_BIN" "${COMMON_FLAGS[@]}" -Xcompiler=-fPIC -c -o "$tmp_obj" "$src"
    "$NVCC_BIN" -lib -o "$tmp_static" "$tmp_obj"
    "$NVCC_BIN" -shared -o "$tmp_shared" "$tmp_obj" "-L$CUDA_LIB_DIR" "-Wl,-rpath,$CUDA_LIB_DIR" "${LINK_FLAGS[@]}"

    mv -f -- "$tmp_obj" "$obj"
    mv -f -- "$tmp_static" "$static_lib"
    mv -f -- "$tmp_shared" "$shared_lib"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd -P)"
SRC_DIR="$ROOT_DIR/kernels/cuda"
BUILD_DIR="$ROOT_DIR/build/cuda"
INCLUDE_DIR="$SRC_DIR/include"
CUDA_ARCH="${CUDA_ARCH:-sm_100}"

[[ "$CUDA_ARCH" =~ ^sm_[0-9]+[A-Za-z0-9_]*$ ]] || fail "Invalid CUDA_ARCH value: $CUDA_ARCH"

if [[ -n "${CUDA_PATH:-}" ]]; then
    CUDA_PATH="$(resolve_dir "$CUDA_PATH")"
else
    if command -v nvcc >/dev/null 2>&1; then
        NVCC_FROM_PATH="$(command -v nvcc)"
        NVCC_REAL="$(readlink -f -- "$NVCC_FROM_PATH" 2>/dev/null || printf '%s\n' "$NVCC_FROM_PATH")"
        CUDA_PATH="$(cd -- "$(dirname -- "$NVCC_REAL")/.." >/dev/null 2>&1 && pwd -P)"
    else
        CUDA_PATH="/usr/local/cuda"
    fi
fi

NVCC_BIN="$CUDA_PATH/bin/nvcc"
[[ -x "$NVCC_BIN" ]] || fail "nvcc not found or not executable: $NVCC_BIN"
[[ -d "$SRC_DIR" ]] || fail "Source directory not found: $SRC_DIR"
[[ -d "$CUDA_PATH/include" ]] || fail "CUDA include directory not found: $CUDA_PATH/include"

CUDA_LIB_DIR="$(find_cuda_lib_dir)" || fail "CUDA library directory not found under: $CUDA_PATH"
require_cuda_library "cudart"

CUDA_ARCH_DEFINE="CUDA_ARCH_$(printf '%s' "$CUDA_ARCH" | tr '[:lower:]' '[:upper:]' | tr -cd '[:alnum:]')"

COMMON_FLAGS=(
    "-arch=$CUDA_ARCH"
    "-O3"
    "--use_fast_math"
    "-D$CUDA_ARCH_DEFINE"
    "--ptxas-options=-v"
    "-lineinfo"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "-std=c++20"
    "-I$CUDA_PATH/include"
)

if [[ -d "$INCLUDE_DIR" ]]; then
    COMMON_FLAGS+=("-I$INCLUDE_DIR")
fi

KERNEL_SOURCES=()
while IFS= read -r -d '' src; do
    KERNEL_SOURCES+=("$src")
done < <(find "$SRC_DIR" -maxdepth 1 -type f -name '*.cu' -print0 | sort -z)

((${#KERNEL_SOURCES[@]} > 0)) || fail "No CUDA source files found in $SRC_DIR"

rm -rf -- "$BUILD_DIR"
mkdir -p -- "$BUILD_DIR"

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cuda-build.XXXXXX")"
cleanup() {
    rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

printf 'Building CUDA kernels for %s...\n' "$CUDA_ARCH"
printf 'ROOT_DIR: %s\n' "$ROOT_DIR"
printf 'CUDA_PATH: %s\n' "$CUDA_PATH"
printf 'CUDA_LIB_DIR: %s\n' "$CUDA_LIB_DIR"

for src in "${KERNEL_SOURCES[@]}"; do
    build_kernel "$src"
done

printf 'CUDA kernels built successfully!\n'
printf 'Output: %s\n' "$BUILD_DIR"
ls -la -- "$BUILD_DIR"
