#!/usr/bin/env bash
#
# GPU validation entrypoint for self-hosted runners with CUDA 12.9
# and the cuVS/CAGRA libraries installed.

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; aborting" >&2
  exit 1
fi

nvidia-smi

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
CUVS_PATH="${CUVS_PATH:-/usr/local/cuvs}"

if [ -f ./scripts/detect_gpu.sh ]; then
  # shellcheck disable=SC1091
  source ./scripts/detect_gpu.sh
fi

if [ -f ./gpu_env.sh ]; then
  # shellcheck disable=SC1091
  source ./gpu_env.sh
fi

export LD_LIBRARY_PATH="$(pwd):$(pwd)/gpu:$(pwd)/libtorch/lib:${CUDA_PATH}/lib64:${CUVS_PATH}/lib:${LD_LIBRARY_PATH:-}"
export CGO_LDFLAGS="-L$(pwd) -L$(pwd)/gpu -L${CUDA_PATH}/lib64 -L${CUVS_PATH}/lib -lcagra_wrapper -lgpu_fast_search -lgpu_memory -lcudart -lcuvs -lcuvs_c"

# Ensure CUDA/cuVS toolkits are discoverable
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found on PATH" >&2
fi

# Build wrapper on first run if the shared object is missing
if [ ! -f ./libcagra_wrapper.so ]; then
  echo "libcagra_wrapper.so missing – invoking build_cagra.sh"
  ./build_cagra.sh
fi

echo "🧪 Running tagged GPU test suite"
PACKAGES=(
  "github.com/lee101/gobed"
  "github.com/lee101/gobed/bed"
)

for pkg in "${PACKAGES[@]}"; do
  echo "  → go test -tags \"cagra gpu\" ${pkg}"
  go test -tags "cagra gpu" -v "${pkg}"
done

echo "✅ GPU checks completed"
