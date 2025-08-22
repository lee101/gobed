#!/bin/bash
set -e

echo "Building GPU ANN ops with CUDA..."

# Source LibTorch environment
source ~/.secretbashrc 2>/dev/null || true

# Use GCC 12 for CUDA compatibility
export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=g++-12

echo "Using compiler: $(gcc-12 --version | head -1)"

# Clean previous build
rm -rf build

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
  -DCMAKE_PREFIX_PATH=$LIBTORCH \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-12 \
  -DCMAKE_CXX_COMPILER=g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-12

# Build
cmake --build . -j$(nproc)

echo "✅ Built libgobed_ann_ops.so"
echo "Library location: $(pwd)/libgobed_ann_ops.so"

# Copy to a standard location
sudo cp libgobed_ann_ops.so /usr/local/lib/ 2>/dev/null || \
  cp libgobed_ann_ops.so ../../ 

echo "✅ Installed to /usr/local/lib/ or parent directory"