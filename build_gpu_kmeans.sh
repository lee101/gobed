#!/bin/bash

# Build GPU K-means CUDA library

echo "🔨 Building GPU K-means library..."

# Set CUDA paths
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Compile CUDA kernel
nvcc -c gpu_kmeans.cu -o gpu_kmeans.o \
    -arch=sm_86 \
    -O3 \
    -use_fast_math \
    --expt-relaxed-constexpr \
    -Xcompiler -fPIC \
    -I$CUDA_PATH/include

# Create static library
ar rcs libgpu_kmeans.a gpu_kmeans.o

# Create shared library
nvcc -shared gpu_kmeans.o -o libgpu_kmeans.so \
    -lcudart -lcublas \
    -L$CUDA_PATH/lib64

echo "✅ GPU K-means library built successfully"
echo "   Static: libgpu_kmeans.a"
echo "   Shared: libgpu_kmeans.so"

# Set library path for testing
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

echo ""
echo "To use GPU K-means:"
echo "  export LD_LIBRARY_PATH=$(pwd):\$LD_LIBRARY_PATH"
echo "  go build -tags gpu"