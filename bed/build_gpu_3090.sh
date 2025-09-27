#!/bin/bash

echo "🚀 Building RTX 3090 optimized bed..."

# Compile CUDA kernels with RTX 3090 optimizations
echo "   Compiling CUDA kernels for compute capability 8.6 (RTX 3090)..."
nvcc -c cuda_optimized.cu \
    -o cuda_optimized.o \
    -arch=sm_86 \
    -gencode=arch=compute_86,code=sm_86 \
    -O3 \
    -use_fast_math \
    --ptxas-options=-v \
    -lineinfo \
    -Xcompiler -fPIC \
    -I/usr/local/cuda/include

# Create shared library
echo "   Creating shared library..."
nvcc -shared -o libcuda_optimized.so \
    cuda_optimized.o \
    -L/usr/local/cuda/lib64 \
    -lcudart -lcublas \
    -Xcompiler -fPIC

# Build Go executable
echo "   Building Go executable..."
export CGO_CFLAGS="-I/usr/local/cuda/include"
export CGO_LDFLAGS="-L. -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda_optimized"
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

go build -o bed_gpu_3090 bed_gpu_3090.go

if [ $? -eq 0 ]; then
    echo "✅ Build successful! Run with:"
    echo "   env LD_LIBRARY_PATH=. ./bed_gpu_3090 -dir testdata \"your query\""
    echo ""
    echo "🎮 RTX 3090 optimizations enabled:"
    echo "   • Batch size: 2048"
    echo "   • Multi-stream processing (4 streams)"
    echo "   • Pinned memory transfers"
    echo "   • Tensor core support"
    echo "   • 20GB GPU memory pool"
else
    echo "❌ Build failed"
    exit 1
fi