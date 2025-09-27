#!/bin/bash

# Build script for bed tool with GPU support

echo "Building bed tool with GPU acceleration..."

# Compile CUDA library
echo "Compiling CUDA kernels..."
nvcc -Xcompiler -fPIC -c -o ../cuda_unique_topk.o ../cuda_unique_topk.cu -arch=sm_86 -O3
nvcc -shared -o ../libcuda_unique_topk.so ../cuda_unique_topk.o -lcudart -lcublas

# Build Go binary
echo "Building Go binary..."
CGO_CFLAGS="-I.." \
CGO_LDFLAGS="-L.. -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda_unique_topk" \
go build -o bed main.go

echo "Build complete!"
echo "Usage: ./bed 'search query'"