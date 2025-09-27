#!/bin/bash
set -e

echo " Testing GPU Search Build"
echo "============================"

# Source environment
source ~/.secretbashrc 2>/dev/null || true

# Check environment
echo -e "\n Environment Check:"
echo "  CUDA: $(nvcc --version | grep release | cut -d' ' -f5)"
echo "  GCC: $(gcc --version | head -1 | cut -d' ' -f3)"
echo "  LibTorch: $LIBTORCH"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Test 1: Simple CUDA compilation
echo -e "\n Test 1: Building standalone CUDA test..."
cat > test_cuda.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel() {
    printf("CUDA kernel running on thread %d\n", threadIdx.x);
}

int main() {
    printf("Testing CUDA...\n");
    test_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    printf(" CUDA works!\n");
    return 0;
}
EOF

nvcc -allow-unsupported-compiler test_cuda.cu -o test_cuda
./test_cuda
rm test_cuda test_cuda.cu

# Test 2: Build CUDA ops without TorchScript
echo -e "\n Test 2: Building CUDA ops (simplified)..."
cd cuda_ops

# Create simple makefile for testing
cat > Makefile << 'EOF'
NVCC = nvcc
NVCC_FLAGS = -allow-unsupported-compiler -O3 --use_fast_math -gencode arch=compute_86,code=sm_86
TORCH_PATH = $(LIBTORCH)
INCLUDES = -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include
LIBS = -L$(TORCH_PATH)/lib -ltorch -ltorch_cuda -lc10 -lc10_cuda

test_ops: i8_dot512.cu
	$(NVCC) $(NVCC_FLAGS) -shared -Xcompiler -fPIC $(INCLUDES) i8_dot512.cu -o test_i8dot.so $(LIBS)
	@echo " Built test_i8dot.so"

clean:
	rm -f *.so *.o
EOF

make test_ops
ls -lh test_i8dot.so

echo -e "\n Build tests passed!"
echo ""
echo "Next steps:"
echo "1. The full build needs Python torch installed for TorchScript"
echo "2. Install with: pip install torch torchvision"
echo "3. Then run: ./build_and_test.sh"