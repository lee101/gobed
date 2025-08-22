# Makefile for Real GPU LibTorch integration with CUDA kernels
LIBTORCH_ROOT = /home/lee/code/gobed/libtorch
CUDA_PATH = /usr/local/cuda-12.0

# Compilers
NVCC = /usr/local/cuda-12.0/bin/nvcc
CXX = g++

# Flags
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_86 --compiler-options -fPIC -allow-unsupported-compiler -ccbin g++-12
CXX_FLAGS = -std=c++17 -O3 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
SHARED_FLAGS = --shared

# Include paths
INCLUDES = -I$(LIBTORCH_ROOT)/include \
           -I$(LIBTORCH_ROOT)/include/torch/csrc/api/include \
           -I$(CUDA_PATH)/include

# Library paths and libraries
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas
TORCH_LIBS = -L$(LIBTORCH_ROOT)/lib -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10
RPATH_FLAGS = -Xlinker -rpath -Xlinker $(LIBTORCH_ROOT)/lib -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64

# Output files
WRAPPER_LIB = libtorch_cgo_wrapper.so
CUDA_KERNEL_OBJ = cuda_kernels.o
WRAPPER_OBJ = torch_gpu_forced.o

# Source files
CUDA_KERNEL_SRC = cuda_kernels.cu
WRAPPER_SRC = torch_gpu_forced.cpp

.PHONY: all clean test diagnose

all: $(WRAPPER_LIB)

# Compile CUDA kernels
$(CUDA_KERNEL_OBJ): $(CUDA_KERNEL_SRC)
	@echo "🔧 Compiling CUDA kernels..."
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<
	@echo "✅ CUDA kernels compiled"

# Compile C++ wrapper
$(WRAPPER_OBJ): $(WRAPPER_SRC) torch_cgo_wrapper.h
	@echo "🔧 Compiling C++ wrapper..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c -o $@ $<
	@echo "✅ C++ wrapper compiled"

# Link everything together
$(WRAPPER_LIB): $(CUDA_KERNEL_OBJ) $(WRAPPER_OBJ)
	@echo "🔗 Linking GPU-accelerated library..."
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ $(CUDA_LIBS) $(TORCH_LIBS) $(RPATH_FLAGS)
	@echo "✅ Built $(WRAPPER_LIB) with Real GPU acceleration!"
	@echo ""
	@echo "🚀 GPU-accelerated library is ready!"
	@echo "   CUDA kernels: ✅"
	@echo "   LibTorch integration: ✅"
	@echo "   Shared library: $(WRAPPER_LIB)"

clean:
	rm -f $(WRAPPER_LIB) $(CUDA_KERNEL_OBJ) $(WRAPPER_OBJ)
	@echo "🧹 Cleaned build artifacts"

test: $(WRAPPER_LIB)
	@echo "🧪 Testing GPU integration..."
	LD_LIBRARY_PATH=$(LIBTORCH_ROOT)/lib:$(CUDA_PATH)/lib64:. go test -v ./...

diagnose:
	@echo "🔍 GPU Diagnostics:"
	@echo "-------------------"
	@nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
	@echo ""
	@echo "📦 CUDA Installation:"
	@ls -la $(CUDA_PATH)/lib64/libcudart.so* 2>/dev/null || echo "❌ CUDA runtime not found"
	@ls -la $(CUDA_PATH)/lib64/libcublas.so* 2>/dev/null || echo "❌ CUBLAS not found"
	@echo ""
	@echo "📚 LibTorch Installation:"
	@ls -la $(LIBTORCH_ROOT)/lib/libtorch_cuda.so 2>/dev/null || echo "❌ LibTorch CUDA not found"
	@echo ""
	@echo "🔧 Compiler Versions:"
	@$(NVCC) --version | head -n 4
	@echo ""
	@$(CXX) --version | head -n 1

help:
	@echo "GPU-Accelerated LibTorch Build System"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@echo "  all      - Build the GPU-accelerated LibTorch wrapper"
	@echo "  clean    - Remove all build artifacts"
	@echo "  test     - Run tests with GPU acceleration"
	@echo "  diagnose - Check GPU and CUDA installation"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Environment:"
	@echo "  CUDA Path: $(CUDA_PATH)"
	@echo "  LibTorch: $(LIBTORCH_ROOT)"
	@echo ""
	@echo "To build: make"
	@echo "To test:  make test"