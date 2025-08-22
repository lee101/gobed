# Makefile for native LibTorch integration with CUDA operations

LIBTORCH_ROOT = /home/lee/code/gobed/libtorch
CUDA_OPS_PATH = /home/lee/code/gobed/gpu_search/cuda_ops/build
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O3
INCLUDES = -I$(LIBTORCH_ROOT)/include -I$(LIBTORCH_ROOT)/include/torch/csrc/api/include -I/usr/local/cuda-12.0/include
LDFLAGS = -L$(LIBTORCH_ROOT)/lib -L/usr/local/cuda-12.0/targets/x86_64-linux/lib -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lcudart -shared

WRAPPER_LIB = libtorch_cgo_wrapper.so
WRAPPER_SRC = torch_gpu_forced.cpp

.PHONY: all clean test build-cuda-ops check-deps

all: $(WRAPPER_LIB)

# Build CUDA operations first
build-cuda-ops:
	@echo "🔧 Building CUDA operations..."
	cd $(CUDA_OPS_PATH)/.. && mkdir -p build && cd build && \
	cmake .. && make -j$(shell nproc)

$(WRAPPER_LIB): $(WRAPPER_SRC) torch_cgo_wrapper.h
	@echo "🔧 Building LibTorch CGO wrapper with CUDA ops..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "✅ Built $(WRAPPER_LIB)"

clean:
	rm -f $(WRAPPER_LIB)
	@echo "🧹 Cleaned build artifacts"

check-deps:
	@echo "🔍 Checking dependencies..."
	@echo "LibTorch path: $(LIBTORCH_ROOT)"
	@ls -la $(LIBTORCH_ROOT)/lib/libtorch.so || echo "❌ LibTorch not found"
	@echo "CUDA operations path: $(CUDA_OPS_PATH)"
	@ls -la $(CUDA_OPS_PATH)/libgobed_ann_ops.so || echo "❌ CUDA ops not found"
	@which nvcc || echo "❌ NVCC not found"
	@nvidia-smi || echo "❌ NVIDIA driver not found"

test: $(WRAPPER_LIB)
	@echo "🧪 Testing native LibTorch integration..."
	LD_LIBRARY_PATH=$(LIBTORCH_ROOT)/lib:$(CUDA_OPS_PATH):. go test -v ./...

install: $(WRAPPER_LIB)
	@echo "📦 Installing shared library..."
	@mkdir -p ../lib
	cp $(WRAPPER_LIB) ../lib/
	@echo "✅ Installed to ../lib/"

help:
	@echo "Available targets:"
	@echo "  all     - Build the LibTorch CGO wrapper"
	@echo "  clean   - Remove build artifacts"
	@echo "  test    - Run tests with the wrapper"
	@echo "  install - Install shared library"
	@echo "  help    - Show this help"