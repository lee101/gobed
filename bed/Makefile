# Makefile for building CUDA libraries and Go binary

# CUDA configuration
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_FLAGS = -O3 -arch=sm_70 -Xcompiler -fPIC -std=c++14
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt

# Output files
CUDA_LIB = libcuda_unique_topk.so
GO_BINARY = bed

.PHONY: all cuda go clean

all: cuda go

# Build CUDA library
cuda: $(CUDA_LIB)

$(CUDA_LIB): cuda_unique_topk.cu
	$(NVCC) $(CUDA_FLAGS) -shared -o $@ $< $(CUDA_LIBS)
	@echo "✅ Built CUDA library: $@"

# Build Go binary with GPU support
go: $(CUDA_LIB)
	@echo "🔨 Building Go binary with GPU support..."
	CGO_LDFLAGS="-L. -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcuda_unique_topk" \
	CGO_CFLAGS="-I$(CUDA_PATH)/include" \
	go build -tags gpu -o $(GO_BINARY) main.go
	@echo "✅ Built Go binary: $(GO_BINARY)"

# Clean build artifacts
clean:
	rm -f $(CUDA_LIB) $(GO_BINARY)
	@echo "🧹 Cleaned build artifacts"

# Test the binary
test: all
	./$(GO_BINARY) -dir . -k 5 --debug "test query"