# Makefile for gobed project

# Variables
BINARY_NAME=gobed
DOCKER_IMAGE=gobed
VERSION=$(shell git describe --tags --always --dirty)
GO=go
GOFLAGS=-v
LDFLAGS=-ldflags "-s -w -X main.Version=$(VERSION)"
COVERAGE_FILE=coverage.out

# GPU/CUDA Variables
CUDA_PATH=/usr/local/cuda-12.0
GPU_LIB_PATH=./gpu
GPU_LIB_NAME=libtorch_cgo_wrapper.so
NVCC_FLAGS=-std=c++17 -O3 -arch=sm_86 --compiler-options -fPIC

# Directories
CMD_DIR=./cmd
BIN_DIR=./bin
DIST_DIR=./dist

# Default target
.PHONY: all
all: clean gpu-build build test

# GPU/CUDA Build targets
.PHONY: gpu-check
gpu-check:
	@echo "Checking GPU/CUDA availability..."
	@command -v nvcc > /dev/null || (echo "❌ NVCC not found. Please install CUDA toolkit."; exit 1)
	@command -v nvidia-smi > /dev/null || (echo "❌ nvidia-smi not found. Please install NVIDIA drivers."; exit 1)
	@echo "✅ CUDA environment detected"
	@nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
	@nvcc --version | head -n 4

.PHONY: gpu-build
gpu-build: gpu-check
	@echo "Building GPU acceleration library..."
	@cd $(GPU_LIB_PATH) && make -f Makefile
	@echo "✅ GPU library built: $(GPU_LIB_PATH)/$(GPU_LIB_NAME)"

.PHONY: gpu-clean
gpu-clean:
	@echo "Cleaning GPU build artifacts..."
	@cd $(GPU_LIB_PATH) && make -f Makefile clean

.PHONY: gpu-test
gpu-test: gpu-build
	@echo "Testing GPU acceleration..."
	@cd $(GPU_LIB_PATH) && make -f Makefile test

.PHONY: gpu-server
gpu-server: gpu-build
	@echo "Building GPU-accelerated server..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=1 LD_LIBRARY_PATH=$(GPU_LIB_PATH):$(CUDA_PATH)/lib64 \
		$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BIN_DIR)/gpu-server $(CMD_DIR)/gpu_server

.PHONY: simple-gpu-server
simple-gpu-server: gpu-build
	@echo "Building simple GPU-accelerated server..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=1 LD_LIBRARY_PATH=$(GPU_LIB_PATH):$(CUDA_PATH)/lib64 \
		$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BIN_DIR)/simple-gpu-server $(CMD_DIR)/simple_gpu_server

.PHONY: cuda-server
cuda-server:
	@echo "Building CUDA-accelerated server..."
	@mkdir -p $(BIN_DIR)
	$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BIN_DIR)/cuda-server $(CMD_DIR)/cuda_server

.PHONY: run-cuda-server
run-cuda-server: cuda-server
	@echo "🚀 Starting CUDA-Accelerated Server..."
	@echo "   Features: Pure CUDA acceleration, real-time stats, demo mode"
	@echo ""
	$(BIN_DIR)/cuda-server

.PHONY: simple-server
simple-server:
	@echo "Building high-performance server..."
	@mkdir -p $(BIN_DIR)
	$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BIN_DIR)/simple-server $(CMD_DIR)/simple_server

.PHONY: run-simple-server
run-simple-server: simple-server
	@echo "🚀 Starting High-Performance GoBeD Server..."
	@echo "   Features: Optimized int8 search, parallel processing, real-time stats"
	@echo ""
	$(BIN_DIR)/simple-server

.PHONY: run-demo
run-demo: simple-server
	@echo "🎪 Starting Server with Performance Demo..."
	$(BIN_DIR)/simple-server -demo -demo-vectors=5000 -stats -load-test

.PHONY: run-cuda-demo
run-cuda-demo: cuda-server
	@echo "🎪 Starting CUDA Server with Demo..."
	$(BIN_DIR)/cuda-server -demo -demo-vectors=5000 -stats

.PHONY: run-simple-gpu-server
run-simple-gpu-server: simple-gpu-server
	@echo "🚀 Starting Simple GPU-Accelerated Server..."
	@echo "   Features: CUDA search, GPU indexing, real-time stats"
	@echo ""
	LD_LIBRARY_PATH=$(GPU_LIB_PATH):$(CUDA_PATH)/lib64 $(BIN_DIR)/simple-gpu-server

.PHONY: run-gpu-server
run-gpu-server: gpu-server
	@echo "Starting GPU-accelerated server..."
	@echo "🚀 GPU Server Features:"
	@echo "   • CUDA-accelerated similarity search"
	@echo "   • GPU memory management and pooling"  
	@echo "   • Batch processing optimization"
	@echo "   • Automatic CPU fallback"
	@echo ""
	LD_LIBRARY_PATH=$(GPU_LIB_PATH):$(CUDA_PATH)/lib64 $(BIN_DIR)/gpu-server

# Build targets
.PHONY: build
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BIN_DIR)
	$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BIN_DIR)/$(BINARY_NAME) $(CMD_DIR)/demo

.PHONY: build-all
build-all:
	@echo "Building all binaries..."
	@mkdir -p $(BIN_DIR)
	@for dir in $(CMD_DIR)/*; do \
		if [ -d "$$dir" ]; then \
			name=$$(basename $$dir); \
			echo "Building $$name..."; \
			$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BIN_DIR)/$(BINARY_NAME)-$$name $$dir || true; \
		fi \
	done

.PHONY: build-release
build-release:
	@echo "Building release binaries..."
	@mkdir -p $(DIST_DIR)
	# Linux AMD64
	GOOS=linux GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-linux-amd64 $(CMD_DIR)/demo
	# Linux ARM64
	GOOS=linux GOARCH=arm64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-linux-arm64 $(CMD_DIR)/demo
	# Darwin AMD64
	GOOS=darwin GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-darwin-amd64 $(CMD_DIR)/demo
	# Darwin ARM64
	GOOS=darwin GOARCH=arm64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-darwin-arm64 $(CMD_DIR)/demo
	# Windows AMD64
	GOOS=windows GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-windows-amd64.exe $(CMD_DIR)/demo

# Test targets
.PHONY: test
test:
	@echo "Running tests..."
	$(GO) test $(GOFLAGS) ./...

.PHONY: test-short
test-short:
	@echo "Running short tests..."
	$(GO) test -short $(GOFLAGS) ./...

.PHONY: test-race
test-race:
	@echo "Running tests with race detector..."
	$(GO) test -race $(GOFLAGS) ./...

.PHONY: test-coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(GO) test -coverprofile=$(COVERAGE_FILE) -covermode=atomic ./...
	$(GO) tool cover -html=$(COVERAGE_FILE) -o coverage.html
	@echo "Coverage report saved to coverage.html"

.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	$(GO) test -tags=integration $(GOFLAGS) ./...

# Benchmark targets
.PHONY: bench
bench:
	@echo "Running benchmarks..."
	$(GO) test -bench=. -benchmem ./...

.PHONY: bench-cpu
bench-cpu:
	@echo "Running CPU benchmarks with profiling..."
	$(GO) test -bench=. -benchmem -cpuprofile=cpu.prof ./...
	$(GO) tool pprof -http=:8080 cpu.prof

.PHONY: bench-mem
bench-mem:
	@echo "Running memory benchmarks with profiling..."
	$(GO) test -bench=. -benchmem -memprofile=mem.prof ./...
	$(GO) tool pprof -http=:8080 mem.prof

# Code quality targets
.PHONY: fmt
fmt:
	@echo "Formatting code..."
	$(GO) fmt ./...
	@echo "Running goimports..."
	@command -v goimports > /dev/null || $(GO) install golang.org/x/tools/cmd/goimports@latest
	goimports -w .

.PHONY: lint
lint:
	@echo "Running linters..."
	@command -v golangci-lint > /dev/null || $(GO) install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	golangci-lint run ./...

.PHONY: vet
vet:
	@echo "Running go vet..."
	$(GO) vet ./...

.PHONY: staticcheck
staticcheck:
	@echo "Running staticcheck..."
	@command -v staticcheck > /dev/null || $(GO) install honnef.co/go/tools/cmd/staticcheck@latest
	staticcheck ./...

.PHONY: security
security:
	@echo "Running security scan..."
	@command -v gosec > /dev/null || $(GO) install github.com/securego/gosec/v2/cmd/gosec@latest
	gosec -fmt sarif -out gosec.sarif ./...
	@command -v govulncheck > /dev/null || $(GO) install golang.org/x/vuln/cmd/govulncheck@latest
	govulncheck ./...

.PHONY: quality
quality: fmt vet lint staticcheck security

# Dependency management
.PHONY: deps
deps:
	@echo "Downloading dependencies..."
	$(GO) mod download

.PHONY: tidy
tidy:
	@echo "Tidying dependencies..."
	$(GO) mod tidy

.PHONY: verify
verify:
	@echo "Verifying dependencies..."
	$(GO) mod verify

.PHONY: update
update:
	@echo "Updating dependencies..."
	$(GO) get -u ./...
	$(GO) mod tidy

# Docker targets
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(VERSION) .
	docker tag $(DOCKER_IMAGE):$(VERSION) $(DOCKER_IMAGE):latest

.PHONY: docker-push
docker-push:
	@echo "Pushing Docker image..."
	docker push $(DOCKER_IMAGE):$(VERSION)
	docker push $(DOCKER_IMAGE):latest

.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	docker run --rm -it $(DOCKER_IMAGE):latest

# Development targets
.PHONY: run
run: build
	@echo "Running $(BINARY_NAME)..."
	$(BIN_DIR)/$(BINARY_NAME)

.PHONY: run-server
run-server: build
	@echo "Running search server..."
	$(GO) run $(CMD_DIR)/search_server/main.go

.PHONY: dev
dev:
	@echo "Starting development mode with hot reload..."
	@command -v air > /dev/null || $(GO) install github.com/cosmtrek/air@latest
	air

.PHONY: install
install: build
	@echo "Installing $(BINARY_NAME)..."
	$(GO) install $(LDFLAGS) $(CMD_DIR)/demo

# Documentation targets
.PHONY: docs
docs:
	@echo "Generating documentation..."
	@command -v godoc > /dev/null || $(GO) install golang.org/x/tools/cmd/godoc@latest
	@echo "Starting godoc server on http://localhost:6060"
	godoc -http=:6060

# Clean targets
.PHONY: clean
clean:
	@echo "Cleaning..."
	@rm -rf $(BIN_DIR) $(DIST_DIR)
	@rm -f $(COVERAGE_FILE) coverage.html
	@rm -f cpu.prof mem.prof
	@rm -f gosec.sarif
	@$(GO) clean -cache -testcache

.PHONY: clean-all
clean-all: clean
	@echo "Deep cleaning..."
	@$(GO) clean -modcache

# CI targets
.PHONY: ci
ci: deps fmt vet lint test

.PHONY: ci-full
ci-full: deps quality test-race test-coverage bench

# Help target
.PHONY: help
help:
	@echo "GoBeD - GPU-Accelerated Vector Search Engine"
	@echo "============================================"
	@echo ""
	@echo "🚀 GPU Acceleration Targets:"
	@echo "  gpu-check      - Check CUDA/GPU availability" 
	@echo "  gpu-build      - Build CUDA acceleration library"
	@echo "  gpu-clean      - Clean GPU build artifacts"
	@echo "  gpu-test       - Test GPU acceleration"
	@echo "  gpu-server     - Build GPU-accelerated server"
	@echo "  run-gpu-server - Run GPU-accelerated server (RECOMMENDED)"
	@echo ""
	@echo "📦 Standard Build Targets:"
	@echo "  all            - Clean, GPU build, build, and test"
	@echo "  build          - Build the main binary"
	@echo "  build-all      - Build all binaries"
	@echo "  build-release  - Build release binaries for all platforms"
	@echo ""
	@echo "🧪 Testing Targets:"
	@echo "  test           - Run tests"
	@echo "  test-short     - Run short tests"
	@echo "  test-race      - Run tests with race detector"
	@echo "  test-coverage  - Run tests with coverage"
	@echo "  bench          - Run benchmarks"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  fmt            - Format code"
	@echo "  lint           - Run linters"
	@echo "  vet            - Run go vet"
	@echo "  quality        - Run all code quality checks"
	@echo ""
	@echo "📋 Dependencies:"
	@echo "  deps           - Download dependencies"
	@echo "  tidy           - Tidy dependencies"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo ""
	@echo "🏃 Development:"
	@echo "  run            - Build and run the binary"
	@echo "  run-server     - Run standard search server"
	@echo "  clean          - Clean build artifacts"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "💡 Quick Start (CUDA Acceleration):"
	@echo "  make gpu-test        # Test pure CUDA acceleration (FASTEST)"
	@echo "  make run-simple-server # Run high-performance server"
	@echo "  make run-demo        # Run performance demonstration"
	@echo ""
	@echo "🚀 Pure CUDA Performance (RECOMMENDED):"
	@echo "  cd gpu && ./cuda_test     # Direct CUDA test (13.4x speedup!)"
	@echo "  cd gpu && ./full_benchmark # Comprehensive benchmarks"

.DEFAULT_GOAL := help