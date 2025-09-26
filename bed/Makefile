# Bed - Blazing Fast Semantic Search Tool
# Makefile for building and testing

.PHONY: all build clean test demo install uninstall help deps gpu-deps

# Build configuration
BINARY_NAME=bed
VERSION=$(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME=$(shell date -u '+%Y-%m-%d_%H:%M:%S')
COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Go build flags
LDFLAGS=-ldflags "-X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME) -X main.commit=$(COMMIT) -s -w"
BUILD_FLAGS=-trimpath $(LDFLAGS)

# GPU build flags (requires CUDA)
GPU_FLAGS=-tags="gpu cuda"

# Default target
all: build

# Build the binary (CPU version)
build:
	@echo "Building bed (CPU version)..."
	go build $(BUILD_FLAGS) -o $(BINARY_NAME) main.go

# Build with GPU support
build-gpu:
	@echo "Building bed with GPU acceleration..."
	@echo "Note: This requires CUDA 12.8+ to be installed"
	go build $(BUILD_FLAGS) $(GPU_FLAGS) -o $(BINARY_NAME)-gpu main.go

# Build both CPU and GPU versions
build-all: build build-gpu

# Install dependencies
deps:
	@echo "Installing Go dependencies..."
	go mod tidy
	go mod download

# Install GPU dependencies
gpu-deps: deps
	@echo "Installing GPU dependencies..."
	@echo "Make sure you have CUDA 12.8+ installed"
	cd .. && make -C gpu all

# Run tests
test:
	@echo "Running tests..."
	go test ./src/... -v

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	go test ./src/... -v -coverprofile=coverage.out
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report saved to coverage.html"

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	go test ./src/... -bench=. -benchmem

# Run the demo
demo: build
	@echo "Running bed demo..."
	cd examples && go run demo.go

# Install bed to system PATH
install: build
	@echo "Installing bed to /usr/local/bin..."
	sudo cp $(BINARY_NAME) /usr/local/bin/
	@echo "bed installed successfully"

# Install GPU version
install-gpu: build-gpu
	@echo "Installing bed-gpu to /usr/local/bin..."
	sudo cp $(BINARY_NAME)-gpu /usr/local/bin/bed
	@echo "bed (GPU version) installed successfully"

# Uninstall from system
uninstall:
	@echo "Removing bed from /usr/local/bin..."
	sudo rm -f /usr/local/bin/bed
	@echo "bed uninstalled successfully"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(BINARY_NAME) $(BINARY_NAME)-gpu
	rm -f coverage.out coverage.html
	rm -rf dist/
	go clean -cache

# Format code
fmt:
	@echo "Formatting Go code..."
	go fmt ./...

# Run linter
lint:
	@echo "Running golangci-lint..."
	golangci-lint run

# Run all quality checks
quality: fmt lint test

# Create release builds for multiple platforms
release: clean
	@echo "Creating release builds..."
	mkdir -p dist
	
	# Linux AMD64
	GOOS=linux GOARCH=amd64 go build $(BUILD_FLAGS) -o dist/$(BINARY_NAME)-linux-amd64 main.go
	
	# Linux ARM64
	GOOS=linux GOARCH=arm64 go build $(BUILD_FLAGS) -o dist/$(BINARY_NAME)-linux-arm64 main.go
	
	# macOS AMD64
	GOOS=darwin GOARCH=amd64 go build $(BUILD_FLAGS) -o dist/$(BINARY_NAME)-darwin-amd64 main.go
	
	# macOS ARM64 (M1/M2)
	GOOS=darwin GOARCH=arm64 go build $(BUILD_FLAGS) -o dist/$(BINARY_NAME)-darwin-arm64 main.go
	
	# Windows AMD64
	GOOS=windows GOARCH=amd64 go build $(BUILD_FLAGS) -o dist/$(BINARY_NAME)-windows-amd64.exe main.go
	
	@echo "Release builds created in dist/"
	ls -la dist/

# Create GPU release builds (Linux only for now)
release-gpu: clean
	@echo "Creating GPU release builds..."
	mkdir -p dist
	
	# Linux AMD64 with GPU
	GOOS=linux GOARCH=amd64 go build $(BUILD_FLAGS) $(GPU_FLAGS) -o dist/$(BINARY_NAME)-gpu-linux-amd64 main.go
	
	@echo "GPU release builds created in dist/"
	ls -la dist/

# Development server (watches for changes and rebuilds)
dev:
	@echo "Starting development mode (requires 'watchexec')..."
	@which watchexec > /dev/null || (echo "Please install watchexec: cargo install watchexec-cli" && exit 1)
	watchexec -e go,mod -r -- make build

# Quick test - build and run basic functionality
quick-test: build
	@echo "Running quick functionality test..."
	./$(BINARY_NAME) --help
	@echo ""
	@echo "Creating test directory..."
	mkdir -p /tmp/bed-test
	echo 'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("Hello world")\n}' > /tmp/bed-test/main.go
	echo 'func processData() error {\n    return nil\n}' > /tmp/bed-test/utils.go
	cd /tmp/bed-test && ../$(PWD)/$(BINARY_NAME) index --verbose
	cd /tmp/bed-test && ../$(PWD)/$(BINARY_NAME) "hello world function" --limit 2
	rm -rf /tmp/bed-test
	@echo "Quick test completed successfully!"

# Performance test with larger codebase
perf-test: build
	@echo "Running performance test on gobed codebase..."
	cd .. && ./bed/$(BINARY_NAME) index --force --verbose
	cd .. && ./bed/$(BINARY_NAME) "GPU acceleration" --limit 5
	cd .. && ./bed/$(BINARY_NAME) "embedding generation" --limit 5
	cd .. && ./bed/$(BINARY_NAME) status

# Docker build
docker:
	@echo "Building Docker image..."
	docker build -t bed:latest .

# Docker GPU build
docker-gpu:
	@echo "Building Docker GPU image..."
	docker build -t bed:gpu -f Dockerfile.gpu .

# Show help
help:
	@echo "Bed - Blazing Fast Semantic Search Tool"
	@echo "======================================="
	@echo ""
	@echo "Available targets:"
	@echo "  build        Build the CPU version"
	@echo "  build-gpu    Build with GPU acceleration (requires CUDA)"
	@echo "  build-all    Build both CPU and GPU versions"
	@echo "  deps         Install Go dependencies"
	@echo "  gpu-deps     Install GPU dependencies"
	@echo "  test         Run tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  bench        Run benchmarks"
	@echo "  demo         Run interactive demo"
	@echo "  install      Install to system PATH"
	@echo "  install-gpu  Install GPU version to system PATH"
	@echo "  uninstall    Remove from system PATH"
	@echo "  clean        Clean build artifacts"
	@echo "  fmt          Format Go code"
	@echo "  lint         Run linter"
	@echo "  quality      Run all quality checks"
	@echo "  release      Create release builds for all platforms"
	@echo "  release-gpu  Create GPU release builds"
	@echo "  quick-test   Quick functionality test"
	@echo "  perf-test    Performance test on gobed codebase"
	@echo "  dev          Development mode (auto-rebuild on changes)"
	@echo "  docker       Build Docker image"
	@echo "  docker-gpu   Build Docker GPU image"
	@echo "  help         Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make build                 # Build CPU version"
	@echo "  make build-gpu             # Build GPU version"
	@echo "  make demo                  # Run demo"
	@echo "  make install               # Install to system"
	@echo "  make quick-test            # Test basic functionality"