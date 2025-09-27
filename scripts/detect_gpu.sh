#!/bin/bash

# GPU Auto-Detection and Environment Setup Script
# Detects CUDA version, sets appropriate compiler, and configures environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 GPU Environment Detection${NC}"

# Auto-detect CUDA version from nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    if [ -z "$CUDA_VERSION" ]; then
        echo -e "${YELLOW}⚠️  nvidia-smi found but couldn't detect CUDA version${NC}"
        CUDA_VERSION=""
    else
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        echo -e "${GREEN}✓ Detected CUDA Version: ${CUDA_VERSION}${NC}"

        # Check GPU availability
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        echo -e "${GREEN}✓ Found ${GPU_COUNT} GPU(s)${NC}"
        nvidia-smi -L | head -3
    fi
else
    echo -e "${YELLOW}⚠️  nvidia-smi not found - GPU support disabled${NC}"
    CUDA_VERSION=""
fi

# Find appropriate CUDA installation
if [ -n "$CUDA_VERSION" ]; then
    CUDA_PATHS=(
        "/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}"
        "/usr/local/cuda-${CUDA_MAJOR}"
        "/usr/local/cuda"
        "/opt/cuda"
        "$HOME/cuda"
    )

    CUDA_PATH=""
    for path in "${CUDA_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
            CUDA_PATH="$path"
            echo -e "${GREEN}✓ Found CUDA installation: ${CUDA_PATH}${NC}"
            break
        fi
    done

    if [ -z "$CUDA_PATH" ]; then
        echo -e "${RED}✗ CUDA ${CUDA_VERSION} detected but installation not found${NC}"
        echo -e "${YELLOW}  Searched in: ${CUDA_PATHS[*]}${NC}"
        CUDA_VERSION=""
    fi
fi

# Auto-select GCC based on CUDA version
if [ -n "$CUDA_VERSION" ]; then
    echo -e "${GREEN}🔧 Configuring compiler for CUDA ${CUDA_VERSION}${NC}"

    # CUDA compatibility matrix
    # CUDA 12.0-12.2: GCC 12
    # CUDA 12.3-12.6: GCC 13
    # CUDA 12.7+: GCC 13

    if [ "$CUDA_MAJOR" = "12" ]; then
        if [ "$CUDA_MINOR" -ge "3" ] && command -v gcc-13 &> /dev/null; then
            CUDA_GCC="gcc-13"
            CUDA_GXX="g++-13"
            GCC_VERSION="13"
        elif command -v gcc-12 &> /dev/null; then
            CUDA_GCC="gcc-12"
            CUDA_GXX="g++-12"
            GCC_VERSION="12"
        else
            CUDA_GCC="gcc"
            CUDA_GXX="g++"
            GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
        fi
    elif [ "$CUDA_MAJOR" = "11" ]; then
        if command -v gcc-11 &> /dev/null; then
            CUDA_GCC="gcc-11"
            CUDA_GXX="g++-11"
            GCC_VERSION="11"
        else
            CUDA_GCC="gcc"
            CUDA_GXX="g++"
            GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
        fi
    else
        CUDA_GCC="gcc"
        CUDA_GXX="g++"
        GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    fi

    echo -e "${GREEN}✓ Using GCC ${GCC_VERSION} (${CUDA_GCC})${NC}"
else
    CUDA_GCC="gcc"
    CUDA_GXX="g++"
fi

# Check for Go installation
if command -v go &> /dev/null; then
    GO_VERSION=$(go version | grep -oP 'go[0-9]+\.[0-9]+(\.[0-9]+)?')
    echo -e "${GREEN}✓ Found Go: ${GO_VERSION}${NC}"
    GO_PATH=$(which go)
else
    echo -e "${YELLOW}⚠️  Go not found, checking common locations${NC}"
    if [ -f "/usr/local/go/bin/go" ]; then
        GO_PATH="/usr/local/go/bin"
        GO_VERSION=$($GO_PATH/go version | grep -oP 'go[0-9]+\.[0-9]+(\.[0-9]+)?')
        echo -e "${GREEN}✓ Found Go at /usr/local/go: ${GO_VERSION}${NC}"
    else
        echo -e "${RED}✗ Go installation not found${NC}"
        GO_PATH=""
    fi
fi

# Generate environment configuration
cat > gpu_env.sh << 'EOF'
#!/bin/bash
# Auto-generated GPU environment configuration
# Source this file: source ./gpu_env.sh

EOF

# Add Go path if found
if [ -n "$GO_PATH" ]; then
    if [ "$GO_PATH" = "/usr/local/go/bin" ]; then
        echo 'export PATH=/usr/local/go/bin:$PATH' >> gpu_env.sh
    fi
fi

# Add CUDA configuration if available
if [ -n "$CUDA_PATH" ]; then
    cat >> gpu_env.sh << EOF
# CUDA Configuration
export CUDA_PATH="${CUDA_PATH}"
export PATH=\$CUDA_PATH/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\${LD_LIBRARY_PATH}
export CGO_CFLAGS="-I\$CUDA_PATH/include"
export CGO_LDFLAGS="-L\$CUDA_PATH/lib64 -lcudart -lcublas -lcublasLt"
export CC="${CUDA_GCC}"
export CXX="${CUDA_GXX}"

# GPU Flags
export USE_GPU=true
export CUDA_ENABLED=true
export GPU_ENABLED=true
export GPU_TAGS="-tags gpu,cuda"

EOF
else
    cat >> gpu_env.sh << EOF
# CPU-only mode (no CUDA found)
export USE_GPU=false
export CUDA_ENABLED=false
export GPU_ENABLED=false
export GPU_TAGS=""
export CC="${CUDA_GCC}"
export CXX="${CUDA_GXX}"

EOF
fi

# Add development flags
cat >> gpu_env.sh << 'EOF'
# Development Configuration
export DEV=true
export DEBUG=true
export INDEXCACHE=true

# Performance tuning
export GOMAXPROCS=3
export GOCACHE=${GOCACHE:-$HOME/.cache/go-build}
export GOMODCACHE=${GOMODCACHE:-$HOME/go/pkg/mod}

# Project-specific database (if needed)
export DATABASE_URL="postgresql://lee:netwrck2024@localhost/netwrck_ai_characters_dev"

# Helper function to build with GPU support
build_with_gpu() {
    if [ "$USE_GPU" = "true" ]; then
        echo "🚀 Building with GPU support..."
        go build $GPU_TAGS "$@"
    else
        echo "🖥️  Building CPU-only version..."
        go build "$@"
    fi
}

# Helper function to run tests with GPU
test_with_gpu() {
    if [ "$USE_GPU" = "true" ]; then
        echo "🚀 Running tests with GPU support..."
        go test $GPU_TAGS "$@"
    else
        echo "🖥️  Running CPU-only tests..."
        go test "$@"
    fi
}

echo "✓ GPU environment configured. Use 'build_with_gpu' and 'test_with_gpu' helpers."
EOF

chmod +x gpu_env.sh

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}           Configuration Summary            ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"

if [ -n "$CUDA_PATH" ]; then
    echo -e "  CUDA:        ${GREEN}✓${NC} ${CUDA_VERSION} at ${CUDA_PATH}"
    echo -e "  Compiler:    ${GREEN}✓${NC} ${CUDA_GCC} (GCC ${GCC_VERSION})"
    echo -e "  GPU Mode:    ${GREEN}✓${NC} Enabled"
else
    echo -e "  CUDA:        ${YELLOW}✗${NC} Not available"
    echo -e "  Compiler:    ${YELLOW}⚠${NC} ${CUDA_GCC}"
    echo -e "  GPU Mode:    ${YELLOW}✗${NC} Disabled (CPU-only)"
fi

if [ -n "$GO_VERSION" ]; then
    echo -e "  Go:          ${GREEN}✓${NC} ${GO_VERSION}"
else
    echo -e "  Go:          ${RED}✗${NC} Not found"
fi

echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}To activate this environment:${NC}"
echo -e "  ${YELLOW}source ./gpu_env.sh${NC}"
echo ""

# Test CUDA compilation if available
if [ -n "$CUDA_PATH" ] && [ -f "$CUDA_PATH/bin/nvcc" ]; then
    echo -e "${GREEN}Testing CUDA compilation...${NC}"

    # Create test CUDA file
    cat > /tmp/cuda_test.cu << 'CUDA_EOF'
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices: %d\n", device_count);

    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU 0: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    }
    return 0;
}
CUDA_EOF

    # Try to compile
    if $CUDA_PATH/bin/nvcc -ccbin $CUDA_GCC -o /tmp/cuda_test /tmp/cuda_test.cu 2>/dev/null; then
        echo -e "${GREEN}✓ CUDA compilation test passed${NC}"
        if [ -f /tmp/cuda_test ]; then
            /tmp/cuda_test
            rm -f /tmp/cuda_test
        fi
    else
        echo -e "${YELLOW}⚠️  CUDA compilation test failed - check compiler compatibility${NC}"
    fi
    rm -f /tmp/cuda_test.cu
fi