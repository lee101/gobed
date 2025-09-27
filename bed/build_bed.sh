#!/bin/bash

set -e

echo "🔨 Building BED search tool..."

# Auto-detect CUDA version from nvidia-smi
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    if [ -n "$CUDA_VERSION" ]; then
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

        echo "✅ Detected CUDA Version: ${CUDA_VERSION}"

        # Find appropriate CUDA installation
        if [ -d "/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}" ]; then
            CUDA_PATH="/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}"
        elif [ -d "/usr/local/cuda-${CUDA_MAJOR}" ]; then
            CUDA_PATH="/usr/local/cuda-${CUDA_MAJOR}"
        elif [ "$CUDA_MAJOR" = "12" ] && [ -d "/usr/local/cuda-12.0" ]; then
            CUDA_PATH="/usr/local/cuda-12.0"
        elif [ -d "/usr/local/cuda" ]; then
            CUDA_PATH="/usr/local/cuda"
        else
            CUDA_PATH=""
        fi

        # Auto-select GCC based on CUDA version
        if [ "$CUDA_MAJOR" = "12" ]; then
            if [ "$CUDA_MINOR" -ge "3" ] && command -v gcc-13 &> /dev/null; then
                echo "✅ Using GCC 13 for CUDA ${CUDA_VERSION}"
                export CUDA_GCC="gcc-13"
                export CUDA_GXX="g++-13"
            elif command -v gcc-12 &> /dev/null; then
                echo "✅ Using GCC 12 for CUDA ${CUDA_VERSION}"
                export CUDA_GCC="gcc-12"
                export CUDA_GXX="g++-12"
            else
                echo "ℹ️  Using system default GCC"
                export CUDA_GCC="gcc"
                export CUDA_GXX="g++"
            fi
        elif [ "$CUDA_MAJOR" = "11" ] && command -v gcc-11 &> /dev/null; then
            echo "✅ Using GCC 11 for CUDA ${CUDA_VERSION}"
            export CUDA_GCC="gcc-11"
            export CUDA_GXX="g++-11"
        else
            echo "ℹ️  Using system default GCC"
            export CUDA_GCC="gcc"
            export CUDA_GXX="g++"
        fi

        if [ -n "$CUDA_PATH" ]; then
            GPU_AVAILABLE=true
        fi
    fi
else
    echo "ℹ️  nvidia-smi not found, building CPU-only version"
fi

# Set CUDA paths if available
if [ "$GPU_AVAILABLE" = true ]; then
    export CUDA_PATH
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}
    export CGO_CFLAGS="-I$CUDA_PATH/include"
    export CGO_LDFLAGS="-L$CUDA_PATH/lib64 -lcudart -lcublas -lcublasLt"
    export USE_GPU=true
    export CUDA_ENABLED=true
    export GPU_ENABLED=true
    export CC="${CUDA_GCC}"
    export CXX="${CUDA_GXX}"
    GPU_TAGS="-tags gpu"

    echo "🚀 Building with GPU support (CUDA path: $CUDA_PATH)"

    # Try to build GPU libraries if they exist
    if [ -d "../gpu_fast" ]; then
        echo "🔨 Building GPU libraries..."
        (cd ../gpu_fast && make -j4 >/dev/null 2>&1) || {
            echo "⚠️  GPU library build failed, continuing with CPU mode"
            GPU_AVAILABLE=false
            GPU_TAGS=""
        }
    fi
else
    export USE_GPU=false
    export CUDA_ENABLED=false
    export GPU_ENABLED=false
    GPU_TAGS=""
    echo "ℹ️  Building CPU-only version"
fi

# Build the binary - use bed_cuda.go which has our working implementation
if [ "$GPU_AVAILABLE" = true ] && [ -n "$GPU_TAGS" ]; then
    echo "🔨 Compiling with GPU support..."
    go build $GPU_TAGS -o bed bed_cuda.go || {
        echo "⚠️  GPU build failed, falling back to CPU build"
        go build -o bed bed_cuda.go
    }
else
    echo "🔨 Compiling CPU version..."
    go build -o bed bed_cuda.go
fi

if [ -f "bed" ]; then
    echo "✅ Build successful! Binary: ./bed"
    echo ""
    echo "Usage:"
    echo "  ./bed [-dir <directory>] [-k <num_results>] [--debug] [--gpu] <query>"
    echo ""
    echo "Examples:"
    echo "  ./bed 'websocket connection'                    # Search current directory"
    echo "  ./bed -dir /path/to/code -k 10 'async function' # Search specific directory"
    echo "  ./bed --debug 'error handling'                  # Show debug info"
    if [ "$GPU_AVAILABLE" = true ]; then
        echo ""
        echo "✅ GPU acceleration is available and enabled by default"
    else
        echo ""
        echo "ℹ️  Running in CPU mode (no GPU detected)"
    fi
else
    echo "❌ Build failed"
    exit 1
fi