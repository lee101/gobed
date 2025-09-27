#!/bin/bash
# CUDA Environment Detection and Setup

# Auto-detect CUDA version from nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

    echo "Detected CUDA Version: ${CUDA_VERSION}"

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
            echo "Using GCC 13 for CUDA ${CUDA_VERSION}"
            export CUDA_GCC="gcc-13"
            export CUDA_GXX="g++-13"
        elif command -v gcc-12 &> /dev/null; then
            echo "Using GCC 12 for CUDA ${CUDA_VERSION}"
            export CUDA_GCC="gcc-12"
            export CUDA_GXX="g++-12"
        else
            echo "Using system default GCC"
            export CUDA_GCC="gcc"
            export CUDA_GXX="g++"
        fi
    elif [ "$CUDA_MAJOR" = "11" ] && command -v gcc-11 &> /dev/null; then
        echo "Using GCC 11 for CUDA ${CUDA_VERSION}"
        export CUDA_GCC="gcc-11"
        export CUDA_GXX="g++-11"
    else
        echo "Using system default GCC"
        export CUDA_GCC="gcc"
        export CUDA_GXX="g++"
    fi
else
    CUDA_PATH=""
    export CUDA_GCC="gcc"
    export CUDA_GXX="g++"
fi

# Set CUDA paths if available
if [ -n "$CUDA_PATH" ]; then
    export CUDA_PATH
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}
    export CGO_CFLAGS="-I$CUDA_PATH/include"
    export CGO_LDFLAGS="-L$CUDA_PATH/lib64 -lcudart -lcublas -lcublasLt"
    export USE_GPU=true
    export CUDA_ENABLED=true
    export GPU_ENABLED=true
    export GPU_ARCH="-arch=sm_86"  # RTX 3090 architecture
    echo "🚀 CUDA ${CUDA_VERSION} environment ready"
else
    export USE_GPU=false
    export CUDA_ENABLED=false
    export GPU_ENABLED=false
    echo "⚠️  CUDA not available, CPU mode only"
fi

# Export all variables for use in other scripts
export CUDA_VERSION CUDA_MAJOR CUDA_MINOR