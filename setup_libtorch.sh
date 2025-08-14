#!/bin/bash
set -e

echo "=================================================================================="
echo "🔥 LibTorch Setup (Optional - for future GPU acceleration)"
echo "=================================================================================="
echo ""

# Detect system and CUDA
CUDA_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        CUDA_AVAILABLE=true
        CUDA_VERSION=$(nvidia-smi | grep -Po 'CUDA Version: \K[0-9.]+' | head -1)
        echo "✅ CUDA detected: Version $CUDA_VERSION"
    fi
else
    echo "ℹ️  No CUDA detected - will install CPU version"
fi
echo ""

# Create directory
mkdir -p libtorch

# Select appropriate LibTorch version
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "Choose LibTorch version:"
    echo "1) CPU only (smaller, ~200MB)"
    echo "2) CUDA 11.8 (GPU acceleration, ~2GB)"
    echo "3) CUDA 12.1 (GPU acceleration, ~2GB)"
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        2)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip"
            echo "📥 Downloading LibTorch with CUDA 11.8 support..."
            ;;
        3)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
            echo "📥 Downloading LibTorch with CUDA 12.1 support..."
            ;;
        *)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
            echo "📥 Downloading LibTorch CPU version..."
            ;;
    esac
else
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
    echo "📥 Downloading LibTorch CPU version..."
fi

LIBTORCH_ZIP="libtorch.zip"

# Download if not exists
if [ ! -f "libtorch/lib/libtorch.so" ]; then
    wget -q --show-progress -O "$LIBTORCH_ZIP" "$LIBTORCH_URL"
    echo "📦 Extracting LibTorch..."
    unzip -q -o "$LIBTORCH_ZIP" -d .
    rm "$LIBTORCH_ZIP"
    echo "✅ LibTorch installed"
else
    echo "✅ LibTorch already installed"
fi

echo ""
echo "=================================================================================="
echo "✨ LibTorch Setup Complete!"
echo "=================================================================================="
echo ""
echo "To use LibTorch in your Go code, set these environment variables:"
echo ""
echo "export LIBTORCH=$PWD/libtorch"
echo "export LD_LIBRARY_PATH=\$LIBTORCH/lib:\$LD_LIBRARY_PATH"
echo "export CGO_CFLAGS=\"-I\$LIBTORCH/include -I\$LIBTORCH/include/torch/csrc/api/include\""
echo "export CGO_LDFLAGS=\"-L\$LIBTORCH/lib -ltorch -ltorch_cpu -lc10\""
echo ""
echo "Note: The current implementation uses pure Go and doesn't require LibTorch."
echo "LibTorch is included for future GPU acceleration support."
echo ""