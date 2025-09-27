#!/bin/bash

echo " Checking libtorch installation and GPU support..."
echo "=================================================="

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "✗ No NVIDIA GPU detected or nvidia-smi not installed"
fi

# Check CUDA version
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA installed"
    nvcc --version | grep "release"
    echo ""
else
    echo "✗ CUDA not found in PATH"
fi

# Check for libtorch
echo "Checking for libtorch..."
if [ -d "/usr/local/lib/libtorch" ]; then
    echo "✓ libtorch found at /usr/local/lib/libtorch"
elif [ -d "$HOME/libtorch" ]; then
    echo "✓ libtorch found at $HOME/libtorch"
elif [ -n "$LIBTORCH" ] && [ -d "$LIBTORCH" ]; then
    echo "✓ libtorch found at $LIBTORCH"
else
    echo "✗ libtorch not found in standard locations"
    echo "  Please set LIBTORCH environment variable or install to /usr/local/lib/libtorch"
fi

# Check CGO flags
echo ""
echo "Recommended CGO flags for GPU support:"
echo "--------------------------------------"
cat << 'EOF'
export CGO_CFLAGS="-I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include"
export CGO_CXXFLAGS="-I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include"
export CGO_LDFLAGS="-L${LIBTORCH}/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:$LD_LIBRARY_PATH"
EOF

echo ""
echo "To download libtorch with CUDA support:"
echo "----------------------------------------"
echo "wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
echo "unzip libtorch-*.zip"
echo "sudo mv libtorch /usr/local/lib/ # or export LIBTORCH=\$PWD/libtorch"