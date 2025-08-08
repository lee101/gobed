#!/bin/bash

# Setup script for LibTorch with Go
echo "🔧 Setting up LibTorch for Go development..."

# Check if we're on CPU or have CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 CUDA detected, will install CUDA version of LibTorch"
    CUDA_AVAILABLE=true
else
    echo "💻 No CUDA detected, installing CPU-only LibTorch"
    CUDA_AVAILABLE=false
fi

# Create libtorch directory
mkdir -p libtorch
cd libtorch

# Download LibTorch based on system
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "📥 Downloading LibTorch with CUDA support..."
    wget -q --show-progress https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
    unzip -q libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
else
    echo "📥 Downloading CPU-only LibTorch..."
    wget -q --show-progress https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip  
    unzip -q libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
fi

# Set environment variables
export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

echo "✅ LibTorch installed at: $LIBTORCH"
echo "📝 Environment variables set:"
echo "   LIBTORCH=$LIBTORCH"
echo "   LD_LIBRARY_PATH includes LibTorch libs"

# Create environment setup script
cat > ../setup_env.sh << 'EOF'
#!/bin/bash
# Source this file to set up LibTorch environment
export LIBTORCH=$(pwd)/libtorch/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export CGO_CPPFLAGS="-I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include"
export CGO_LDFLAGS="-L${LIBTORCH}/lib -ltorch -ltorch_cpu"
echo "🔧 LibTorch environment configured"
echo "   LIBTORCH=$LIBTORCH"
EOF

chmod +x ../setup_env.sh

cd ..

echo ""
echo "🎉 LibTorch setup complete!"
echo "💡 To use LibTorch, run: source setup_env.sh"
echo "🧪 Then you can build Go programs with LibTorch support"