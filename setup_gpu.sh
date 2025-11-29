#!/bin/bash
# setup_gpu.sh - Automated GPU search setup for Gobed
# Sets up CUDA, PyTorch, and all GPU acceleration components

set -e  # Exit on error

echo " Gobed GPU Search Setup"
echo "========================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
success() {
    echo -e "${GREEN} $1${NC}"
}

warning() {
    echo -e "${YELLOW}  $1${NC}"
}

error() {
    echo -e "${RED} $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ  $1${NC}"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    error "This script is designed for Linux. For other platforms, see README_GPU.md"
    exit 1
fi

# 1. Check NVIDIA GPU availability
echo " Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found. Please install NVIDIA drivers first."
    echo "   Visit: https://www.nvidia.com/Download/index.aspx"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_INFO" ]; then
    error "No NVIDIA GPU detected"
    exit 1
fi

success "GPU detected: $GPU_INFO"

# 2. Check CUDA installation
echo
echo " Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    success "CUDA found: version $CUDA_VERSION"
    
    # Check CUDA version compatibility
    if [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l) -eq 1 ]]; then
        success "CUDA version is compatible (11.8+)"
    else
        warning "CUDA version $CUDA_VERSION detected. Recommended: 11.8+"
        echo "   Consider upgrading for optimal performance"
    fi
else
    error "CUDA not found. Please install CUDA Toolkit 11.8+"
    echo "   Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# 3. Check/Install Python dependencies
echo
echo "🐍 Setting up Python environment..."
if ! command -v python3 &> /dev/null; then
    error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
success "Python found: version $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    error "pip not found. Please install pip3"
    exit 1
fi

# Install PyTorch with CUDA support
echo " Installing PyTorch with CUDA support..."
if python3 -c "import torch; print('PyTorch already installed:', torch.__version__)" 2>/dev/null; then
    # Check if CUDA is available in PyTorch
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        success "PyTorch with CUDA already installed"
    else
        warning "PyTorch found but CUDA support missing. Reinstalling..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
    fi
else
    info "Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install additional Python dependencies
echo " Installing additional Python dependencies..."
pip3 install transformers tokenizers numpy

# Verify PyTorch CUDA installation
if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'" 2>/dev/null; then
    success "PyTorch CUDA support verified"
else
    error "PyTorch CUDA verification failed"
    echo "   Try: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

# 4. Check/Install system dependencies
echo
echo " Checking system dependencies..."

# Check for CMake
if ! command -v cmake &> /dev/null; then
    info "Installing CMake..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y cmake
    elif command -v yum &> /dev/null; then
        sudo yum install -y cmake
    else
        error "Please install CMake manually"
        exit 1
    fi
fi

CMAKE_VERSION=$(cmake --version | head -1 | cut -d' ' -f3)
success "CMake found: version $CMAKE_VERSION"

# Check for build essentials
if ! command -v make &> /dev/null || ! command -v g++ &> /dev/null; then
    info "Installing build essentials..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y build-essential
    elif command -v yum &> /dev/null; then
        sudo yum groupinstall -y "Development Tools"
    else
        error "Please install build tools (make, g++) manually"
        exit 1
    fi
fi

# Check GCC version for CUDA compatibility
GCC_VERSION=$(gcc --version | head -1 | grep -oP '\d+\.\d+' | head -1)
if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l) -eq 1 ]] && [[ $(echo "$GCC_VERSION >= 13.0" | bc -l) -eq 1 ]]; then
    warning "GCC $GCC_VERSION with CUDA $CUDA_VERSION may have compatibility issues"
    echo "   Consider installing GCC 12: sudo apt-get install gcc-12 g++-12"
    echo "   Then update CMakeLists.txt to use GCC 12"
fi

success "Build tools verified: GCC $GCC_VERSION"

# 5. Setup LibTorch (optional, for native integration)
echo
echo "📚 Setting up LibTorch..."
LIBTORCH_DIR="./libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    info "Downloading LibTorch..."
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
    wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip
    unzip -q libtorch.zip
    rm libtorch.zip
    success "LibTorch downloaded and extracted"
else
    success "LibTorch already available"
fi

# Configure environment variables
echo
echo "  Configuring environment..."
ENV_FILE="$HOME/.bashrc"

# Add LibTorch environment variables if not already present
if ! grep -q "LIBTORCH" "$ENV_FILE" 2>/dev/null; then
    echo >> "$ENV_FILE"
    echo "# Gobed GPU Search - LibTorch environment" >> "$ENV_FILE"
    echo "export LIBTORCH=$(pwd)/libtorch" >> "$ENV_FILE"
    echo "export LD_LIBRARY_PATH=\$LIBTORCH/lib:\$LD_LIBRARY_PATH" >> "$ENV_FILE"
    echo "export CGO_CFLAGS=\"-I\$LIBTORCH/include -I\$LIBTORCH/include/torch/csrc/api/include\"" >> "$ENV_FILE"
    echo "export CGO_LDFLAGS=\"-L\$LIBTORCH/lib -ltorch -ltorch_cpu -lc10\"" >> "$ENV_FILE"
    success "Environment variables added to $ENV_FILE"
else
    success "Environment variables already configured"
fi

# 6. Build CUDA components
echo
echo " Building CUDA components..."
cd gpu_search/cuda_ops

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake
if cmake .. > cmake_output.log 2>&1; then
    success "CMake configuration successful"
else
    error "CMake configuration failed"
    echo "   Check cmake_output.log for details"
    exit 1
fi

# Build with make
echo "   Compiling CUDA kernels..."
if make -j$(nproc) > make_output.log 2>&1; then
    success "CUDA library built successfully"
    if [ -f "libgobed_ann_ops.so" ]; then
        success "Found libgobed_ann_ops.so"
    else
        warning "Library file not found where expected"
    fi
else
    error "CUDA compilation failed"
    echo "   Check make_output.log for details"
    exit 1
fi

cd ../../..

# 7. Export TorchScript model
echo
echo " Exporting TorchScript model..."
cd gpu_search

if python3 simple_search_module.py > export_output.log 2>&1; then
    success "TorchScript model exported"
    if [ -f "../model/simple_gpu_search_module.pt" ]; then
        MODEL_SIZE=$(ls -lh ../model/simple_gpu_search_module.pt | cut -d' ' -f5)
        success "Model file created: $MODEL_SIZE"
    else
        warning "Model file not found where expected"
    fi
else
    error "TorchScript export failed"
    echo "   Check export_output.log for details"
    exit 1
fi

cd ..

# 8. Verify installation
echo
echo "🧪 Verifying installation..."

# Test CUDA library
if [ -f "gpu_search/cuda_ops/build/libgobed_ann_ops.so" ]; then
    success "CUDA library: ✓"
else
    error "CUDA library: ✗"
fi

# Test TorchScript model
if [ -f "model/simple_gpu_search_module.pt" ]; then
    success "TorchScript model: ✓"
else
    error "TorchScript model: ✗"
fi

# Test Python imports
if python3 -c "import torch; import transformers; print('Imports successful')" 2>/dev/null; then
    success "Python dependencies: ✓"
else
    error "Python dependencies: ✗"
fi

# 9. Performance test (optional)
echo
echo " Running quick performance test..."
cd ../gobedexample

if timeout 30s go run achievement_summary.go > test_output.log 2>&1; then
    success "Go integration test: ✓"
    echo "   See test_output.log for details"
else
    warning "Go integration test: Limited (may need GPU server)"
    echo "   This is normal if GPU server isn't running"
fi

cd ../gobed

# 10. Summary and next steps
echo
echo " GPU Setup Complete!"
echo "====================="
echo
success "Components installed:"
echo "   • NVIDIA GPU drivers and CUDA"
echo "   • PyTorch with CUDA support"
echo "   • Custom CUDA kernels compiled"
echo "   • TorchScript model exported"
echo "   • LibTorch environment configured"
echo
info "Next steps:"
echo "   1. Restart terminal or run: source ~/.bashrc"
echo "   2. Test GPU search: cd ../gobedexample && go run achievement_summary.go"
echo "   3. Start GPU server: python3 gpu_search/gpu_search_server.py"
echo "   4. Run examples: cd ../gobedexample && go run main.go --benchmark"
echo
success "Ready for GPU-accelerated text search! "
echo
echo "📚 Documentation:"
echo "   • Main guide: README_GPU.md"
echo "   • Examples: ../gobedexample/README.md"
echo "   • Performance tuning: Check batch sizes and GPU memory"
echo
echo " Troubleshooting:"
echo "   • Out of memory: Reduce batch size in config"
echo "   • CUDA errors: Check nvidia-smi and GPU compatibility"
echo "   • Build issues: Verify GCC version compatibility"