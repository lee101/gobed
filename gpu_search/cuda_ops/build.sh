#!/bin/bash
set -e

echo "Building GPU ANN ops with CUDA..."
echo "================================="

# Get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Detect CUDA installation
echo "Detecting CUDA installation..."
source "$SCRIPT_DIR/detect_cuda.sh"

CUDA_PATH=$(detect_cuda)
if [ $? -ne 0 ]; then
    echo "ERROR: No CUDA installation found!"
    echo "Please install CUDA or set CUDA_HOME environment variable"
    exit 1
fi

CUDA_VERSION=$(get_cuda_version "$CUDA_PATH")
CUDA_ARCH_FLAGS=$(get_cuda_arch_flags "$CUDA_VERSION")
GCC_COMPILER=$(get_compatible_gcc "$CUDA_VERSION")

echo "✓ Found CUDA $CUDA_VERSION at $CUDA_PATH"
echo "✓ Using compiler: $GCC_COMPILER"
echo "✓ Target GPU architectures: $CUDA_ARCH_FLAGS"

# Verify GCC is available
if ! command -v $GCC_COMPILER &> /dev/null; then
    echo "WARNING: $GCC_COMPILER not found, trying default gcc"
    GCC_COMPILER=gcc
fi

# Get GCC version for CXX
CXX_COMPILER=${GCC_COMPILER/gcc/g++}

# Verify CXX compiler is available
if ! command -v $CXX_COMPILER &> /dev/null; then
    echo "WARNING: $CXX_COMPILER not found, trying default g++"
    CXX_COMPILER=g++
fi

echo "✓ Using C++ compiler: $CXX_COMPILER"

# LibTorch-free build - no need for LibTorch environment

# Set up environment
export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

# Set compiler environment
export CC="$GCC_COMPILER"
export CXX="$CXX_COMPILER"
export CUDAHOSTCXX="$CXX_COMPILER"

echo ""
echo "Environment Configuration:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CC: $CC"
echo "  CXX: $CXX"
echo "  CUDAHOSTCXX: $CUDAHOSTCXX"

# Verify nvcc is accessible
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH after setting CUDA_HOME"
    exit 1
fi

echo "  NVCC: $(which nvcc)"
echo ""

# Clean previous build
echo "Cleaning previous build..."
rm -rf build

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake_args=(
    ".."
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_C_COMPILER=$GCC_COMPILER"
    "-DCMAKE_CXX_COMPILER=$CXX_COMPILER"
    "-DCMAKE_CUDA_HOST_COMPILER=$CXX_COMPILER"
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH_FLAGS"
)

# Add CUDA-version specific flags
if [[ "$CUDA_VERSION" == "12."* ]] && [[ "$GCC_COMPILER" != *"gcc-12"* ]]; then
    echo "NOTE: CUDA 12.x with non-GCC-12 compiler may need additional flags"
fi

cmake "${cmake_args[@]}"

# Build
echo ""
echo "Building library..."
cmake --build . -j$(nproc) --verbose

echo ""
echo " Successfully built libgobed_ann_ops.so (LibTorch-free)"
echo "Library location: $(pwd)/libgobed_ann_ops.so"

# Copy to a standard location
echo ""
echo "Installing library..."
if [ -w /usr/local/lib ]; then
    sudo cp libgobed_ann_ops.so /usr/local/lib/ 2>/dev/null && \
        echo " Installed to /usr/local/lib/" || \
        echo " Could not install to /usr/local/lib/ (permission denied)"
fi

# Always copy to parent directory for local use
cp libgobed_ann_ops.so ../../ 
echo " Copied to parent directory for local use"

echo ""
echo "Build Summary:"
echo "=============="
echo "  CUDA Version: $CUDA_VERSION"
echo "  Compiler: $GCC_COMPILER"
echo "  Library: $(pwd)/libgobed_ann_ops.so"
echo ""
echo "To use this library:"
echo "  1. Add to LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$(pwd):\$LD_LIBRARY_PATH"
echo "  2. Or copy to your project directory"
echo ""