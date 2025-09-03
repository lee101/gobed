# GPU Search Build Instructions

## Prerequisites

1. **CUDA Toolkit** (11.x or 12.x)
   - CUDA 12.0-12.8 (recommended for best performance)
   - CUDA 11.x also supported
   
2. **GCC Compiler**
   - CUDA 12.x: GCC 12 or 11 (gcc-12 recommended)
   - CUDA 11.x: GCC 11 or 10
   
3. **LibTorch** (PyTorch C++ library)
   - Compatible with your CUDA version
   
4. **CMake** (>= 3.18)

## Quick Build

```bash
cd gpu_search/cuda_ops
./build.sh
```

The build script will automatically:
- Detect your CUDA installation (prioritizes newer versions)
- Select the appropriate GCC compiler
- Configure GPU architecture targets
- Build the CUDA ops library

## Manual Build

If you need more control:

```bash
cd gpu_search/cuda_ops

# 1. Check your CUDA setup
./detect_cuda.sh --info

# 2. Set environment (example for CUDA 12.0)
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. Set compiler (CUDA version dependent)
export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=g++-12

# 4. Build
mkdir -p build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=$LIBTORCH \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90"
  
cmake --build . -j$(nproc)
```

## CUDA Version Compatibility

| CUDA Version | Supported GCC | GPU Architectures | Notes |
|-------------|--------------|-------------------|-------|
| 12.3-12.8 | GCC 13, 12, 11 | 60-90 (up to Hopper) | Best performance |
| 12.0-12.2 | GCC 12, 11 | 60-90 (up to Hopper) | Recommended |
| 11.8 | GCC 11, 10 | 60-89 (up to Ada) | Good compatibility |
| 11.4-11.7 | GCC 11, 10 | 60-86 (up to Ampere) | Legacy support |

## GPU Architecture Codes

| Code | GPU Generation | Example GPUs |
|------|---------------|--------------|
| 60, 61 | Pascal | GTX 1060, 1070, 1080 |
| 70 | Volta | V100 |
| 75 | Turing | RTX 2060, 2070, 2080 |
| 80, 86 | Ampere | RTX 3060, 3070, 3080, 3090, A100 |
| 89 | Ada Lovelace | RTX 4070, 4080, 4090 |
| 90 | Hopper | H100 |

## Troubleshooting

### Multiple CUDA Versions

If you have multiple CUDA versions installed:

```bash
# List all CUDA installations
ls -la /usr/local/cuda*

# Use specific version
export CUDA_HOME=/usr/local/cuda-12.0
./build.sh
```

### GCC Version Issues

```bash
# Install specific GCC version (Ubuntu/Debian)
sudo apt-get install gcc-12 g++-12

# Install specific GCC version (RHEL/CentOS)
sudo yum install gcc-toolset-12
```

### Build Errors

1. **"nvcc fatal: Unsupported gpu architecture"**
   - Your CUDA version doesn't support the requested GPU architecture
   - Solution: Let the build script auto-detect architectures

2. **"error: #error -- unsupported GNU version"**
   - GCC version incompatible with CUDA
   - Solution: Use the GCC version recommended by detect_cuda.sh

3. **"cannot find -lcudart"**
   - CUDA libraries not in path
   - Solution: Set LD_LIBRARY_PATH as shown above

### Verify Installation

```bash
# Check if library was built
ls -la build/libgobed_ann_ops.so

# Check CUDA info
nvidia-smi

# Run detection script
./detect_cuda.sh --info
```

## Complete Build Process

For a full build including Go client:

```bash
cd gpu_search
./build_and_test.sh
```

This will:
1. Build CUDA ops
2. Generate TorchScript modules
3. Build Go client
4. Run tests

## Environment Variables

Set these in your `.bashrc` or `.bash_profile` for permanent configuration:

```bash
# CUDA (adjust version as needed)
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# LibTorch
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```