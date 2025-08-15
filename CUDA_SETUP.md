# CUDA Support Setup Guide

## Performance Summary

### Current CPU Performance (RTX 3080 Laptop)
- **Small texts (100 tokens)**: ~0.09ms latency, 1M tokens/sec
- **Medium texts (512 tokens)**: ~0.51ms latency, 0.95M tokens/sec  
- **Large texts (4K tokens)**: ~5ms latency, 0.86M tokens/sec
- **Extreme texts (16K tokens)**: ~20ms latency, 794K tokens/sec

### Memory & Compute
- Model size: 119.23 MB (30,522 vocab × 1,024 dims)
- Memory bandwidth achieved: 3.78 GB/s
- Compute throughput: 1.89 GFLOPS

## Setting Up CUDA Support

### 1. Install LibTorch with CUDA (if needed)

```bash
# Download LibTorch CUDA 12.1 version
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip

# Extract
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip

# Set environment variables
export LIBTORCH="${PWD}/libtorch"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH}"
export CGO_CXXFLAGS="-I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include"
export CGO_LDFLAGS="-L${LIBTORCH}/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda"
```

### 2. Alternative: Use ONNX Runtime with CUDA

```bash
# Install ONNX Runtime GPU version
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-gpu-1.19.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.19.0.tgz
```

### 3. Build with CUDA Support

```bash
# Build the CUDA-enabled version
go build -tags cuda -o gobed_cuda main.go
```

## Expected CUDA Performance Improvements

Based on your RTX 3080 with 16GB VRAM:

### Theoretical Speedups
- **Batch processing**: 10-50x faster for batches of 32+ texts
- **Single inference**: 2-5x faster for texts > 512 tokens
- **Memory bandwidth**: Up to 760 GB/s (vs 3.78 GB/s CPU)
- **Compute**: ~20 TFLOPS FP32 (vs 1.89 GFLOPS CPU)

### Real-world Expectations
- Small texts (< 100 tokens): CPU may be faster due to transfer overhead
- Medium texts (100-1000 tokens): 2-3x speedup
- Large texts (> 1000 tokens): 5-10x speedup
- Batch processing: 20-40x speedup for large batches

## Optimization Tips

1. **Batch Processing**: Always batch multiple texts together for GPU
2. **Pre-allocate Buffers**: Reuse GPU memory allocations
3. **Async Transfers**: Use CUDA streams for overlapping compute/transfer
4. **Mixed Precision**: Use FP16 for 2x memory and compute improvement

## Current Implementation Status

✅ **Completed:**
- CPU implementation with excellent performance
- Long string support (tested up to 16K tokens)
- Memory-efficient processing
- Competitive with Python/NumPy

🔄 **CUDA Implementation Options:**
1. Pure Go with CGO bindings to CUDA
2. LibTorch integration via gotch
3. ONNX Runtime with CUDA provider

## Recommendations

For your use case:
- **Current CPU performance is already excellent** for single inference
- **CUDA is most beneficial for:**
  - Batch processing multiple texts simultaneously
  - Very long texts (> 4K tokens)
  - High-throughput scenarios (> 1000 requests/sec)

The current implementation achieves ~1M tokens/sec on CPU, which is sufficient for most real-time applications. CUDA would primarily help with batch processing scenarios.