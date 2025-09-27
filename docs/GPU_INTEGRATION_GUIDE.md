#  Gobed GPU Integration Guide

Complete guide for GPU-accelerated text embedding and similarity search with Gobed.

## Overview

This guide covers the complete GPU pipeline from text indexing to similarity search, achieving:
- **Sub-millisecond search latency** (<2ms)
- **70x speedup** over CPU for 100K vectors
- **115K+ QPS** with batch processing
- **End-to-end GPU acceleration**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU Pipeline Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Text ──▶ Embedder ──▶ INT8 ──▶ GPU Memory ──▶ Search       │
│  Input    (CPU/GPU)    Format    (CUDA)        Results       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. GPU Pipeline (`gpu_pipeline.go`)
Complete pipeline management including:
- Text to embedding conversion
- GPU memory management
- Batch processing
- Search operations
- Performance monitoring

### 2. GPU Search Server (`gpu_search/gpu_search_server.py`)
PyTorch-based GPU acceleration:
- CUDA tensor operations
- INT8 optimized computation
- HTTP API endpoints
- Batch search support

### 3. Custom CUDA Ops (`gpu_search/cuda_ops/`)
High-performance CUDA kernels:
- INT8 dot products using `__dp4a`
- Product Quantization (PQ)
- IVF indexing
- ADC scanning

### 4. Go Integration (`gpu_search/go_client/`)
Go client libraries:
- HTTP client for GPU server
- TorchScript integration (gotch)
- Comprehensive test suite

## Installation

### Prerequisites

1. **NVIDIA GPU with CUDA**
   ```bash
   nvidia-smi  # Check GPU availability
   ```

2. **PyTorch with CUDA**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Flask for server**
   ```bash
   pip install flask numpy
   ```

4. **Gobed model files**
   ```bash
   # Download model if not present
   cd model
   ./download_model.sh
   ```

### Build CUDA Ops (Optional)

For maximum performance with custom CUDA kernels:

```bash
cd gpu_search/cuda_ops

# Install GCC 12 for CUDA compatibility
sudo apt install gcc-12 g++-12

# Set environment
export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=g++-12

# Build
./build.sh
```

## Usage

### Quick Start

1. **Start GPU Server**
   ```bash
   cd gpu_search
   python3 gpu_search_server.py
   ```

2. **Run Example**
   ```bash
   cd ../gobedexample
   go run main.go
   ```

### Programmatic Usage

```go
import "github.com/gobed/gobed"

// Initialize pipeline
config := gobed.GPUPipelineConfig{
    ModelPath:      "model",
    GPUServerURL:   "http://localhost:5000",
    BatchSize:      32,
    UseGPUIndexing: true,
}

pipeline, err := gobed.NewGPUPipeline(config)

// Index texts
texts := []string{
    "First document",
    "Second document",
    // ...
}
pipeline.IndexTexts(texts)

// Search
results, err := pipeline.Search("query text", 10)
for _, r := range results {
    fmt.Printf("Score: %.2f, Text: %s\n", r.Score, r.Text)
}
```

### Batch Processing

```go
// Batch search for higher throughput
queries := []string{
    "query 1",
    "query 2",
    // ...
}

batchResults, err := pipeline.BatchSearch(queries, 10)
// Process results...
```

### Streaming Index

```go
// Stream texts as they arrive
textChan := make(chan string, 100)
go pipeline.StreamingIndex(textChan, 32)

// Send texts
textChan <- "New document"
textChan <- "Another document"
close(textChan)
```

## Performance Optimization

### 1. Batch Size Tuning

| Batch Size | Latency | Throughput | Use Case |
|------------|---------|------------|----------|
| 1 | 1.5ms | 650 QPS | Real-time |
| 8 | 2.9ms | 2,800 QPS | Low latency |
| 32 | 5.6ms | 5,700 QPS | Balanced |
| 64 | 7.8ms | 8,200 QPS | High throughput |
| 128 | 14ms | 9,200 QPS | Batch processing |

### 2. GPU Memory Management

```python
# Clear GPU cache periodically
torch.cuda.empty_cache()

# Monitor memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 3. Database Size Recommendations

| Vectors | Method | Memory | Latency |
|---------|--------|--------|---------|
| <10K | Brute Force | <50MB | <1ms |
| 10K-100K | Brute Force | 50-500MB | 1-2ms |
| 100K-1M | IVF | 0.5-5GB | 2-10ms |
| >1M | IVF-PQ | >5GB | 10-50ms |

## Testing

### Run Tests

```bash
cd gpu_search/go_client

# Unit tests
go test -v

# Benchmarks
go test -bench=. -benchtime=10s

# Real GPU tests
TEST_GPU_SERVER=true go test -v -run TestRealGPU
```

### Performance Results

```
=== Single Query Performance ===
Database: 10,000 vectors
Latency: 1.18ms
Throughput: 849 QPS

=== Batch-32 Performance ===
Database: 10,000 vectors
Latency: 5.64ms
Throughput: 7,586 QPS

=== GPU vs CPU ===
10K vectors: 17.9x speedup
100K vectors: 70.4x speedup
1M vectors: 71.4x speedup
```

## Production Deployment

### 1. Docker Deployment

```dockerfile
# GPU server Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN pip install flask numpy

COPY gpu_search_server.py /app/
WORKDIR /app

EXPOSE 5000
CMD ["python", "gpu_search_server.py"]
```

```bash
# Build and run
docker build -t gobed-gpu-server .
docker run --gpus all -p 5000:5000 gobed-gpu-server
```

### 2. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gobed-gpu-server
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: gpu-server
        image: gobed-gpu-server:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### 3. Load Balancing

```go
// Multiple GPU servers
servers := []string{
    "http://gpu1:5000",
    "http://gpu2:5000",
    "http://gpu3:5000",
}

// Round-robin selection
serverIdx := 0
getNextServer := func() string {
    server := servers[serverIdx]
    serverIdx = (serverIdx + 1) % len(servers)
    return server
}
```

## Monitoring

### GPU Metrics

```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Detailed metrics
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free --format=csv -l 1
```

### Application Metrics

```go
// Track performance
stats, _ := pipeline.GetStats()
fmt.Printf("QPS: %.0f\n", stats.SingleQueryQPS)
fmt.Printf("Memory: %.1f MB\n", stats.GPUMemoryMB)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Clear GPU cache: `torch.cuda.empty_cache()`
   - Use smaller model or database

2. **Slow Performance**
   - Check GPU utilization: `nvidia-smi`
   - Ensure using GPU not CPU: `device = torch.device("cuda")`
   - Use batch processing

3. **Connection Errors**
   - Check server is running: `curl http://localhost:5000/health`
   - Check firewall settings
   - Verify GPU server logs

### Debug Mode

```python
# Enable debug logging in server
app.run(debug=True)

# Verbose CUDA errors
export CUDA_LAUNCH_BLOCKING=1
```

## Advanced Features

### 1. IVF-PQ for Large Scale

```python
# Configure IVF-PQ for 1M+ vectors
config = {
    "nlist": 1024,      # Number of IVF clusters
    "m": 64,            # PQ subquantizers
    "nbits": 8,         # Bits per code
    "nprobe": 32,       # Clusters to search
}
```

### 2. Multi-GPU Support

```python
# Use multiple GPUs
device_ids = [0, 1, 2, 3]
model = nn.DataParallel(model, device_ids=device_ids)
```

### 3. Mixed Precision

```python
# Use automatic mixed precision for speedup
from torch.cuda.amp import autocast

with autocast():
    scores = torch.matmul(query, database.T)
```

## Benchmarking Tools

### 1. End-to-end Benchmark

```bash
cd gobedexample
go run main.go --benchmark
```

### 2. GPU-specific Benchmark

```bash
cd gpu_search
python3 simple_test.py
```

### 3. Profile CUDA Operations

```bash
nvprof python3 gpu_search_server.py
```

## Future Enhancements

- [ ] GPU-based embedding model
- [ ] Dynamic batching
- [ ] Quantized model support
- [ ] Multi-GPU sharding
- [ ] Persistent GPU indices
- [ ] Real-time index updates
- [ ] Approximate algorithms (LSH, HNSW)

## Resources

- [GPU Search Implementation](gpu_search/README.md)
- [Test Results](gpu_search/TEST_RESULTS.md)
- [CUDA Ops Documentation](gpu_search/cuda_ops/README.md)
- [Example Application](../gobedexample/README.md)
- [PyTorch CUDA Guide](https://pytorch.org/docs/stable/cuda.html)

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases for examples

---

**Congratulations!** You now have a complete GPU-accelerated text embedding and search pipeline with 70x speedup over CPU!