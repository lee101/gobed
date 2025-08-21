# GPU Bulk Indexing Performance Expectations

## Hardware Requirements
- NVIDIA GPU with CUDA 11.8+ support
- Minimum 8GB VRAM for batch sizes up to 64
- 16GB+ VRAM recommended for batch sizes 128+

## Expected Performance Metrics

### GPU vs CPU Speedup
Based on typical transformer models (BERT-like):

| Document Count | Batch Size | GPU (docs/sec) | CPU (docs/sec) | Speedup |
|---------------|------------|----------------|----------------|---------|
| 100           | 8          | 80-120         | 10-15          | 6-8x    |
| 500           | 32         | 300-400        | 10-15          | 20-30x  |
| 1000          | 64         | 500-700        | 10-15          | 35-50x  |
| 5000          | 128        | 800-1200       | 10-15          | 60-80x  |

### Batch Size Impact
Optimal batch sizes depend on GPU memory:

- **RTX 3060 (12GB)**: Batch size 32-64
- **RTX 3080 (16GB)**: Batch size 64-128  
- **RTX 4090 (24GB)**: Batch size 128-256
- **A100 (40GB)**: Batch size 256-512

### Memory Usage
Approximate VRAM usage per batch:

| Batch Size | Model Size | VRAM Usage |
|------------|------------|------------|
| 32         | 110M params| ~2GB       |
| 64         | 110M params| ~4GB       |
| 128        | 110M params| ~8GB       |
| 256        | 110M params| ~16GB      |

### Throughput Expectations

For a typical BERT-base model on modern GPUs:

**RTX 3080 (16GB)**
- Small batches (1-8): 50-100 docs/sec
- Medium batches (32-64): 300-500 docs/sec
- Large batches (128): 600-800 docs/sec

**RTX 4090 (24GB)**
- Small batches (1-8): 80-150 docs/sec
- Medium batches (32-64): 500-800 docs/sec
- Large batches (128-256): 1000-1500 docs/sec

**A100 (40GB)**
- Small batches (1-8): 100-200 docs/sec
- Medium batches (32-64): 800-1200 docs/sec
- Large batches (256-512): 2000-3000 docs/sec

## Optimization Tips

1. **Batch Size**: Larger batches = better GPU utilization
   - Start with batch size 32
   - Increase until OOM, then back off by 20%

2. **Sequence Length**: Shorter sequences = faster processing
   - Truncate to 128 tokens if possible
   - Use 512 only when necessary

3. **Mixed Precision**: Use FP16 for 2x speedup
   - Requires GPU with Tensor Cores (RTX 20xx+)
   - Minimal accuracy loss for embeddings

4. **Memory Pinning**: Pre-allocate pinned memory for faster transfers
   - Reduces CPU->GPU transfer overhead
   - Especially important for small batches

## Benchmarking Commands

Run the full benchmark suite:
```bash
go test -bench=BenchmarkBulkGPUIndexing -benchtime=10s
```

Compare GPU vs CPU:
```bash
go test -bench=BenchmarkGPUvsCPU -benchtime=10s
```

Test memory usage:
```bash
go test -run=TestGPUMemoryUsage -v
```

Verify GPU/CPU parity:
```bash
go test -run=TestGPUCPUEmbeddingParity -v
```

## Monitoring GPU Usage

During indexing, monitor with:
```bash
nvidia-smi -l 1  # Update every second
```

Or use the built-in monitoring:
```go
progressChan, err := idx.AddDocumentsWithMonitoring(docs)
for progress := range progressChan {
    fmt.Printf("GPU: %.1f%%, Memory: %.2f GB, Speed: %.0f docs/sec\n",
        progress.GPUUtilization,
        float64(progress.GPUMemoryUsed)/(1024*1024*1024),
        progress.DocsPerSecond)
}
```