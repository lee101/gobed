# 🚀 GPU Indexing Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Configuration](#advanced-configuration)
5. [Performance Tuning](#performance-tuning)
6. [Production Deployment](#production-deployment)
7. [API Reference](#api-reference)

## Quick Start

```go
import "github.com/lee101/gobed/gpu"

// Create GPU-accelerated indexer with INT8 quantization
indexer := gpu.NewIndexer(gpu.Config{
    Precision:  gpu.INT8,
    BatchSize:  5000,
    Device:     gpu.AutoDetect(), // Automatically use best available GPU
})

// Add vectors (automatically batched and quantized)
indexer.AddVectors(vectors)

// Search with automatic GPU acceleration
results := indexer.Search(query, k)
```

## Installation

### Prerequisites

1. **NVIDIA GPU** with CUDA support (optional but recommended)
2. **CUDA Toolkit** (for GPU acceleration)
3. **LibTorch** (for tensor operations)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/lee101/gobed
cd gobed

# 2. Run setup script (automatically detects CUDA)
./setup_libtorch.sh

# 3. Set environment variables
export LIBTORCH=$PWD/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
export CGO_CFLAGS="-I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include"
export CGO_LDFLAGS="-L$LIBTORCH/lib -ltorch -ltorch_cpu -lc10 -ltorch_cuda"

# 4. Install Go package
go get github.com/lee101/gobed/gpu
```

### Verify Installation

```bash
# Run GPU detection test
go run cmd/gpu_simulation/main.go

# Check CUDA availability
nvidia-smi
```

## Basic Usage

### 1. Simple Indexing

```go
package main

import (
    "fmt"
    "github.com/lee101/gobed/gpu"
)

func main() {
    // Initialize GPU indexer
    indexer := gpu.NewIndexer(gpu.DefaultConfig())
    
    // Add vectors (dimension 384)
    vectors := [][]float32{
        {0.1, 0.2, 0.3, ...}, // 384 dimensions
        {0.4, 0.5, 0.6, ...},
        // ... more vectors
    }
    
    err := indexer.AddVectors(vectors)
    if err != nil {
        panic(err)
    }
    
    // Search for similar vectors
    query := []float32{0.2, 0.3, 0.4, ...} // 384 dimensions
    k := 10 // Find top 10 similar vectors
    
    indices, scores := indexer.Search(query, k)
    
    for i := 0; i < len(indices); i++ {
        fmt.Printf("Rank %d: Vector %d (score: %.4f)\n", 
            i+1, indices[i], scores[i])
    }
}
```

### 2. Token Embedding Pipeline

```go
// Create embedding model
embedModel := gpu.NewEmbeddingModel(gpu.EmbedConfig{
    VocabSize:  250000,
    EmbedDim:   384,
    MaxSeqLen:  512,
    UseINT8:    true, // Enable INT8 quantization
})

// Convert tokens to embeddings
tokens := []int64{101, 2023, 2003, 1037, 3231, 102} // Token IDs
embedding := embedModel.Embed(tokens)

// Batch processing for multiple sequences
tokenBatch := [][]int64{
    {101, 2023, 2003, ...},
    {101, 2054, 2003, ...},
    // ... more sequences
}
embeddings := embedModel.BatchEmbed(tokenBatch)
```

### 3. Document Indexing

```go
type Document struct {
    ID      int
    Text    string
    Tokens  []int64
}

// Create document indexer
docIndexer := gpu.NewDocumentIndexer(gpu.DocConfig{
    EmbedModel: embedModel,
    IndexSize:  1000000,
    Precision:  gpu.INT8,
})

// Add documents
documents := []Document{
    {ID: 1, Text: "Machine learning is fascinating"},
    {ID: 2, Text: "GPU acceleration improves performance"},
    // ... more documents
}

// Tokenize and index documents
for _, doc := range documents {
    doc.Tokens = tokenizer.Encode(doc.Text)
    docIndexer.AddDocument(doc)
}

// Search documents
query := "deep learning GPU optimization"
results := docIndexer.SearchDocuments(query, 10)
```

## Advanced Configuration

### Precision Modes

```go
// FP32 - Highest accuracy, most memory
indexerFP32 := gpu.NewIndexer(gpu.Config{
    Precision: gpu.FP32,
})

// FP16 - Good balance of speed and accuracy
indexerFP16 := gpu.NewIndexer(gpu.Config{
    Precision: gpu.FP16,
})

// INT8 - Best speed and memory efficiency
indexerINT8 := gpu.NewIndexer(gpu.Config{
    Precision: gpu.INT8,
    Quantization: gpu.QuantConfig{
        Mode:      gpu.Symmetric,  // or Asymmetric
        PerVector: true,           // Per-vector scaling
    },
})
```

### Search Algorithms

```go
// Brute-force (exact search)
indexer := gpu.NewIndexer(gpu.Config{
    SearchType: gpu.BruteForce,
})

// IVF (approximate search for large datasets)
indexer := gpu.NewIndexer(gpu.Config{
    SearchType:   gpu.IVF,
    NumCentroids: 1000,  // sqrt(num_vectors) is good default
    NProbe:       10,    // Number of clusters to search
})

// HNSW (hierarchical navigable small world)
indexer := gpu.NewIndexer(gpu.Config{
    SearchType: gpu.HNSW,
    M:          16,  // Number of connections
    EfConstruct: 200,  // Construction parameter
})
```

### Batch Processing

```go
// Configure batch sizes for optimal GPU utilization
config := gpu.Config{
    EmbedBatchSize:  100,   // Embedding batch size
    IndexBatchSize:  5000,  // Indexing batch size
    SearchBatchSize: 1000,  // Search batch size
}

// Batch search
queries := [][]float32{query1, query2, query3, ...}
batchResults := indexer.BatchSearch(queries, k)
```

### Multi-GPU Support

```go
// Use specific GPU
indexer := gpu.NewIndexer(gpu.Config{
    Device: gpu.Device(0), // Use GPU 0
})

// Distribute across multiple GPUs
multiGPU := gpu.NewMultiGPUIndexer(gpu.MultiConfig{
    Devices:    []gpu.Device{0, 1, 2, 3},
    Sharding:   gpu.RoundRobin,
    Replication: false,
})
```

## Performance Tuning

### Memory Management

```go
// Pre-allocate memory for better performance
indexer := gpu.NewIndexer(gpu.Config{
    MaxVectors:  1000000,
    PreAllocate: true,
    PinnedMemory: true, // Use pinned memory for faster transfers
})

// Monitor memory usage
stats := indexer.GetMemoryStats()
fmt.Printf("GPU Memory: %.2f MB used, %.2f MB free\n", 
    stats.UsedMB, stats.FreeMB)
```

### Optimization Tips

```go
// 1. Optimal batch sizes based on GPU memory
batchSize := gpu.CalculateOptimalBatchSize(
    gpuMemoryGB: 24,
    vectorDim:   384,
    precision:   gpu.INT8,
)

// 2. Stream processing for continuous data
stream := gpu.NewStreamProcessor(gpu.StreamConfig{
    BufferSize:  10000,
    NumStreams:  4,
    AsyncUpload: true,
})

// 3. Cache frequently accessed vectors
indexer.EnableCache(gpu.CacheConfig{
    MaxSize:      10000,
    EvictionMode: gpu.LRU,
})
```

### Profiling

```go
// Enable profiling
indexer.EnableProfiling()

// Run operations
indexer.AddVectors(vectors)
results := indexer.Search(query, k)

// Get profiling results
profile := indexer.GetProfile()
fmt.Printf("Embedding: %.2fms\n", profile.EmbedTime)
fmt.Printf("Quantization: %.2fms\n", profile.QuantizeTime)
fmt.Printf("Search: %.2fms\n", profile.SearchTime)
fmt.Printf("Total: %.2fms\n", profile.TotalTime)
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential

# Install Go
RUN wget https://go.dev/dl/go1.21.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin

# Setup LibTorch
WORKDIR /app
COPY setup_libtorch.sh .
RUN ./setup_libtorch.sh

# Copy application
COPY . .

# Build
RUN go build -o gpu-indexer cmd/main.go

# Run
CMD ["./gpu-indexer"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-indexer
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: indexer
        image: gpu-indexer:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU
          requests:
            memory: "32Gi"
            cpu: "8"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: INDEX_PRECISION
          value: "INT8"
        - name: BATCH_SIZE
          value: "5000"
```

### Health Checks

```go
// Implement health check endpoint
http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
    health := indexer.HealthCheck()
    
    if health.Healthy {
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(map[string]interface{}{
            "status": "healthy",
            "gpu_available": health.GPUAvailable,
            "memory_usage": health.MemoryUsage,
            "indexed_vectors": health.NumVectors,
        })
    } else {
        w.WriteHeader(http.StatusServiceUnavailable)
        json.NewEncoder(w).Encode(map[string]string{
            "status": "unhealthy",
            "error": health.Error,
        })
    }
})
```

## API Reference

### Core Types

```go
// Config - Main configuration
type Config struct {
    Precision      Precision     // FP32, FP16, or INT8
    Device         Device        // GPU device to use
    MaxVectors     int          // Maximum vectors to index
    BatchSize      int          // Batch size for operations
    SearchType     SearchType   // BruteForce, IVF, or HNSW
    PreAllocate    bool         // Pre-allocate GPU memory
    UseINT8        bool         // Enable INT8 quantization
}

// Precision modes
const (
    FP32 Precision = iota
    FP16
    INT8
)

// Search types
const (
    BruteForce SearchType = iota
    IVF
    HNSW
)
```

### Main Functions

```go
// NewIndexer creates a new GPU indexer
func NewIndexer(config Config) *Indexer

// AddVectors adds vectors to the index
func (idx *Indexer) AddVectors(vectors [][]float32) error

// Search finds k nearest neighbors
func (idx *Indexer) Search(query []float32, k int) (indices []int, scores []float32)

// BatchSearch searches multiple queries
func (idx *Indexer) BatchSearch(queries [][]float32, k int) ([][]int, [][]float32)

// Save saves the index to disk
func (idx *Indexer) Save(path string) error

// Load loads the index from disk
func (idx *Indexer) Load(path string) error

// Close releases GPU resources
func (idx *Indexer) Close()
```

### Embedding Functions

```go
// NewEmbeddingModel creates embedding model
func NewEmbeddingModel(config EmbedConfig) *EmbeddingModel

// Embed converts tokens to embedding
func (m *EmbeddingModel) Embed(tokens []int64) []float32

// BatchEmbed processes multiple sequences
func (m *EmbeddingModel) BatchEmbed(tokenBatch [][]int64) [][]float32
```

## Benchmarking

```go
// Run comprehensive benchmark
results := gpu.Benchmark(gpu.BenchmarkConfig{
    NumVectors:  []int{10000, 100000, 1000000},
    Dimensions:  384,
    NumQueries:  1000,
    Precisions:  []Precision{FP32, FP16, INT8},
    Devices:     []Device{CPU, GPU},
})

// Print results
results.PrintSummary()
results.SaveCSV("benchmark_results.csv")
```

## Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

2. **Out of GPU memory**
   ```go
   // Reduce batch size
   config.BatchSize = 1000
   
   // Use INT8 quantization
   config.Precision = gpu.INT8
   
   // Enable memory pooling
   config.MemoryPool = true
   ```

3. **Slow performance**
   ```go
   // Check GPU utilization
   stats := indexer.GetGPUStats()
   if stats.Utilization < 80 {
       // Increase batch size
       config.BatchSize *= 2
   }
   ```

## Examples Repository

Full working examples available at: https://github.com/lee101/gobedexample

```bash
git clone https://github.com/lee101/gobedexample
cd gobedexample
go run examples/gpu_indexing.go
```