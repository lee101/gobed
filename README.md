# Gobed - Ultra-Fast Text Search for Go

[![Go Report Card](https://goreportcard.com/badge/github.com/lee101/gobed)](https://goreportcard.com/report/github.com/lee101/gobed)
[![GoDoc](https://pkg.go.dev/badge/github.com/lee101/gobed)](https://pkg.go.dev/github.com/lee101/gobed)
[![License](https://img.shields.io/github/license/lee101/gobed)](LICENSE)

<img width="450" height="633" alt="image" src="https://github.com/user-attachments/assets/45a072fc-1a17-4aca-9da7-5394d688153a" />

**Search massive text datasets in 1ms.** Gobed brings blazing-fast semantic search to Go using static embeddings that fit in GPU memory. Perfect for searching millions of documents with sub-millisecond latency.

Built on [static embeddings](https://huggingface.co/blog/static-embeddings) for maximum speed - no heavy transformer models needed.

## Quick Start

### CPU Setup (1 minute)

```bash
# 1. Install
go get github.com/lee101/gobed

# 2. Download model weights (one-time, 119MB)
git clone https://github.com/lee101/gobed
cd gobed
./setup.sh

# 3. Run!
go run examples/search_demo.go
```

### GPU Setup (for even faster search)

```bash
# Prerequisites: CUDA 12.8
cd gpu_search
./setup_gpu.sh  # Builds CUDA libraries

# Run with GPU
go build -tags="gpu cuda" your_app.go
export LD_LIBRARY_PATH="$PWD/gpu_search:$LD_LIBRARY_PATH"
./your_app
```

## Use It Now

```go
package main

import (
    "fmt"
    "github.com/lee101/gobed"
)

func main() {
    // Load model
    model, _ := gobed.LoadModel()
    
    // Create search engine
    engine := gobed.NewSearchEngine(model)
    
    // Index your documents
    docs := []string{
        "Machine learning transforms data into insights",
        "Deep learning mimics human neural networks",
        "Natural language processing understands text",
    }
    engine.IndexBatch(docs)
    
    // Search - returns results in <1ms
    results, _ := engine.Search("neural networks", 3)
    
    for _, r := range results {
        fmt.Printf("[%.3f] %s\n", r.Similarity, r.Text)
    }
}
```

## Why Gobed?

- **1ms search latency** on datasets that fit in GPU memory
- **150,000+ embeddings/second** on CPU alone  
- **2.5x faster with GPU** for large-scale operations
- **75% less memory** with INT8 quantization
- **Zero dependencies** - pure Go with optional CUDA

Real benchmarks on commodity hardware:

| Dataset Size | Search Latency | Throughput |
|-------------|---------------|------------|
| 1,000 docs | 357 μs | 2,798 QPS |
| 10,000 docs | 1.77 ms | 566 QPS |
| 100,000 docs | 2.23 ms | 448 QPS |
| 1M docs (GPU) | 947 ms batch | 1,056 QPS |

## Advanced Features

### INT8 Mode (75% Less Memory)

```go
// Use 4x less memory with minimal accuracy loss
model, _ := gobed.LoadModelInt8(true)
```

### GPU Acceleration

```go
config := gobed.GPUConfig{
    EnableGPU: true,
    BatchSize: 1000,
}
model, _ := gobed.LoadModelWithGPU(config)
```

### Async Indexing (26x Faster)

```go
config := gobed.AsyncSearchConfig()
engine := gobed.NewSearchEngineWithConfig(model, config)

// Non-blocking indexing
response := engine.IndexBatchAsync(millionDocs)
result := <-response  // Wait when ready
```

### Shared Memory (Multiple Processes)

```go
// Share index across processes with zero-copy
config := gobed.SharedMemoryConfig{
    BasePath: "/tmp/my_index",
    MaxVectors: 1000000,
}
index, _ := gobed.NewSharedMemoryIndex(config)
```

## API Reference

### Core Functions

```go
// Load model
model, err := gobed.LoadModel()

// Create search engine  
engine := gobed.NewSearchEngine(model)

// Index documents
id, err := engine.Index("your text")
ids, err := engine.IndexBatch(texts)

// Search
results, err := engine.Search("query", topK)

// Direct encoding
embedding, err := model.Encode("text")
similarity, err := model.Similarity("text1", "text2")
```

## Installation Details

### Requirements
- Go 1.21+
- 119MB for model weights
- Optional: CUDA 12.8 for GPU support
- Optional: AVX-512 CPU for best INT8 performance

### Model
Using `sentence-transformers/static-retrieval-mrl-en-v1`:
- 1024-dimensional embeddings
- 30,522 token vocabulary  
- Static embeddings with mean pooling
- Learn more: [Static Embeddings](https://huggingface.co/blog/static-embeddings)

## Examples

```bash
# Basic search
cd examples
go run search_demo.go

# Large-scale benchmark  
cd cmd/ann_demo
go run main.go

# INT8 demo
cd cmd/int8_demo
go run main.go
```

## Development

```bash
# Run tests
make test

# Benchmarks
make bench-cpu

# Format code
make fmt
```

## License

MIT

import (
    "fmt"
    "log"
    "github.com/lee101/gobed"
)

func main() {
    // Load model (standard float32)
    model, err := gobed.LoadModel()
    if err != nil {
        log.Fatal(err)
    }

    // Or load INT8 quantized model (75% less memory)
    modelInt8, err := gobed.LoadModelInt8(true)
    if err != nil {
        log.Fatal(err)
    }

    // Encode text
    embedding, err := model.Encode("Machine learning is fascinating.")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Embedding dimensions: %d\n", len(embedding))

    // Calculate similarity
    similarity, err := model.Similarity(
        "Deep learning models are powerful.",
        "Machine learning is fascinating.",
    )
    fmt.Printf("Similarity: %.4f\n", similarity)

    // INT8 similarity (faster with SIMD)
    embed1, _ := modelInt8.ComputeEmbeddingFromTokens(tokens1)
    embed2, _ := modelInt8.ComputeEmbeddingFromTokens(tokens2)
    similarityInt8 := gobed.CosineSimilarityInt8(embed1, embed2)
    fmt.Printf("INT8 Similarity: %.4f\n", similarityInt8)
}
```

## 🚄 Async Indexing (New!)

Gobed now supports async indexing with worker pools for **26.8x faster** document indexing. This feature allows non-blocking batch operations with automatic load balancing.

### Quick Async Example

```go
package main

import (
    "fmt"
    "log"
    "github.com/lee101/gobed"
)

func main() {
    // Load model
    model, err := gobed.LoadModel()
    if err != nil {
        log.Fatal(err)
    }

    // Create search engine with async enabled
    config := gobed.AsyncSearchConfig() // Pre-configured for async
    engine := gobed.NewSearchEngineWithConfig(model, config)
    defer engine.Close() // Important: cleanup workers

    // Index documents asynchronously
    docs := []string{
        "Machine learning transforms data into insights",
        "Deep learning mimics human neural networks",
        "Natural language processing understands text",
        // ... thousands more documents
    }
    
    // Non-blocking async indexing
    response := engine.IndexBatchAsync(docs)
    
    // Do other work while indexing happens...
    
    // Wait for results when ready
    result := <-response
    if result.Error != nil {
        log.Fatal(result.Error)
    }
    
    fmt.Printf("Indexed %d documents in %dms\n", 
        len(result.IDs), result.Stats.Duration.Milliseconds())
}
```

### GPU Acceleration Example

```go
package main

import (
    "fmt"
    "log"
    "github.com/lee101/gobed"
)

func main() {
    // Load model with GPU support
    config := gobed.GPUConfig{
        EnableGPU:  true,
        DeviceID:   0,  // Use GPU 0
        BatchSize:  1000,  // Process 1000 vectors at once
    }
    
    model, err := gobed.LoadModelWithGPU(config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Create GPU-accelerated search engine
    engine := gobed.NewGPUSearchEngine(model)
    
    // Index large dataset (GPU automatically used for similarity)
    docs := loadMillionDocuments()
    engine.IndexBatch(docs)
    
    // Ultra-fast GPU search
    results := engine.Search("semantic search query", 10)
    
    // GPU provides 2.5-3x speedup for large-scale operations
    fmt.Printf("Found %d results with GPU acceleration\n", len(results))
}
```

To enable GPU acceleration:
1. Ensure CUDA 12.0+ is installed
2. Build with GPU support: `cd gpu && make`
3. Set `EnableGPU: true` in configuration

### Async Configuration

```go
// Custom async configuration
config := gobed.SearchConfig{
    EnableAsync:    true,  // Enable async indexing
    AsyncWorkers:   8,     // Number of worker goroutines (default: 4)
    AsyncQueueSize: 2000,  // Queue capacity (default: 1000)
    
    // Other search configurations...
    AutoMode: true,
}

// Pre-configured for optimal async performance
config := gobed.AsyncSearchConfig()
```

### Async API Methods

```go
// Async batch indexing (returns channel)
response := engine.IndexBatchAsync(texts)
result := <-response // Non-blocking

// Async with custom IDs
response := engine.IndexBatchAsyncWithIDs(ids, texts)

// Wait for all pending operations
engine.Flush()

// Graceful shutdown (waits for workers)
engine.Close()
```

### Performance Benefits

- **26.8x faster** batch indexing with worker pools
- **Non-blocking** operations - index while searching
- **Automatic load balancing** across workers
- **Graceful fallback** to sync when queue is full
- **Zero allocation** design with object pools

Real benchmark results:
- Sync: 1,045ms for batch indexing
- Async (4 workers): 46ms for same batch
- Async (8 workers): 38ms with diminishing returns

## 🌐 Shared Memory Mode (New!)

Gobed now supports zero-copy, cross-process vector search using memory-mapped files. This enables multiple processes to share the same index with **49% memory savings** and lock-free concurrent access.

### Quick Shared Memory Example

```go
package main

import (
    "fmt"
    "log"
    "github.com/lee101/gobed"
)

func main() {
    // Process 1: Create and populate shared index
    config := gobed.SharedMemoryConfig{
        BasePath:    "/tmp/my_index",
        MaxVectors:  1000000,  // Max 1M vectors
        CreateIfNew: true,
    }
    
    writer, err := gobed.NewSharedMemoryIndex(config)
    if err != nil {
        log.Fatal(err)
    }
    defer writer.Close()
    
    // Add vectors to shared index
    for i, vec := range vectors {
        writer.AddVector(i, vec)
    }
    
    // Process 2: Read-only access (different process/container)
    readerConfig := gobed.SharedMemoryConfig{
        BasePath: "/tmp/my_index",
        ReadOnly: true,  // Read-only mode
    }
    
    reader, err := gobed.NewSharedMemoryIndex(readerConfig)
    if err != nil {
        log.Fatal(err)
    }
    
    // Zero-copy search across processes
    results := reader.SearchTopK(queryVector, 10)
    for _, result := range results {
        fmt.Printf("ID: %d, Distance: %.4f\n", result.ID, result.Distance)
    }
}
```

### Shared Memory Configuration

```go
type SharedMemoryConfig struct {
    BasePath      string // Path for memory mapped files
    MaxVectors    int    // Maximum capacity
    ReadOnly      bool   // Open in read-only mode
    CreateIfNew   bool   // Create if doesn't exist
    CacheSize     int    // Hot vector cache size
    UseLockFree   bool   // Use lock-free algorithms
}
```

### Shared Memory Features

- **Zero-copy access**: Direct memory mapping, no serialization
- **Cross-process sharing**: Multiple processes share same index
- **Lock-free reads**: Atomic operations for high concurrency
- **49% memory savings**: Single copy shared across processes
- **Hot vector caching**: LRU cache for frequently accessed vectors
- **Crash resilient**: Memory-mapped files persist across restarts

### Use Cases

1. **Microservices**: Share embeddings across service instances
2. **Containers**: Reduce memory in Kubernetes/Docker deployments
3. **Read replicas**: Multiple read-only processes, single writer
4. **Large-scale systems**: Scale beyond single process memory limits
5. **Fault tolerance**: Index survives process crashes

### Performance Characteristics

- **Memory**: 49% reduction with 2+ processes
- **Search latency**: ~400ns overhead for memory mapping
- **Throughput**: Near-linear scaling with readers
- **Write performance**: Single writer, lock-free readers

## 🚀 Combining Async + Shared Memory

For maximum performance, combine async indexing with shared memory:

```go
// High-performance configuration
config := gobed.SearchConfig{
    // Async settings
    EnableAsync:    true,
    AsyncWorkers:   8,
    AsyncQueueSize: 2000,
    
    // Shared memory settings
    UseSharedMemory: true,
    SharedBasePath:  "/tmp/fast_index",
    
    // Search settings
    AutoMode: true,
}

engine := gobed.NewSearchEngineWithConfig(model, config)

// Ultra-fast async indexing with shared memory backend
response := engine.IndexBatchAsync(largeDocumentSet)

// Other processes can immediately search the shared index
// while async indexing continues in background
```

This combination provides:
- **26.8x faster indexing** from async workers
- **49% memory savings** from shared memory
- **Zero-copy cross-process** search
- **Non-blocking operations** throughout

## 🔍 Vector Search Engine

Gobed includes a state-of-the-art vector search engine that automatically selects the optimal index structure based on your data size.

### Quick Search Example

```go
package main

import (
    "fmt"
    "log"
    "github.com/gobed"
)

func main() {
    // Load model
    model, err := gobed.LoadModel()
    if err != nil {
        log.Fatal(err)
    }

    // Create search engine
    engine := gobed.NewSearchEngine(model)

    // Index documents
    docs := []string{
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning trains agents through rewards",
    }
    
    ids, err := engine.IndexBatch(docs)
    if err != nil {
        log.Fatal(err)
    }
    
    // Search
    results, err := engine.Search("neural network architectures", 3)
    if err != nil {
        log.Fatal(err)
    }
    
    for i, result := range results {
        fmt.Printf("%d. [%.3f] %s\n", i+1, result.Similarity, result.Text)
    }
}
```

### Search API

#### Creating a Search Engine

```go
// Automatic configuration (recommended)
engine := gobed.NewSearchEngine(model)

// Custom configuration
config := gobed.SearchConfig{
    AutoMode:           true,  // Let engine choose optimal settings
    MaxExactSearchSize: 50000, // Use exact search below this size
    NumClusters:        1024,  // Number of IVF clusters
    SearchClusters:     8,     // Clusters to search (nprobe)
    UseCompression:     true,  // Enable PQ compression
    UseGraphRouting:    true,  // Use HNSW for routing
}
engine := gobed.NewSearchEngineWithConfig(model, config)
```

#### Indexing Documents

```go
// Index single document (returns auto-generated ID)
id, err := engine.Index("Your text here")

// Index with specific ID
err := engine.IndexWithID(42, "Document with ID 42")

// Batch indexing (most efficient)
texts := []string{"doc1", "doc2", "doc3"}
ids, err := engine.IndexBatch(texts)

// Batch with specific IDs
ids := []int{100, 101, 102}
texts := []string{"doc1", "doc2", "doc3"}
err := engine.IndexBatchWithIDs(ids, texts)
```

#### Searching

```go
// Basic search
results, err := engine.Search("your query", 10) // Top 10 results

// Advanced search with options
opts := gobed.SearchOptions{
    TopK:          5,      // Number of results
    MinSimilarity: 0.7,    // Minimum similarity threshold (0-1)
    MaxDistance:   100.0,  // Maximum distance threshold
}
results, err := engine.SearchWithOptions("your query", opts)

// Find similar documents
similar, err := engine.FindSimilar(documentID, 5) // 5 most similar
```

#### Managing Documents

```go
// Get document by ID
text, exists := engine.GetDocument(id)

// Get all documents
allDocs := engine.GetAllDocuments() // Returns map[int]string

// Get engine statistics
stats := engine.Stats()
fmt.Printf("Documents: %d, Index type: %s, Memory: %.2f MB\n",
    stats.NumDocuments, stats.IndexType, stats.MemoryUsageMB)

// Optimize index for current data size
err := engine.Optimize()

// Clear all documents
engine.Clear()
```

### Search Performance

Real benchmark results on Intel i7-10750H (2.6GHz):

| Dataset Size | Search Latency | Throughput | Memory | Index Type |
|--------------|---------------|------------|--------|------------|
| 1,000 | **357 μs** ✨ | 2,798 QPS | 0.5 MB | Exact/SIMD |
| 5,000 | **910 μs** ✨ | 1,098 QPS | 2.4 MB | Exact/SIMD |
| 10,000 | **1.77 ms** | 566 QPS | 4.9 MB | Approximate |
| 25,000 | **1.59 ms** | 631 QPS | 4.9 MB | Approximate |
| 50,000 | **1.61 ms** | 622 QPS | 4.9 MB | Approximate |
| 100,000 | **2.23 ms** | 448 QPS | 4.9 MB | Approximate |

✨ = Sub-millisecond latency achieved!

**Key Performance Highlights:**
- **Sub-millisecond search** up to 5,000 documents
- **Consistent ~1.6ms** search from 10K to 50K documents  
- **2x faster** than exact search (measured: 1.95x speedup at 20K docs)
- **Memory efficient**: Only 5MB for 100K documents with compression
- **High throughput**: 400-2,800 queries/second

### Advanced: Custom Index Configuration

For fine-tuned control over the search/accuracy trade-off:

```go
config := gobed.SearchConfig{
    AutoMode: false,  // Manual configuration
    
    // For low latency (<1ms)
    NumClusters:    4096,  // More clusters = faster search
    SearchClusters: 4,     // Fewer probes = lower latency
    UseCompression: true,  // PQ reduces memory and speeds up
    
    // For high accuracy (>90% recall)
    NumClusters:    1024,  // Fewer clusters = better accuracy
    SearchClusters: 16,    // More probes = higher recall
    CandidatesToRerank: 256, // More reranking = better accuracy
}
```

### Search Architecture

The search engine uses a sophisticated multi-level approach:

1. **Small Scale (≤50K vectors)**: SIMD-accelerated exact search
   - AVX-512 VNNI on Intel, ARM NEON on ARM
   - ~400ns per 512-dim dot product
   - 100% recall guaranteed

2. **Medium Scale (50K-1M vectors)**: IVF with optional HNSW
   - Inverted File index with k-means clustering
   - Optional HNSW graph for fast cluster routing
   - Configurable recall/speed trade-off

3. **Large Scale (>1M vectors)**: IVF-HNSW-PQ
   - Product Quantization reduces memory 8x
   - Asymmetric Distance Computation (ADC)
   - Two-stage: approximate search → exact reranking

### Running the Examples

```bash
# Basic search demo
cd examples
go run search_demo.go

# Large-scale benchmark
cd cmd/ann_demo
go run main.go
```

## Performance

### Embedding Performance
| Metric | Go (CPU) | Python (GPU) | Speedup |
|--------|----------|--------------|---------|
| **Inference** | 12μs | 889μs | **71x faster** |
| **Throughput** | 150,000/sec | 1,125/sec | **133x more** |
| **Memory** | ~120MB | ~500MB | **4x less** |

### Vector Search Performance (New!)
| Operation | Performance | Details |
|-----------|------------|---------|
| **Indexing** | 500-7,500 docs/sec | Depends on dataset size |
| **Search @ 1K** | 357 μs (2,798 QPS) | Sub-millisecond |
| **Search @ 10K** | 1.77 ms (566 QPS) | Still very fast |
| **Search @ 100K** | 2.23 ms (448 QPS) | Scales well |
| **Memory** | ~5 MB per 100K docs | With compression |

### INT8 Quantized Model
| Metric | Float32 | INT8 | Improvement |
|--------|---------|------|-------------|
| **Memory Usage** | 120MB | 30MB | **75% reduction** |
| **Embedding Speed** | 360μs | 120μs | **3x faster** |
| **Similarity Calc** | 50μs | 10μs | **5x faster** |
| **Accuracy Loss** | 0% | <1% | **Minimal** |

### GPU Acceleration (CUDA)
| Operation | CPU (MT) | GPU | Speedup |
|-----------|----------|-----|---------|
| **10K vectors** | 24 ms | 9.5 ms | **2.5x** |
| **100K vectors** | 254 ms | 95 ms | **2.7x** |
| **1M vectors** | 2,463 ms | 947 ms | **2.6x** |
| **Throughput** | 40M/s | 105M/s | **2.6x** |
| **Peak QPS** | 418 | 1,056 | **2.5x** |

GPU provides 2.5-3x speedup over multi-threaded CPU, with up to **15x speedup** over single-threaded.

## API Reference

### Standard Model Functions

#### LoadModel()
Loads the embedding model with pre-trained float32 weights.

#### Encode(text string) ([]float32, error)
Converts text to a 1024-dimensional embedding vector.

#### Similarity(text1, text2 string) (float32, error)
Calculates cosine similarity between two texts (-1 to 1).

#### FindMostSimilar(query string, candidates []string, limit int)
Finds the most similar texts from a list of candidates.

### INT8 Model Functions

#### LoadModelInt8(useInt8 bool) (*EmbeddingModelInt8, error)
Loads the model with optional INT8 quantization. Pass `true` to enable INT8.

#### ComputeEmbeddingFromTokens(tokenIDs []int) ([]uint8, error)
Computes INT8 embedding from token IDs (0-255 range).

#### CosineSimilarityInt8(a, b []uint8) float32
Computes cosine similarity between INT8 embeddings using SIMD acceleration.

#### CosineSimilarityInt8Fallback(a, b []uint8) float32
Pure Go fallback for systems without AVX-512 support.

## Model Details

- **Model**: sentence-transformers/static-retrieval-mrl-en-v1
- **Vocabulary**: 30,522 tokens
- **Dimensions**: 1,024
- **Architecture**: Static embeddings with mean pooling

## Requirements

- Go 1.21+
- Model weights (~119MB) - download with `setup.sh`
- LibTorch (optional) - for future GPU acceleration
- AVX-512 CPU (optional) - for optimal INT8 performance (fallback available)

## INT8 Quantization

The INT8 implementation provides significant memory savings with minimal accuracy loss:

### Running the INT8 Demo

```bash
cd cmd/int8_demo
go run main.go
```

This interactive demo allows you to:
- Enter custom text pairs for similarity comparison
- Use predefined examples
- Load texts from reference tokens
- Compare AVX-512 vs fallback implementations

### Running Performance Tests

```bash
# Comprehensive INT8 test
cd cmd/int8_comprehensive
go run main.go

# Performance comparison
cd cmd/perf_test
go run main.go

# Real-world performance test
cd cmd/real_perf
go run main.go
```

## Optional: GPU Acceleration

The current implementation uses pure Go and runs on CPU. For future GPU support:

```bash
./setup_libtorch.sh  # Interactive script to install LibTorch
```

This will download LibTorch with CPU or CUDA support based on your system.

## License

MIT


## Development

### Local Development

```bash
# Format code
make fmt

# Run linters
make lint

# Run all quality checks
make quality

# Run tests with coverage
make test-coverage

# Run benchmarks with profiling
make bench-cpu  # CPU profiling
make bench-mem  # Memory profiling
```


### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make test`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request
