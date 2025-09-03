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
./setup_gpu.sh  # Automated GPU setup

# Or manual build:
cd gpu_search/cuda_ops
./build.sh

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
// Load model normally
model, _ := gobed.LoadModel()

// Create GPU-accelerated search engine
engine := gobed.NewGPUSearchEngine(model)

// Or with custom config:
import "github.com/lee101/gobed/gpu"
config := gpu.GPUSearchConfig{
    EnableGPU: true,
    DeviceID:  0,
    BatchSize: 1000,
}
engine := gpu.NewGPUSearchEngineWithConfig(model, config)
```

### Async Indexing (26x Faster)

```go
config := gobed.AsyncSearchConfig()
engine := gobed.NewSearchEngineWithConfig(model, config)

// Non-blocking indexing
response := engine.IndexBatchAsync(millionDocs)
result := <-response  // Wait when ready
// Note: result.Stats.ProcessingTime contains duration
```

### Shared Memory (Multiple Processes)

```go
// Share index across processes with zero-copy
config := gobed.SearchConfig{
    UseSharedMemory: true,
    SharedBasePath: "/tmp/my_index",
    MaxVectors: 1000000,
}
engine := gobed.NewSearchEngineWithConfig(model, config)
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
- Optional: AVX-512 CPU for INT8 mode (required for INT8, will crash without it)

### Model
Using `sentence-transformers/static-retrieval-mrl-en-v1`:
- 1024-dimensional embeddings
- 30,522 token vocabulary  
- Static embeddings with mean pooling
- Learn more: [Static Embeddings](https://huggingface.co/blog/static-embeddings)

### Important Notes

**Model Location**: The model files (`real_model.safetensors` and `tokenizer.json`) must be in a `model/` directory relative to where your code runs. The `setup.sh` script handles this automatically.

**INT8 Mode**: Requires a CPU with AVX-512 support. Will crash with "illegal instruction" error on older CPUs. Check your CPU with `lscpu | grep avx512` on Linux.

**GPU Package**: The published Go package has GPU build dependencies. For now, clone the repository locally instead of using `go get` if you need GPU support:
```bash
git clone https://github.com/lee101/gobed
cd gobed
# Use replace directive in your go.mod
go mod edit -replace github.com/lee101/gobed=./gobed
```

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