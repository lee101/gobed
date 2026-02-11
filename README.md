# Gobed - Ultra-Fast Text Search for Go

[![Go Report Card](https://goreportcard.com/badge/github.com/lee101/gobed)](https://goreportcard.com/report/github.com/lee101/gobed)
[![GoDoc](https://pkg.go.dev/badge/github.com/lee101/gobed)](https://pkg.go.dev/github.com/lee101/gobed)
MIT [![License](https://img.shields.io/github/license/lee101/gobed)](LICENSE)

<img width="450" height="633" alt="image" src="https://github.com/user-attachments/assets/45a072fc-1a17-4aca-9da7-5394d688153a" />

**Semantic search for Go with efficient int8 embeddings.** Gobed provides semantic search using compressed static embeddings. Features automatic GPU detection, int8 quantization for memory efficiency, and 7.9x model compression.

##  Performance Achievements

- **6.39s average search time** on 243K documents (current)
- **1.7 queries/sec** throughput with parallel processing
- **Int8 quantization** - 7.9x compression, 87.4% space saved
- **0.151ms embedding latency** with 6,629 embeddings/sec
- **15MB memory usage** for full model vs 119MB original

Built on [static embeddings](https://huggingface.co/blog/static-embeddings) with GPU kernel fusion for maximum speed.

## Quick Start

### CPU Setup 

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

## Bed CLI – Semantic Filesystem Search

`bed` is the command-line front end that applies Gobed embeddings to your local
projects. It can index and search using CPU-only mode or take advantage of a
CUDA-enabled GPU (via cuVS/CAGRA) for sub-millisecond querying.

### Install v1

```bash
go install github.com/lee101/gobed/cmd/bed@v1.0.0
# or:
go install github.com/lee101/gobed/bed/cmd/bed@v1.0.0
```

### Quick Start (GPU-accelerated)

```bash
# 1. Install CUDA 12.8 and fetch the Gobed model
./setup.sh

# 2. Run bed with GPU support (CAGRA + CUDA)
export LD_LIBRARY_PATH="$(pwd)/gpu:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}"
bed --gpu "memory leak in handler"  # searches the current directory
```

Useful sub-commands:

```bash
# Index a project (stores the embedding index for faster repeat searches)
bed index /path/to/project

# Run a GPU search against an indexed project
bed --gpu --limit 15 "database connection"

# CPU fallback
bed "keyword"

# Live index mode
bed index . --watch

# Performance + quality
bed bench . --queries 200 --ndcg
```

### Benchmark Against `testdata/`

We added a Go benchmark that indexes the repository's `testdata/` directory and
measures semantic search throughput:

```bash
cd bed
go test ./src -bench BedSearch -run ^$
```

The benchmark indexes the sample files once and then repeatedly searches using
`SimpleSearchEngine`, reporting `queries_per_second` so you can compare CPU and
GPU configurations on your machine.

## Advanced Features

### INT8 Mode (75% Less Memory)

```go
// Use 4x less memory with minimal accuracy loss
model, _ := gobed.LoadModelInt8(true)
```

### GPU Acceleration (RTX 3090 Optimized)

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
    UseInt8:   true,  // 4x memory reduction
}
engine := gpu.NewGPUSearchEngineWithConfig(model, config)
```

#### GPU Implementation Features

- **Ultra-Fast Static Embeddings** (`cuda_ultra_fast.cu`)
  - Simple token→vector lookup (not BERT)
  - Pre-quantized int8 embedding table
  - Automatic IVF clustering at 50K+ documents

- **Fused Kernels** (`cuda_fused_embed_search.cu`)
  - Single-pass: embed + average + quantize
  - No intermediate memory writes
  - Direct GPU search pipeline

- **RTX 3090 Optimizations**
  - 164KB shared memory per SM fully utilized
  - 6MB L2 cache for persistent data
  - Warp shuffle reductions
  - Multi-stream processing (4 concurrent)

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
- Optional: AVX-512 CPU for INT8 mode

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
