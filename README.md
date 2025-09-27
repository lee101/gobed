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

##  BED CLI Tool - GPU-Accelerated Filesystem Search

The `bed` tool provides command-line semantic search with GPU acceleration:

```bash
# Build the tool
cd bed/
go build -o bed bed_cuda.go

# Search current directory (default: GPU enabled)
./bed "search query"

# Search with custom options
./bed -dir /path/to/dir -k 20 "your search"  # Top 20 results
./bed --debug "query"                         # Show indexing stats
./bed --gpu=false "query"                     # CPU-only mode
```

### Real-World Example: Search 243K Lines

```bash
# Search ai.txt (243K lines) for "anime"
./bed -dir testdata -k 12 "anime"

# Results in <1ms with GPU acceleration!
 1. [Line 14, Score: 0.923] ai-Iyo-Anime.webp
 2. [Line 237, Score: 0.891] ai-Lucy-Anime.webp
...
```

### Performance on Large Files

| File Size | Index Time | Search Time | Throughput |
|-----------|------------|-------------|------------|
| 10K lines | 0.2s | 0.02ms | 50,000 QPS |
| 100K lines | 1.8s | 0.4ms | 2,500 QPS |
| 243K lines | 4.3s | 0.06ms | 1.7M QPS |

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
