# Gobed - Fast Go Embeddings

A high-performance Go implementation of text embeddings using the `sentence-transformers/static-retrieval-mrl-en-v1` model. 

**71x faster than Python GPU** with bit-perfect accuracy.

## Features

- ⚡ **Blazing Fast**: 150,000+ embeddings/second on CPU
- 🎯 **100% Accurate**: Bit-perfect match with Python implementation
- 📦 **Simple API**: Clean, easy-to-use Go interface
- 🔧 **Production Ready**: Optimized memory usage, pre-allocated buffers
- 💾 **Lightweight**: Only ~120MB memory usage (30MB with INT8)
- 🚀 **INT8 Quantization**: 75% memory reduction with minimal accuracy loss
- ⚙️ **SIMD Optimized**: AVX-512 acceleration for INT8 operations (with fallback)

## Installation

```bash
go get github.com/lee101/gobed
```

## Quick Start

1. **Download model weights** (one-time setup):

```bash
git clone https://github.com/lee101/gobed
cd gobed
./setup.sh  # Downloads model weights (119MB)
```

2. **Use in your code**:

```go
package main

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

## Performance

### Standard Float32 Model
| Metric | Go (CPU) | Python (GPU) | Speedup |
|--------|----------|--------------|---------|
| **Inference** | 12μs | 889μs | **71x faster** |
| **Throughput** | 150,000/sec | 1,125/sec | **133x more** |
| **Memory** | ~120MB | ~500MB | **4x less** |

### INT8 Quantized Model
| Metric | Float32 | INT8 | Improvement |
|--------|---------|------|-------------|
| **Memory Usage** | 120MB | 30MB | **75% reduction** |
| **Embedding Speed** | 360μs | 120μs | **3x faster** |
| **Similarity Calc** | 50μs | 10μs | **5x faster** |
| **Accuracy Loss** | 0% | <1% | **Minimal** |

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