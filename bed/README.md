# bed - Semantic Search CLI with NDCG@10 Eval

Fast semantic search for directories with GPU-accelerated CAGRA indexing and NDCG@10 benchmarking.

## Features

- **Directory Search**: Index and search through code/docs directories
- **GPU CAGRA**: CUDA-accelerated graph-based ANN search
- **NDCG@10 Eval**: Standard IR benchmark metric for ranking quality
- **Recall@K**: Traditional retrieval quality metric
- **Performance**: P50/P95 latency, QPS metrics

## Quick Start

```bash
# Build CLI
cd /home/lee/code/gobed
go build -o bed-cli cmd/bed/main.go

# Search a directory
./bed-cli -dir ./docs -q "GPU acceleration" -k 10

# Run benchmark with NDCG@10
./bed-cli -dir ./docs -bench -queries 100 -k 10
```

## API Usage

```go
import "github.com/lee101/gobed/bed"

// CPU eval with NDCG@10
cfg := bed.EvalConfig{K: 10, NumQueries: 100, Warmup: 10}
result, _ := bed.RunEval(model, docs, cfg)
fmt.Printf("NDCG@10: %.4f, Recall@10: %.4f\n", result.NDCGAtK, result.RecallAtK)

// GPU CAGRA eval
gpuCfg := gobed.DefaultGPUCagraConfig()
gpuResult, _ := bed.RunEvalGPU(model, docs, cfg, gpuCfg)
fmt.Printf("NDCG@10: %.4f\n", gpuResult.NDCGAtK)
```

## Metrics

- **NDCG@K**: Normalized Discounted Cumulative Gain - ranks by relevance scores
- **Recall@K**: Fraction of relevant docs in top-K
- **P50/P95**: Latency percentiles
- **QPS**: Queries per second

## GPU Setup

```bash
# Build GPU CAGRA library
make gpu-build

# Build with GPU tags
go build -tags="gpu" -o bed-cli cmd/bed/main.go

# Set library path
export LD_LIBRARY_PATH=/home/lee/code/gobed/gpu:$LD_LIBRARY_PATH

# Run with GPU
./bed-cli -bench -gpu -queries 1000
```

## Files

- `ndcg.go` - NDCG@K metric implementation
- `eval.go` - CPU evaluation harness
- `eval_gpu.go` - GPU evaluation harness
- `fsindexer.go` - Directory/file indexer
- `cmd/bed/main.go` - CLI tool

## Test

```bash
go run cmd/test_ndcg/main.go
```

Output:
```
✓ NDCG@5: 0.9995
✓ Recall@5: 1.0000
✓ P50: 0.08ms, QPS: 11988.1
```
