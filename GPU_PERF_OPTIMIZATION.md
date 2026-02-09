# GPU Performance Optimization Results

## Baseline (Brute Force - fused_cagra.cu)

| Size | Build(ms) | Latency(us) | P95(us) | QPS | NDCG@10 | Recall@10 |
|------|-----------|-------------|---------|-----|---------|-----------|
| 10k | 6 | 2,034 | 2,422 | 492 | 0.9982 | 0.9900 |
| 25k | 11 | 3,284 | 3,773 | 304 | 0.9985 | 0.9880 |
| 50k | 36 | 5,310 | 5,890 | 188 | 0.9987 | 0.9930 |
| 100k | 70 | 9,441 | 10,138 | 106 | 0.9993 | 0.9950 |

**Key characteristics:**
- Perfect NDCG@10 (> 0.99) - expected for brute force
- Latency scales linearly O(n) with dataset size
- Build is fast (~1.4M vec/s)

## IVF Optimization (fused_cagra_opt.cu)

### Configuration: 64 lists, nprobe=24 (37.5% coverage)

| Size | Build(ms) | Latency(us) | P95(us) | QPS | NDCG@10 | Recall@10 |
|------|-----------|-------------|---------|-----|---------|-----------|
| 10k | 165 | 523 | 861 | 1,912 | 0.8030 | 0.7020 |
| 25k | 29 | 1,052 | 1,447 | 951 | 0.8265 | 0.7030 |
| 50k | 45 | 1,784 | 2,441 | 561 | 0.8636 | 0.7350 |
| 100k | 83 | 3,241 | 4,155 | 309 | 0.8931 | 0.7800 |

### Speedup at 100k
- **Latency**: 9.4ms → 3.2ms = **2.9x faster**
- **QPS**: 106 → 309 = **2.9x higher**
- **NDCG@10**: 0.99 → 0.89 = **10% quality drop** (unacceptable)

## Analysis

### Why IVF quality is low
1. **Uniform centroid sampling** - Not proper KMeans clustering
2. **Fixed nprobe** - Need dynamic probe count based on query
3. **Simple clustering** - Need multi-iteration KMeans

### Bottlenecks identified in brute force
1. **Serial final merge** (thread 0 only) - O(blockDim.x * k^2)
2. **Insert-sort top-k** - O(k) per candidate
3. **No work sharing** - Each query uses full block

## Recommendations

### Short-term (sub-ms at 100k)
1. **Enable native cuVS CAGRA** - Compile libcagra_wrapper.so with cuVS
2. **Tune CAGRA params** - graph_degree=64, max_iters=64

### Medium-term (quality improvements for IVF)
1. **Implement proper KMeans** - 5-10 iterations
2. **Residual quantization** - Better distance computation
3. **Dynamic nprobe** - Based on query difficulty

### Long-term (1M+ scale)
1. **Hierarchical IVF** - IVF-PQ or IVF-HNSW
2. **Multi-GPU** - Shard by cluster
3. **Streaming** - Incremental index updates

## Files created
- `cmd/gpu_perf_test/cagra_scale_eval.go` - CAGRA baseline benchmark
- `cmd/gpu_perf_test/fused_scale_bench.go` - Fused CAGRA benchmark
- `cmd/gpu_perf_test/ivf_bench.go` - IVF benchmark
- `fused_cagra_opt.cu` - IVF-optimized kernel

## Usage

```bash
# Run baseline benchmark
LD_LIBRARY_PATH=.:/usr/local/cuda/lib64 ./fused_bench -n 100000 -q 100

# Run IVF benchmark
LD_LIBRARY_PATH=.:/usr/local/cuda/lib64 ./ivf_bench -n 100000 -q 100
```
