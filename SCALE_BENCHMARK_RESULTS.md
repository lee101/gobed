# Scale Benchmark Results: 1M-5M Vectors

## Executive Summary

Testing gobed search at scale (1M-5M AI characters) shows:
- **Embedding generation**: 196K/sec (5us/embed) - can embed 1M chars in ~5s
- **IVF search at 500K**: 715 QPS parallel (1.4ms avg)
- **IVF search at 1M**: Est. 300-400 QPS (2-3ms avg)
- **GPU CAGRA**: Sub-millisecond expected (requires CUVS lib)

## Detailed Results

### Embedding Generation (Int8 Model)

| Metric | Value |
|--------|-------|
| Avg latency | 5.085us |
| Throughput | 196,623 embeddings/sec |
| Model size | 15MB |
| Vector dim | 512 |

**Time to embed full datasets:**
- 1M characters: ~5.1 seconds
- 5M characters: ~25.5 seconds

### Search Performance (CPU IVF)

#### 100K Vectors
| Method | Avg Latency | QPS |
|--------|-------------|-----|
| Brute force (serial) | 31ms | 32 |
| Brute force (parallel) | 3.2ms | 313 |
| IVF256-p8 (parallel) | 417us | 2,397 |
| IVF512-p16 (parallel) | 494us | 2,025 |

#### 500K Vectors
| Method | Avg Latency | P99 | QPS |
|--------|-------------|-----|-----|
| IVF256-p8 (parallel) | 1.4ms | 1.65ms | 715 |
| IVF1024-p32 (parallel) | 5.5ms | 7.8ms | 183 |
| IVF1024-p16 | 6.7ms | 10ms | 149 |

### Memory Usage

| Scale | Raw Vectors | With Index | Bytes/Vector |
|-------|-------------|------------|--------------|
| 100K | 48.8 MB | ~60 MB | ~600 |
| 500K | 244 MB | ~300 MB | ~600 |
| 1M | 488 MB | ~600 MB | ~600 |
| 5M | 2.44 GB | ~3 GB | ~600 |

Memory formula: `n * 512 bytes (vectors) + n * 4 bytes (scales) + IVF overhead (~16%)`

### IVF Training Time (CPU)

| Vectors | nlist | Training Time |
|---------|-------|---------------|
| 100K | 256 | 20s |
| 100K | 512 | 40s |
| 500K | 256 | 1m42s |
| 500K | 1024 | 7m31s |
| 1M | 1024 | ~15min (est) |

**Note**: Training is one-time cost. Index can be cached/persisted.

## Projections for 1M-5M Scale

### 1M Vectors (Estimated)
- **Embedding time**: 5.1 seconds
- **IVF training**: 15-20 minutes (one-time)
- **Search latency**: 2-4ms with IVF1024-p32
- **Memory**: ~600 MB
- **QPS**: 250-400

### 5M Vectors (Estimated)
- **Embedding time**: 25.5 seconds
- **IVF training**: 60-90 minutes (one-time)
- **Search latency**: 5-10ms with IVF4096-p64
- **Memory**: ~3 GB
- **QPS**: 100-200

### With GPU CAGRA (Projected)
Based on CAGRA architecture docs:
- **Index build**: ~30s for 1M, ~2min for 5M
- **Search latency**: <1ms even at 5M
- **QPS**: 10,000+
- **Memory**: Same vectors + CAGRA graph (~2x vectors)

## Optimization Recommendations

### 1. Enable GPU CAGRA
Link CUVS library for production:
```bash
# Install NVIDIA CUVS
apt-get install libcuvs
# Link in build
CGO_LDFLAGS="-lcuvs -lcuvs_c"
```

Expected improvement: 10-100x search speedup

### 2. Pre-compute Index at Build Time
- Embed all characters during startup/deploy
- Cache CAGRA/IVF index to disk
- Load pre-built index for instant startup

### 3. Batch Embeddings
Current: Sequential embedding
Optimize: Parallel batch embedding (8 workers)
```go
func EmbedBatchInt8Optimized(texts []string, workers int)
```

### 4. Reduce Memory Allocations
- Use sync.Pool for query vectors
- Pre-allocate result slices
- Reuse score buffers

### 5. IVF Parameter Tuning
For 1M vectors:
- nlist: 1000-2000 (sqrt(n))
- nprobe: 16-32 (1-3% of nlist)

For 5M vectors:
- nlist: 2000-4000
- nprobe: 32-64

### 6. Hybrid CPU/GPU Strategy
- Small queries (<100): CPU IVF
- Large queries (>100 batch): GPU CAGRA
- Background: GPU for index updates

## Current Bottlenecks

1. **IVF Training**: K-means is slow for large nlist
   - Solution: K-means++ with parallel assignment (already in pkg/ann/ivf)
   - Solution: Train on sample, not full dataset

2. **CPU Search at Scale**: Linear with vectors searched
   - Solution: GPU CAGRA for sub-ms latency
   - Solution: Lower nprobe for speed vs accuracy tradeoff

3. **Memory**: ~600 bytes/vector
   - Solution: PQ compression (8x reduction possible)
   - Solution: Mmap for larger-than-RAM indexes

## Conclusion

For production 1M-5M character search:
- **CPU-only**: Achievable but limited to 200-700 QPS at 2-10ms latency
- **GPU CAGRA**: Required for <1ms latency at scale
- **Embedding**: Not a bottleneck (196K/sec)
- **Index build**: One-time cost, can be cached

Recommended architecture:
1. Embed characters at startup (5s for 1M)
2. Build CAGRA index (30s for 1M)
3. Serve search via GPU (<1ms per query)
4. Incremental updates via CPU IVF fallback
