# GPU-Accelerated K-Means Implementation Plan

## Current Bottleneck Analysis
- **Problem**: K-means training takes 2m34s for 240k vectors (154s)
- **Target**: <3 seconds (50x speedup needed)
- **Root cause**: CPU-based K-means with O(n*k*d*iterations) complexity

## Proposed Solution: CUDA K-Means

### 1. Architecture
```
CPU (Current)                  GPU (Proposed)
=============                  ==============
240k vectors                   240k vectors
  ↓                              ↓
K-means (25 iter)             CUDA K-means
  - Distance calc (CPU)         - Distance (GPU)
  - Assignment (CPU)             - Assignment (GPU)
  - Update (CPU)                 - Update (GPU)
  ↓                              ↓
154 seconds                    <3 seconds
```

### 2. Implementation Steps

#### Phase 1: CUDA Kernel for Distance Computation
```cuda
__global__ void computeDistances(
    int8_t* vectors,      // [n, 512] int8 vectors
    int8_t* centroids,    // [k, 512] int8 centroids
    float* scales,        // [n] vector scales
    float* centroidScales,// [k] centroid scales
    float* distances,     // [n, k] output distances
    int n, int k
) {
    // Use tensor cores for int8 dot products
    // Batch process with shared memory
    // Warp-level primitives for reduction
}
```

#### Phase 2: Assignment Kernel
```cuda
__global__ void assignClusters(
    float* distances,     // [n, k] distances
    int* assignments,     // [n] cluster assignments
    int n, int k
) {
    // Find argmin per vector
    // Coalesced memory access
    // Use CUB for reduction
}
```

#### Phase 3: Centroid Update Kernel
```cuda
__global__ void updateCentroids(
    int8_t* vectors,      // [n, 512] vectors
    int* assignments,     // [n] assignments
    int8_t* newCentroids, // [k, 512] output
    int n, int k
) {
    // Atomic operations for accumulation
    // Shared memory for partial sums
    // Quantize back to int8
}
```

### 3. Optimizations

#### Memory Optimizations
- **Pinned memory** for CPU-GPU transfers
- **Unified memory** for large datasets
- **Texture memory** for centroid caching
- **Shared memory** for tile-based computation

#### Compute Optimizations
- **Tensor cores** for int8 operations (RTX 3090)
- **Warp shuffle** for reductions
- **Grid-stride loops** for load balancing
- **Mixed precision** (int8 compute, fp32 accumulation)

#### Algorithmic Optimizations
- **Mini-batch K-means** for huge datasets
- **K-means++** initialization on GPU
- **Early termination** when converged
- **Elkan's algorithm** to skip distance computations

### 4. Integration with gobed

```go
// gpu_kmeans.go
type GPUKMeans struct {
    handle    unsafe.Pointer // CUDA context
    k         int
    maxIters  int
    device    int
}

func NewGPUKMeans(k, maxIters int) *GPUKMeans {
    // Initialize CUDA
    // Allocate device memory
    // Load kernels
}

func (km *GPUKMeans) Fit(vectors []Vec512, scales []float32) {
    // Transfer to GPU
    // Run CUDA kernels
    // Transfer results back
}
```

### 5. Performance Targets

| Dataset Size | Current (CPU) | Target (GPU) | Speedup |
|-------------|---------------|--------------|---------|
| 10k         | 3.2s          | 0.06s        | 53x     |
| 50k         | 16s           | 0.3s         | 53x     |
| 240k        | 154s          | 2.9s         | 53x     |
| 1M          | 640s (est)    | 12s          | 53x     |

### 6. Testing Strategy

1. **Unit tests**: Compare GPU vs CPU results
2. **Benchmark suite**: Various dataset sizes
3. **Correctness validation**: Clustering quality metrics
4. **Memory leak detection**: cuda-memcheck
5. **Profile optimization**: nvprof/nsight

### 7. Fallback Strategy

```go
func (e *Engine) Train(vectors []Vec512) error {
    if IsCUDAAvailable() && len(vectors) > 10000 {
        // Use GPU K-means
        km := NewGPUKMeans(e.config.NList, 25)
        km.Fit(vectors, scales)
    } else {
        // Fall back to CPU
        km := NewKMeans(e.config.NList, 25)
        km.Fit(vectors, scales)
    }
}
```

### 8. Additional Optimizations

#### AVX-512 for CPU Path
- Use AVX-512 VNNI for int8 operations
- Vectorized distance computation
- Parallel assignment with OpenMP

#### Int16 Tokenizer
- Custom tokenizer avoiding external dependencies
- SIMD-optimized vocabulary lookup
- Batch tokenization for throughput

#### Pipeline Optimizations
- Overlap embedding computation with indexing
- Streaming index updates
- Progressive K-means refinement

## Implementation Priority

1. **CRITICAL**: GPU K-means (50x speedup needed)
2. **HIGH**: Int16 tokenizer benchmark & optimization
3. **MEDIUM**: AVX-512 optimizations for CPU fallback
4. **LOW**: Progressive indexing improvements

## Next Steps

1. Create CUDA kernel prototypes
2. Benchmark distance computation kernel
3. Integrate with gobed IVF training
4. Validate correctness at scale
5. Profile and optimize memory transfers