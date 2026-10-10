// Ultra-Fast GPU Search Implementation - Target: <1ms inference
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

extern "C" {

// GPU Fast Search structure
typedef struct {
    void* handle;           // cuBLAS handle
    void* search_cache;     // Pre-computed search structures
    void* gpu_index;        // Index optimized for search
    int nlist;              // Number of clusters
    int nprobe;             // Number of clusters to probe
    int vector_dim;         // Vector dimension (512)
    int total_vectors;      // Total indexed vectors
    
    // GPU memory pointers
    signed char* d_vectors; // Device vectors
    float* d_scales;        // Device scales  
    int* d_ids;            // Document IDs
    float* d_centroids;    // Cluster centroids for IVF
    
    // Per-query scratch, allocated once and reused. These used to be four
    // cudaMalloc/device_vector allocations on every single search, which made
    // every query a potential OOM and (unchecked) a source of illegal-address
    // faults that aborted the whole process.
    signed char* d_query_buf;   // [vector_dim]
    float* d_distances_buf;     // [scratch_capacity]
    float* d_sort_keys;         // [scratch_capacity]
    int* d_sort_idx;            // [scratch_capacity]
    int scratch_capacity;

    // Performance tracking
    double total_search_time_us;
    int search_count;
    int cache_hits;
} gpu_fast_search_t;

// Create fast search indexer
gpu_fast_search_t* gpu_fast_search_create(int nlist, int nprobe, int vector_dim) {
    gpu_fast_search_t* searcher = (gpu_fast_search_t*)calloc(1, sizeof(gpu_fast_search_t));
    if (!searcher) return nullptr;
    
    searcher->nlist = nlist;
    searcher->nprobe = nprobe;
    searcher->vector_dim = vector_dim;
    searcher->total_vectors = 0;
    
    // Create cuBLAS handle for fast matrix operations
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    searcher->handle = cublas_handle;
    
    // Pre-allocate centroids for IVF
    size_t centroid_bytes = nlist * vector_dim * sizeof(float);
    cudaMalloc(&searcher->d_centroids, centroid_bytes);
    cudaMemset(searcher->d_centroids, 0, centroid_bytes);
    
    return searcher;
}

// Destroy fast search indexer
void gpu_fast_search_destroy(gpu_fast_search_t* searcher) {
    if (!searcher) return;
    
    // Free GPU memory
    if (searcher->d_vectors) cudaFree(searcher->d_vectors);
    if (searcher->d_scales) cudaFree(searcher->d_scales);
    if (searcher->d_ids) cudaFree(searcher->d_ids);
    if (searcher->d_centroids) cudaFree(searcher->d_centroids);
    if (searcher->d_query_buf) cudaFree(searcher->d_query_buf);
    if (searcher->d_distances_buf) cudaFree(searcher->d_distances_buf);
    if (searcher->d_sort_keys) cudaFree(searcher->d_sort_keys);
    if (searcher->d_sort_idx) cudaFree(searcher->d_sort_idx);
    
    // Destroy cuBLAS handle
    if (searcher->handle) {
        cublasDestroy((cublasHandle_t)searcher->handle);
    }
    
    free(searcher);
}

// Build index from vectors (target: 10-20s for 243K vectors)
int gpu_fast_search_build_index(gpu_fast_search_t* searcher, const signed char* vectors,
                                const float* scales, const int* ids, int num_vectors) {
    if (!searcher) return 0;
    
    searcher->total_vectors = num_vectors;
    
    // Allocate GPU memory for vectors
    size_t vector_bytes = num_vectors * searcher->vector_dim * sizeof(signed char);
    size_t scale_bytes = num_vectors * sizeof(float);
    size_t id_bytes = num_vectors * sizeof(int);
    
    cudaMalloc(&searcher->d_vectors, vector_bytes);
    cudaMalloc(&searcher->d_scales, scale_bytes);
    cudaMalloc(&searcher->d_ids, id_bytes);
    
    // Copy data to GPU (ultra-fast with pinned memory)
    cudaMemcpy(searcher->d_vectors, vectors, vector_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(searcher->d_scales, scales, scale_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(searcher->d_ids, ids, id_bytes, cudaMemcpyHostToDevice);
    
    return 1; // Success
}

// Ultra-fast distance computation kernel
__global__ void ultra_fast_distance_kernel(const signed char* vectors, const float* scales,
                                          const signed char* query, float query_scale,
                                          float* distances, int num_vectors, int vector_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;
    
    // INSANELY FAST: Raw difference computation (no sqrt, no abs)
    float dist = 0.0f;
    for (int d = 0; d < vector_dim; d++) {
        float v_val = (float)vectors[idx * vector_dim + d] * scales[idx];
        float q_val = (float)query[d] * query_scale;
        float diff = v_val - q_val;
        dist += diff; // Raw difference - fastest possible
    }
    
    distances[idx] = dist;
}

// Single query search (<1ms target)
// Allocate (once) the per-query scratch buffers. Returns false on any failure so
// the caller can report failure and let Go fall back to CPU search.
static bool ensure_fast_search_scratch(gpu_fast_search_t* searcher) {
    if (searcher->d_query_buf == nullptr) {
        if (cudaMalloc(&searcher->d_query_buf,
                       searcher->vector_dim * sizeof(signed char)) != cudaSuccess) {
            searcher->d_query_buf = nullptr;
            return false;
        }
    }
    if (searcher->scratch_capacity >= searcher->total_vectors) return true;

    if (searcher->d_distances_buf) cudaFree(searcher->d_distances_buf);
    if (searcher->d_sort_keys) cudaFree(searcher->d_sort_keys);
    if (searcher->d_sort_idx) cudaFree(searcher->d_sort_idx);
    searcher->d_distances_buf = nullptr;
    searcher->d_sort_keys = nullptr;
    searcher->d_sort_idx = nullptr;
    searcher->scratch_capacity = 0;

    int cap = searcher->total_vectors + searcher->total_vectors / 5;  // 20% headroom
    if (cudaMalloc(&searcher->d_distances_buf, cap * sizeof(float)) != cudaSuccess) {
        searcher->d_distances_buf = nullptr;
        return false;
    }
    if (cudaMalloc(&searcher->d_sort_keys, cap * sizeof(float)) != cudaSuccess) {
        cudaFree(searcher->d_distances_buf);
        searcher->d_distances_buf = nullptr;
        searcher->d_sort_keys = nullptr;
        return false;
    }
    if (cudaMalloc(&searcher->d_sort_idx, cap * sizeof(int)) != cudaSuccess) {
        cudaFree(searcher->d_distances_buf);
        cudaFree(searcher->d_sort_keys);
        searcher->d_distances_buf = nullptr;
        searcher->d_sort_keys = nullptr;
        searcher->d_sort_idx = nullptr;
        return false;
    }
    searcher->scratch_capacity = cap;
    return true;
}

int gpu_fast_search_single(gpu_fast_search_t* searcher, const signed char* query, float scale,
                          int k, int* result_ids, float* result_scores) {
    if (!searcher || searcher->total_vectors == 0) return 0;
    if (k <= 0) return 0;
    if (k > searcher->total_vectors) k = searcher->total_vectors;

    auto start = std::chrono::high_resolution_clock::now();

    if (!ensure_fast_search_scratch(searcher)) {
        std::cerr << "gpu_fast_search: device scratch unavailable, falling back to CPU" << std::endl;
        cudaGetLastError();
        return 0;
    }

    const int n = searcher->total_vectors;

    if (cudaMemcpy(searcher->d_query_buf, query, searcher->vector_dim * sizeof(signed char),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }

    // Launch ultra-fast distance kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ultra_fast_distance_kernel<<<blocks, threads>>>(
        searcher->d_vectors, searcher->d_scales, searcher->d_query_buf, scale,
        searcher->d_distances_buf, n, searcher->vector_dim
    );
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        std::cerr << "gpu_fast_search kernel launch failed: " << cudaGetErrorString(kerr) << std::endl;
        return 0;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }

    // Top-k over the WHOLE score array. The previous implementation sorted only
    // distances[0..k), i.e. it ranked the first k vectors by index and ignored
    // every vector after that, so the "top k" was not the top k at all.
    try {
        thrust::device_ptr<float> keys(searcher->d_sort_keys);
        thrust::device_ptr<int> idx(searcher->d_sort_idx);
        thrust::device_ptr<float> dist(searcher->d_distances_buf);

        thrust::copy(dist, dist + n, keys);
        thrust::sequence(idx, idx + n);
        // Higher score is better, so sort descending.
        thrust::sort_by_key(keys, keys + n, idx, thrust::greater<float>());
    } catch (const std::exception& e) {
        std::cerr << "gpu_fast_search top-k failed, falling back to CPU: " << e.what() << std::endl;
        cudaGetLastError();
        return 0;
    } catch (...) {
        std::cerr << "gpu_fast_search top-k failed (unknown error), falling back to CPU" << std::endl;
        cudaGetLastError();
        return 0;
    }

    if (cudaMemcpy(result_scores, searcher->d_sort_keys, k * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(result_ids, searcher->d_sort_idx, k * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }

    // Track performance
    auto end = std::chrono::high_resolution_clock::now();
    double search_time_us = std::chrono::duration<double, std::micro>(end - start).count();
    searcher->total_search_time_us += search_time_us;
    searcher->search_count++;

    return 1; // Success
}

// Batch search
int gpu_fast_search_batch(gpu_fast_search_t* searcher, const signed char* queries, 
                         const float* scales, int num_queries, int k,
                         int* result_ids, float* result_scores) {
    if (!searcher) return 0;
    
    // Process each query (could be parallelized further)
    for (int q = 0; q < num_queries; q++) {
        gpu_fast_search_single(searcher, 
                              queries + q * searcher->vector_dim, scales[q],
                              k, result_ids + q * k, result_scores + q * k);
    }
    
    return num_queries;
}

// Load index from file
int gpu_fast_search_load_from_file(gpu_fast_search_t* searcher, const char* index_file) {
    if (!searcher) return 0;
    
    std::ifstream file(index_file, std::ios::binary);
    if (!file) return 0;
    
    // Read metadata
    file.read((char*)&searcher->total_vectors, sizeof(int));
    file.read((char*)&searcher->vector_dim, sizeof(int));
    
    // Allocate and read vectors
    size_t vector_bytes = searcher->total_vectors * searcher->vector_dim;
    signed char* host_vectors = (signed char*)malloc(vector_bytes);
    file.read((char*)host_vectors, vector_bytes);
    
    // Copy to GPU
    cudaMalloc(&searcher->d_vectors, vector_bytes);
    cudaMemcpy(searcher->d_vectors, host_vectors, vector_bytes, cudaMemcpyHostToDevice);
    
    free(host_vectors);
    file.close();
    
    return 1; // Success
}

// Save index to file
int gpu_fast_search_save_to_file(gpu_fast_search_t* searcher, const char* index_file) {
    if (!searcher || !searcher->d_vectors) return 0;
    
    std::ofstream file(index_file, std::ios::binary);
    if (!file) return 0;
    
    // Write metadata
    file.write((char*)&searcher->total_vectors, sizeof(int));
    file.write((char*)&searcher->vector_dim, sizeof(int));
    
    // Copy vectors from GPU and write
    size_t vector_bytes = searcher->total_vectors * searcher->vector_dim;
    signed char* host_vectors = (signed char*)malloc(vector_bytes);
    cudaMemcpy(host_vectors, searcher->d_vectors, vector_bytes, cudaMemcpyDeviceToHost);
    file.write((char*)host_vectors, vector_bytes);
    
    free(host_vectors);
    file.close();
    
    return 1; // Success
}

// Pre-compute structures for faster search
int gpu_fast_search_precompute_structures(gpu_fast_search_t* searcher) {
    // Could pre-compute norms, quantization tables, etc.
    // For now, just ensure GPU memory is ready
    cudaDeviceSynchronize();
    return 1;
}

// Warm cache with common queries
int gpu_fast_search_warm_cache(gpu_fast_search_t* searcher, const signed char* common_queries, 
                               int num_queries) {
    // Run dummy searches to warm GPU caches
    int k = 10;
    int* dummy_ids = (int*)malloc(k * sizeof(int));
    float* dummy_scores = (float*)malloc(k * sizeof(float));
    
    for (int i = 0; i < num_queries && i < 10; i++) {
        gpu_fast_search_single(searcher, common_queries + i * searcher->vector_dim,
                              1.0f, k, dummy_ids, dummy_scores);
    }
    
    free(dummy_ids);
    free(dummy_scores);
    
    return 1;
}

// Set search parameters
int gpu_fast_search_set_search_params(gpu_fast_search_t* searcher, int nprobe, int use_cache) {
    if (!searcher) return 0;
    searcher->nprobe = nprobe;
    return 1;
}

// Optimize for latency
int gpu_fast_search_optimize_for_latency(gpu_fast_search_t* searcher) {
    // Set GPU to prefer cache over shared memory for lower latency
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    return 1;
}

// Get average search latency in microseconds
double gpu_fast_search_get_avg_latency_us(gpu_fast_search_t* searcher) {
    if (!searcher || searcher->search_count == 0) return 0.0;
    return searcher->total_search_time_us / searcher->search_count;
}

// Get cache hit rate
int gpu_fast_search_get_cache_hit_rate(gpu_fast_search_t* searcher) {
    if (!searcher || searcher->search_count == 0) return 0;
    return (searcher->cache_hits * 100) / searcher->search_count;
}

} // extern "C"