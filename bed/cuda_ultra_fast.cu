// Ultra-optimized CUDA kernels for maximum performance on RTX 3090
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

namespace cg = cooperative_groups;

// RTX 3090 optimal parameters
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 16
#define THREADS_PER_BLOCK 512
#define VECTOR_DIM 512
#define INT8_VECTOR_SIZE 4  // Process 4 int8s at once
#define TILE_SIZE 256
#define SHARED_MEM_SIZE 49152  // 48KB shared memory
#define L2_CACHE_SIZE 6291456  // 6MB L2 cache

// Memory alignment for coalesced access
#define ALIGN_BYTES 128
#define ALIGN_UP(x) (((x) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES)

// GPU memory pool with unified memory
struct UltraGPUIndex {
    // Primary data structures
    int8_t* __restrict__ embeddings;        // Aligned int8 embeddings
    float* __restrict__ norms;              // Pre-computed L2 norms
    float* __restrict__ inv_norms;          // Pre-computed 1/norm for faster division

    // Async indexing structures
    int8_t* __restrict__ staging_buffer;    // Pinned memory for async transfers
    cudaStream_t index_stream;               // Dedicated stream for indexing
    cudaEvent_t index_event;                 // Event for synchronization

    // Search structures
    float* __restrict__ score_buffer;       // Reusable score buffer
    int* __restrict__ index_buffer;         // Reusable index buffer

    // Metadata
    size_t num_docs;
    size_t max_docs;
    int vector_dim;

    // cuBLAS handle for GEMM operations
    cublasHandle_t cublas_handle;

    // Texture memory for frequently accessed data
    cudaTextureObject_t embedding_tex;
    cudaTextureObject_t norm_tex;
};

// Ultra-fast int8 dot product using vector instructions
__device__ __forceinline__ float int8_dot_product_vectorized(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int dim
) {
    float sum = 0.0f;

    // Process 4 int8s at once using int32 loads
    const int32_t* a32 = reinterpret_cast<const int32_t*>(a);
    const int32_t* b32 = reinterpret_cast<const int32_t*>(b);

    #pragma unroll 8
    for (int i = 0; i < dim / 4; i++) {
        int32_t va = a32[i];
        int32_t vb = b32[i];

        // Extract and multiply 4 int8 values at once
        sum += __int2float_rn((va & 0xFF) * (vb & 0xFF));
        sum += __int2float_rn(((va >> 8) & 0xFF) * ((vb >> 8) & 0xFF));
        sum += __int2float_rn(((va >> 16) & 0xFF) * ((vb >> 16) & 0xFF));
        sum += __int2float_rn(((va >> 24) & 0xFF) * ((vb >> 24) & 0xFF));
    }

    return sum;
}

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Ultra-optimized similarity kernel with tensor cores (simulated for int8)
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
ultra_similarity_kernel(
    const int8_t* __restrict__ query,
    const int8_t* __restrict__ embeddings,
    const float* __restrict__ inv_norms,
    float* __restrict__ scores,
    int num_docs,
    int dim
) {
    // Shared memory for query caching
    __shared__ int8_t shared_query[512];
    __shared__ float shared_scores[THREADS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int doc_idx = blockIdx.x * THREADS_PER_BLOCK + tid;

    // Cooperative loading of query to shared memory
    if (tid < dim) {
        shared_query[tid] = query[tid];
    }
    __syncthreads();

    if (doc_idx >= num_docs) {
        shared_scores[tid] = -FLT_MAX;
        return;
    }

    // Compute dot product using vectorized operations
    const int8_t* doc_embedding = embeddings + doc_idx * dim;
    float dot_product = 0.0f;
    float query_norm_sq = 0.0f;

    // Process multiple elements per thread
    #pragma unroll 16
    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        int8_t q = shared_query[i];
        int8_t d = doc_embedding[i];

        // Use FMA instructions for better throughput
        dot_product = __fmaf_rn(__int2float_rn(q), __int2float_rn(d), dot_product);
        query_norm_sq = __fmaf_rn(__int2float_rn(q), __int2float_rn(q), query_norm_sq);
    }

    // Warp-level reduction
    dot_product = warp_reduce_sum(dot_product);
    query_norm_sq = warp_reduce_sum(query_norm_sq);

    // First thread in warp writes result
    if (lane_id == 0) {
        float inv_query_norm = __frsqrt_rn(query_norm_sq);
        float score = dot_product * inv_query_norm * inv_norms[doc_idx];
        shared_scores[warp_id] = score;
    }

    __syncthreads();

    // Coalesced write
    if (tid < gridDim.x && tid < num_docs) {
        scores[blockIdx.x * THREADS_PER_BLOCK + tid] = shared_scores[tid];
    }
}

// Batch processing with multi-stream execution
__global__ void batch_ultra_similarity(
    const int8_t* __restrict__ queries,
    const int8_t* __restrict__ embeddings,
    const float* __restrict__ inv_norms,
    float* __restrict__ scores,
    int batch_size,
    int num_docs,
    int dim
) {
    extern __shared__ char shared_mem[];
    int8_t* shared_queries = (int8_t*)shared_mem;

    const int query_idx = blockIdx.y;
    const int doc_base = blockIdx.x * blockDim.x;
    const int doc_idx = doc_base + threadIdx.x;
    const int tid = threadIdx.x;

    if (query_idx >= batch_size) return;

    // Load query to shared memory cooperatively
    const int8_t* query = queries + query_idx * dim;
    for (int i = tid; i < dim; i += blockDim.x) {
        shared_queries[i] = query[i];
    }
    __syncthreads();

    if (doc_idx >= num_docs) return;

    // Compute similarity with vectorized operations
    const int8_t* doc = embeddings + doc_idx * dim;
    float dot = 0.0f, q_norm = 0.0f;

    #pragma unroll 8
    for (int i = 0; i < dim; i += 4) {
        int32_t q4 = *reinterpret_cast<const int32_t*>(&shared_queries[i]);
        int32_t d4 = *reinterpret_cast<const int32_t*>(&doc[i]);

        for (int j = 0; j < 4; j++) {
            int8_t qv = (q4 >> (j * 8)) & 0xFF;
            int8_t dv = (d4 >> (j * 8)) & 0xFF;
            dot = __fmaf_rn(qv, dv, dot);
            q_norm = __fmaf_rn(qv, qv, q_norm);
        }
    }

    float score = dot * __frsqrt_rn(q_norm) * inv_norms[doc_idx];
    scores[query_idx * num_docs + doc_idx] = score;
}

// Async indexing with memory mapping support
extern "C" void* ultra_gpu_create(int max_docs, int dim) {
    UltraGPUIndex* index = new UltraGPUIndex();

    index->max_docs = max_docs;
    index->vector_dim = dim;
    index->num_docs = 0;

    // Allocate aligned GPU memory
    size_t embedding_size = ALIGN_UP(max_docs * dim * sizeof(int8_t));
    size_t norm_size = ALIGN_UP(max_docs * sizeof(float));

    cudaMalloc(&index->embeddings, embedding_size);
    cudaMalloc(&index->norms, norm_size);
    cudaMalloc(&index->inv_norms, norm_size);
    cudaMalloc(&index->score_buffer, max_docs * sizeof(float));
    cudaMalloc(&index->index_buffer, max_docs * sizeof(int));

    // Allocate pinned memory for async transfers
    cudaMallocHost(&index->staging_buffer, 1024 * dim * sizeof(int8_t));

    // Create dedicated stream for indexing
    cudaStreamCreate(&index->index_stream);
    cudaEventCreate(&index->index_event);

    // Initialize cuBLAS
    cublasCreate(&index->cublas_handle);
    cublasSetStream(index->cublas_handle, index->index_stream);

    // Set L2 cache persistence
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, L2_CACHE_SIZE);

    return index;
}

// Async batch indexing
extern "C" int ultra_gpu_add_batch_async(
    void* handle,
    const int8_t* embeddings,
    int num_vectors
) {
    UltraGPUIndex* index = (UltraGPUIndex*)handle;

    if (index->num_docs + num_vectors > index->max_docs) {
        return -1;
    }

    size_t offset = index->num_docs * index->vector_dim;
    size_t size = num_vectors * index->vector_dim * sizeof(int8_t);

    // Async copy to GPU
    cudaMemcpyAsync(
        index->embeddings + offset,
        embeddings,
        size,
        cudaMemcpyHostToDevice,
        index->index_stream
    );

    // Compute norms asynchronously
    dim3 blocks((num_vectors + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    compute_norms_kernel<<<blocks, threads, 0, index->index_stream>>>(
        index->embeddings + offset,
        index->norms + index->num_docs,
        index->inv_norms + index->num_docs,
        num_vectors,
        index->vector_dim
    );

    // Record event for synchronization
    cudaEventRecord(index->index_event, index->index_stream);

    index->num_docs += num_vectors;
    return 0;
}

// Ultra-fast search with top-k selection
extern "C" int ultra_gpu_search(
    void* handle,
    const int8_t* query,
    int k,
    int* indices,
    float* scores
) {
    UltraGPUIndex* index = (UltraGPUIndex*)handle;

    // Ensure indexing is complete
    cudaEventSynchronize(index->index_event);

    // Allocate query on GPU
    int8_t* d_query;
    cudaMalloc(&d_query, index->vector_dim * sizeof(int8_t));
    cudaMemcpy(d_query, query, index->vector_dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch ultra-optimized kernel
    dim3 blocks((index->num_docs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    ultra_similarity_kernel<<<blocks, threads>>>(
        d_query,
        index->embeddings,
        index->inv_norms,
        index->score_buffer,
        index->num_docs,
        index->vector_dim
    );

    // Use CUB for efficient top-k selection
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        index->score_buffer, index->score_buffer,
        index->index_buffer, index->index_buffer,
        index->num_docs
    );

    void* temp_storage;
    cudaMalloc(&temp_storage, temp_storage_bytes);

    // Initialize indices
    thrust::sequence(thrust::device, index->index_buffer, index->index_buffer + index->num_docs);

    cub::DeviceRadixSort::SortPairsDescending(
        temp_storage, temp_storage_bytes,
        index->score_buffer, index->score_buffer,
        index->index_buffer, index->index_buffer,
        index->num_docs
    );

    // Copy top-k results
    cudaMemcpy(indices, index->index_buffer, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(scores, index->score_buffer, k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(temp_storage);

    return 0;
}

// Cleanup
extern "C" void ultra_gpu_destroy(void* handle) {
    UltraGPUIndex* index = (UltraGPUIndex*)handle;

    cudaFree(index->embeddings);
    cudaFree(index->norms);
    cudaFree(index->inv_norms);
    cudaFree(index->score_buffer);
    cudaFree(index->index_buffer);
    cudaFreeHost(index->staging_buffer);

    cudaStreamDestroy(index->index_stream);
    cudaEventDestroy(index->index_event);
    cublasDestroy(index->cublas_handle);

    delete index;
}

// Helper kernel for norm computation
__global__ void compute_norms_kernel(
    const int8_t* __restrict__ embeddings,
    float* __restrict__ norms,
    float* __restrict__ inv_norms,
    int num_docs,
    int dim
) {
    const int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (doc_idx >= num_docs) return;

    const int8_t* embedding = embeddings + doc_idx * dim;
    float norm_sq = 0.0f;

    #pragma unroll 16
    for (int i = 0; i < dim; i++) {
        float val = __int2float_rn(embedding[i]);
        norm_sq = __fmaf_rn(val, val, norm_sq);
    }

    float norm = __fsqrt_rn(norm_sq);
    norms[doc_idx] = norm;
    inv_norms[doc_idx] = __frcp_rn(norm + 1e-8f);
}