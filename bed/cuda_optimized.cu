// cuda_optimized.cu - RTX 3090 optimized CUDA kernels for bed
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// RTX 3090 optimal parameters
#define WARP_SIZE 32
#define MAX_THREADS 1024
#define TILE_SIZE 128
#define BATCH_SIZE 2048  // Optimal for 3090
#define VECTOR_DIM 384
#define MAX_DOCS 10000000  // 10M documents
#define NUM_STREAMS 4

// RTX 3090 optimized parameters
#define USE_FAST_MATH 1

// GPU memory pool
struct GPUMemoryPool {
    int8_t* doc_embeddings;      // Document embeddings
    float* doc_norms;            // Pre-computed norms
    float* scores;               // Similarity scores
    int* indices;                // Result indices

    // Pinned memory for faster transfers
    int8_t* pinned_input;
    float* pinned_output;

    size_t total_docs;
    size_t max_docs;

    // CUDA streams for overlap
    cudaStream_t streams[NUM_STREAMS];

    // cuBLAS handle for matrix ops
    cublasHandle_t cublas_handle;
};

// Optimized cosine similarity kernel for int8
__global__ void cosine_similarity_int8_optimized(
    const int8_t* __restrict__ query,
    const int8_t* __restrict__ docs,
    const float* __restrict__ doc_norms,
    float* __restrict__ scores,
    int num_docs,
    int dim
) {
    // Use shared memory for query caching
    extern __shared__ int8_t shared_query[];

    const int tid = threadIdx.x;
    const int doc_idx = blockIdx.x * blockDim.x + tid;

    // Cooperative load of query to shared memory
    if (tid < dim) {
        shared_query[tid] = query[tid];
    }
    __syncthreads();

    if (doc_idx >= num_docs) return;

    // Compute dot product using vector instructions
    float dot_product = 0.0f;
    float query_norm_sq = 0.0f;

    // Unroll loop for better performance
    #pragma unroll 8
    for (int i = tid; i < dim; i += blockDim.x) {
        int8_t q = shared_query[i];
        int8_t d = docs[doc_idx * dim + i];

        dot_product += float(q) * float(d);
        query_norm_sq += float(q) * float(q);
    }

    // Warp-level reduction using shuffle instructions
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        dot_product += __shfl_down_sync(0xFFFFFFFF, dot_product, offset);
        query_norm_sq += __shfl_down_sync(0xFFFFFFFF, query_norm_sq, offset);
    }

    // Write result for first thread in warp
    if (tid % WARP_SIZE == 0) {
        float query_norm = sqrtf(query_norm_sq);
        float doc_norm = doc_norms[doc_idx];
        scores[doc_idx] = dot_product / (query_norm * doc_norm + 1e-8f);
    }
}

// Batch processing kernel for multiple queries
__global__ void batch_cosine_similarity(
    const int8_t* __restrict__ queries,  // [batch_size, dim]
    const int8_t* __restrict__ docs,     // [num_docs, dim]
    const float* __restrict__ doc_norms,
    float* __restrict__ scores,          // [batch_size, num_docs]
    int batch_size,
    int num_docs,
    int dim
) {
    const int query_idx = blockIdx.y;
    const int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (query_idx >= batch_size || doc_idx >= num_docs) return;

    const int8_t* query = queries + query_idx * dim;
    const int8_t* doc = docs + doc_idx * dim;

    // Compute similarity
    float dot_product = 0.0f;
    float query_norm = 0.0f;

    for (int i = 0; i < dim; ++i) {
        float q = float(query[i]);
        float d = float(doc[i]);
        dot_product += q * d;
        query_norm += q * q;
    }

    query_norm = sqrtf(query_norm);
    float score = dot_product / (query_norm * doc_norms[doc_idx] + 1e-8f);

    scores[query_idx * num_docs + doc_idx] = score;
}

// Pre-compute document norms for efficiency
__global__ void compute_norms(
    const int8_t* __restrict__ docs,
    float* __restrict__ norms,
    int num_docs,
    int dim
) {
    const int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (doc_idx >= num_docs) return;

    float norm_sq = 0.0f;
    const int8_t* doc = docs + doc_idx * dim;

    #pragma unroll 8
    for (int i = 0; i < dim; ++i) {
        float val = float(doc[i]);
        norm_sq += val * val;
    }

    norms[doc_idx] = sqrtf(norm_sq);
}

// Top-K selection using CUB
__global__ void select_top_k(
    const float* scores,
    int* indices,
    float* top_scores,
    int num_docs,
    int k
) {
    // Simplified top-k - in production use CUB RadixSort
    extern __shared__ float shared_scores[];

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Initialize indices
    if (tid < k) {
        indices[tid] = tid;
        top_scores[tid] = scores[tid];
    }
    __syncthreads();

    // Find top-k using simple selection
    for (int i = k + tid; i < num_docs; i += stride) {
        float score = scores[i];

        // Find minimum in current top-k
        int min_idx = 0;
        float min_score = top_scores[0];
        for (int j = 1; j < k; ++j) {
            if (top_scores[j] < min_score) {
                min_score = top_scores[j];
                min_idx = j;
            }
        }

        // Replace if better
        if (score > min_score) {
            top_scores[min_idx] = score;
            indices[min_idx] = i;
        }
    }
}

extern "C" {

// Initialize GPU memory pool
void* gpu_pool_create(int max_docs, int dim) {
    GPUMemoryPool* pool = (GPUMemoryPool*)malloc(sizeof(GPUMemoryPool));

    pool->max_docs = max_docs;
    pool->total_docs = 0;

    // Allocate GPU memory (use 20GB for documents on 3090)
    size_t doc_mem_size = (size_t)max_docs * dim * sizeof(int8_t);
    size_t norm_mem_size = max_docs * sizeof(float);
    size_t score_mem_size = BATCH_SIZE * max_docs * sizeof(float);

    cudaMalloc(&pool->doc_embeddings, doc_mem_size);
    cudaMalloc(&pool->doc_norms, norm_mem_size);
    cudaMalloc(&pool->scores, score_mem_size);
    cudaMalloc(&pool->indices, BATCH_SIZE * 100 * sizeof(int));

    // Allocate pinned memory for faster transfers
    cudaMallocHost(&pool->pinned_input, BATCH_SIZE * dim * sizeof(int8_t));
    cudaMallocHost(&pool->pinned_output, BATCH_SIZE * 100 * sizeof(float));

    // Create streams for concurrent operations
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&pool->streams[i]);
    }

    // Initialize cuBLAS
    cublasCreate(&pool->cublas_handle);

    printf("GPU Pool initialized: %.2f GB allocated on RTX 3090\n",
           (doc_mem_size + norm_mem_size + score_mem_size) / (1024.0 * 1024.0 * 1024.0));

    return pool;
}

// Add documents in batches for efficiency
int gpu_pool_add_batch(void* handle, const int8_t* docs, int num_docs, int dim) {
    GPUMemoryPool* pool = (GPUMemoryPool*)handle;

    if (pool->total_docs + num_docs > pool->max_docs) {
        return -1;
    }

    // Copy to GPU using pinned memory and streams
    size_t offset = pool->total_docs * dim;
    size_t batch_bytes = num_docs * dim * sizeof(int8_t);

    // Use multiple streams for large batches
    int docs_per_stream = (num_docs + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int s = 0; s < NUM_STREAMS && s * docs_per_stream < num_docs; ++s) {
        int stream_offset = s * docs_per_stream;
        int stream_docs = (docs_per_stream < num_docs - stream_offset) ? docs_per_stream : (num_docs - stream_offset);
        size_t stream_bytes = stream_docs * dim * sizeof(int8_t);

        cudaMemcpyAsync(
            pool->doc_embeddings + offset + stream_offset * dim,
            docs + stream_offset * dim,
            stream_bytes,
            cudaMemcpyHostToDevice,
            pool->streams[s % NUM_STREAMS]
        );

        // Compute norms in parallel
        int blocks = (stream_docs + 255) / 256;
        compute_norms<<<blocks, 256, 0, pool->streams[s % NUM_STREAMS]>>>(
            pool->doc_embeddings + offset + stream_offset * dim,
            pool->doc_norms + pool->total_docs + stream_offset,
            stream_docs,
            dim
        );
    }

    // Synchronize streams
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(pool->streams[s]);
    }

    pool->total_docs += num_docs;
    return 0;
}

// Optimized batch search using all GPU resources
int gpu_pool_search_batch(
    void* handle,
    const int8_t* queries,
    int batch_size,
    int k,
    int* out_indices,
    float* out_scores
) {
    GPUMemoryPool* pool = (GPUMemoryPool*)handle;

    // Optimal configuration for RTX 3090
    int threads_per_block = 256;
    int blocks_per_query = (pool->total_docs + threads_per_block - 1) / threads_per_block;

    dim3 grid(blocks_per_query, batch_size);
    dim3 block(threads_per_block);

    // Copy queries to GPU using pinned memory
    cudaMemcpyAsync(
        pool->pinned_input,
        queries,
        batch_size * VECTOR_DIM * sizeof(int8_t),
        cudaMemcpyHostToDevice,
        pool->streams[0]
    );

    // Launch batch similarity kernel
    batch_cosine_similarity<<<grid, block, 0, pool->streams[0]>>>(
        pool->pinned_input,
        pool->doc_embeddings,
        pool->doc_norms,
        pool->scores,
        batch_size,
        pool->total_docs,
        VECTOR_DIM
    );

    // Top-K selection for each query
    for (int q = 0; q < batch_size; ++q) {
        int stream_idx = q % NUM_STREAMS;

        select_top_k<<<1, 512, k * sizeof(float), pool->streams[stream_idx]>>>(
            pool->scores + q * pool->total_docs,
            pool->indices + q * k,
            out_scores + q * k,
            pool->total_docs,
            k
        );
    }

    // Copy results back
    cudaMemcpyAsync(
        out_indices,
        pool->indices,
        batch_size * k * sizeof(int),
        cudaMemcpyDeviceToHost,
        pool->streams[0]
    );

    cudaMemcpyAsync(
        out_scores,
        pool->scores,
        batch_size * k * sizeof(float),
        cudaMemcpyDeviceToHost,
        pool->streams[0]
    );

    cudaStreamSynchronize(pool->streams[0]);

    return 0;
}

// Single query search (for compatibility)
int gpu_pool_search(
    void* handle,
    const int8_t* query,
    int k,
    int* out_indices,
    float* out_scores
) {
    return gpu_pool_search_batch(handle, query, 1, k, out_indices, out_scores);
}

// Cleanup
void gpu_pool_destroy(void* handle) {
    GPUMemoryPool* pool = (GPUMemoryPool*)handle;

    cudaFree(pool->doc_embeddings);
    cudaFree(pool->doc_norms);
    cudaFree(pool->scores);
    cudaFree(pool->indices);

    cudaFreeHost(pool->pinned_input);
    cudaFreeHost(pool->pinned_output);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(pool->streams[i]);
    }

    cublasDestroy(pool->cublas_handle);

    free(pool);
}

// Get GPU memory usage
void gpu_memory_info(size_t* free_mem, size_t* total_mem) {
    cudaMemGetInfo(free_mem, total_mem);
}

} // extern "C"