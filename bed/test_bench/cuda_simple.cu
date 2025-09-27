// Simple CUDA library for scalable document search
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

// Simple search context without document count restrictions
struct SimpleSearchContext {
    int8_t* d_docs;
    float* d_scores;
    int* d_indices;
    int max_docs;
    int dim;
    cudaStream_t stream;
};

extern "C" {

// Fast int8 dot product kernel
__global__ void simple_int8_similarity_kernel(
    const int8_t* query,
    const int8_t* docs,
    float* scores,
    int num_docs,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_docs) return;

    const int8_t* doc = docs + idx * dim;
    int32_t dot = 0;

    // Vectorized accumulation (4 elements at a time)
    for (int i = 0; i < dim; i += 4) {
        if (i + 3 < dim) {
            dot += (int32_t)doc[i] * query[i] +
                   (int32_t)doc[i+1] * query[i+1] +
                   (int32_t)doc[i+2] * query[i+2] +
                   (int32_t)doc[i+3] * query[i+3];
        } else {
            // Handle remaining elements
            for (int j = i; j < dim; j++) {
                dot += (int32_t)doc[j] * query[j];
            }
            break;
        }
    }

    scores[idx] = (float)dot;
}

// Simple top-k selection kernel
__global__ void simple_topk_kernel(
    const float* scores,
    int* indices,
    int num_docs,
    int k
) {
    // Initialize indices
    for (int i = threadIdx.x; i < num_docs; i += blockDim.x) {
        indices[i] = i;
    }
    __syncthreads();

    // Simple bubble sort for top-k (works well for small k)
    for (int i = 0; i < k && i < num_docs; i++) {
        for (int j = threadIdx.x + i + 1; j < num_docs; j += blockDim.x) {
            if (scores[indices[j]] > scores[indices[i]]) {
                // Swap indices
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        __syncthreads();
    }
}

// Create simple search context
void* simple_search_create(int max_docs, int dim) {
    SimpleSearchContext* ctx = new SimpleSearchContext();
    ctx->max_docs = max_docs;
    ctx->dim = dim;

    // Allocate GPU memory
    cudaMalloc(&ctx->d_docs, max_docs * dim * sizeof(int8_t));
    cudaMalloc(&ctx->d_scores, max_docs * sizeof(float));
    cudaMalloc(&ctx->d_indices, max_docs * sizeof(int));

    // Create stream for async operations
    cudaStreamCreate(&ctx->stream);

    return ctx;
}

// Destroy search context
void simple_search_destroy(void* handle) {
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;

    cudaFree(ctx->d_docs);
    cudaFree(ctx->d_scores);
    cudaFree(ctx->d_indices);
    cudaStreamDestroy(ctx->stream);

    delete ctx;
}

// Add vectors (no limit checking - caller must ensure capacity)
int simple_search_add_vectors(void* handle, const int8_t* docs, int num_docs) {
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;

    // Copy all documents to GPU (overwrite any existing)
    cudaMemcpyAsync(
        ctx->d_docs,
        docs,
        num_docs * ctx->dim * sizeof(int8_t),
        cudaMemcpyHostToDevice,
        ctx->stream
    );

    return num_docs;
}

// Perform search
int simple_search_query(
    void* handle,
    const int8_t* query,
    int k,
    int* out_indices,
    float* out_scores
) {
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;

    // Allocate temporary GPU memory for query
    int8_t* d_query;
    cudaMalloc(&d_query, ctx->dim * sizeof(int8_t));

    // Copy query to GPU
    cudaMemcpyAsync(
        d_query,
        query,
        ctx->dim * sizeof(int8_t),
        cudaMemcpyHostToDevice,
        ctx->stream
    );

    // For this simple version, we'll assume all slots are used
    // In a real implementation, you'd track the actual number of documents
    int num_docs = ctx->max_docs;

    // Launch similarity kernel
    int threads_per_block = 256;
    int blocks = (num_docs + threads_per_block - 1) / threads_per_block;

    simple_int8_similarity_kernel<<<blocks, threads_per_block, 0, ctx->stream>>>(
        d_query,
        ctx->d_docs,
        ctx->d_scores,
        num_docs,
        ctx->dim
    );

    // Launch top-k kernel
    simple_topk_kernel<<<1, min(1024, num_docs), 0, ctx->stream>>>(
        ctx->d_scores,
        ctx->d_indices,
        num_docs,
        k
    );

    // Copy results back
    cudaMemcpyAsync(
        out_indices,
        ctx->d_indices,
        k * sizeof(int),
        cudaMemcpyDeviceToHost,
        ctx->stream
    );

    cudaMemcpyAsync(
        out_scores,
        ctx->d_scores,
        k * sizeof(float),
        cudaMemcpyDeviceToHost,
        ctx->stream
    );

    // Synchronize to ensure completion
    cudaStreamSynchronize(ctx->stream);

    // Clean up
    cudaFree(d_query);

    return k;
}

} // extern "C"