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
    int8_t* d_query;  // Preallocated query buffer
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

// Optimized top-k: each thread finds local top-k, then merge
// Much faster than bubble sort O(k*n) -> O(n/threads + threads*k^2)
__global__ void simple_topk_kernel(
    const float* scores,
    int* indices,
    int num_docs,
    int k
) {
    extern __shared__ char smem[];
    float* local_scores = (float*)smem;
    int* local_indices = (int*)(local_scores + blockDim.x * k);

    int tid = threadIdx.x;
    int chunk_size = (num_docs + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, num_docs);

    // Initialize thread-local top-k
    float* my_scores = local_scores + tid * k;
    int* my_indices = local_indices + tid * k;
    for (int i = 0; i < k; i++) {
        my_scores[i] = -1e30f;
        my_indices[i] = -1;
    }

    // Find local top-k for this thread's chunk
    for (int i = start; i < end; i++) {
        float score = scores[i];
        if (score > my_scores[k-1]) {
            // Insert into sorted position
            int pos = k - 1;
            while (pos > 0 && score > my_scores[pos-1]) {
                my_scores[pos] = my_scores[pos-1];
                my_indices[pos] = my_indices[pos-1];
                pos--;
            }
            my_scores[pos] = score;
            my_indices[pos] = i;
        }
    }
    __syncthreads();

    // Thread 0 merges all local top-k results
    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            indices[i] = -1;
        }
        float final_scores[64];  // Max k=64
        for (int i = 0; i < k && i < 64; i++) {
            final_scores[i] = -1e30f;
        }

        // Merge from all threads
        for (int t = 0; t < blockDim.x; t++) {
            float* t_scores = local_scores + t * k;
            int* t_indices = local_indices + t * k;
            for (int i = 0; i < k; i++) {
                if (t_indices[i] >= 0 && t_scores[i] > final_scores[k-1]) {
                    int pos = k - 1;
                    while (pos > 0 && t_scores[i] > final_scores[pos-1]) {
                        final_scores[pos] = final_scores[pos-1];
                        indices[pos] = indices[pos-1];
                        pos--;
                    }
                    final_scores[pos] = t_scores[i];
                    indices[pos] = t_indices[i];
                }
            }
        }
    }
}

// Create simple search context
void* simple_search_create(int max_docs, int dim) {
    // Check for CUDA availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        return nullptr;
    }

    SimpleSearchContext* ctx = new SimpleSearchContext();
    ctx->max_docs = max_docs;
    ctx->dim = dim;

    // Allocate GPU memory with error checking
    err = cudaMalloc(&ctx->d_docs, max_docs * dim * sizeof(int8_t));
    if (err != cudaSuccess) {
        delete ctx;
        return nullptr;
    }

    err = cudaMalloc(&ctx->d_scores, max_docs * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_docs);
        delete ctx;
        return nullptr;
    }

    err = cudaMalloc(&ctx->d_indices, max_docs * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_docs);
        cudaFree(ctx->d_scores);
        delete ctx;
        return nullptr;
    }

    // Preallocate query buffer
    err = cudaMalloc(&ctx->d_query, dim * sizeof(int8_t));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_docs);
        cudaFree(ctx->d_scores);
        cudaFree(ctx->d_indices);
        delete ctx;
        return nullptr;
    }

    // Create stream for async operations
    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        cudaFree(ctx->d_docs);
        cudaFree(ctx->d_scores);
        cudaFree(ctx->d_indices);
        cudaFree(ctx->d_query);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

// Destroy search context
void simple_search_destroy(void* handle) {
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;

    cudaFree(ctx->d_docs);
    cudaFree(ctx->d_scores);
    cudaFree(ctx->d_indices);
    cudaFree(ctx->d_query);
    cudaStreamDestroy(ctx->stream);

    delete ctx;
}

// Add vectors (no limit checking - caller must ensure capacity)
int simple_search_add_vectors(void* handle, const int8_t* docs, int num_docs) {
    if (!handle) return -1;
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;

    // Copy all documents to GPU (overwrite any existing)
    cudaError_t err = cudaMemcpyAsync(
        ctx->d_docs,
        docs,
        num_docs * ctx->dim * sizeof(int8_t),
        cudaMemcpyHostToDevice,
        ctx->stream
    );

    if (err != cudaSuccess) {
        return -1;
    }

    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) {
        return -1;
    }

    return 0;
}

// Perform search (zero-allocation hot path)
int simple_search_query(
    void* handle,
    const int8_t* query,
    int k,
    int* out_indices,
    float* out_scores
) {
    if (!handle) return -1;
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;

    // Copy query to preallocated GPU buffer
    cudaMemcpyAsync(ctx->d_query, query, ctx->dim * sizeof(int8_t),
                    cudaMemcpyHostToDevice, ctx->stream);

    int num_docs = ctx->max_docs;

    // Launch similarity kernel
    int threads_per_block = 256;
    int blocks = (num_docs + threads_per_block - 1) / threads_per_block;

    simple_int8_similarity_kernel<<<blocks, threads_per_block, 0, ctx->stream>>>(
        ctx->d_query, ctx->d_docs, ctx->d_scores, num_docs, ctx->dim);

    // Launch top-k kernel with shared memory for local top-k
    int topk_threads = min(256, num_docs);
    size_t smem_size = topk_threads * k * (sizeof(float) + sizeof(int));
    simple_topk_kernel<<<1, topk_threads, smem_size, ctx->stream>>>(
        ctx->d_scores, ctx->d_indices, num_docs, k);

    // Copy results back
    cudaMemcpyAsync(out_indices, ctx->d_indices, k * sizeof(int),
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaMemcpyAsync(out_scores, ctx->d_scores, k * sizeof(float),
                    cudaMemcpyDeviceToHost, ctx->stream);

    cudaStreamSynchronize(ctx->stream);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

} // extern "C"