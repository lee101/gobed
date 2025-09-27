// Simplified CUDA implementation with basic topk
#include <cuda_runtime.h>
#include <algorithm>

struct SimpleSearchContext {
    int8_t* d_embeddings;
    int8_t* d_query;
    float* d_scores;
    int num_vectors;
    int dim;
};

extern "C" {

// Simple int8 dot product kernel
__global__ void simple_similarity_kernel(
    const int8_t* query,
    const int8_t* embeddings,
    float* scores,
    int num_vectors,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;

    const int8_t* emb = embeddings + idx * dim;
    int sum = 0;

    for (int i = 0; i < dim; i++) {
        sum += query[i] * emb[i];
    }

    scores[idx] = (float)sum;
}

void* simple_search_create(int max_vectors, int dim) {
    SimpleSearchContext* ctx = new SimpleSearchContext();
    ctx->num_vectors = 0;
    ctx->dim = dim;

    cudaMalloc(&ctx->d_embeddings, max_vectors * dim * sizeof(int8_t));
    cudaMalloc(&ctx->d_query, dim * sizeof(int8_t));
    cudaMalloc(&ctx->d_scores, max_vectors * sizeof(float));

    return ctx;
}

void simple_search_destroy(void* handle) {
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;
    cudaFree(ctx->d_embeddings);
    cudaFree(ctx->d_query);
    cudaFree(ctx->d_scores);
    delete ctx;
}

int simple_search_add_vectors(void* handle, const int8_t* vectors, int num_vectors) {
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;
    ctx->num_vectors = num_vectors;

    cudaMemcpy(
        ctx->d_embeddings,
        vectors,
        num_vectors * ctx->dim * sizeof(int8_t),
        cudaMemcpyHostToDevice
    );

    return 0;
}

int simple_search_query(void* handle, const int8_t* query, int k, int* indices, float* scores) {
    SimpleSearchContext* ctx = (SimpleSearchContext*)handle;

    // Copy query to GPU
    cudaMemcpy(
        ctx->d_query,
        query,
        ctx->dim * sizeof(int8_t),
        cudaMemcpyHostToDevice
    );

    // Launch similarity kernel
    int threads_per_block = 256;
    int blocks = (ctx->num_vectors + threads_per_block - 1) / threads_per_block;

    simple_similarity_kernel<<<blocks, threads_per_block>>>(
        ctx->d_query,
        ctx->d_embeddings,
        ctx->d_scores,
        ctx->num_vectors,
        ctx->dim
    );

    cudaDeviceSynchronize();

    // Copy scores back to CPU for simple topk
    float* h_scores = new float[ctx->num_vectors];
    cudaMemcpy(
        h_scores,
        ctx->d_scores,
        ctx->num_vectors * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Simple CPU-based topk selection
    std::vector<std::pair<float, int>> score_index_pairs;
    for (int i = 0; i < ctx->num_vectors; i++) {
        score_index_pairs.push_back({h_scores[i], i});
    }

    // Sort by score (descending)
    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Fill results
    for (int i = 0; i < k && i < ctx->num_vectors; i++) {
        scores[i] = score_index_pairs[i].first;
        indices[i] = score_index_pairs[i].second;
    }

    delete[] h_scores;
    return 0;
}

} // extern "C"