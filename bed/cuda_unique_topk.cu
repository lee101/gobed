// CUDA library for unique top-k search with deduplication
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unordered_set>
#include <algorithm>

// Structure to hold search state
struct UniqueTopKSearch {
    int8_t* d_docs;      // Device documents
    float* d_scores;     // Device scores
    int* d_indices;      // Device indices

    int max_docs;
    int dim;
    int max_k;
    int num_docs;

    cublasHandle_t cublas_handle;
};

extern "C" {

// Create search index
void* create_unique_topk_search(int max_docs, int dim, int max_k) {
    UniqueTopKSearch* search = new UniqueTopKSearch;
    search->max_docs = max_docs;
    search->dim = dim;
    search->max_k = max_k;
    search->num_docs = 0;

    // Allocate GPU memory
    size_t docs_size = (size_t)max_docs * dim * sizeof(int8_t);
    cudaMalloc(&search->d_docs, docs_size);
    cudaMalloc(&search->d_scores, max_docs * sizeof(float));
    cudaMalloc(&search->d_indices, max_docs * sizeof(int));

    // Initialize cuBLAS
    cublasCreate(&search->cublas_handle);

    return search;
}

// Destroy search index
void destroy_unique_topk_search(void* handle) {
    UniqueTopKSearch* search = (UniqueTopKSearch*)handle;

    cudaFree(search->d_docs);
    cudaFree(search->d_scores);
    cudaFree(search->d_indices);
    cublasDestroy(search->cublas_handle);

    delete search;
}

// Add documents to index
void add_documents_topk(void* handle, const int8_t* docs, int num_docs, int dim) {
    UniqueTopKSearch* search = (UniqueTopKSearch*)handle;

    if (search->num_docs + num_docs > search->max_docs) {
        printf("Warning: Exceeding max documents\n");
        num_docs = search->max_docs - search->num_docs;
    }

    // Copy documents to GPU
    size_t offset = (size_t)search->num_docs * dim * sizeof(int8_t);
    size_t size = (size_t)num_docs * dim * sizeof(int8_t);
    cudaMemcpy(search->d_docs + search->num_docs * dim, docs, size, cudaMemcpyHostToDevice);

    search->num_docs += num_docs;
}

// Kernel to compute int8 dot products
__global__ void compute_int8_similarities(
    const int8_t* docs,
    const int8_t* query,
    float* scores,
    int* indices,
    int num_docs,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_docs) return;

    // Compute dot product
    int32_t dot = 0;
    const int8_t* doc = docs + idx * dim;

    // Unroll loop for better performance
    int i = 0;
    for (; i < dim - 3; i += 4) {
        dot += (int32_t)doc[i] * query[i];
        dot += (int32_t)doc[i+1] * query[i+1];
        dot += (int32_t)doc[i+2] * query[i+2];
        dot += (int32_t)doc[i+3] * query[i+3];
    }
    for (; i < dim; i++) {
        dot += (int32_t)doc[i] * query[i];
    }

    // Normalize to [-1, 1] range
    scores[idx] = (float)dot / (127.0f * dim);
    indices[idx] = idx;
}

// Structure for sorting
struct ScoreIndex {
    float score;
    int index;
};

// Comparison function for sorting
bool compare_scores(const ScoreIndex& a, const ScoreIndex& b) {
    return a.score > b.score; // Descending order
}

// Search for top-k unique results
int search_topk_unique(
    void* handle,
    const int8_t* query,
    int dim,
    int k,
    int* out_indices,
    float* out_scores
) {
    UniqueTopKSearch* search = (UniqueTopKSearch*)handle;

    if (search->num_docs == 0) {
        return 0;
    }

    // Copy query to GPU
    int8_t* d_query;
    cudaMalloc(&d_query, dim * sizeof(int8_t));
    cudaMemcpy(d_query, query, dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Compute similarities
    int threads = 256;
    int blocks = (search->num_docs + threads - 1) / threads;

    compute_int8_similarities<<<blocks, threads>>>(
        search->d_docs,
        d_query,
        search->d_scores,
        search->d_indices,
        search->num_docs,
        dim
    );

    // Copy scores and indices back to host
    std::vector<float> h_scores(search->num_docs);
    std::vector<int> h_indices(search->num_docs);

    cudaMemcpy(h_scores.data(), search->d_scores,
               search->num_docs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices.data(), search->d_indices,
               search->num_docs * sizeof(int), cudaMemcpyDeviceToHost);

    // Create score-index pairs and sort
    std::vector<ScoreIndex> score_pairs(search->num_docs);
    for (int i = 0; i < search->num_docs; i++) {
        score_pairs[i].score = h_scores[i];
        score_pairs[i].index = h_indices[i];
    }

    // Sort by score
    std::sort(score_pairs.begin(), score_pairs.end(), compare_scores);

    // Deduplicate results based on document index
    // This ensures we don't return the same line multiple times
    std::unordered_set<int> seen_indices;
    int num_results = 0;

    for (const auto& pair : score_pairs) {
        if (num_results >= k) break;

        // Check if we've seen this document index
        if (seen_indices.find(pair.index) == seen_indices.end()) {
            out_indices[num_results] = pair.index;
            out_scores[num_results] = pair.score;
            seen_indices.insert(pair.index);
            num_results++;
        }
    }

    // Clean up
    cudaFree(d_query);

    return num_results;
}

} // extern "C"