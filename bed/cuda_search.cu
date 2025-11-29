// cuda_search.cu - Simplified CUDA search implementation
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

extern "C" {

// Simple CUDA search structure
struct CUDASearch {
    int8_t* d_database;    // GPU database vectors
    float* d_scores;       // GPU scores buffer
    int num_vectors;
    int dim;
    cublasHandle_t cublas_handle;
    bool initialized;
};

// Create CUDA search index
void* cuda_search_create(int max_vectors, int dim) {
    CUDASearch* search = new CUDASearch();
    search->num_vectors = 0;
    search->dim = dim;
    search->initialized = false;

    // Allocate GPU memory
    size_t db_size = max_vectors * dim * sizeof(int8_t);
    size_t scores_size = max_vectors * sizeof(float);

    if (cudaMalloc(&search->d_database, db_size) != cudaSuccess) {
        delete search;
        return nullptr;
    }

    if (cudaMalloc(&search->d_scores, scores_size) != cudaSuccess) {
        cudaFree(search->d_database);
        delete search;
        return nullptr;
    }

    // Create cuBLAS handle
    if (cublasCreate(&search->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        cudaFree(search->d_database);
        cudaFree(search->d_scores);
        delete search;
        return nullptr;
    }

    search->initialized = true;
    return search;
}

// Destroy CUDA search index
void cuda_search_destroy(void* handle) {
    if (!handle) return;

    CUDASearch* search = static_cast<CUDASearch*>(handle);
    if (search->initialized) {
        cublasDestroy(search->cublas_handle);
        cudaFree(search->d_database);
        cudaFree(search->d_scores);
    }
    delete search;
}

// Add vectors to database
int cuda_search_add_vectors(void* handle, const int8_t* vectors, int num_vectors) {
    if (!handle) return -1;

    CUDASearch* search = static_cast<CUDASearch*>(handle);

    // Copy vectors to GPU
    size_t size = num_vectors * search->dim * sizeof(int8_t);
    cudaError_t err = cudaMemcpy(search->d_database, vectors, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        return -1;
    }

    search->num_vectors = num_vectors;
    return 0;
}

// Simple int8 dot product kernel
__global__ void compute_int8_dot_products(
    const int8_t* query,
    const int8_t* database,
    float* scores,
    int num_vectors,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;

    const int8_t* db_vec = database + idx * dim;
    int32_t sum = 0;

    // Compute dot product
    #pragma unroll 8
    for (int i = 0; i < dim; i++) {
        sum += (int32_t)query[i] * (int32_t)db_vec[i];
    }

    // Normalize based on INT8 range and dimension
    scores[idx] = (float)sum / (127.0f * 127.0f);  // Normalize INT8 dot product
}

// Search for similar vectors
int cuda_search_query(
    void* handle,
    const int8_t* query,
    int k,
    int* indices,
    float* scores
) {
    if (!handle) return -1;

    CUDASearch* search = static_cast<CUDASearch*>(handle);

    // Allocate GPU memory for query
    int8_t* d_query;
    cudaMalloc(&d_query, search->dim * sizeof(int8_t));
    cudaMemcpy(d_query, query, search->dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (search->num_vectors + threads - 1) / threads;

    compute_int8_dot_products<<<blocks, threads>>>(
        d_query,
        search->d_database,
        search->d_scores,
        search->num_vectors,
        search->dim
    );

    cudaDeviceSynchronize();

    // Copy scores back to host
    float* h_scores = new float[search->num_vectors];
    cudaMemcpy(h_scores, search->d_scores, search->num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    // Find top-k (simple CPU sort for now)
    std::vector<std::pair<float, int>> score_pairs;
    for (int i = 0; i < search->num_vectors; i++) {
        score_pairs.push_back({h_scores[i], i});
    }

    std::partial_sort(
        score_pairs.begin(),
        score_pairs.begin() + std::min(k, search->num_vectors),
        score_pairs.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;  // Descending order
        }
    );

    // Fill results
    int actual_k = std::min(k, search->num_vectors);
    for (int i = 0; i < actual_k; i++) {
        scores[i] = score_pairs[i].first;
        indices[i] = score_pairs[i].second;
    }

    // Cleanup
    delete[] h_scores;
    cudaFree(d_query);

    return 0;
}

// Get GPU memory stats
void cuda_search_stats(void* handle, size_t* memory_used) {
    if (!handle || !memory_used) return;

    CUDASearch* search = static_cast<CUDASearch*>(handle);
    *memory_used = search->num_vectors * search->dim * sizeof(int8_t);
}

} // extern "C"