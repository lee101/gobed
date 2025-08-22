#ifndef TORCH_CGO_WRAPPER_H
#define TORCH_CGO_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// LibTorch indexer handle (opaque pointer)
typedef void* TorchIndexerHandle;

// Configuration structure
typedef struct {
    int vector_dim;
    int num_subquantizers;
    int codebook_size;
    int ivf_clusters;
    int probe_lists;
    int rerank_k;
    int device_id;
} IndexConfig;

// Search result structure
typedef struct {
    int* ids;
    float* scores;
    int count;
} SearchResult;

// Statistics structure
typedef struct {
    int num_vectors;
    int vector_dim;
    int ivf_clusters;
    int pq_subquantizers;
    float gpu_memory_mb;
    int is_trained;
    int index_built;
} IndexStats;

// Core functions
TorchIndexerHandle torch_indexer_create(IndexConfig config);
void torch_indexer_destroy(TorchIndexerHandle handle);

// Training and indexing
int torch_indexer_train(TorchIndexerHandle handle, signed char* vectors, int n_vectors, int vector_dim);
int torch_indexer_add_vectors(TorchIndexerHandle handle, signed char* vectors, int n_vectors, int vector_dim);

// Search operations
SearchResult torch_indexer_search(TorchIndexerHandle handle, signed char* query, int vector_dim, int k);
SearchResult* torch_indexer_batch_search(TorchIndexerHandle handle, signed char* queries, int n_queries, int vector_dim, int k);

// Utility functions
IndexStats torch_indexer_get_stats(TorchIndexerHandle handle);
int torch_indexer_save(TorchIndexerHandle handle, const char* path);
int torch_indexer_load(TorchIndexerHandle handle, const char* path);

// Memory management
void torch_search_result_free(SearchResult* result);
void torch_batch_results_free(SearchResult* results, int count);

// Version and capability checks
const char* torch_get_version();
int torch_cuda_is_available();
int torch_cuda_device_count();

#ifdef __cplusplus
}
#endif

#endif // TORCH_CGO_WRAPPER_H