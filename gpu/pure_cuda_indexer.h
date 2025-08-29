#ifndef PURE_CUDA_INDEXER_H
#define PURE_CUDA_INDEXER_H

#ifdef __cplusplus
extern "C" {
#endif

// Create and destroy index
void* cuda_index_create(int vector_dim, int vocab_size, int embed_dim);
void cuda_index_destroy(void* index);

// Load embedding table (for token to embedding mapping)
int cuda_load_embeddings(void* index, const float* embeddings);

// Load quantized int8 embeddings to GPU for faster processing
int cuda_load_embeddings_int8(void* index, const signed char* embeddings, const float* scales);

// Add int8 vectors to index
int cuda_index_add(void* index, const signed char* vectors, const float* scales, int num_vectors);

// Search with token IDs (generates embedding on GPU)
int cuda_search_with_tokens(
    void* index,
    const int* token_ids,
    int seq_len,
    int* result_indices,
    float* result_scores,
    int k
);

// Search with pre-computed int8 embedding
int cuda_search_with_embedding(
    void* index,
    const signed char* query,
    float query_scale,
    int* result_indices,
    float* result_scores,
    int k
);

// Utility functions
size_t cuda_get_memory_usage();
int cuda_is_available();

// Bulk operations for optimized indexing
int cuda_bulk_index_tokens(
    void* index,
    const int* token_sequences,  // [batch_size x max_seq_len] flattened
    const int* seq_lengths,      // [batch_size] actual lengths
    int batch_size,
    int max_seq_len
);

// Configuration
void cuda_set_max_tokens(void* index, int max_tokens);

#ifdef __cplusplus
}
#endif

#endif // PURE_CUDA_INDEXER_H