// pure_cuda_indexer.cu - Pure CUDA implementation without LibTorch
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>

// Structure to hold GPU index data
struct CUDAIndex {
    int8_t* d_database;     // Database vectors on GPU [num_vectors x dim]
    float* d_scales;        // Scale factors for each vector
    int num_vectors;
    int vector_dim;
    size_t allocated_size;
    
    // Embedding table for token lookups
    float* d_embeddings;    // Token embeddings [vocab_size x embed_dim]
    int vocab_size;
    int embed_dim;
    int max_tokens;         // Max tokens to process (default 512)
    
    // Bulk indexing cache
    float* d_batch_embeddings;  // Cached batch embeddings for bulk operations
    int batch_capacity;
    
    cublasHandle_t cublas_handle;
    bool initialized;
    
    CUDAIndex() : d_database(nullptr), d_scales(nullptr), d_embeddings(nullptr), 
                  num_vectors(0), vector_dim(0), allocated_size(0),
                  vocab_size(0), embed_dim(0), max_tokens(512),
                  d_batch_embeddings(nullptr), batch_capacity(0), initialized(false) {}
};

// Optimized vectorized int8 dot product kernel
__global__ void int8_dot_product_kernel(
    const int8_t* query,      // [1 x dim]
    const int8_t* database,   // [num_vectors x dim]
    float* scores,            // [num_vectors]
    float query_scale,
    const float* db_scales,   // [num_vectors]
    int num_vectors,
    int dim
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= num_vectors) return;
    
    // Use vectorized loads when possible (4 int8 values at once)
    int32_t sum = 0;
    const int8_t* db_vec = database + vec_idx * dim;
    
    // Process 4 elements at a time using int32 loads
    int vec_len = dim / 4;
    const int32_t* query_vec = reinterpret_cast<const int32_t*>(query);
    const int32_t* db_vec_int32 = reinterpret_cast<const int32_t*>(db_vec);
    
    for (int i = 0; i < vec_len; i++) {
        int32_t q_packed = query_vec[i];
        int32_t d_packed = db_vec_int32[i];
        
        // Extract individual int8 values and multiply
        int8_t q0 = (int8_t)(q_packed & 0xFF);
        int8_t q1 = (int8_t)((q_packed >> 8) & 0xFF);
        int8_t q2 = (int8_t)((q_packed >> 16) & 0xFF);
        int8_t q3 = (int8_t)((q_packed >> 24) & 0xFF);
        
        int8_t d0 = (int8_t)(d_packed & 0xFF);
        int8_t d1 = (int8_t)((d_packed >> 8) & 0xFF);
        int8_t d2 = (int8_t)((d_packed >> 16) & 0xFF);
        int8_t d3 = (int8_t)((d_packed >> 24) & 0xFF);
        
        sum += (int32_t)q0 * (int32_t)d0;
        sum += (int32_t)q1 * (int32_t)d1;
        sum += (int32_t)q2 * (int32_t)d2;
        sum += (int32_t)q3 * (int32_t)d3;
    }
    
    // Handle remaining elements
    for (int i = vec_len * 4; i < dim; i++) {
        sum += (int32_t)query[i] * (int32_t)db_vec[i];
    }
    
    // Apply scaling and store
    scores[vec_idx] = (float)sum * query_scale * db_scales[vec_idx];
}

// Optimized CUDA kernel for token embedding lookup and average pooling
// Only computes first output_dim dimensions (e.g., 512 instead of full 1024)
__global__ void embed_and_pool_optimized_kernel(
    const int* token_ids,       // [seq_len]
    const float* embeddings,    // [vocab_size x embed_dim]
    float* output,              // [output_dim] - only first 512 dimensions
    int seq_len,
    int embed_dim,              // Full embedding dimension (1024)
    int output_dim,             // Output dimension (512)
    int vocab_size,
    int max_tokens = 512        // Truncate to first 512 tokens
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= output_dim) return;  // Only compute first output_dim dimensions
    
    float sum = 0.0f;
    int valid_tokens = 0;
    
    // Truncate to max_tokens
    int actual_seq_len = min(seq_len, max_tokens);
    
    // Only access first output_dim dimensions of each embedding
    for (int i = 0; i < actual_seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            sum += embeddings[token_id * embed_dim + dim_idx];
            valid_tokens++;
        }
    }
    
    // Average pooling
    output[dim_idx] = valid_tokens > 0 ? sum / valid_tokens : 0.0f;
}

// Legacy kernel for backward compatibility
__global__ void embed_and_pool_kernel(
    const int* token_ids,       // [seq_len]
    const float* embeddings,    // [vocab_size x embed_dim]
    float* output,              // [embed_dim]
    int seq_len,
    int embed_dim,
    int vocab_size,
    int max_tokens = 512        // Truncate to first 512 tokens
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= embed_dim) return;
    
    float sum = 0.0f;
    int valid_tokens = 0;
    
    // Truncate to max_tokens
    int actual_seq_len = min(seq_len, max_tokens);
    
    for (int i = 0; i < actual_seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            sum += embeddings[token_id * embed_dim + dim_idx];
            valid_tokens++;
        }
    }
    
    // Average pooling
    output[dim_idx] = valid_tokens > 0 ? sum / valid_tokens : 0.0f;
}

// Optimized two-stage quantization: Step 1 - Find maximum absolute value using reduction
__global__ void find_max_abs_kernel(
    const float* input,
    float* max_val,
    int dim
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory with bounds check
    sdata[tid] = (i < dim) ? fabsf(input[i]) : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicMax((int*)max_val, __float_as_int(sdata[0]));
    }
}

// Optimized two-stage quantization: Step 2 - Apply quantization with known scale
__global__ void apply_quantization_kernel(
    const float* input,      // [embed_dim]
    int8_t* output,         // [embed_dim]
    float scale_factor,     // Pre-computed scale factor
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    
    // Quantize with vectorized operations when possible
    float scaled = input[idx] * scale_factor;
    scaled = fminf(127.0f, fmaxf(-128.0f, scaled));
    output[idx] = (int8_t)__float2int_rn(scaled); // Use fast float to int conversion
}

// Legacy single-kernel quantization for backward compatibility
__global__ void quantize_to_int8_kernel(
    const float* input,      // [embed_dim]
    int8_t* output,         // [embed_dim]
    float* scale,          // [1]
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    
    // Find max absolute value for scaling (inefficient but compatible)
    float max_val = 0.0f;
    for (int i = 0; i < dim; i++) {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    
    // Compute scale factor
    float scale_factor = max_val > 0 ? 127.0f / max_val : 1.0f;
    if (idx == 0) {
        *scale = 1.0f / scale_factor; // Store inverse for later multiplication
    }
    
    // Quantize
    float scaled = input[idx] * scale_factor;
    scaled = fminf(127.0f, fmaxf(-128.0f, scaled));
    output[idx] = (int8_t)__float2int_rn(scaled);
}

// Optimized parallel top-k selection for small k (k <= 32)
__global__ void topk_parallel_kernel(
    const float* scores,     // [num_vectors]
    int* indices,           // [k]
    float* top_scores,      // [k]
    int num_vectors,
    int k
) {
    extern __shared__ char shared_mem[];
    float* shared_scores = (float*)shared_mem;
    int* shared_indices = (int*)(shared_scores + k);
    
    int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < k) {
        shared_scores[tid] = -FLT_MAX;
        shared_indices[tid] = -1;
    }
    __syncthreads();
    
    // Each thread processes multiple elements to find local top-k
    for (int i = tid; i < num_vectors; i += blockDim.x) {
        float score = scores[i];
        
        // Try to insert into shared top-k list
        for (int j = 0; j < k; j++) {
            if (score > shared_scores[j]) {
                // Use atomic operations to prevent race conditions
                float old_score = atomicExch(&shared_scores[j], score);
                int old_idx = atomicExch(&shared_indices[j], i);
                
                // Cascade the displaced element down
                score = old_score;
                i = old_idx;
                if (score == -FLT_MAX) break; // No more cascading needed
            }
        }
    }
    __syncthreads();
    
    // Copy results back to global memory
    if (tid < k) {
        top_scores[tid] = shared_scores[tid];
        indices[tid] = shared_indices[tid];
    }
}

// Simple top-k selection kernel using single thread for correctness (fallback)
__global__ void topk_selection_kernel(
    const float* scores,     // [num_vectors]
    int* indices,           // [k]
    float* top_scores,      // [k]
    int num_vectors,
    int k
) {
    // Use only thread 0 for sequential processing to avoid race conditions
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Initialize results with worst scores
    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        top_scores[i] = -FLT_MAX;
    }
    
    // Find top-k scores sequentially
    for (int i = 0; i < num_vectors; i++) {
        float score = scores[i];
        
        // Find position to insert if score is large enough
        int insert_pos = -1;
        for (int j = 0; j < k; j++) {
            if (score > top_scores[j]) {
                insert_pos = j;
                break;
            }
        }
        
        // Insert and shift if needed
        if (insert_pos >= 0) {
            // Shift smaller scores down
            for (int j = k - 1; j > insert_pos; j--) {
                top_scores[j] = top_scores[j - 1];
                indices[j] = indices[j - 1];
            }
            // Insert new score
            top_scores[insert_pos] = score;
            indices[insert_pos] = i;
        }
    }
}

extern "C" {

// Create a new CUDA index
void* cuda_index_create(int vector_dim, int vocab_size, int embed_dim) {
    CUDAIndex* index = new CUDAIndex();
    index->vector_dim = vector_dim;
    index->vocab_size = vocab_size;
    index->embed_dim = embed_dim;
    
    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&index->cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle" << std::endl;
        delete index;
        return nullptr;
    }
    
    index->initialized = true;
    std::cout << "✅ Pure CUDA indexer created (dim=" << vector_dim 
              << ", vocab=" << vocab_size << ", embed=" << embed_dim << ")" << std::endl;
    return index;
}

// Load embedding table to GPU
int cuda_load_embeddings(void* index_ptr, const float* embeddings) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (!index || !index->initialized) return 0;
    
    size_t embed_size = index->vocab_size * index->embed_dim * sizeof(float);
    
    // Allocate and copy embeddings to GPU
    if (index->d_embeddings) {
        cudaFree(index->d_embeddings);
    }
    
    cudaError_t err = cudaMalloc(&index->d_embeddings, embed_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for embeddings: " 
                  << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    
    err = cudaMemcpy(index->d_embeddings, embeddings, embed_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy embeddings to GPU: " 
                  << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    
    std::cout << "✅ Loaded embedding table to GPU (" 
              << (embed_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
    return 1;
}

// Add vectors to the index
int cuda_index_add(void* index_ptr, const int8_t* vectors, const float* scales, int num_vectors) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (!index || !index->initialized) return 0;
    
    size_t new_size = (index->num_vectors + num_vectors) * index->vector_dim * sizeof(int8_t);
    
    // Reallocate if needed
    if (new_size > index->allocated_size) {
        int8_t* new_database;
        float* new_scales;
        
        cudaMalloc(&new_database, new_size * 2); // Allocate with growth factor
        cudaMalloc(&new_scales, (index->num_vectors + num_vectors) * sizeof(float) * 2);
        
        if (index->d_database && index->num_vectors > 0) {
            // Copy existing data
            cudaMemcpy(new_database, index->d_database, 
                      index->num_vectors * index->vector_dim * sizeof(int8_t),
                      cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_scales, index->d_scales,
                      index->num_vectors * sizeof(float),
                      cudaMemcpyDeviceToDevice);
            cudaFree(index->d_database);
            cudaFree(index->d_scales);
        }
        
        index->d_database = new_database;
        index->d_scales = new_scales;
        index->allocated_size = new_size * 2;
    }
    
    // Copy new vectors
    size_t offset = index->num_vectors * index->vector_dim;
    cudaMemcpy(index->d_database + offset, vectors,
              num_vectors * index->vector_dim * sizeof(int8_t),
              cudaMemcpyHostToDevice);
    
    cudaMemcpy(index->d_scales + index->num_vectors, scales,
              num_vectors * sizeof(float),
              cudaMemcpyHostToDevice);
    
    index->num_vectors += num_vectors;
    
    // Get GPU memory info
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    
    std::cout << "✅ Added " << num_vectors << " vectors to GPU index. "
              << "Total: " << index->num_vectors << " vectors, "
              << "GPU memory: " << (used_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    return 1;
}

// Search with token IDs (generates embedding on GPU)
int cuda_search_with_tokens(
    void* index_ptr,
    const int* token_ids,
    int seq_len,
    int* result_indices,
    float* result_scores,
    int k
) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (!index || !index->initialized || index->num_vectors == 0) return 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate GPU memory for tokens and embedding
    int* d_tokens;
    float* d_embedding;
    int8_t* d_query;
    float* d_query_scale;
    float* d_scores;
    
    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMalloc(&d_embedding, index->embed_dim * sizeof(float));
    cudaMalloc(&d_query, index->vector_dim * sizeof(int8_t));
    cudaMalloc(&d_query_scale, sizeof(float));
    cudaMalloc(&d_scores, index->num_vectors * sizeof(float));
    
    // Copy tokens to GPU
    cudaMemcpy(d_tokens, token_ids, seq_len * sizeof(int), cudaMemcpyHostToDevice);
    
    // Generate embedding from tokens - only compute first 512 dimensions
    int block_size = 256;
    int output_dim = index->vector_dim;  // Only compute 512 dimensions
    int grid_size = (output_dim + block_size - 1) / block_size;
    embed_and_pool_optimized_kernel<<<grid_size, block_size>>>(
        d_tokens, index->d_embeddings, d_embedding,
        seq_len, index->embed_dim, output_dim, index->vocab_size, index->max_tokens
    );
    
    // Quantize embedding to int8
    quantize_to_int8_kernel<<<grid_size, block_size>>>(
        d_embedding, d_query, d_query_scale, index->vector_dim
    );
    
    // Compute similarities
    grid_size = (index->num_vectors + block_size - 1) / block_size;
    
    // Get query scale from device
    float query_scale;
    cudaMemcpy(&query_scale, d_query_scale, sizeof(float), cudaMemcpyDeviceToHost);
    
    int8_dot_product_kernel<<<grid_size, block_size>>>(
        d_query, index->d_database, d_scores,
        query_scale, index->d_scales,
        index->num_vectors, index->vector_dim
    );
    
    // Find top-k
    int* d_indices;
    float* d_top_scores;
    cudaMalloc(&d_indices, k * sizeof(int));
    cudaMalloc(&d_top_scores, k * sizeof(float));
    
    size_t shared_mem_size = k * (sizeof(float) + sizeof(int));
    topk_selection_kernel<<<1, 256, shared_mem_size>>>(
        d_scores, d_indices, d_top_scores, index->num_vectors, k
    );
    
    // Copy results back
    cudaMemcpy(result_indices, d_indices, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_scores, d_top_scores, k * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_embedding);
    cudaFree(d_query);
    cudaFree(d_query_scale);
    cudaFree(d_scores);
    cudaFree(d_indices);
    cudaFree(d_top_scores);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "🚀 GPU search completed in " << duration.count() << " μs" << std::endl;
    
    return k;
}

// Search with pre-computed int8 embedding
int cuda_search_with_embedding(
    void* index_ptr,
    const int8_t* query,
    float query_scale,
    int* result_indices,
    float* result_scores,
    int k
) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (!index || !index->initialized || index->num_vectors == 0) return 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate GPU memory
    int8_t* d_query;
    float* d_scores;
    cudaMalloc(&d_query, index->vector_dim * sizeof(int8_t));
    cudaMalloc(&d_scores, index->num_vectors * sizeof(float));
    
    // Copy query to GPU
    cudaMemcpy(d_query, query, index->vector_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    
    // Compute similarities
    int block_size = 256;
    int grid_size = (index->num_vectors + block_size - 1) / block_size;
    
    int8_dot_product_kernel<<<grid_size, block_size>>>(
        d_query, index->d_database, d_scores,
        query_scale, index->d_scales,
        index->num_vectors, index->vector_dim
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "❌ Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_query);
        cudaFree(d_scores);
        return 0;
    }
    
    // Find top-k
    int* d_indices;
    float* d_top_scores;
    cudaMalloc(&d_indices, k * sizeof(int));
    cudaMalloc(&d_top_scores, k * sizeof(float));
    
    size_t shared_mem_size = k * (sizeof(float) + sizeof(int));
    topk_selection_kernel<<<1, 256, shared_mem_size>>>(
        d_scores, d_indices, d_top_scores, index->num_vectors, k
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(result_indices, d_indices, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_scores, d_top_scores, k * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_query);
    cudaFree(d_scores);
    cudaFree(d_indices);
    cudaFree(d_top_scores);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "⚡ GPU search completed in " << duration.count() << " μs ("
              << index->num_vectors << " vectors searched)" << std::endl;
    
    return k;
}

// Destroy the index
void cuda_index_destroy(void* index_ptr) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (!index) return;
    
    if (index->d_database) cudaFree(index->d_database);
    if (index->d_scales) cudaFree(index->d_scales);
    if (index->d_embeddings) cudaFree(index->d_embeddings);
    if (index->initialized) cublasDestroy(index->cublas_handle);
    
    delete index;
    std::cout << "🧹 CUDA index destroyed" << std::endl;
}

// Get GPU memory usage
size_t cuda_get_memory_usage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem - free_mem;
}

// Check CUDA availability
int cuda_is_available() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

// Bulk index token sequences - keeps everything on GPU
int cuda_bulk_index_tokens(
    void* index_ptr,
    const int* token_sequences,  // [batch_size x max_seq_len] flattened
    const int* seq_lengths,      // [batch_size] actual lengths
    int batch_size,
    int max_seq_len
) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (!index || !index->initialized || !index->d_embeddings) return 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Ensure we have enough space for batch embeddings
    size_t batch_embed_size = batch_size * index->embed_dim * sizeof(float);
    size_t batch_int8_size = batch_size * index->vector_dim * sizeof(int8_t);
    
    if (index->batch_capacity < batch_size) {
        if (index->d_batch_embeddings) cudaFree(index->d_batch_embeddings);
        cudaMalloc(&index->d_batch_embeddings, batch_embed_size);
        index->batch_capacity = batch_size;
    }
    
    // Allocate temporary arrays
    int* d_tokens;
    float* d_embeddings_batch;
    int8_t* d_vectors_batch;
    float* d_scales_batch;
    int* d_seq_lengths;
    
    cudaMalloc(&d_tokens, batch_size * max_seq_len * sizeof(int));
    cudaMalloc(&d_embeddings_batch, batch_embed_size);
    cudaMalloc(&d_vectors_batch, batch_int8_size);
    cudaMalloc(&d_scales_batch, batch_size * sizeof(float));
    cudaMalloc(&d_seq_lengths, batch_size * sizeof(int));
    
    // Copy token sequences to GPU
    cudaMemcpy(d_tokens, token_sequences, batch_size * max_seq_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq_lengths, seq_lengths, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Process each sequence in batch
    int block_size = 256;
    for (int i = 0; i < batch_size; i++) {
        int* seq_tokens = d_tokens + i * max_seq_len;
        float* seq_embedding = d_embeddings_batch + i * index->embed_dim;
        int8_t* seq_vector = d_vectors_batch + i * index->vector_dim;
        float* seq_scale = d_scales_batch + i;
        
        // Generate embedding - only compute first 512 dimensions
        int output_dim = index->vector_dim;  // Only 512 dimensions
        int grid_size = (output_dim + block_size - 1) / block_size;
        embed_and_pool_optimized_kernel<<<grid_size, block_size>>>(
            seq_tokens, index->d_embeddings, seq_embedding,
            seq_lengths[i], index->embed_dim, output_dim, index->vocab_size, index->max_tokens
        );
        
        // Quantize to int8
        quantize_to_int8_kernel<<<grid_size, block_size>>>(
            seq_embedding, seq_vector, seq_scale, index->vector_dim
        );
    }
    
    // Expand database if needed
    size_t new_total = index->num_vectors + batch_size;
    if (new_total * index->vector_dim > index->allocated_size / sizeof(int8_t)) {
        size_t new_size = new_total * 2 * index->vector_dim * sizeof(int8_t);
        int8_t* new_database;
        float* new_scales;
        
        cudaMalloc(&new_database, new_size);
        cudaMalloc(&new_scales, new_total * 2 * sizeof(float));
        
        if (index->d_database && index->num_vectors > 0) {
            cudaMemcpy(new_database, index->d_database, 
                      index->num_vectors * index->vector_dim * sizeof(int8_t), 
                      cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_scales, index->d_scales,
                      index->num_vectors * sizeof(float),
                      cudaMemcpyDeviceToDevice);
            cudaFree(index->d_database);
            cudaFree(index->d_scales);
        }
        
        index->d_database = new_database;
        index->d_scales = new_scales;
        index->allocated_size = new_size;
    }
    
    // Copy batch vectors to database (all stays on GPU)
    cudaMemcpy(index->d_database + index->num_vectors * index->vector_dim,
               d_vectors_batch, batch_int8_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(index->d_scales + index->num_vectors,
               d_scales_batch, batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    index->num_vectors += batch_size;
    
    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_embeddings_batch);
    cudaFree(d_vectors_batch);
    cudaFree(d_scales_batch);
    cudaFree(d_seq_lengths);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "⚡ Bulk indexed " << batch_size << " sequences on GPU in " 
              << duration.count() << " ms. Total vectors: " << index->num_vectors 
              << ", GPU memory: " << (cuda_get_memory_usage() / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    return batch_size;
}

// Set max tokens for truncation
void cuda_set_max_tokens(void* index_ptr, int max_tokens) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (index) {
        index->max_tokens = max_tokens;
        std::cout << "📏 Max tokens set to " << max_tokens << std::endl;
    }
}

} // extern "C"