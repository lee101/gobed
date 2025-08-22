// cuda_kernels.cu - CUDA kernels for similarity search
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

// CUDA kernel for int8 similarity search
__global__ void cuda_similarity_kernel(
    const int8_t* queries,      // [num_queries, dim]
    const int8_t* database,     // [num_vectors, dim] 
    float* scores,              // [num_queries, num_vectors]
    int num_queries,
    int num_vectors, 
    int dim
) {
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || vector_idx >= num_vectors) return;
    
    // Compute dot product similarity
    int sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += static_cast<int>(queries[query_idx * dim + i]) * 
               static_cast<int>(database[vector_idx * dim + i]);
    }
    
    scores[query_idx * num_vectors + vector_idx] = static_cast<float>(sum);
}

// Top-K selection kernel
__global__ void cuda_topk_kernel(
    const float* scores,        // [num_queries, num_vectors]
    int* top_indices,          // [num_queries, k]
    float* top_scores,         // [num_queries, k] 
    int num_queries,
    int num_vectors,
    int k
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;
    
    const float* query_scores = scores + query_idx * num_vectors;
    int* query_indices = top_indices + query_idx * k;
    float* query_top_scores = top_scores + query_idx * k;
    
    // Initialize with first k elements
    for (int i = 0; i < k && i < num_vectors; i++) {
        query_indices[i] = i;
        query_top_scores[i] = query_scores[i];
    }
    
    // Find k largest elements
    for (int vec = k; vec < num_vectors; vec++) {
        float current_score = query_scores[vec];
        
        // Find minimum in current top-k
        int min_idx = 0;
        for (int i = 1; i < k; i++) {
            if (query_top_scores[i] < query_top_scores[min_idx]) {
                min_idx = i;
            }
        }
        
        // Replace if current score is larger
        if (current_score > query_top_scores[min_idx]) {
            query_top_scores[min_idx] = current_score;
            query_indices[min_idx] = vec;
        }
    }
    
    // Sort the top-k results in descending order
    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            if (query_top_scores[i] < query_top_scores[j]) {
                // Swap scores
                float temp_score = query_top_scores[i];
                query_top_scores[i] = query_top_scores[j];
                query_top_scores[j] = temp_score;
                
                // Swap indices
                int temp_idx = query_indices[i];
                query_indices[i] = query_indices[j];
                query_indices[j] = temp_idx;
            }
        }
    }
}

// C interface for kernel launches
void launch_similarity_kernel(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim
) {
    dim3 block_size(32, 16);  
    dim3 grid_size(
        (num_vectors + block_size.x - 1) / block_size.x,
        (num_queries + block_size.y - 1) / block_size.y
    );
    
    cuda_similarity_kernel<<<grid_size, block_size>>>(
        queries, database, scores, num_queries, num_vectors, dim
    );
}

void launch_topk_kernel(
    const float* scores,
    int* top_indices,
    float* top_scores,
    int num_queries,
    int num_vectors,
    int k
) {
    dim3 block_size(256);
    dim3 grid_size((num_queries + block_size.x - 1) / block_size.x);
    
    cuda_topk_kernel<<<grid_size, block_size>>>(
        scores, top_indices, top_scores, num_queries, num_vectors, k
    );
}

} // extern "C"