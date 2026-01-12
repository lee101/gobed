// pure_cuda_indexer.cu - Pure CUDA implementation without LibTorch
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <chrono>
#include <cstring>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace cg = cooperative_groups;

// Structure to hold GPU index data
struct CUDAIndex {
    int8_t* d_database;     // Database vectors on GPU [num_vectors x dim]
    float* d_scales;        // Scale factors for each vector
    int num_vectors;
    int vector_dim;
    size_t allocated_size;

    // Embedding table for token lookups
    float* d_embeddings;    // Token embeddings [vocab_size x embed_dim]
    int8_t* d_embeddings_int8;  // Quantized int8 embeddings [vocab_size x embed_dim]
    float* d_embedding_scales;   // Scale factors for each embedding [vocab_size]
    int vocab_size;
    int embed_dim;
    int max_tokens;         // Max tokens to process (default 512)
    bool use_int8_embeddings;  // Whether to use int8 embeddings

    // Bulk indexing cache
    float* d_batch_embeddings;  // Cached batch embeddings for bulk operations
    int batch_capacity;

    // Pre-allocated search buffers (avoid cudaMalloc per query)
    int8_t* d_query_buffer;     // [vector_dim] - reused per query
    float* d_scores_buffer;     // [max_search_vectors] - similarity scores
    int* d_indices_buffer;      // [max_k] - top-k indices
    float* d_topk_scores_buffer; // [max_k] - top-k scores
    int search_buffer_capacity; // current max vectors the score buffer can hold
    int topk_buffer_capacity;   // current max k the topk buffers can hold

    // CUDA streams for async execution
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;

    cublasHandle_t cublas_handle;
    bool initialized;

    CUDAIndex() : d_database(nullptr), d_scales(nullptr), d_embeddings(nullptr),
                  d_embeddings_int8(nullptr), d_embedding_scales(nullptr),
                  num_vectors(0), vector_dim(0), allocated_size(0),
                  vocab_size(0), embed_dim(0), max_tokens(512),
                  use_int8_embeddings(false), d_batch_embeddings(nullptr),
                  batch_capacity(0),
                  d_query_buffer(nullptr), d_scores_buffer(nullptr),
                  d_indices_buffer(nullptr), d_topk_scores_buffer(nullptr),
                  search_buffer_capacity(0), topk_buffer_capacity(0),
                  compute_stream(0), transfer_stream(0),
                  initialized(false) {}
};

// Custom heap-based top-k selection (much faster than full sort)
template<int K>
__global__ void topk_heap_kernel(
    const float* __restrict__ scores,
    int* __restrict__ indices,
    float* __restrict__ top_scores,
    int num_vectors
) {
    // Use a min-heap to track top K elements
    __shared__ float heap_scores[K];
    __shared__ int heap_indices[K];
    
    if (threadIdx.x == 0) {
        // Initialize heap with smallest values
        for (int i = 0; i < K; i++) {
            heap_scores[i] = -FLT_MAX;
            heap_indices[i] = -1;
        }
    }
    __syncthreads();
    
    // Each thread processes a portion of scores
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < num_vectors; i += stride) {
        float score = scores[i];
        
        // Check if this score belongs in top-k
        if (score > heap_scores[0]) {
            // Use atomic operations to update heap
            if (threadIdx.x == 0) {
                // Replace minimum with new value
                heap_scores[0] = score;
                heap_indices[0] = i;
                
                // Heapify down
                int pos = 0;
                while (2 * pos + 1 < K) {
                    int child = 2 * pos + 1;
                    if (child + 1 < K && heap_scores[child + 1] < heap_scores[child]) {
                        child++;
                    }
                    if (heap_scores[pos] > heap_scores[child]) {
                        // Swap
                        float tmp_score = heap_scores[pos];
                        int tmp_idx = heap_indices[pos];
                        heap_scores[pos] = heap_scores[child];
                        heap_indices[pos] = heap_indices[child];
                        heap_scores[child] = tmp_score;
                        heap_indices[child] = tmp_idx;
                        pos = child;
                    } else {
                        break;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Copy heap to output (sorted)
    if (threadIdx.x < K) {
        top_scores[threadIdx.x] = heap_scores[threadIdx.x];
        indices[threadIdx.x] = heap_indices[threadIdx.x];
    }
}

// Tensor Core accelerated int8 matrix multiplication (Ampere/Hopper)
// Note: Full tensor core implementation would use nvcuda::wmma API
__global__ void int8_gemm_tensor_core_kernel(
    const int8_t* __restrict__ query,      // [1 x dim] treated as [1 x dim x 1]
    const int8_t* __restrict__ database,   // [num_vectors x dim]
    float* __restrict__ scores,            // [num_vectors]
    float query_scale,
    const float* __restrict__ db_scales,
    int num_vectors,
    int dim
) {
    // Optimized for tensor core alignment (16-byte boundaries)
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= num_vectors) return;
    
    const int8_t* db_vec = database + vec_idx * dim;
    int32_t sum = 0;
    
    // Process 16 elements at a time for tensor core alignment
    #pragma unroll
    for (int i = 0; i < dim; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16 && i + j < dim; j++) {
            sum += (int32_t)query[i + j] * (int32_t)db_vec[i + j];
        }
    }
    
    scores[vec_idx] = (float)sum * query_scale * db_scales[vec_idx];
}

// Ultra-optimized int8 dot product with int4 vectorization
__global__ void int8_dot_product_vectorized_kernel(
    const int8_t* __restrict__ query,
    const int8_t* __restrict__ database,
    float* __restrict__ scores,
    float query_scale,
    const float* __restrict__ db_scales,
    int num_vectors,
    int dim
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= num_vectors) return;
    
    const int8_t* db_vec = database + vec_idx * dim;
    
    // Use int4 (128-bit) loads for maximum bandwidth
    int32_t sum = 0;
    const int4* query_int4 = reinterpret_cast<const int4*>(query);
    const int4* db_int4 = reinterpret_cast<const int4*>(db_vec);
    int num_int4 = dim / 16;  // Each int4 contains 16 int8 values
    
    #pragma unroll 4
    for (int i = 0; i < num_int4; i++) {
        int4 q = query_int4[i];
        int4 d = db_int4[i];
        
        // Process 16 int8 values at once
        const int8_t* q_bytes = reinterpret_cast<const int8_t*>(&q);
        const int8_t* d_bytes = reinterpret_cast<const int8_t*>(&d);
        
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            sum += (int32_t)q_bytes[j] * (int32_t)d_bytes[j];
        }
    }
    
    // Handle remaining elements
    for (int i = num_int4 * 16; i < dim; i++) {
        sum += (int32_t)query[i] * (int32_t)db_vec[i];
    }
    
    scores[vec_idx] = (float)sum * query_scale * db_scales[vec_idx];
}

// Warp-cooperative dot product for better cache utilization
__global__ void int8_dot_product_warp_kernel(
    const int8_t* __restrict__ query,
    const int8_t* __restrict__ database,
    float* __restrict__ scores,
    float query_scale,
    const float* __restrict__ db_scales,
    int num_vectors,
    int dim
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    if (warp_id >= num_vectors) return;
    
    const int8_t* db_vec = database + warp_id * dim;
    int32_t partial_sum = 0;
    
    // Each thread processes a portion of the vector
    for (int i = lane_id; i < dim; i += 32) {
        partial_sum += (int32_t)query[i] * (int32_t)db_vec[i];
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += warp.shfl_down(partial_sum, offset);
    }
    
    // Lane 0 writes the result
    if (lane_id == 0) {
        scores[warp_id] = (float)partial_sum * query_scale * db_scales[warp_id];
    }
}

// Original kernel for compatibility
__global__ void int8_dot_product_kernel(
    const int8_t* query,
    const int8_t* database,
    float* scores,
    float query_scale,
    const float* db_scales,
    int num_vectors,
    int dim
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= num_vectors) return;
    
    int32_t sum = 0;
    const int8_t* db_vec = database + vec_idx * dim;
    
    // Simple unrolled loop
    #pragma unroll 8
    for (int i = 0; i < dim; i++) {
        sum += (int32_t)query[i] * (int32_t)db_vec[i];
    }
    
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

// Fused kernel: embedding lookup + pooling + quantization + dot product
__global__ void fused_embed_search_kernel(
    const int* __restrict__ token_ids,
    const int8_t* __restrict__ embeddings,
    const int8_t* __restrict__ database,
    float* __restrict__ scores,
    const float* __restrict__ db_scales,
    int seq_len,
    int num_vectors,
    int embed_dim,
    int output_dim,
    int vocab_size,
    int max_tokens
) {
    extern __shared__ char shared_mem[];
    int8_t* shared_query = (int8_t*)shared_mem;
    
    int vec_idx = blockIdx.x;
    if (vec_idx >= num_vectors) return;
    
    // Phase 1: Cooperatively compute query embedding
    int tid = threadIdx.x;
    int dims_per_thread = (output_dim + blockDim.x - 1) / blockDim.x;
    
    for (int d = tid * dims_per_thread; d < (tid + 1) * dims_per_thread && d < output_dim; d++) {
        int32_t sum = 0;
        int valid_tokens = 0;
        int actual_seq_len = min(seq_len, max_tokens);
        
        for (int i = 0; i < actual_seq_len; i++) {
            int token_id = token_ids[i];
            if (token_id >= 0 && token_id < vocab_size) {
                sum += embeddings[token_id * embed_dim + d];
                valid_tokens++;
            }
        }
        
        // Average and quantize inline
        float avg = valid_tokens > 0 ? (float)sum / valid_tokens : 0.0f;
        int8_t quantized = (int8_t)fmaxf(-128.0f, fminf(127.0f, avg));
        shared_query[d] = quantized;
    }
    
    __syncthreads();
    
    // Phase 2: Compute dot product with database vector
    const int8_t* db_vec = database + vec_idx * output_dim;
    int32_t dot_product = 0;
    
    // Each thread computes partial sum
    for (int d = tid; d < output_dim; d += blockDim.x) {
        dot_product += (int32_t)shared_query[d] * (int32_t)db_vec[d];
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
    }
    
    // Write result (first thread of each warp)
    if (tid % 32 == 0) {
        atomicAdd(&scores[vec_idx], (float)dot_product * db_scales[vec_idx] / 127.0f);
    }
}

// Ultra-fast shared memory version for small sequences
__global__ void embed_and_pool_int8_shared_kernel(
    const int* __restrict__ token_ids,
    const int8_t* __restrict__ embeddings,
    int8_t* __restrict__ output,
    float* output_scale,
    int seq_len,
    int embed_dim,
    int output_dim,
    int vocab_size,
    int max_tokens = 512
) {
    extern __shared__ int32_t shared_sums[];
    
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory
    if (dim_idx < output_dim) {
        shared_sums[tid] = 0;
    }
    __syncthreads();
    
    if (dim_idx >= output_dim) return;
    
    int32_t local_sum = 0;
    int actual_seq_len = min(seq_len, max_tokens);
    
    // Unrolled loop for better performance
    #pragma unroll 8
    for (int i = 0; i < actual_seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            local_sum += (int32_t)embeddings[token_id * embed_dim + dim_idx];
        }
    }
    
    // Store to shared memory
    shared_sums[tid] = local_sum;
    __syncthreads();
    
    // Final averaging and quantization
    if (dim_idx < output_dim) {
        int32_t avg = shared_sums[tid] / actual_seq_len;
        avg = min(127, max(-128, avg));
        output[dim_idx] = (int8_t)avg;
    }
    
    if (tid == 0) {
        *output_scale = 1.0f / 127.0f;
    }
}

// Warp-optimized version using warp shuffle instructions
__global__ void embed_and_pool_int8_warp_kernel(
    const int* __restrict__ token_ids,
    const int8_t* __restrict__ embeddings,
    int8_t* __restrict__ output,
    float* output_scale,
    int seq_len,
    int embed_dim,
    int output_dim,
    int vocab_size,
    int max_tokens = 512
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= output_dim) return;
    
    int32_t sum = 0;
    int actual_seq_len = min(seq_len, max_tokens);
    
    // Process tokens with warp-level coordination
    for (int i = warp.thread_rank(); i < actual_seq_len; i += warp.size()) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            sum += (int32_t)embeddings[token_id * embed_dim + dim_idx];
        }
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += warp.shfl_down(sum, offset);
    }
    
    // Thread 0 of each warp writes result
    if (warp.thread_rank() == 0) {
        int32_t avg = sum / actual_seq_len;
        avg = min(127, max(-128, avg));
        output[dim_idx] = (int8_t)avg;
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output_scale = 1.0f / 127.0f;
    }
}

// Ultra-optimized int8 quantized embedding lookup and pooling
// Uses int8 embeddings and performs accumulation in int32 for speed
__global__ void embed_and_pool_int8_kernel(
    const int* token_ids,       // [seq_len]
    const int8_t* embeddings,   // [vocab_size x embed_dim] - quantized embeddings
    int8_t* output,             // [output_dim] - quantized output
    float* output_scale,        // Output scale factor
    const float* embed_scales,  // [vocab_size] - scale for each embedding
    int seq_len,
    int embed_dim,              // Full embedding dimension (1024)
    int output_dim,             // Output dimension (512)
    int vocab_size,
    int max_tokens = 512        // Truncate to first 512 tokens
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= output_dim) return;
    
    // Use int32 for accumulation to avoid overflow
    int32_t sum = 0;
    int valid_tokens = 0;
    float scale_sum = 0.0f;
    
    // Truncate to max_tokens
    int actual_seq_len = min(seq_len, max_tokens);
    
    // Accumulate int8 values
    for (int i = 0; i < actual_seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            // Load int8 embedding value
            int8_t embed_val = embeddings[token_id * embed_dim + dim_idx];
            sum += (int32_t)embed_val;
            scale_sum += embed_scales[token_id];
            valid_tokens++;
        }
    }
    
    if (valid_tokens > 0) {
        // Average the accumulated int32 value
        float avg = (float)sum / (float)valid_tokens;
        float avg_scale = scale_sum / valid_tokens;
        
        // Quantize the average back to int8
        // Note: This is a simplified quantization, you might want more sophisticated approach
        int32_t quantized = __float2int_rn(avg);
        quantized = min(127, max(-128, quantized));
        output[dim_idx] = (int8_t)quantized;
        
        // Store scale factor (only first thread)
        if (dim_idx == 0) {
            *output_scale = avg_scale;
        }
    } else {
        output[dim_idx] = 0;
        if (dim_idx == 0) {
            *output_scale = 1.0f;
        }
    }
}

// Vectorized int8 embedding lookup with warp-level optimizations
__global__ void embed_and_pool_int8_vectorized_kernel(
    const int* token_ids,       // [seq_len]
    const int8_t* embeddings,   // [vocab_size x embed_dim] - quantized embeddings
    int8_t* output,             // [output_dim] - quantized output
    float* output_scale,        // Output scale factor
    int seq_len,
    int embed_dim,              // Full embedding dimension (1024)
    int output_dim,             // Output dimension (512)
    int vocab_size,
    int max_tokens = 512        // Truncate to first 512 tokens
) {
    // Process 4 dimensions per thread for better memory coalescing
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_base = thread_idx * 4;
    
    if (dim_base >= output_dim) return;
    
    // Use int32 for accumulation of 4 values
    int32_t sum[4] = {0, 0, 0, 0};
    int valid_tokens = 0;
    
    // Truncate to max_tokens
    int actual_seq_len = min(seq_len, max_tokens);
    
    // Process tokens
    for (int i = 0; i < actual_seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            // Load 4 int8 values at once (vectorized load)
            const int8_t* embed_ptr = embeddings + token_id * embed_dim + dim_base;
            
            // Process up to 4 dimensions
            #pragma unroll
            for (int j = 0; j < 4 && (dim_base + j) < output_dim; j++) {
                sum[j] += (int32_t)embed_ptr[j];
            }
            valid_tokens++;
        }
    }
    
    // Write results
    if (valid_tokens > 0) {
        int8_t* output_ptr = output + dim_base;
        
        #pragma unroll
        for (int j = 0; j < 4 && (dim_base + j) < output_dim; j++) {
            // Average and quantize
            int32_t avg = sum[j] / valid_tokens;
            avg = min(127, max(-128, avg));
            output_ptr[j] = (int8_t)avg;
        }
        
        // Simple scale (first thread only)
        if (thread_idx == 0) {
            *output_scale = 1.0f / 127.0f;  // Simple fixed scale
        }
    }
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
    
    // Create CUDA streams for async execution
    cudaStreamCreate(&index->compute_stream);
    cudaStreamCreate(&index->transfer_stream);
    
    index->initialized = true;
    // Indexer created successfully
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
    
    index->use_int8_embeddings = false;
    // Embedding table loaded to GPU
    return 1;
}

// Load quantized int8 embeddings to GPU for faster processing
int cuda_load_embeddings_int8(void* index_ptr, const int8_t* embeddings, const float* scales) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (!index || !index->initialized) return 0;
    
    size_t embed_size = index->vocab_size * index->embed_dim * sizeof(int8_t);
    size_t scale_size = index->vocab_size * sizeof(float);
    
    // Free old embeddings if any
    if (index->d_embeddings_int8) {
        cudaFree(index->d_embeddings_int8);
    }
    if (index->d_embedding_scales) {
        cudaFree(index->d_embedding_scales);
    }
    
    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&index->d_embeddings_int8, embed_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for int8 embeddings: " 
                  << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    
    err = cudaMalloc(&index->d_embedding_scales, scale_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for embedding scales: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(index->d_embeddings_int8);
        index->d_embeddings_int8 = nullptr;
        return 0;
    }
    
    // Copy to GPU
    err = cudaMemcpy(index->d_embeddings_int8, embeddings, embed_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy int8 embeddings to GPU: " 
                  << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    
    err = cudaMemcpy(index->d_embedding_scales, scales, scale_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy embedding scales to GPU: " 
                  << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    
    index->use_int8_embeddings = true;
    // Int8 embedding table loaded to GPU with 75% memory savings
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
    
    // Vectors added to GPU index
    
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
    
    if (index->use_int8_embeddings && index->d_embeddings_int8) {
        // Use int8 shared memory kernel for better performance
        int grid_size = (output_dim / 4 + block_size - 1) / block_size;
        embed_and_pool_int8_shared_kernel<<<grid_size, block_size, 
            block_size * index->embed_dim * sizeof(int8_t)>>>(
            d_tokens, index->d_embeddings_int8, 
            d_query, d_query_scale, seq_len, index->embed_dim, output_dim, 
            index->vocab_size, index->max_tokens
        );
        // No need for separate quantization step - already int8!
        cudaDeviceSynchronize();
    } else {
        // Use float embeddings
        int grid_size = (output_dim + block_size - 1) / block_size;
        embed_and_pool_optimized_kernel<<<grid_size, block_size>>>(
            d_tokens, index->d_embeddings, d_embedding,
            seq_len, index->embed_dim, output_dim, index->vocab_size, index->max_tokens
        );
        
        // Quantize embedding to int8
        quantize_to_int8_kernel<<<grid_size, block_size>>>(
            d_embedding, d_query, d_query_scale, index->vector_dim
        );
    }
    
    // Compute similarities using optimized vectorized kernel
    int grid_size = (index->num_vectors + block_size - 1) / block_size;
    
    // Get query scale from device
    float query_scale;
    cudaMemcpy(&query_scale, d_query_scale, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Use vectorized kernel for better performance
    int8_dot_product_vectorized_kernel<<<grid_size, block_size>>>(
        d_query, index->d_database, d_scores,
        query_scale, index->d_scales,
        index->num_vectors, index->vector_dim
    );
    
    // Find top-k using optimized heap kernel
    int* d_indices;
    float* d_top_scores;
    cudaMalloc(&d_indices, k * sizeof(int));
    cudaMalloc(&d_top_scores, k * sizeof(float));
    
    // Use heap-based top-k for better performance
    if (k == 10) {
        topk_heap_kernel<10><<<1, 256, 0, index->compute_stream>>>(
            d_scores, d_indices, d_top_scores, index->num_vectors
        );
    } else if (k == 20) {
        topk_heap_kernel<20><<<1, 256, 0, index->compute_stream>>>(
            d_scores, d_indices, d_top_scores, index->num_vectors
        );
    } else {
        // Fallback to original kernel
        size_t shared_mem_size = k * (sizeof(float) + sizeof(int));
        topk_selection_kernel<<<1, 256, shared_mem_size, index->compute_stream>>>(
            d_scores, d_indices, d_top_scores, index->num_vectors, k
        );
    }
    
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
    
    // GPU search completed
    
    return k;
}

// Helper: ensure search buffers are allocated for given capacity
static void ensure_search_buffers(CUDAIndex* index, int num_vectors, int k) {
    // Allocate/reallocate query buffer if needed
    if (index->d_query_buffer == nullptr) {
        cudaMalloc(&index->d_query_buffer, index->vector_dim * sizeof(int8_t));
    }

    // Allocate/reallocate scores buffer if capacity insufficient
    if (index->search_buffer_capacity < num_vectors) {
        if (index->d_scores_buffer) cudaFree(index->d_scores_buffer);
        // Allocate with 20% headroom for growth
        int new_cap = num_vectors + num_vectors / 5;
        cudaMalloc(&index->d_scores_buffer, new_cap * sizeof(float));
        index->search_buffer_capacity = new_cap;
    }

    // Allocate/reallocate topk buffers if k increased
    if (index->topk_buffer_capacity < k) {
        if (index->d_indices_buffer) cudaFree(index->d_indices_buffer);
        if (index->d_topk_scores_buffer) cudaFree(index->d_topk_scores_buffer);
        int new_k = k + 64; // headroom
        cudaMalloc(&index->d_indices_buffer, new_k * sizeof(int));
        cudaMalloc(&index->d_topk_scores_buffer, new_k * sizeof(float));
        index->topk_buffer_capacity = new_k;
    }
}

// Fast parallel top-k using thrust (radix sort partial)
static void thrust_topk(float* d_scores, int* d_indices, float* d_top_scores, int n, int k, cudaStream_t stream) {
    // Create index array
    thrust::device_vector<int> idx(n);
    thrust::sequence(thrust::cuda::par.on(stream), idx.begin(), idx.end());

    // Wrap scores in device_ptr
    thrust::device_ptr<float> scores_ptr(d_scores);

    // Partial sort (only sorts enough to get top k) - use negative for descending
    thrust::device_vector<float> neg_scores(n);
    thrust::transform(thrust::cuda::par.on(stream), scores_ptr, scores_ptr + n, neg_scores.begin(),
                      thrust::negate<float>());

    // Sort by negative scores (gives descending order)
    thrust::sort_by_key(thrust::cuda::par.on(stream), neg_scores.begin(), neg_scores.end(), idx.begin());

    // Copy top k results
    cudaMemcpyAsync(d_indices, thrust::raw_pointer_cast(idx.data()), k * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    // Negate back to get original scores
    thrust::transform(thrust::cuda::par.on(stream), neg_scores.begin(), neg_scores.begin() + k,
                      thrust::device_ptr<float>(d_top_scores), thrust::negate<float>());
}

// Search with pre-computed int8 embedding (optimized: reuses pre-allocated buffers)
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

    // Ensure buffers are allocated (only allocates once, then reused)
    ensure_search_buffers(index, index->num_vectors, k);

    // Copy query to pre-allocated GPU buffer (async on transfer stream)
    cudaMemcpyAsync(index->d_query_buffer, query, index->vector_dim * sizeof(int8_t),
                    cudaMemcpyHostToDevice, index->transfer_stream);
    cudaStreamSynchronize(index->transfer_stream);

    // Compute similarities using optimized vectorized kernel
    int block_size = 256;
    int grid_size = (index->num_vectors + block_size - 1) / block_size;

    int8_dot_product_vectorized_kernel<<<grid_size, block_size, 0, index->compute_stream>>>(
        index->d_query_buffer, index->d_database, index->d_scores_buffer,
        query_scale, index->d_scales,
        index->num_vectors, index->vector_dim
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    // Use fast thrust-based top-k (O(n log n) but highly parallel)
    thrust_topk(index->d_scores_buffer, index->d_indices_buffer, index->d_topk_scores_buffer,
                index->num_vectors, k, index->compute_stream);

    cudaStreamSynchronize(index->compute_stream);

    // Copy results back (async)
    cudaMemcpyAsync(result_indices, index->d_indices_buffer, k * sizeof(int), cudaMemcpyDeviceToHost, index->transfer_stream);
    cudaMemcpyAsync(result_scores, index->d_topk_scores_buffer, k * sizeof(float), cudaMemcpyDeviceToHost, index->transfer_stream);
    cudaStreamSynchronize(index->transfer_stream);

    return k;
}

// Destroy the index
void cuda_index_destroy(void* index_ptr) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (!index) return;

    if (index->d_database) cudaFree(index->d_database);
    if (index->d_scales) cudaFree(index->d_scales);
    if (index->d_embeddings) cudaFree(index->d_embeddings);
    if (index->d_embeddings_int8) cudaFree(index->d_embeddings_int8);
    if (index->d_embedding_scales) cudaFree(index->d_embedding_scales);

    // Free pre-allocated search buffers
    if (index->d_query_buffer) cudaFree(index->d_query_buffer);
    if (index->d_scores_buffer) cudaFree(index->d_scores_buffer);
    if (index->d_indices_buffer) cudaFree(index->d_indices_buffer);
    if (index->d_topk_scores_buffer) cudaFree(index->d_topk_scores_buffer);

    // Destroy CUDA streams
    if (index->compute_stream) cudaStreamDestroy(index->compute_stream);
    if (index->transfer_stream) cudaStreamDestroy(index->transfer_stream);

    if (index->initialized) cublasDestroy(index->cublas_handle);

    delete index;
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
    
    // Bulk indexing completed
    
    return batch_size;
}

// Set max tokens for truncation
void cuda_set_max_tokens(void* index_ptr, int max_tokens) {
    CUDAIndex* index = static_cast<CUDAIndex*>(index_ptr);
    if (index) {
        index->max_tokens = max_tokens;
        // Max tokens updated
    }
}

} // extern "C"