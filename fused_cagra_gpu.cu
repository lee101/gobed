// fused_cagra_gpu.cu - Robust GPU CAGRA implementation
// nvcc -O3 -arch=sm_80 --compiler-options "-fPIC" --shared -o libfused_cagra.so fused_cagra_gpu.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EMBED_DIM 512
#define BLOCK_SIZE 256
#define MAX_TOP_K 50
#define WARP_SIZE 32

struct FusedContext {
    int8_t* d_embed_weights;
    float* d_embed_scales;
    int8_t* d_database;
    float* d_db_scales;
    int vocab_size;
    int num_vectors;
    int top_k;
    int embed_dim;
};

// Optimized int8 dot product using dp4a intrinsic
__device__ __forceinline__ int32_t dot_product_int8(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int dim) {

    int32_t sum = 0;

    // Use dp4a for 4-element dot products (requires sm_61+)
    #pragma unroll 8
    for (int i = 0; i < dim; i += 4) {
        // Pack 4 int8 values into int32
        int32_t a_packed = *reinterpret_cast<const int32_t*>(a + i);
        int32_t b_packed = *reinterpret_cast<const int32_t*>(b + i);
        sum = __dp4a(a_packed, b_packed, sum);
    }

    return sum;
}

// Parallel reduction to find max value in shared memory
__device__ float reduce_max_shared(float* sdata, int tid) {
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    return sdata[0];
}

// Kernel for embedding generation and search
__global__ void fused_embed_search_kernel(
    const uint16_t* __restrict__ token_batch,
    const int* __restrict__ token_lengths,
    int batch_size,
    int max_tokens,
    const int8_t* __restrict__ embed_weights,
    const float* __restrict__ embed_scales,
    int vocab_size,
    const int8_t* __restrict__ database,
    const float* __restrict__ db_scales,
    int num_vectors,
    int embed_dim,
    float* __restrict__ output_distances,
    int* __restrict__ output_indices,
    int top_k) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;

    // Allocate shared memory
    extern __shared__ char shared_mem[];
    float* query_float = (float*)shared_mem;
    int8_t* query_int8 = (int8_t*)(query_float + EMBED_DIM);
    float* shared_max = (float*)(query_int8 + EMBED_DIM);

    // Get tokens for this batch
    const uint16_t* tokens = token_batch + batch_idx * max_tokens;
    int num_tokens = token_lengths[batch_idx];

    // Step 1: Generate query embedding
    // Initialize to zero
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        query_float[i] = 0.0f;
    }
    __syncthreads();

    // Accumulate token embeddings
    for (int t = 0; t < num_tokens && t < max_tokens; t++) {
        uint16_t token = tokens[t];
        if (token < vocab_size) {
            const int8_t* token_embed = embed_weights + token * EMBED_DIM;
            float scale = embed_scales[token];

            for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
                float val = (float)token_embed[i] * scale;
                atomicAdd(&query_float[i], val);
            }
        }
    }
    __syncthreads();

    // Average embeddings
    if (num_tokens > 0) {
        float inv_count = 1.0f / num_tokens;
        for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
            query_float[i] *= inv_count;
        }
    }
    __syncthreads();

    // Find max absolute value for quantization
    for (int i = tid; i < blockDim.x; i += blockDim.x) {
        shared_max[i] = 0.0f;
    }
    __syncthreads();

    float local_max = 0.0f;
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(query_float[i]));
    }
    shared_max[tid] = local_max;

    float max_val = reduce_max_shared(shared_max, tid);
    float query_scale = max_val / 127.0f + 1e-8f;

    // Quantize to int8
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        int val = __float2int_rn(query_float[i] / query_scale);
        query_int8[i] = max(-128, min(127, val));
    }
    __syncthreads();

    // Step 2: Search database
    // We interpret output buffer as similarities (higher is better)
    float* batch_dists = output_distances + batch_idx * top_k;
    int* batch_indices = output_indices + batch_idx * top_k;

    // Use shared memory for collecting results
    extern __shared__ char extra_shared[];
    float* all_dists = (float*)(extra_shared + EMBED_DIM * sizeof(float) + EMBED_DIM * sizeof(int8_t) + BLOCK_SIZE * sizeof(float));
    int* all_indices = (int*)(all_dists + BLOCK_SIZE * top_k);

    // Initialize thread-local top-k (track highest similarity)
    for (int i = 0; i < top_k; i++) {
        all_dists[tid * top_k + i] = -FLT_MAX;
        all_indices[tid * top_k + i] = -1;
    }
    __syncthreads();

    // Each thread processes different vectors
    for (int vid = tid; vid < num_vectors; vid += blockDim.x) {
        const int8_t* db_vec = database + vid * EMBED_DIM;

        // Compute dot product
        int32_t dot = dot_product_int8(query_int8, db_vec, EMBED_DIM);

        // Convert to similarity (positive, higher is better)
        float sim = (float)dot * query_scale * db_scales[vid];

        // Update thread-local top-k
        float* thread_dists = &all_dists[tid * top_k];
        int* thread_indices = &all_indices[tid * top_k];

        for (int k = 0; k < top_k; k++) {
            if (sim > thread_dists[k]) {
                // Shift and insert
                for (int j = top_k - 1; j > k; j--) {
                    thread_dists[j] = thread_dists[j-1];
                    thread_indices[j] = thread_indices[j-1];
                }
                thread_dists[k] = sim;
                thread_indices[k] = vid;
                break;
            }
        }
    }
    __syncthreads();

    // Final merge by thread 0
    if (tid == 0) {
        // Initialize output (track highest similarity)
        for (int i = 0; i < top_k; i++) {
            batch_dists[i] = -FLT_MAX;
            batch_indices[i] = -1;
        }

        // Merge all thread results
        for (int t = 0; t < blockDim.x; t++) {
            for (int i = 0; i < top_k; i++) {
                float dist = all_dists[t * top_k + i];
                int idx = all_indices[t * top_k + i];

                if (idx >= 0 && idx < num_vectors) {
                    // Insert into final top-k
                    for (int k = 0; k < top_k; k++) {
                        if (dist > batch_dists[k]) {
                            // Shift and insert
                            for (int j = top_k - 1; j > k; j--) {
                                batch_dists[j] = batch_dists[j-1];
                                batch_indices[j] = batch_indices[j-1];
                            }
                            batch_dists[k] = dist;
                            batch_indices[k] = idx;
                            break;
                        }
                    }
                }
            }
        }
    }
}

// Host-side functions
extern "C" {

void* create_fused_context(
    int8_t* embed_weights,
    float* embed_scales,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales,
    int num_vectors,
    int top_k) {

    printf("Creating CAGRA context: vocab=%d, vectors=%d, dim=%d, top_k=%d\n",
           vocab_size, num_vectors, embed_dim, top_k);

    FusedContext* ctx = new FusedContext;
    ctx->vocab_size = vocab_size;
    ctx->num_vectors = num_vectors;
    ctx->top_k = top_k;
    ctx->embed_dim = embed_dim;

    // Allocate GPU memory
    size_t embed_size = vocab_size * embed_dim * sizeof(int8_t);
    size_t db_size = num_vectors * embed_dim * sizeof(int8_t);

    cudaError_t err;

    err = cudaMalloc(&ctx->d_embed_weights, embed_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate embed_weights: %s\n", cudaGetErrorString(err));
        delete ctx;
        return nullptr;
    }

    err = cudaMalloc(&ctx->d_embed_scales, vocab_size * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate embed_scales: %s\n", cudaGetErrorString(err));
        cudaFree(ctx->d_embed_weights);
        delete ctx;
        return nullptr;
    }

    err = cudaMalloc(&ctx->d_database, db_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate database: %s\n", cudaGetErrorString(err));
        cudaFree(ctx->d_embed_weights);
        cudaFree(ctx->d_embed_scales);
        delete ctx;
        return nullptr;
    }

    err = cudaMalloc(&ctx->d_db_scales, num_vectors * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate db_scales: %s\n", cudaGetErrorString(err));
        cudaFree(ctx->d_embed_weights);
        cudaFree(ctx->d_embed_scales);
        cudaFree(ctx->d_database);
        delete ctx;
        return nullptr;
    }

    // Copy data to GPU
    cudaMemcpy(ctx->d_embed_weights, embed_weights, embed_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_embed_scales, embed_scales, vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_database, database, db_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_db_scales, db_scales, num_vectors * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in create_fused_context: %s\n", cudaGetErrorString(err));
    }

    return ctx;
}

void build_cagra_graph(void* context, int degree) {
    // Placeholder - using brute force for now
    cudaDeviceSynchronize();
}

void fused_search(
    void* context,
    uint16_t* token_batch,
    int* token_lengths,
    int batch_size,
    int max_tokens,
    float* output_distances,
    int* output_indices) {

    FusedContext* ctx = (FusedContext*)context;

    // Allocate device memory
    uint16_t* d_tokens;
    int* d_lengths;
    float* d_distances;
    int* d_indices;

    cudaMalloc(&d_tokens, batch_size * max_tokens * sizeof(uint16_t));
    cudaMalloc(&d_lengths, batch_size * sizeof(int));
    cudaMalloc(&d_distances, batch_size * ctx->top_k * sizeof(float));
    cudaMalloc(&d_indices, batch_size * ctx->top_k * sizeof(int));

    // Copy inputs
    cudaMemcpy(d_tokens, token_batch, batch_size * max_tokens * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, token_lengths, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate shared memory size
    size_t shared_size = EMBED_DIM * sizeof(float) +           // query_float
                        EMBED_DIM * sizeof(int8_t) +            // query_int8
                        BLOCK_SIZE * sizeof(float) +            // shared_max
                        BLOCK_SIZE * ctx->top_k * sizeof(float) +  // all_dists
                        BLOCK_SIZE * ctx->top_k * sizeof(int);     // all_indices

    // Launch kernel
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);

    fused_embed_search_kernel<<<grid, block, shared_size>>>(
        d_tokens, d_lengths,
        batch_size, max_tokens,
        ctx->d_embed_weights, ctx->d_embed_scales, ctx->vocab_size,
        ctx->d_database, ctx->d_db_scales, ctx->num_vectors,
        ctx->embed_dim,
        d_distances, d_indices,
        ctx->top_k
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fused_search kernel: %s\n", cudaGetErrorString(err));
    }

    // Copy results back
    cudaMemcpy(output_distances, d_distances, batch_size * ctx->top_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_indices, d_indices, batch_size * ctx->top_k * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_tokens);
    cudaFree(d_lengths);
    cudaFree(d_distances);
    cudaFree(d_indices);
}

void destroy_fused_context(void* context) {
    FusedContext* ctx = (FusedContext*)context;
    if (ctx) {
        cudaFree(ctx->d_embed_weights);
        cudaFree(ctx->d_embed_scales);
        cudaFree(ctx->d_database);
        cudaFree(ctx->d_db_scales);
        delete ctx;
    }
}

} // extern "C"
