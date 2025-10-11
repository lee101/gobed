// fused_cagra_simple.cu - Simplified, robust CAGRA implementation
// nvcc -O3 -arch=sm_80 -shared -fPIC -o libfused_cagra.so fused_cagra_simple.cu

#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define EMBED_DIM 512
#define BLOCK_SIZE 256
#define MAX_TOP_K 50

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

__device__ float compute_int8_cosine_distance(
    const int8_t* vec1, float scale1,
    const int8_t* vec2, float scale2,
    int dim) {

    int32_t dot = 0;
    int32_t norm1 = 0;
    int32_t norm2 = 0;

    #pragma unroll 8
    for (int i = 0; i < dim; i++) {
        int32_t v1 = vec1[i];
        int32_t v2 = vec2[i];
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }

    float cosine = (float)dot / (sqrtf((float)norm1) * sqrtf((float)norm2) + 1e-8f);
    return 1.0f - cosine;
}

__global__ void embed_and_search_kernel(
    const uint16_t* tokens,
    const int* token_lengths,
    int max_tokens,
    const int8_t* embed_weights,
    const float* embed_scales,
    int vocab_size,
    const int8_t* database,
    const float* db_scales,
    int num_vectors,
    int embed_dim,
    float* output_distances,
    int* output_indices,
    int top_k) {

    extern __shared__ char shared_mem[];
    float* query_embed_float = (float*)shared_mem;
    int8_t* query_embed = (int8_t*)(query_embed_float + embed_dim);

    // Shared auxiliaries for robust reductions and merge
    __shared__ int s_max_abs_i;      // int representation of max(abs(value))
    __shared__ float s_query_scale;  // shared query scale
    __shared__ float s_thread_dists[BLOCK_SIZE][10];
    __shared__ int   s_thread_indices[BLOCK_SIZE][10];

    int tid = threadIdx.x;

    // Step 1: Generate query embedding by averaging token embeddings
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        query_embed_float[i] = 0.0f;
    }
    __syncthreads();

    // Determine this block's slice
    int num_tokens = token_lengths[blockIdx.x];
    const uint16_t* tokens_b = tokens + blockIdx.x * max_tokens;

    // Fast-path: single-token query -> direct copy yields exact match potential
    if (num_tokens == 1) {
        uint16_t token = tokens_b[0];
        if (token < vocab_size) {
            const int8_t* token_embed = embed_weights + token * embed_dim;
            // Copy int8 directly to query_embed and set scale to token scale
            for (int i = tid; i < embed_dim; i += blockDim.x) {
                query_embed[i] = token_embed[i];
            }
            __syncthreads();
            if (tid == 0) { s_query_scale = embed_scales[token]; }
            __syncthreads();
        }
    } else {
        // Accumulate embeddings
        for (int t = 0; t < num_tokens && t < 20; t++) {
            uint16_t token = tokens_b[t];
            if (token < vocab_size) {
                const int8_t* token_embed = embed_weights + token * embed_dim;
                float scale = embed_scales[token];

                for (int i = tid; i < embed_dim; i += blockDim.x) {
                    float val = (float)token_embed[i] * scale;
                    atomicAdd(&query_embed_float[i], val);
                }
            }
        }
    }
    __syncthreads();

    // Average and quantize
    if (num_tokens > 0) {
        if (num_tokens == 1) {
            // Initialize per-thread top-k
            float thread_best_dists[10];
            int   thread_best_indices[10];
            for (int i = 0; i < top_k && i < 10; i++) { thread_best_dists[i] = FLT_MAX; thread_best_indices[i] = -1; }

            for (int vid = tid; vid < num_vectors; vid += blockDim.x) {
                const int8_t* db_vec = database + vid * embed_dim;
                int32_t dot = 0;
                #pragma unroll 8
                for (int i = 0; i < embed_dim; i++) { dot += (int32_t)query_embed[i] * (int32_t)db_vec[i]; }
                float dist = -(float)dot * s_query_scale * db_scales[vid];
                for (int i = 0; i < top_k && i < 10; i++) {
                    if (dist < thread_best_dists[i]) {
                        for (int j = min(9, top_k-1); j > i; j--) { thread_best_dists[j] = thread_best_dists[j-1]; thread_best_indices[j] = thread_best_indices[j-1]; }
                        thread_best_dists[i] = dist; thread_best_indices[i] = vid; break;
                    }
                }
            }
            __syncthreads();
            // Publish thread results to shared arrays
            for (int i = 0; i < top_k && i < 10; i++) {
                s_thread_dists[tid][i] = thread_best_dists[i];
                s_thread_indices[tid][i] = thread_best_indices[i];
            }
            __syncthreads();
            if (tid == 0) {
                for (int i = 0; i < top_k; i++) { output_distances[i] = FLT_MAX; output_indices[i] = -1; }
                for (int t = 0; t < blockDim.x; t++) {
                    for (int i = 0; i < top_k && i < 10; i++) {
                        int idx = s_thread_indices[t][i]; if (idx < 0) continue;
                        float dist = s_thread_dists[t][i];
                        for (int j = 0; j < top_k; j++) {
                            if (dist < output_distances[j]) {
                                for (int k2 = top_k-1; k2 > j; k2--) { output_distances[k2] = output_distances[k2-1]; output_indices[k2] = output_indices[k2-1]; }
                                output_distances[j] = dist; output_indices[j] = idx; break;
                            }
                        }
                    }
                }
            }
            return;
        }

        // Reduce max(abs) across block to determine scale
        if (tid == 0) s_max_abs_i = __float_as_int(0.0f);
        __syncthreads();

        float local_max = 0.0f;
        for (int i = tid; i < embed_dim; i += blockDim.x) {
            query_embed_float[i] /= num_tokens;
            float abs_val = fabsf(query_embed_float[i]);
            if (abs_val > local_max) local_max = abs_val;
        }
        // Atomically reduce local maxima into shared integer
        atomicMax(&s_max_abs_i, __float_as_int(local_max));
        __syncthreads();

        if (tid == 0) {
            float max_abs = __int_as_float(s_max_abs_i);
            s_query_scale = max_abs / 127.0f + 1e-8f; // Avoid div by zero
        }
        __syncthreads();

        // Quantize to int8 using shared query scale
        for (int i = tid; i < embed_dim; i += blockDim.x) {
            int val = __float2int_rn(query_embed_float[i] / s_query_scale);
            val = max(-128, min(127, val));
            query_embed[i] = (int8_t)val;
        }
        __syncthreads();

        // Step 2: Search - each thread handles different database vectors
        float thread_best_dists[10];
        int   thread_best_indices[10];

        for (int i = 0; i < top_k && i < 10; i++) {
            thread_best_dists[i] = FLT_MAX;
            thread_best_indices[i] = -1;
        }

        // Each thread processes a subset of vectors
        for (int vid = tid; vid < num_vectors; vid += blockDim.x) {
            const int8_t* db_vec = database + vid * embed_dim;

            // Simple dot product distance
            int32_t dot = 0;
            #pragma unroll 8
            for (int i = 0; i < embed_dim; i++) {
                dot += (int32_t)query_embed[i] * (int32_t)db_vec[i];
            }

            float dist = -(float)dot * s_query_scale * db_scales[vid];

            // Update thread-local top-k
            for (int i = 0; i < top_k && i < 10; i++) {
                if (dist < thread_best_dists[i]) {
                    // Shift and insert
                    for (int j = min(9, top_k-1); j > i; j--) {
                        thread_best_dists[j] = thread_best_dists[j-1];
                        thread_best_indices[j] = thread_best_indices[j-1];
                    }
                    thread_best_dists[i] = dist;
                    thread_best_indices[i] = vid;
                    break;
                }
            }
        }

        __syncthreads();

        // Publish thread-local bests to shared memory
        for (int i = 0; i < top_k && i < 10; i++) {
            s_thread_dists[tid][i] = thread_best_dists[i];
            s_thread_indices[tid][i] = thread_best_indices[i];
        }
        __syncthreads();

        // Step 3: Global reduction - only thread 0 merges results
        if (tid == 0) {
            // Initialize output
            for (int i = 0; i < top_k; i++) {
                output_distances[i] = FLT_MAX;
                output_indices[i] = -1;
            }

            // Merge all thread results
            for (int t = 0; t < blockDim.x; t++) {
                for (int i = 0; i < top_k && i < 10; i++) {
                    int idx = s_thread_indices[t][i];
                    if (idx >= 0) {
                        float dist = s_thread_dists[t][i];
                        // Insert into final top-k
                        for (int j = 0; j < top_k; j++) {
                            if (dist < output_distances[j]) {
                                for (int k = top_k - 1; k > j; k--) {
                                    output_distances[k] = output_distances[k-1];
                                    output_indices[k] = output_indices[k-1];
                                }
                                output_distances[j] = dist;
                                output_indices[j] = idx;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

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

    FusedContext* ctx = new FusedContext;

    // Allocate device memory
    size_t embed_size = vocab_size * embed_dim * sizeof(int8_t);
    size_t db_size = num_vectors * embed_dim * sizeof(int8_t);

    cudaMalloc(&ctx->d_embed_weights, embed_size);
    cudaMalloc(&ctx->d_embed_scales, vocab_size * sizeof(float));
    cudaMalloc(&ctx->d_database, db_size);
    cudaMalloc(&ctx->d_db_scales, num_vectors * sizeof(float));

    // Copy to device
    cudaMemcpy(ctx->d_embed_weights, embed_weights, embed_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_embed_scales, embed_scales, vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_database, database, db_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_db_scales, db_scales, num_vectors * sizeof(float), cudaMemcpyHostToDevice);

    ctx->vocab_size = vocab_size;
    ctx->num_vectors = num_vectors;
    ctx->top_k = top_k;
    ctx->embed_dim = embed_dim;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in create_fused_context: %s\n", cudaGetErrorString(err));
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void build_cagra_graph(void* context, int degree) {
    // For now, we're using brute force search, so no graph to build
    // This is a placeholder for future graph-based optimization
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

    // Allocate device memory for inputs/outputs
    uint16_t* d_tokens;
    int* d_lengths;
    float* d_distances;
    int* d_indices;

    cudaMalloc(&d_tokens, batch_size * max_tokens * sizeof(uint16_t));
    cudaMalloc(&d_lengths, batch_size * sizeof(int));
    cudaMalloc(&d_distances, batch_size * ctx->top_k * sizeof(float));
    cudaMalloc(&d_indices, batch_size * ctx->top_k * sizeof(int));

    cudaMemcpy(d_tokens, token_batch, batch_size * max_tokens * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, token_lengths, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Shared memory size per block: float embeddings + int8 quantized
    size_t shared_size = ctx->embed_dim * (sizeof(float) + sizeof(int8_t));

    // Launch one block per query for better GPU utilization
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    embed_and_search_kernel<<<grid, block, shared_size>>>(
        d_tokens,
        d_lengths,
        max_tokens,
        ctx->d_embed_weights,
        ctx->d_embed_scales,
        ctx->vocab_size,
        ctx->d_database,
        ctx->d_db_scales,
        ctx->num_vectors,
        ctx->embed_dim,
        d_distances,
        d_indices,
        ctx->top_k
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fused_search: %s\n", cudaGetErrorString(err));
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
