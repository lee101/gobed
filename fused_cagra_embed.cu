// fused_cagra_fixed.cu - Fixed CAGRA implementation
// Compile: nvcc -O3 -arch=sm_80 -shared -fPIC -o libfused_cagra.so fused_cagra_fixed.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>
#include <stdio.h>

#define WARP_SIZE 32
#define CAGRA_DEGREE 32
#define EMBED_DIM 512
#define BLOCK_DIM 256
#define WARPS_PER_BLOCK (BLOCK_DIM / WARP_SIZE)
#define MAX_PROBES 64
#define SHARED_CACHE_SIZE 48

struct QuantParams {
    float scale;
    int8_t zero_point;
};

struct CAGRANode {
    int32_t neighbors[CAGRA_DEGREE];
    float distances[CAGRA_DEGREE];
};

struct FusedContext {
    int8_t* embed_weights;
    QuantParams* embed_scales;
    CAGRANode* cagra_graph;
    int8_t* database;
    QuantParams* db_scales;
    int vocab_size;
    int num_vectors;
    int top_k;
};

__device__ __forceinline__ int32_t int8_dot_product_512(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b) {

    int32_t sum = 0;
    #pragma unroll 4
    for (int i = 0; i < EMBED_DIM; i += 4) {
        int32_t a_packed = *reinterpret_cast<const int32_t*>(a + i);
        int32_t b_packed = *reinterpret_cast<const int32_t*>(b + i);
        sum = __dp4a(a_packed, b_packed, sum);
    }
    return sum;
}

__device__ __forceinline__ void warp_reduce_min(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

__device__ void generate_embedding_int8(
    const uint16_t* tokens,
    int num_tokens,
    const int8_t* __restrict__ embed_weights,
    const QuantParams* __restrict__ embed_scales,
    int vocab_size,
    int8_t* __restrict__ query_embed,
    float& query_scale) {

    int tid = threadIdx.x;

    __shared__ float embed_accum[EMBED_DIM];

    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        embed_accum[i] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < num_tokens && t < 20; t++) {
        uint16_t token = tokens[t];
        if (token >= vocab_size) continue;

        const int8_t* token_embed = embed_weights + token * EMBED_DIM;
        float scale = embed_scales[token].scale;

        for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
            atomicAdd(&embed_accum[i], float(token_embed[i]) * scale);
        }
    }
    __syncthreads();

    if (num_tokens > 0) {
        for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
            embed_accum[i] /= num_tokens;
        }
    }
    __syncthreads();

    __shared__ float shared_max_vals[BLOCK_DIM];
    float local_max = 0.0f;
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        float abs_val = fabsf(embed_accum[i]);
        if (abs_val > local_max) {
            local_max = abs_val;
        }
    }
    shared_max_vals[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = shared_max_vals[tid + offset];
            if (other > shared_max_vals[tid]) {
                shared_max_vals[tid] = other;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        query_scale = shared_max_vals[0] / 127.0f + 1e-8f;
    }
    __syncthreads();

    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        query_embed[i] = int8_t(embed_accum[i] / query_scale);
    }
    __syncthreads();
}

__device__ void cagra_search_fixed(
    const int8_t* __restrict__ query,
    float query_scale,
    const CAGRANode* __restrict__ graph,
    const int8_t* __restrict__ database,
    const QuantParams* __restrict__ db_scales,
    int num_vectors,
    float* top_k_dists,
    int* top_k_indices,
    int k) {

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Initialize results
    for (int i = tid; i < k; i += blockDim.x) {
        top_k_dists[i] = FLT_MAX;
        top_k_indices[i] = -1;
    }
    __syncthreads();

    // CAGRA graph traversal with multiple entry points
    // Use modular hashing for visited set - hash node IDs to fit in 8K bitmap
    __shared__ int visited_set[8192];  // Bitmap for 256K unique tracked nodes (with collisions)
    __shared__ float best_dists[64];   // Candidate pool
    __shared__ int best_indices[64];
    __shared__ int num_candidates;

    // Clear visited set
    for (int i = tid; i < 8192; i += blockDim.x) {
        visited_set[i] = 0;
    }
    if (tid == 0) num_candidates = 0;
    __syncthreads();

    // Start from multiple random entry points
    int num_entries = min(8, num_vectors);
    int entry_stride = num_vectors / num_entries;
    const int VISIT_MASK = 8192 * 32 - 1;  // 262143 - modular hash to fit in bitmap

    // Each warp handles one entry point initially
    if (warp_id < num_entries && lane_id == 0) {
        int entry = warp_id * entry_stride;
        int hash_entry = entry & VISIT_MASK;  // Modular hash
        atomicOr(&visited_set[hash_entry / 32], 1 << (hash_entry % 32));

        const int8_t* db_vec = database + entry * EMBED_DIM;
        int32_t dot = int8_dot_product_512(query, db_vec);
        float dist = -dot * query_scale * db_scales[entry].scale;

        int slot = atomicAdd(&num_candidates, 1);
        if (slot < 64) {
            best_dists[slot] = dist;
            best_indices[slot] = entry;
        }
    }
    __syncthreads();

    // Graph traversal iterations
    for (int iter = 0; iter < MAX_PROBES; iter++) {
        // Find best unvisited candidate
        __shared__ int best_candidate;
        __shared__ float best_candidate_dist;

        if (tid == 0) {
            best_candidate = -1;
            best_candidate_dist = FLT_MAX;
            for (int i = 0; i < num_candidates && i < 64; i++) {
                if (best_indices[i] >= 0 && best_dists[i] < best_candidate_dist) {
                    best_candidate_dist = best_dists[i];
                    best_candidate = best_indices[i];
                }
            }
        }
        __syncthreads();

        if (best_candidate < 0) break;

        // Explore neighbors of best candidate
        int node_idx = best_candidate;
        if (node_idx >= 0 && node_idx < num_vectors) {
            // Each thread in warp handles different neighbor
            int neighbor_idx = lane_id;
            if (neighbor_idx < CAGRA_DEGREE) {
                int neighbor = graph[node_idx].neighbors[neighbor_idx];
                if (neighbor >= 0 && neighbor < num_vectors) {
                    // Check if visited using modular hash
                    int hash_neighbor = neighbor & VISIT_MASK;
                    int word = hash_neighbor / 32;
                    int bit = hash_neighbor % 32;
                    int old = atomicOr(&visited_set[word], 1 << bit);
                    bool was_visited = (old >> bit) & 1;

                    if (!was_visited) {
                        const int8_t* db_vec = database + neighbor * EMBED_DIM;
                        int32_t dot = int8_dot_product_512(query, db_vec);
                        float dist = -dot * query_scale * db_scales[neighbor].scale;

                        // Add to candidates
                        int slot = atomicAdd(&num_candidates, 1);
                        if (slot < 64) {
                            best_dists[slot] = dist;
                            best_indices[slot] = neighbor;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Extract top-k from candidates
    if (tid == 0) {
        for (int c = 0; c < num_candidates && c < 64; c++) {
            float dist = best_dists[c];
            int idx = best_indices[c];
            if (idx >= 0) {
                for (int j = 0; j < k; j++) {
                    if (dist < top_k_dists[j]) {
                        for (int l = k-1; l > j; l--) {
                            top_k_dists[l] = top_k_dists[l-1];
                            top_k_indices[l] = top_k_indices[l-1];
                        }
                        top_k_dists[j] = dist;
                        top_k_indices[j] = idx;
                        break;
                    }
                }
            }
        }
    }
    __syncthreads();
}

__global__ void fused_embed_cagra_search_kernel(
    const uint16_t* __restrict__ token_batch,
    const int* __restrict__ token_lengths,
    int batch_size,
    int max_tokens,
    const FusedContext* ctx,  // Pass as pointer
    float* __restrict__ output_distances,
    int* __restrict__ output_indices) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    __shared__ int8_t query_embed[EMBED_DIM];
    __shared__ float query_scale;
    __shared__ float top_k_dists[50];  // Max top_k
    __shared__ int top_k_indices[50];

    const uint16_t* tokens = token_batch + batch_idx * max_tokens;
    int num_tokens = token_lengths[batch_idx];

    generate_embedding_int8(
        tokens, num_tokens,
        ctx->embed_weights, ctx->embed_scales, ctx->vocab_size,
        query_embed, query_scale
    );

    float* batch_dists = output_distances + batch_idx * ctx->top_k;
    int* batch_indices = output_indices + batch_idx * ctx->top_k;

    cagra_search_fixed(
        query_embed, query_scale,
        ctx->cagra_graph, ctx->database, ctx->db_scales,
        ctx->num_vectors,
        top_k_dists, top_k_indices, ctx->top_k
    );

    // Copy results
    int tid = threadIdx.x;
    for (int i = tid; i < ctx->top_k; i += blockDim.x) {
        batch_dists[i] = top_k_dists[i];
        batch_indices[i] = top_k_indices[i];
    }
}

// Optimized k-NN graph builder with sampling for large datasets
__global__ void build_knn_graph_kernel(
    const int8_t* __restrict__ database,
    const QuantParams* __restrict__ db_scales,
    int num_vectors,
    int embed_dim,
    CAGRANode* __restrict__ graph,
    int degree) {

    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_vectors) return;

    const int8_t* query_vec = database + vid * embed_dim;
    float query_scale = db_scales[vid].scale;

    float distances[CAGRA_DEGREE];
    int indices[CAGRA_DEGREE];

    for (int i = 0; i < degree; i++) {
        distances[i] = FLT_MAX;
        indices[i] = -1;
    }

    // For large datasets, use sampling + local refinement
    // Sample stride based on dataset size to limit comparisons
    int sample_size = min(num_vectors, 10000);  // Compare against 10K samples max
    int stride = max(1, num_vectors / sample_size);

    // First pass: sample-based search
    for (int s = 0; s < sample_size; s++) {
        int other = (s * stride + vid * 17) % num_vectors;  // Pseudo-random but deterministic
        if (other == vid) continue;

        const int8_t* other_vec = database + other * embed_dim;
        float other_scale = db_scales[other].scale;

        int32_t dot = int8_dot_product_512(query_vec, other_vec);
        float dist = -dot * query_scale * other_scale;

        for (int i = 0; i < degree; i++) {
            if (dist < distances[i]) {
                for (int j = degree-1; j > i; j--) {
                    distances[j] = distances[j-1];
                    indices[j] = indices[j-1];
                }
                distances[i] = dist;
                indices[i] = other;
                break;
            }
        }
    }

    // Store in graph
    for (int i = 0; i < degree; i++) {
        graph[vid].neighbors[i] = indices[i];
        graph[vid].distances[i] = distances[i];
    }
}

extern "C" {

void* create_fused_context(
    int8_t* embed_weights,
    float* embed_scales_raw,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales_raw,
    int num_vectors,
    int top_k) {

    FusedContext* ctx = new FusedContext;
    FusedContext* d_ctx;

    cudaMalloc(&ctx->embed_weights, vocab_size * embed_dim * sizeof(int8_t));
    cudaMalloc(&ctx->embed_scales, vocab_size * sizeof(QuantParams));
    cudaMalloc(&ctx->database, num_vectors * embed_dim * sizeof(int8_t));
    cudaMalloc(&ctx->db_scales, num_vectors * sizeof(QuantParams));
    cudaMalloc(&ctx->cagra_graph, num_vectors * sizeof(CAGRANode));

    cudaMemcpy(ctx->embed_weights, embed_weights,
               vocab_size * embed_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->database, database,
               num_vectors * embed_dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    QuantParams* h_embed_scales = new QuantParams[vocab_size];
    QuantParams* h_db_scales = new QuantParams[num_vectors];

    for (int i = 0; i < vocab_size; i++) {
        h_embed_scales[i].scale = embed_scales_raw[i];
        h_embed_scales[i].zero_point = 0;
    }
    for (int i = 0; i < num_vectors; i++) {
        h_db_scales[i].scale = db_scales_raw[i];
        h_db_scales[i].zero_point = 0;
    }

    cudaMemcpy(ctx->embed_scales, h_embed_scales,
               vocab_size * sizeof(QuantParams), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->db_scales, h_db_scales,
               num_vectors * sizeof(QuantParams), cudaMemcpyHostToDevice);

    delete[] h_embed_scales;
    delete[] h_db_scales;

    ctx->vocab_size = vocab_size;
    ctx->num_vectors = num_vectors;
    ctx->top_k = top_k;

    return ctx;
}

void build_cagra_graph(void* context, int degree) {
    FusedContext* ctx = (FusedContext*)context;

    // Build k-NN graph on GPU
    int threads = 256;
    int blocks = (ctx->num_vectors + threads - 1) / threads;

    build_knn_graph_kernel<<<blocks, threads>>>(
        ctx->database,
        ctx->db_scales,
        ctx->num_vectors,
        EMBED_DIM,
        ctx->cagra_graph,  // Use the actual graph pointer
        CAGRA_DEGREE
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in build_cagra_graph: %s\n", cudaGetErrorString(err));
    }
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

    uint16_t* d_tokens;
    int* d_lengths;
    float* d_distances;
    int* d_indices;

    cudaMalloc(&d_tokens, batch_size * max_tokens * sizeof(uint16_t));
    cudaMalloc(&d_lengths, batch_size * sizeof(int));
    cudaMalloc(&d_distances, batch_size * ctx->top_k * sizeof(float));
    cudaMalloc(&d_indices, batch_size * ctx->top_k * sizeof(int));

    cudaMemcpy(d_tokens, token_batch,
               batch_size * max_tokens * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, token_lengths,
               batch_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(batch_size);
    dim3 block(BLOCK_DIM);

    // Copy context to device
    FusedContext* d_ctx;
    cudaMalloc(&d_ctx, sizeof(FusedContext));
    cudaMemcpy(d_ctx, ctx, sizeof(FusedContext), cudaMemcpyHostToDevice);

    fused_embed_cagra_search_kernel<<<grid, block>>>(
        d_tokens, d_lengths,
        batch_size, max_tokens,
        d_ctx,  // Pass device context pointer
        d_distances, d_indices
    );

    // CRITICAL: Add synchronization
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fused_search: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output_distances, d_distances,
               batch_size * ctx->top_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_indices, d_indices,
               batch_size * ctx->top_k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_ctx);
    cudaFree(d_tokens);
    cudaFree(d_lengths);
    cudaFree(d_distances);
    cudaFree(d_indices);
}

void destroy_fused_context(void* context) {
    FusedContext* ctx = (FusedContext*)context;
    cudaFree(ctx->embed_weights);
    cudaFree(ctx->embed_scales);
    cudaFree(ctx->database);
    cudaFree(ctx->db_scales);
    cudaFree(ctx->cagra_graph);
    delete ctx;
}

} // extern "C"
