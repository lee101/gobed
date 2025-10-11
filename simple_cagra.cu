// simple_cagra.cu - Simplified CAGRA kernel for testing
// Compile: nvcc -O3 -arch=sm_86 --shared --compiler-options '-fPIC' -o libsimple_cagra.so simple_cagra.cu

#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>

#define WARP_SIZE 32
#define EMBED_DIM 512
#define BLOCK_DIM 256

// Simplified structures
struct SimpleQuantParams {
    float scale;
};

struct SimpleContext {
    int8_t* embed_weights;      // [vocab_size x embed_dim]
    SimpleQuantParams* embed_scales;
    int8_t* database;          // [num_vectors x embed_dim]
    SimpleQuantParams* db_scales;
    int vocab_size;
    int num_vectors;
    int top_k;
};

// Simple INT8 dot product
__device__ __forceinline__ int32_t simple_dot_product(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b) {

    int32_t sum = 0;
    for (int i = 0; i < EMBED_DIM; i += 4) {
        // Use vectorized operations when possible
        int32_t a_packed = *reinterpret_cast<const int32_t*>(a + i);
        int32_t b_packed = *reinterpret_cast<const int32_t*>(b + i);

        // Manual INT8 multiplication and accumulation
        int8_t a0 = (a_packed) & 0xFF;
        int8_t a1 = (a_packed >> 8) & 0xFF;
        int8_t a2 = (a_packed >> 16) & 0xFF;
        int8_t a3 = (a_packed >> 24) & 0xFF;

        int8_t b0 = (b_packed) & 0xFF;
        int8_t b1 = (b_packed >> 8) & 0xFF;
        int8_t b2 = (b_packed >> 16) & 0xFF;
        int8_t b3 = (b_packed >> 24) & 0xFF;

        sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }
    return sum;
}

// Simple embedding generation
__device__ void simple_generate_embedding(
    const uint16_t* tokens,
    int num_tokens,
    const int8_t* __restrict__ embed_weights,
    const SimpleQuantParams* __restrict__ embed_scales,
    int vocab_size,
    int8_t* __restrict__ query_embed,
    float& query_scale) {

    int tid = threadIdx.x;

    // Use shared memory for accumulation
    __shared__ float embed_accum[EMBED_DIM];

    // Initialize
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        embed_accum[i] = 0.0f;
    }
    __syncthreads();

    // Simple average of token embeddings
    for (int t = 0; t < num_tokens; t++) {
        uint16_t token = tokens[t];
        if (token >= vocab_size) continue;

        const int8_t* token_embed = embed_weights + token * EMBED_DIM;
        float scale = embed_scales[token].scale;

        for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
            float val = token_embed[i] * scale;
            atomicAdd(&embed_accum[i], val);
        }
    }
    __syncthreads();

    // Average and quantize
    if (tid == 0) {
        float max_val = 0.0f;
        for (int i = 0; i < EMBED_DIM; i++) {
            embed_accum[i] /= num_tokens;
            max_val = fmaxf(max_val, fabsf(embed_accum[i]));
        }
        query_scale = max_val / 127.0f + 1e-8f; // Avoid division by zero
    }
    __syncthreads();

    // Quantize
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        float val = embed_accum[i] / query_scale;
        query_embed[i] = (int8_t)(__float2int_rn(fminf(fmaxf(val, -127.0f), 127.0f)));
    }
    __syncthreads();
}

// Simple brute-force search (no graph for now)
__device__ void simple_brute_force_search(
    const int8_t* __restrict__ query,
    float query_scale,
    const int8_t* __restrict__ database,
    const SimpleQuantParams* __restrict__ db_scales,
    int num_vectors,
    float* __restrict__ top_k_dists,
    int* __restrict__ top_k_indices,
    int k) {

    int tid = threadIdx.x;

    // Initialize results
    if (tid < k) {
        top_k_dists[tid] = FLT_MAX;
        top_k_indices[tid] = -1;
    }
    __syncthreads();

    // Each thread processes some vectors
    for (int vec_id = tid; vec_id < num_vectors; vec_id += blockDim.x) {
        const int8_t* db_vec = database + vec_id * EMBED_DIM;

        // Compute distance
        int32_t dot = simple_dot_product(query, db_vec);
        float similarity = dot * query_scale * db_scales[vec_id].scale;
        float distance = -similarity; // Convert to distance (lower is better)

        // Update top-k (simple bubble sort approach)
        for (int i = 0; i < k; i++) {
            if (distance < top_k_dists[i]) {
                // Shift elements
                for (int j = k - 1; j > i; j--) {
                    top_k_dists[j] = top_k_dists[j - 1];
                    top_k_indices[j] = top_k_indices[j - 1];
                }
                top_k_dists[i] = distance;
                top_k_indices[i] = vec_id;
                break;
            }
        }
    }
    __syncthreads();
}

// Simplified kernel
__global__ void simple_fused_kernel(
    const uint16_t* __restrict__ token_batch,
    const int* __restrict__ token_lengths,
    int batch_size,
    int max_tokens,
    SimpleContext ctx,
    float* __restrict__ output_distances,
    int* __restrict__ output_indices) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Shared memory for query embedding
    __shared__ int8_t query_embed[EMBED_DIM];
    __shared__ float query_scale;

    // Generate embedding
    const uint16_t* tokens = token_batch + batch_idx * max_tokens;
    int num_tokens = token_lengths[batch_idx];

    simple_generate_embedding(
        tokens, num_tokens,
        ctx.embed_weights, ctx.embed_scales, ctx.vocab_size,
        query_embed, query_scale
    );

    // Search
    float* batch_dists = output_distances + batch_idx * ctx.top_k;
    int* batch_indices = output_indices + batch_idx * ctx.top_k;

    simple_brute_force_search(
        query_embed, query_scale,
        ctx.database, ctx.db_scales, ctx.num_vectors,
        batch_dists, batch_indices, ctx.top_k
    );
}

// C interface
extern "C" {

void* create_simple_context(
    int8_t* embed_weights,
    float* embed_scales_raw,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales_raw,
    int num_vectors,
    int top_k) {

    SimpleContext* ctx = new SimpleContext;

    // Allocate device memory
    cudaMalloc(&ctx->embed_weights, vocab_size * embed_dim * sizeof(int8_t));
    cudaMalloc(&ctx->embed_scales, vocab_size * sizeof(SimpleQuantParams));
    cudaMalloc(&ctx->database, num_vectors * embed_dim * sizeof(int8_t));
    cudaMalloc(&ctx->db_scales, num_vectors * sizeof(SimpleQuantParams));

    // Copy data
    cudaMemcpy(ctx->embed_weights, embed_weights,
               vocab_size * embed_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->database, database,
               num_vectors * embed_dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Convert scales
    SimpleQuantParams* h_embed_scales = new SimpleQuantParams[vocab_size];
    SimpleQuantParams* h_db_scales = new SimpleQuantParams[num_vectors];

    for (int i = 0; i < vocab_size; i++) {
        h_embed_scales[i].scale = embed_scales_raw[i];
    }
    for (int i = 0; i < num_vectors; i++) {
        h_db_scales[i].scale = db_scales_raw[i];
    }

    cudaMemcpy(ctx->embed_scales, h_embed_scales,
               vocab_size * sizeof(SimpleQuantParams), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->db_scales, h_db_scales,
               num_vectors * sizeof(SimpleQuantParams), cudaMemcpyHostToDevice);

    delete[] h_embed_scales;
    delete[] h_db_scales;

    ctx->vocab_size = vocab_size;
    ctx->num_vectors = num_vectors;
    ctx->top_k = top_k;

    return ctx;
}

void simple_search(
    void* context,
    uint16_t* token_batch,
    int* token_lengths,
    int batch_size,
    int max_tokens,
    float* output_distances,
    int* output_indices) {

    SimpleContext* ctx = (SimpleContext*)context;

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
    cudaMemcpy(d_tokens, token_batch,
               batch_size * max_tokens * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, token_lengths,
               batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid(batch_size);
    dim3 block(BLOCK_DIM);

    simple_fused_kernel<<<grid, block>>>(
        d_tokens, d_lengths,
        batch_size, max_tokens,
        *ctx,
        d_distances, d_indices
    );

    // Copy results
    cudaMemcpy(output_distances, d_distances,
               batch_size * ctx->top_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_indices, d_indices,
               batch_size * ctx->top_k * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_lengths);
    cudaFree(d_distances);
    cudaFree(d_indices);
}

void destroy_simple_context(void* context) {
    SimpleContext* ctx = (SimpleContext*)context;
    cudaFree(ctx->embed_weights);
    cudaFree(ctx->embed_scales);
    cudaFree(ctx->database);
    cudaFree(ctx->db_scales);
    delete ctx;
}

} // extern "C"