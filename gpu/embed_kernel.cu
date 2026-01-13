// embed_kernel.cu - Ultra-optimized GPU embedding kernel with per-token scales
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Optimized kernel: Each thread handles one dimension across all tokens
// Uses shared memory for token IDs and coalesced memory access
__global__ void embed_tokens_scaled_kernel(
    const int8_t* __restrict__ embedding_table,  // [vocab_size x 512]
    const float* __restrict__ scales,            // [vocab_size]
    const int16_t* __restrict__ tokens,          // [num_tokens]
    float* __restrict__ output,                  // [512]
    int num_tokens,
    int vocab_size
) {
    __shared__ int16_t s_tokens[256];  // Cache tokens in shared memory
    __shared__ float s_scales[256];     // Cache scales

    int dim = threadIdx.x + blockIdx.x * blockDim.x;
    if (dim >= 512) return;

    float sum = 0.0f;
    int valid = 0;

    // Process tokens in chunks that fit in shared memory
    for (int chunk_start = 0; chunk_start < num_tokens; chunk_start += 256) {
        int chunk_size = min(256, num_tokens - chunk_start);

        // Cooperatively load tokens and scales into shared memory
        if (threadIdx.x < chunk_size) {
            int16_t tok = tokens[chunk_start + threadIdx.x];
            s_tokens[threadIdx.x] = tok;
            if (tok >= 0 && tok < vocab_size) {
                s_scales[threadIdx.x] = scales[tok];
            } else {
                s_scales[threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        // Accumulate embeddings for this chunk
        #pragma unroll 8
        for (int i = 0; i < chunk_size; i++) {
            int16_t tok = s_tokens[i];
            if (tok >= 0 && tok < vocab_size) {
                int8_t val = embedding_table[tok * 512 + dim];
                sum += float(val) * s_scales[i];
                if (dim == 0) valid++;  // Only count once
            }
        }
        __syncthreads();
    }

    // Average (use block reduction for valid count if needed)
    __shared__ int s_valid;
    if (dim == 0) {
        s_valid = valid;
    }
    __syncthreads();

    if (s_valid > 0) {
        output[dim] = sum / float(s_valid);
    } else {
        output[dim] = 0.0f;
    }
}

// Batch version: Process multiple queries in parallel
// Each block handles one query
__global__ void embed_tokens_batch_kernel(
    const int8_t* __restrict__ embedding_table,  // [vocab_size x 512]
    const float* __restrict__ scales,            // [vocab_size]
    const int16_t* __restrict__ all_tokens,      // [batch_size x max_tokens] flattened
    const int* __restrict__ token_counts,        // [batch_size] actual token count per query
    float* __restrict__ outputs,                 // [batch_size x 512]
    int max_tokens,
    int vocab_size,
    int batch_size
) {
    int query_idx = blockIdx.y;
    if (query_idx >= batch_size) return;

    int dim = threadIdx.x + blockIdx.x * blockDim.x;
    if (dim >= 512) return;

    const int16_t* tokens = all_tokens + query_idx * max_tokens;
    int num_tokens = token_counts[query_idx];
    float* output = outputs + query_idx * 512;

    float sum = 0.0f;
    int valid = 0;

    for (int i = 0; i < num_tokens; i++) {
        int16_t tok = tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            int8_t val = embedding_table[tok * 512 + dim];
            sum += float(val) * scales[tok];
            if (dim == 0) valid++;
        }
    }

    // Broadcast valid count
    __shared__ int s_valid;
    if (threadIdx.x == 0) {
        s_valid = valid;
    }
    __syncthreads();

    if (s_valid > 0) {
        output[dim] = sum / float(s_valid);
    } else {
        output[dim] = 0.0f;
    }
}

// Ultra-optimized version using int4 vectorized loads for embedding table
// Processes 4 dimensions per thread
__global__ void embed_tokens_vec4_kernel(
    const int8_t* __restrict__ embedding_table,  // [vocab_size x 512]
    const float* __restrict__ scales,            // [vocab_size]
    const int16_t* __restrict__ tokens,          // [num_tokens]
    float* __restrict__ output,                  // [512]
    int num_tokens,
    int vocab_size
) {
    int dim_base = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (dim_base >= 512) return;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int valid = 0;

    for (int i = 0; i < num_tokens; i++) {
        int16_t tok = tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            float scale = scales[tok];
            const int8_t* emb = embedding_table + tok * 512 + dim_base;

            // Vectorized load (4 bytes)
            sum0 += float(emb[0]) * scale;
            sum1 += float(emb[1]) * scale;
            sum2 += float(emb[2]) * scale;
            sum3 += float(emb[3]) * scale;

            if (dim_base == 0) valid++;
        }
    }

    // Get valid count
    __shared__ int s_valid;
    if (dim_base == 0) {
        s_valid = valid;
    }
    __syncthreads();

    if (s_valid > 0) {
        float inv = 1.0f / float(s_valid);
        output[dim_base] = sum0 * inv;
        output[dim_base + 1] = sum1 * inv;
        output[dim_base + 2] = sum2 * inv;
        output[dim_base + 3] = sum3 * inv;
    } else {
        output[dim_base] = 0.0f;
        output[dim_base + 1] = 0.0f;
        output[dim_base + 2] = 0.0f;
        output[dim_base + 3] = 0.0f;
    }
}

// Integer-only accumulation version (no float until final step)
// Uses int32 accumulation which is faster on some hardware
__global__ void embed_tokens_int32_kernel(
    const int8_t* __restrict__ embedding_table,  // [vocab_size x 512]
    const int16_t* __restrict__ scale_q,         // [vocab_size] quantized scales (Q8)
    const int16_t* __restrict__ tokens,          // [num_tokens]
    float* __restrict__ output,                  // [512]
    float global_scale,                          // Global dequant scale
    int num_tokens,
    int vocab_size
) {
    int dim = threadIdx.x + blockIdx.x * blockDim.x;
    if (dim >= 512) return;

    int32_t sum = 0;
    int valid = 0;

    for (int i = 0; i < num_tokens; i++) {
        int16_t tok = tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            int8_t val = embedding_table[tok * 512 + dim];
            int16_t scale = scale_q[tok];
            // int8 * int16 -> int32 accumulation
            sum += int32_t(val) * int32_t(scale);
            if (dim == 0) valid++;
        }
    }

    __shared__ int s_valid;
    if (dim == 0) {
        s_valid = valid;
    }
    __syncthreads();

    if (s_valid > 0) {
        // Convert to float at the end
        output[dim] = float(sum) * global_scale / float(s_valid);
    } else {
        output[dim] = 0.0f;
    }
}

// Warp-cooperative version - each warp handles one dimension's tokens
// Better for small token counts
__global__ void embed_tokens_warp_kernel(
    const int8_t* __restrict__ embedding_table,
    const float* __restrict__ scales,
    const int16_t* __restrict__ tokens,
    float* __restrict__ output,
    int num_tokens,
    int vocab_size
) {
    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int lane_id = threadIdx.x % 32;
    int dim = warp_id;

    if (dim >= 512) return;

    float sum = 0.0f;
    int valid = 0;

    // Each lane processes subset of tokens
    for (int i = lane_id; i < num_tokens; i += 32) {
        int16_t tok = tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            int8_t val = embedding_table[tok * 512 + dim];
            sum += float(val) * scales[tok];
            valid++;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        valid += __shfl_down_sync(0xffffffff, valid, offset);
    }

    // Lane 0 writes result
    if (lane_id == 0 && valid > 0) {
        output[dim] = sum / float(valid);
    } else if (lane_id == 0) {
        output[dim] = 0.0f;
    }
}

extern "C" {

// C interface for embedding lookup
typedef struct {
    int8_t* d_embedding_table;
    float* d_scales;
    int16_t* d_scale_q;  // Quantized scales for int32 path
    float global_scale;
    int vocab_size;
    int embed_dim;
    cudaStream_t stream;

    // Pre-allocated buffers
    int16_t* d_tokens;
    float* d_output;
    int max_tokens;
} EmbedContext;

void* embed_context_create(int vocab_size) {
    EmbedContext* ctx = new EmbedContext();
    ctx->vocab_size = vocab_size;
    ctx->embed_dim = 512;
    ctx->max_tokens = 256;
    ctx->global_scale = 1.0f / 127.0f;

    cudaStreamCreate(&ctx->stream);
    cudaMalloc(&ctx->d_tokens, ctx->max_tokens * sizeof(int16_t));
    cudaMalloc(&ctx->d_output, 512 * sizeof(float));

    return ctx;
}

int embed_context_load_table(void* ctx_ptr, const int8_t* embeddings, const float* scales) {
    EmbedContext* ctx = (EmbedContext*)ctx_ptr;

    size_t table_size = ctx->vocab_size * 512 * sizeof(int8_t);
    size_t scale_size = ctx->vocab_size * sizeof(float);

    cudaError_t err = cudaMalloc(&ctx->d_embedding_table, table_size);
    if (err != cudaSuccess) return 0;

    err = cudaMalloc(&ctx->d_scales, scale_size);
    if (err != cudaSuccess) return 0;

    cudaMemcpy(ctx->d_embedding_table, embeddings, table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_scales, scales, scale_size, cudaMemcpyHostToDevice);

    return 1;
}

int embed_tokens_gpu(void* ctx_ptr, const int16_t* tokens, int num_tokens, float* output) {
    EmbedContext* ctx = (EmbedContext*)ctx_ptr;

    if (num_tokens > ctx->max_tokens) {
        cudaFree(ctx->d_tokens);
        ctx->max_tokens = num_tokens + 64;
        cudaMalloc(&ctx->d_tokens, ctx->max_tokens * sizeof(int16_t));
    }

    cudaMemcpyAsync(ctx->d_tokens, tokens, num_tokens * sizeof(int16_t),
                    cudaMemcpyHostToDevice, ctx->stream);

    // Use vec4 kernel for best performance
    int threads = 128;  // 512 / 4 = 128 threads
    embed_tokens_vec4_kernel<<<1, threads, 0, ctx->stream>>>(
        ctx->d_embedding_table,
        ctx->d_scales,
        ctx->d_tokens,
        ctx->d_output,
        num_tokens,
        ctx->vocab_size
    );

    cudaMemcpyAsync(output, ctx->d_output, 512 * sizeof(float),
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    return 1;
}

// Batch embedding for multiple queries
int embed_tokens_batch_gpu(
    void* ctx_ptr,
    const int16_t* all_tokens,  // [batch_size x max_tokens]
    const int* token_counts,    // [batch_size]
    float* outputs,             // [batch_size x 512]
    int batch_size,
    int max_tokens
) {
    EmbedContext* ctx = (EmbedContext*)ctx_ptr;

    // Allocate batch buffers
    int16_t* d_all_tokens;
    int* d_counts;
    float* d_outputs;

    cudaMalloc(&d_all_tokens, batch_size * max_tokens * sizeof(int16_t));
    cudaMalloc(&d_counts, batch_size * sizeof(int));
    cudaMalloc(&d_outputs, batch_size * 512 * sizeof(float));

    cudaMemcpyAsync(d_all_tokens, all_tokens, batch_size * max_tokens * sizeof(int16_t),
                    cudaMemcpyHostToDevice, ctx->stream);
    cudaMemcpyAsync(d_counts, token_counts, batch_size * sizeof(int),
                    cudaMemcpyHostToDevice, ctx->stream);

    // Launch batch kernel
    dim3 blocks((512 + 255) / 256, batch_size);
    dim3 threads(256);

    embed_tokens_batch_kernel<<<blocks, threads, 0, ctx->stream>>>(
        ctx->d_embedding_table,
        ctx->d_scales,
        d_all_tokens,
        d_counts,
        d_outputs,
        max_tokens,
        ctx->vocab_size,
        batch_size
    );

    cudaMemcpyAsync(outputs, d_outputs, batch_size * 512 * sizeof(float),
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    cudaFree(d_all_tokens);
    cudaFree(d_counts);
    cudaFree(d_outputs);

    return batch_size;
}

void embed_context_destroy(void* ctx_ptr) {
    EmbedContext* ctx = (EmbedContext*)ctx_ptr;
    if (!ctx) return;

    if (ctx->d_embedding_table) cudaFree(ctx->d_embedding_table);
    if (ctx->d_scales) cudaFree(ctx->d_scales);
    if (ctx->d_scale_q) cudaFree(ctx->d_scale_q);
    if (ctx->d_tokens) cudaFree(ctx->d_tokens);
    if (ctx->d_output) cudaFree(ctx->d_output);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);

    delete ctx;
}

} // extern "C"
