// fused_cagra_opt.cu - Optimized GPU search with IVF clustering
// nvcc -O3 -arch=sm_80 --compiler-options "-fPIC" --shared -o libfused_cagra_opt.so fused_cagra_opt.cu

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
#define NUM_LISTS 64
#define NPROBE 24

struct OptContext {
    int8_t* d_embed_weights;
    float* d_embed_scales;
    int8_t* d_database;
    float* d_db_scales;
    int vocab_size;
    int num_vectors;
    int top_k;
    int embed_dim;

    // IVF structures
    int8_t* d_centroids;      // [NUM_LISTS * EMBED_DIM]
    float* d_centroid_scales; // [NUM_LISTS]
    int* d_list_offsets;      // [NUM_LISTS + 1]
    int* d_list_ids;          // [num_vectors] - vector ID for each position
    int* d_list_lengths;      // [NUM_LISTS]
    int8_t* d_list_vectors;   // [num_vectors * EMBED_DIM] - reordered by list
    float* d_list_scales;     // [num_vectors] - reordered by list
    bool ivf_built;

    // Preallocated buffers
    uint16_t* d_tokens;
    int* d_lengths;
    float* d_distances;
    int* d_indices;
    int* d_probe_lists;       // [batch_size * NPROBE]
    int max_batch_size;
    int max_tokens;
    cudaStream_t stream;
};

// Optimized int8 dot product using dp4a
__device__ __forceinline__ int32_t dot_int8_512(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b) {
    int32_t sum = 0;
    #pragma unroll 16
    for (int i = 0; i < 512; i += 4) {
        int32_t a_packed = *reinterpret_cast<const int32_t*>(a + i);
        int32_t b_packed = *reinterpret_cast<const int32_t*>(b + i);
        sum = __dp4a(a_packed, b_packed, sum);
    }
    return sum;
}

// Warp-level top-k merge using shuffle
__device__ void warp_topk_merge(float* dists, int* indices, int k) {
    unsigned mask = 0xFFFFFFFF;
    int lane = threadIdx.x % WARP_SIZE;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        for (int i = 0; i < k; i++) {
            float other_dist = __shfl_down_sync(mask, dists[i], offset);
            int other_idx = __shfl_down_sync(mask, indices[i], offset);

            if (lane < offset && other_dist < dists[k-1]) {
                // Insert and maintain sorted order
                for (int j = 0; j < k; j++) {
                    if (other_dist < dists[j]) {
                        for (int m = k - 1; m > j; m--) {
                            dists[m] = dists[m-1];
                            indices[m] = indices[m-1];
                        }
                        dists[j] = other_dist;
                        indices[j] = other_idx;
                        break;
                    }
                }
            }
        }
    }
}

// Find nprobe nearest centroids
__device__ void find_probe_lists(
    const int8_t* query_int8,
    float query_scale,
    const int8_t* centroids,
    const float* centroid_scales,
    int num_lists,
    int nprobe,
    int* probe_lists,
    float* probe_dists) {

    int tid = threadIdx.x;
    __shared__ float s_dists[NUM_LISTS];
    __shared__ int s_indices[NUM_LISTS];

    // Compute distances to all centroids in parallel
    for (int c = tid; c < num_lists; c += blockDim.x) {
        int32_t dot = dot_int8_512(query_int8, centroids + c * EMBED_DIM);
        s_dists[c] = -(float)dot * query_scale * centroid_scales[c];
        s_indices[c] = c;
    }
    __syncthreads();

    // Simple selection of top nprobe (thread 0)
    if (tid == 0) {
        for (int i = 0; i < nprobe; i++) {
            int min_idx = i;
            float min_dist = s_dists[i];
            for (int j = i + 1; j < num_lists; j++) {
                if (s_dists[j] < min_dist) {
                    min_dist = s_dists[j];
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                float tmp_d = s_dists[i];
                int tmp_i = s_indices[i];
                s_dists[i] = s_dists[min_idx];
                s_indices[i] = s_indices[min_idx];
                s_dists[min_idx] = tmp_d;
                s_indices[min_idx] = tmp_i;
            }
            probe_lists[i] = s_indices[i];
            probe_dists[i] = s_dists[i];
        }
    }
    __syncthreads();
}

// IVF Search kernel - only searches nprobe lists
__global__ void ivf_search_kernel(
    const int8_t* __restrict__ query_batch,
    const float* __restrict__ query_scales,
    int batch_size,
    const int8_t* __restrict__ centroids,
    const float* __restrict__ centroid_scales,
    int num_lists,
    const int* __restrict__ list_offsets,
    const int* __restrict__ list_ids,
    const int8_t* __restrict__ list_vectors,
    const float* __restrict__ list_scales,
    int nprobe,
    float* __restrict__ output_distances,
    int* __restrict__ output_indices,
    int top_k) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    int8_t* query_int8 = (int8_t*)shared_mem;
    float* s_probe_dists = (float*)(query_int8 + EMBED_DIM);
    int* s_probe_lists = (int*)(s_probe_dists + nprobe);
    float* s_all_dists = (float*)(s_probe_lists + nprobe);
    int* s_all_indices = (int*)(s_all_dists + BLOCK_SIZE * top_k);

    // Load query
    const int8_t* query = query_batch + batch_idx * EMBED_DIM;
    float query_scale = query_scales[batch_idx];
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        query_int8[i] = query[i];
    }
    __syncthreads();

    // Find probe lists
    find_probe_lists(query_int8, query_scale, centroids, centroid_scales,
                     num_lists, nprobe, s_probe_lists, s_probe_dists);

    // Initialize thread-local top-k
    for (int i = 0; i < top_k; i++) {
        s_all_dists[tid * top_k + i] = FLT_MAX;
        s_all_indices[tid * top_k + i] = -1;
    }
    __syncthreads();

    // Search each probe list
    for (int p = 0; p < nprobe; p++) {
        int list_id = s_probe_lists[p];
        int list_start = list_offsets[list_id];
        int list_end = list_offsets[list_id + 1];
        int list_len = list_end - list_start;

        // Each thread processes part of the list
        for (int i = tid; i < list_len; i += blockDim.x) {
            int vec_pos = list_start + i;
            int vec_id = list_ids[vec_pos];
            const int8_t* vec = list_vectors + vec_pos * EMBED_DIM;
            float vec_scale = list_scales[vec_pos];

            int32_t dot = dot_int8_512(query_int8, vec);
            float dist = -(float)dot * query_scale * vec_scale;

            // Update thread-local top-k
            float* my_dists = &s_all_dists[tid * top_k];
            int* my_indices = &s_all_indices[tid * top_k];

            if (dist < my_dists[top_k - 1]) {
                for (int k = 0; k < top_k; k++) {
                    if (dist < my_dists[k]) {
                        for (int j = top_k - 1; j > k; j--) {
                            my_dists[j] = my_dists[j-1];
                            my_indices[j] = my_indices[j-1];
                        }
                        my_dists[k] = dist;
                        my_indices[k] = vec_id;
                        break;
                    }
                }
            }
        }
    }
    __syncthreads();

    // Parallel merge: use warp-level reduction first
    float local_dists[MAX_TOP_K];
    int local_indices[MAX_TOP_K];
    for (int i = 0; i < top_k; i++) {
        local_dists[i] = s_all_dists[tid * top_k + i];
        local_indices[i] = s_all_indices[tid * top_k + i];
    }

    // Warp reduction
    warp_topk_merge(local_dists, local_indices, top_k);

    // Write warp results back
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) {
        for (int i = 0; i < top_k; i++) {
            s_all_dists[warp_id * top_k + i] = local_dists[i];
            s_all_indices[warp_id * top_k + i] = local_indices[i];
        }
    }
    __syncthreads();

    // Final merge by first warp (8 warps -> 1)
    if (tid < WARP_SIZE) {
        int num_warps = BLOCK_SIZE / WARP_SIZE;
        for (int i = 0; i < top_k; i++) {
            local_dists[i] = FLT_MAX;
            local_indices[i] = -1;
        }

        for (int w = tid; w < num_warps; w += WARP_SIZE) {
            for (int i = 0; i < top_k; i++) {
                float d = s_all_dists[w * top_k + i];
                int idx = s_all_indices[w * top_k + i];
                if (d < local_dists[top_k - 1]) {
                    for (int k = 0; k < top_k; k++) {
                        if (d < local_dists[k]) {
                            for (int j = top_k - 1; j > k; j--) {
                                local_dists[j] = local_dists[j-1];
                                local_indices[j] = local_indices[j-1];
                            }
                            local_dists[k] = d;
                            local_indices[k] = idx;
                            break;
                        }
                    }
                }
            }
        }

        warp_topk_merge(local_dists, local_indices, top_k);

        if (lane == 0) {
            float* out_d = output_distances + batch_idx * top_k;
            int* out_i = output_indices + batch_idx * top_k;
            for (int i = 0; i < top_k; i++) {
                out_d[i] = local_dists[i];
                out_i[i] = local_indices[i];
            }
        }
    }
}

// KMeans clustering kernel for building IVF
__global__ void assign_clusters_kernel(
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scales,
    int num_vectors,
    const int8_t* __restrict__ centroids,
    const float* __restrict__ centroid_scales,
    int num_clusters,
    int* __restrict__ assignments) {

    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_vectors) return;

    const int8_t* vec = vectors + vid * EMBED_DIM;
    float vec_scale = scales[vid];

    int best_cluster = 0;
    float best_dist = FLT_MAX;

    for (int c = 0; c < num_clusters; c++) {
        const int8_t* cent = centroids + c * EMBED_DIM;
        int32_t dot = dot_int8_512(vec, cent);
        float dist = -(float)dot * vec_scale * centroid_scales[c];
        if (dist < best_dist) {
            best_dist = dist;
            best_cluster = c;
        }
    }

    assignments[vid] = best_cluster;
}

extern "C" {

void* create_opt_context(
    int8_t* embed_weights,
    float* embed_scales,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales,
    int num_vectors,
    int top_k) {

    printf("[OPT] Creating context: vocab=%d, vectors=%d, dim=%d, top_k=%d\n",
           vocab_size, num_vectors, embed_dim, top_k);

    OptContext* ctx = new OptContext;
    ctx->vocab_size = vocab_size;
    ctx->num_vectors = num_vectors;
    ctx->top_k = top_k;
    ctx->embed_dim = embed_dim;
    ctx->ivf_built = false;

    size_t embed_size = vocab_size * embed_dim * sizeof(int8_t);
    size_t db_size = num_vectors * embed_dim * sizeof(int8_t);

    cudaMalloc(&ctx->d_embed_weights, embed_size);
    cudaMalloc(&ctx->d_embed_scales, vocab_size * sizeof(float));
    cudaMalloc(&ctx->d_database, db_size);
    cudaMalloc(&ctx->d_db_scales, num_vectors * sizeof(float));

    cudaMemcpy(ctx->d_embed_weights, embed_weights, embed_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_embed_scales, embed_scales, vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_database, database, db_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_db_scales, db_scales, num_vectors * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate IVF structures
    cudaMalloc(&ctx->d_centroids, NUM_LISTS * embed_dim * sizeof(int8_t));
    cudaMalloc(&ctx->d_centroid_scales, NUM_LISTS * sizeof(float));
    cudaMalloc(&ctx->d_list_offsets, (NUM_LISTS + 1) * sizeof(int));
    cudaMalloc(&ctx->d_list_ids, num_vectors * sizeof(int));
    cudaMalloc(&ctx->d_list_lengths, NUM_LISTS * sizeof(int));
    cudaMalloc(&ctx->d_list_vectors, db_size);
    cudaMalloc(&ctx->d_list_scales, num_vectors * sizeof(float));

    ctx->max_batch_size = 32;
    ctx->max_tokens = 128;
    cudaMalloc(&ctx->d_tokens, ctx->max_batch_size * ctx->max_tokens * sizeof(uint16_t));
    cudaMalloc(&ctx->d_lengths, ctx->max_batch_size * sizeof(int));
    cudaMalloc(&ctx->d_distances, ctx->max_batch_size * top_k * sizeof(float));
    cudaMalloc(&ctx->d_indices, ctx->max_batch_size * top_k * sizeof(int));
    cudaMalloc(&ctx->d_probe_lists, ctx->max_batch_size * NPROBE * sizeof(int));

    cudaStreamCreate(&ctx->stream);

    return ctx;
}

void build_ivf_index(void* context) {
    OptContext* ctx = (OptContext*)context;
    if (ctx->ivf_built) return;

    printf("[OPT] Building IVF index with %d lists...\n", NUM_LISTS);

    int* h_assignments = (int*)malloc(ctx->num_vectors * sizeof(int));
    int* d_assignments;
    cudaMalloc(&d_assignments, ctx->num_vectors * sizeof(int));

    // Initialize centroids from random subset
    int8_t* h_centroids = (int8_t*)malloc(NUM_LISTS * EMBED_DIM * sizeof(int8_t));
    float* h_centroid_scales = (float*)malloc(NUM_LISTS * sizeof(float));
    int8_t* h_database = (int8_t*)malloc(ctx->num_vectors * EMBED_DIM * sizeof(int8_t));
    float* h_db_scales = (float*)malloc(ctx->num_vectors * sizeof(float));

    cudaMemcpy(h_database, ctx->d_database, ctx->num_vectors * EMBED_DIM * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db_scales, ctx->d_db_scales, ctx->num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    // Select centroids (uniform sampling)
    int stride = ctx->num_vectors / NUM_LISTS;
    for (int i = 0; i < NUM_LISTS; i++) {
        int src_idx = i * stride;
        memcpy(h_centroids + i * EMBED_DIM, h_database + src_idx * EMBED_DIM, EMBED_DIM);
        h_centroid_scales[i] = h_db_scales[src_idx];
    }

    cudaMemcpy(ctx->d_centroids, h_centroids, NUM_LISTS * EMBED_DIM * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_centroid_scales, h_centroid_scales, NUM_LISTS * sizeof(float), cudaMemcpyHostToDevice);

    // Assign vectors to clusters
    int threads = 256;
    int blocks = (ctx->num_vectors + threads - 1) / threads;
    assign_clusters_kernel<<<blocks, threads>>>(
        ctx->d_database, ctx->d_db_scales, ctx->num_vectors,
        ctx->d_centroids, ctx->d_centroid_scales, NUM_LISTS,
        d_assignments);
    cudaDeviceSynchronize();

    cudaMemcpy(h_assignments, d_assignments, ctx->num_vectors * sizeof(int), cudaMemcpyDeviceToHost);

    // Count list lengths
    int* h_list_lengths = (int*)calloc(NUM_LISTS, sizeof(int));
    for (int i = 0; i < ctx->num_vectors; i++) {
        h_list_lengths[h_assignments[i]]++;
    }

    // Compute offsets
    int* h_list_offsets = (int*)malloc((NUM_LISTS + 1) * sizeof(int));
    h_list_offsets[0] = 0;
    for (int i = 0; i < NUM_LISTS; i++) {
        h_list_offsets[i + 1] = h_list_offsets[i] + h_list_lengths[i];
    }

    // Reorder vectors by list
    int* h_list_ids = (int*)malloc(ctx->num_vectors * sizeof(int));
    int8_t* h_list_vectors = (int8_t*)malloc(ctx->num_vectors * EMBED_DIM * sizeof(int8_t));
    float* h_list_scales = (float*)malloc(ctx->num_vectors * sizeof(float));
    int* list_pos = (int*)calloc(NUM_LISTS, sizeof(int));

    for (int i = 0; i < ctx->num_vectors; i++) {
        int list_id = h_assignments[i];
        int pos = h_list_offsets[list_id] + list_pos[list_id];
        h_list_ids[pos] = i;
        memcpy(h_list_vectors + pos * EMBED_DIM, h_database + i * EMBED_DIM, EMBED_DIM);
        h_list_scales[pos] = h_db_scales[i];
        list_pos[list_id]++;
    }

    // Copy to device
    cudaMemcpy(ctx->d_list_offsets, h_list_offsets, (NUM_LISTS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_list_ids, h_list_ids, ctx->num_vectors * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_list_lengths, h_list_lengths, NUM_LISTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_list_vectors, h_list_vectors, ctx->num_vectors * EMBED_DIM * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_list_scales, h_list_scales, ctx->num_vectors * sizeof(float), cudaMemcpyHostToDevice);

    ctx->ivf_built = true;

    // Cleanup
    free(h_assignments);
    free(h_centroids);
    free(h_centroid_scales);
    free(h_database);
    free(h_db_scales);
    free(h_list_lengths);
    free(h_list_offsets);
    free(h_list_ids);
    free(h_list_vectors);
    free(h_list_scales);
    free(list_pos);
    cudaFree(d_assignments);

    printf("[OPT] IVF index built with %d lists, nprobe=%d\n", NUM_LISTS, NPROBE);
}

void opt_search(
    void* context,
    int8_t* query_batch,
    float* query_scales,
    int batch_size,
    float* output_distances,
    int* output_indices) {

    OptContext* ctx = (OptContext*)context;

    if (!ctx->ivf_built) {
        build_ivf_index(context);
    }

    // Copy queries to device
    int8_t* d_queries;
    float* d_query_scales;
    cudaMalloc(&d_queries, batch_size * EMBED_DIM * sizeof(int8_t));
    cudaMalloc(&d_query_scales, batch_size * sizeof(float));
    cudaMemcpy(d_queries, query_batch, batch_size * EMBED_DIM * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_scales, query_scales, batch_size * sizeof(float), cudaMemcpyHostToDevice);

    size_t shared_size = EMBED_DIM * sizeof(int8_t) +
                        NPROBE * sizeof(float) +
                        NPROBE * sizeof(int) +
                        BLOCK_SIZE * ctx->top_k * sizeof(float) +
                        BLOCK_SIZE * ctx->top_k * sizeof(int);

    ivf_search_kernel<<<batch_size, BLOCK_SIZE, shared_size, ctx->stream>>>(
        d_queries, d_query_scales, batch_size,
        ctx->d_centroids, ctx->d_centroid_scales, NUM_LISTS,
        ctx->d_list_offsets, ctx->d_list_ids,
        ctx->d_list_vectors, ctx->d_list_scales,
        NPROBE,
        ctx->d_distances, ctx->d_indices,
        ctx->top_k
    );

    cudaMemcpy(output_distances, ctx->d_distances, batch_size * ctx->top_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_indices, ctx->d_indices, batch_size * ctx->top_k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_queries);
    cudaFree(d_query_scales);
}

void destroy_opt_context(void* context) {
    OptContext* ctx = (OptContext*)context;
    if (ctx) {
        cudaFree(ctx->d_embed_weights);
        cudaFree(ctx->d_embed_scales);
        cudaFree(ctx->d_database);
        cudaFree(ctx->d_db_scales);
        cudaFree(ctx->d_centroids);
        cudaFree(ctx->d_centroid_scales);
        cudaFree(ctx->d_list_offsets);
        cudaFree(ctx->d_list_ids);
        cudaFree(ctx->d_list_lengths);
        cudaFree(ctx->d_list_vectors);
        cudaFree(ctx->d_list_scales);
        cudaFree(ctx->d_tokens);
        cudaFree(ctx->d_lengths);
        cudaFree(ctx->d_distances);
        cudaFree(ctx->d_indices);
        cudaFree(ctx->d_probe_lists);
        cudaStreamDestroy(ctx->stream);
        delete ctx;
    }
}

} // extern "C"
