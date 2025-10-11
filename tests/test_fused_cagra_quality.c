#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 256
#define EMBED_DIM 512
#define NUM_VECTORS 128
#define TOP_K 10
#define NUM_QUERIES 64
#define MAX_TOKENS 4

extern void* create_fused_context(
    int8_t* embed_weights,
    float* embed_scales,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales,
    int num_vectors,
    int top_k);

extern void build_cagra_graph(void* context, int degree);

extern void fused_search(
    void* context,
    uint16_t* token_batch,
    int* token_lengths,
    int batch_size,
    int max_tokens,
    float* output_distances,
    int* output_indices);

extern void destroy_fused_context(void* context);

static void generate_vocab(int8_t* embed_weights, float* embed_scales) {
    for (int t = 0; t < VOCAB_SIZE; ++t) {
        embed_scales[t] = 0.02f * (1.0f + (float)((t % 17) + 1) * 0.05f);
        for (int j = 0; j < EMBED_DIM; ++j) {
            int val = ((t * 131 + j * 37) % 255) - 127;
            embed_weights[t * EMBED_DIM + j] = (int8_t)val;
        }
    }
}

static void generate_database(const int8_t* embed_weights, const float* embed_scales, int8_t* database, float* db_scales) {
    for (int v = 0; v < NUM_VECTORS; ++v) {
        int token = v % VOCAB_SIZE;
        db_scales[v] = embed_scales[token] * (1.0f + (float)((v % 13) * 0.01f));
        const int8_t* src = embed_weights + token * EMBED_DIM;
        int8_t* dst = database + v * EMBED_DIM;
        for (int j = 0; j < EMBED_DIM; ++j) {
            dst[j] = src[j];
        }
    }
}

static void insert_topk(float* dists, int* indices, int k, float dist, int idx) {
    for (int i = 0; i < k; ++i) {
        if (dist < dists[i]) {
            for (int j = k - 1; j > i; --j) {
                dists[j] = dists[j - 1];
                indices[j] = indices[j - 1];
            }
            dists[i] = dist;
            indices[i] = idx;
            return;
        }
    }
}

static void prepare_query(const uint16_t* tokens, int length, const int8_t* embed_weights, const float* embed_scales, int8_t* out_query, float* out_scale) {
    float accum[EMBED_DIM];
    for (int i = 0; i < EMBED_DIM; ++i) accum[i] = 0.0f;
    for (int t = 0; t < length; ++t) {
        uint16_t token = tokens[t];
        const int8_t* token_vec = embed_weights + token * EMBED_DIM;
        float scale = embed_scales[token];
        for (int j = 0; j < EMBED_DIM; ++j) {
            accum[j] += (float)token_vec[j] * scale;
        }
    }
    if (length > 0) {
        float inv = 1.0f / (float)length;
        for (int i = 0; i < EMBED_DIM; ++i) accum[i] *= inv;
    }
    float max_val = 0.0f;
    for (int i = 0; i < EMBED_DIM; ++i) {
        float v = fabsf(accum[i]);
        if (v > max_val) max_val = v;
    }
    float scale = max_val / 127.0f + 1e-8f;
    *out_scale = scale;
    for (int i = 0; i < EMBED_DIM; ++i) {
        float raw = accum[i] / scale;
        if (raw > 127.0f) raw = 127.0f;
        if (raw < -128.0f) raw = -128.0f;
        int val = (int)raw;
        out_query[i] = (int8_t)val;
    }
}

static void cpu_reference(const uint16_t* token_batch,
                          const int* token_lengths,
                          const int8_t* embed_weights,
                          const float* embed_scales,
                          const int8_t* database,
                          const float* db_scales,
                          float* out_distances,
                          int* out_indices) {
    int8_t query_vec[EMBED_DIM];
    float query_scale;
    for (int q = 0; q < NUM_QUERIES; ++q) {
        float* q_dists = out_distances + q * TOP_K;
        int* q_indices = out_indices + q * TOP_K;
        for (int i = 0; i < TOP_K; ++i) {
            q_dists[i] = FLT_MAX;
            q_indices[i] = -1;
        }
        const uint16_t* tokens = token_batch + q * MAX_TOKENS;
        prepare_query(tokens, token_lengths[q], embed_weights, embed_scales, query_vec, &query_scale);
        for (int v = 0; v < NUM_VECTORS; ++v) {
            const int8_t* db_vec = database + v * EMBED_DIM;
            int32_t dot = 0;
            for (int j = 0; j < EMBED_DIM; ++j) {
                dot += (int32_t)query_vec[j] * (int32_t)db_vec[j];
            }
            float dist = -(float)dot * query_scale * db_scales[v];
            insert_topk(q_dists, q_indices, TOP_K, dist, v);
        }
    }
}

int main(void) {
    int8_t* embed_weights = (int8_t*)malloc(VOCAB_SIZE * EMBED_DIM * sizeof(int8_t));
    float* embed_scales = (float*)malloc(VOCAB_SIZE * sizeof(float));
    int8_t* database = (int8_t*)malloc(NUM_VECTORS * EMBED_DIM * sizeof(int8_t));
    float* db_scales = (float*)malloc(NUM_VECTORS * sizeof(float));
    uint16_t* token_batch = (uint16_t*)calloc(NUM_QUERIES * MAX_TOKENS, sizeof(uint16_t));
    int* token_lengths = (int*)malloc(NUM_QUERIES * sizeof(int));
    float* gpu_distances = (float*)malloc(NUM_QUERIES * TOP_K * sizeof(float));
    int* gpu_indices = (int*)malloc(NUM_QUERIES * TOP_K * sizeof(int));
    float* cpu_distances = (float*)malloc(NUM_QUERIES * TOP_K * sizeof(float));
    int* cpu_indices = (int*)malloc(NUM_QUERIES * TOP_K * sizeof(int));

    if (!embed_weights || !embed_scales || !database || !db_scales ||
        !token_batch || !token_lengths || !gpu_distances || !gpu_indices ||
        !cpu_distances || !cpu_indices) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    generate_vocab(embed_weights, embed_scales);
    generate_database(embed_weights, embed_scales, database, db_scales);

    for (int q = 0; q < NUM_QUERIES; ++q) {
        token_batch[q * MAX_TOKENS] = (uint16_t)(q % VOCAB_SIZE);
        token_lengths[q] = 1;
    }

    cpu_reference(token_batch, token_lengths, embed_weights, embed_scales, database, db_scales, cpu_distances, cpu_indices);

    void* ctx = create_fused_context(embed_weights, embed_scales, VOCAB_SIZE, EMBED_DIM, database, db_scales, NUM_VECTORS, TOP_K);
    if (!ctx) {
        fprintf(stderr, "create_fused_context failed\n");
        return 1;
    }

    build_cagra_graph(ctx, 32);

    fused_search(ctx, token_batch, token_lengths, NUM_QUERIES, MAX_TOKENS, gpu_distances, gpu_indices);

    int failures = 0;
    for (int q = 0; q < NUM_QUERIES; ++q) {
        int* q_idx = gpu_indices + q * TOP_K;
        float* q_dist = gpu_distances + q * TOP_K;
        for (int i = 0; i < TOP_K - 1; ++i) {
            for (int j = i + 1; j < TOP_K; ++j) {
                if (q_idx[i] == q_idx[j]) {
                    fprintf(stderr, "duplicate index in query %d: %d appears twice\n", q, q_idx[i]);
                    failures = 1;
                    break;
                }
            }
            if (failures) break;
        }
        for (int i = 1; i < TOP_K; ++i) {
            if (q_dist[i] < q_dist[i - 1] - 1e-5f) {
                fprintf(stderr, "distances not sorted for query %d at position %d\n", q, i);
                failures = 1;
                break;
            }
        }
        if (failures) break;
        int* ref_idx = cpu_indices + q * TOP_K;
        float* ref_dist = cpu_distances + q * TOP_K;
        for (int i = 0; i < 5; ++i) {
            if (q_idx[i] != ref_idx[i]) {
                fprintf(stderr, "index mismatch query %d pos %d gpu=%d cpu=%d\n", q, i, q_idx[i], ref_idx[i]);
                failures = 1;
                break;
            }
            float diff = fabsf(q_dist[i] - ref_dist[i]);
            if (diff > 1e-3f) {
                fprintf(stderr, "distance mismatch query %d pos %d diff=%f\n", q, i, diff);
                failures = 1;
                break;
            }
        }
        if (failures) break;
    }

    destroy_fused_context(ctx);

    free(embed_weights);
    free(embed_scales);
    free(database);
    free(db_scales);
    free(token_batch);
    free(token_lengths);
    free(gpu_distances);
    free(gpu_indices);
    free(cpu_distances);
    free(cpu_indices);

    if (failures) {
        return 1;
    }

    printf("fused_cagra_quality ok\n");
    return 0;
}
