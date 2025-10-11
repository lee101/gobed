//go:build cagra_native
// +build cagra_native

/*
 * CAGRA (CUDA And Graph-based Retrieval Algorithm) wrapper for gobed
 * Provides ultra-fast approximate nearest neighbor search on GPU
 * Targets: <1ms search latency for 1M+ vectors
 */

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/distance/pairwise_distance.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include "cagra_wrapper.h"

// CAGRA index wrapper (define the struct body; typedef is in header)
struct cagra_wrapper_t {
    cuvsResources_t resources;
    cuvsCagraIndex_t index;
    cuvsCagraIndexParams_t index_params;
    cuvsCagraSearchParams_t search_params;

    // Data storage
    int8_t* d_dataset;      // GPU dataset (int8 quantized)
    float* d_dataset_fp32;  // GPU dataset (float32 for CAGRA)
    float* d_scales;        // Quantization scales

    // Configuration
    int num_vectors;
    int dim;
    int graph_degree;       // CAGRA graph connectivity
    int intermediate_graph_degree;
    int max_iterations;

    // Stats
    cagra_stats_t stats;

    // Caching
    int cache_enabled;
    char cache_path[256];
};

// simple integer clamp helpers
#ifndef INT_MIN
#define INT_MIN (-2147483647 - 1)
#endif
#ifndef INT_MAX
#define INT_MAX 2147483647
#endif
#define CLAMP_INT(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

// forward declaration
void cagra_destroy(cagra_wrapper_t* wrapper);

// Helpers to read environment overrides safely
static int getenv_int(const char* name, int defval) {
    const char* v = getenv(name);
    if (!v || !*v) return defval;
    // simple integer parse
    int sign = 1; long val = 0; const char* p = v;
    if (*p == '+') { p++; } else if (*p == '-') { sign = -1; p++; }
    while (*p && isdigit((unsigned char)*p)) { val = val*10 + (*p - '0'); p++; }
    return (int)(sign * val);
}

static float getenv_float(const char* name, float defval) {
    const char* v = getenv(name);
    if (!v || !*v) return defval;
    return (float)atof(v);
}

// Helper function to convert int8 to float32 with scaling
__global__ void int8_to_float32_kernel(
    const int8_t* __restrict__ int8_data,
    const float* __restrict__ scales,
    float* __restrict__ float32_data,
    int n, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * dim) return;

    int vec_idx = idx / dim;
    float scale = scales[vec_idx];
    float32_data[idx] = (float)int8_data[idx] * scale;
}

// Helper function to convert float32 query to int8 (for consistency)
__global__ void float32_to_int8_kernel(
    const float* __restrict__ float32_data,
    int8_t* __restrict__ int8_data,
    float* __restrict__ scales,
    int n, int dim
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= n) return;

    // Find max absolute value for quantization
    float max_abs = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float val = float32_data[vec_idx * dim + d];
        max_abs = fmaxf(max_abs, fabsf(val));
    }

    // Reduce within warp
    for (int offset = 16; offset > 0; offset /= 2) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    }

    // Write scale
    if (threadIdx.x == 0) {
        scales[vec_idx] = max_abs / 127.0f;
    }
    __syncthreads();

    // Quantize to int8
    float inv_scale = 127.0f / max_abs;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float val = float32_data[vec_idx * dim + d];
        int quantized = __float2int_rn(val * inv_scale);
        quantized = max(-128, min(127, quantized));
        int8_data[vec_idx * dim + d] = (int8_t)quantized;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// Create CAGRA wrapper
cagra_wrapper_t* cagra_create(int max_vectors, int dim) {
    cagra_wrapper_t* wrapper = (cagra_wrapper_t*)calloc(1, sizeof(cagra_wrapper_t));
    if (!wrapper) return NULL;

    wrapper->dim = dim;
    wrapper->num_vectors = 0;

    // CAGRA configuration for ultra-fast search
    wrapper->graph_degree = getenv_int("CAGRA_GRAPH_DEGREE", 64);  // Higher = better quality, more memory
    wrapper->intermediate_graph_degree = getenv_int("CAGRA_INTERMEDIATE_DEGREE", 128);
    wrapper->max_iterations = getenv_int("CAGRA_MAX_ITERS", 200);

    // Create cuVS resources
    if (cuvsResourcesCreate(&wrapper->resources) != CUVS_SUCCESS) {
        free(wrapper);
        return NULL;
    }

    // Create index params
    if (cuvsCagraIndexParamsCreate(&wrapper->index_params) != CUVS_SUCCESS) {
        cuvsResourcesDestroy(wrapper->resources);
        free(wrapper);
        return NULL;
    }

    // Set index parameters for speed and correct metric (Inner Product for similarity)
    wrapper->index_params->graph_degree = wrapper->graph_degree;
    wrapper->index_params->intermediate_graph_degree = wrapper->intermediate_graph_degree;
    wrapper->index_params->build_algo = IVF_PQ;  // Fast build algorithm per cuVS C API
    // Ensure we use inner-product similarity to match gobed tests/expectations
    // (higher score means more similar). CAGRA returns distances; with InnerProduct this is
    // typically -dot. We will convert to positive similarity at the Go layer if needed.
    wrapper->index_params->metric = InnerProduct;

    // Create search params
    if (cuvsCagraSearchParamsCreate(&wrapper->search_params) != CUVS_SUCCESS) {
        cuvsCagraIndexParamsDestroy(wrapper->index_params);
        cuvsResourcesDestroy(wrapper->resources);
        free(wrapper);
        return NULL;
    }

    // Set search parameters with env overrides for tuning
    wrapper->search_params->max_queries = getenv_int("CAGRA_MAX_QUERIES", 1);
    {
        int algo = getenv_int("CAGRA_ALGO", (int)SINGLE_CTA);
        // Accept values from enum cuvsCagraSearchAlgo
        if (algo != (int)SINGLE_CTA && algo != (int)MULTI_CTA && algo != (int)MULTI_KERNEL && algo != (int)AUTO) {
            algo = (int)SINGLE_CTA;
        }
        wrapper->search_params->algo = (enum cuvsCagraSearchAlgo)algo;
    }
    int maxit = getenv_int("CAGRA_SEARCH_MAX_ITERS", 64);
    int minit = getenv_int("CAGRA_SEARCH_MIN_ITERS", 0);
    wrapper->search_params->max_iterations = maxit;
    wrapper->search_params->min_iterations = minit;
    wrapper->search_params->team_size = getenv_int("CAGRA_TEAM_SIZE", 8);
    int itopk = getenv_int("CAGRA_ITOPK_SIZE", 64);
    if (itopk < 1) itopk = 64;
    wrapper->search_params->itopk_size = itopk;
    int sw = getenv_int("CAGRA_SEARCH_WIDTH", 1);
    if (sw < 1) sw = 1;
    wrapper->search_params->search_width = sw;

    // Allocate GPU memory
    size_t dataset_bytes = max_vectors * dim * sizeof(int8_t);
    size_t dataset_fp32_bytes = max_vectors * dim * sizeof(float);
    size_t scales_bytes = max_vectors * sizeof(float);

    cudaMalloc((void**)&wrapper->d_dataset, dataset_bytes);
    cudaMalloc((void**)&wrapper->d_dataset_fp32, dataset_fp32_bytes);
    cudaMalloc((void**)&wrapper->d_scales, scales_bytes);

    if (!wrapper->d_dataset || !wrapper->d_dataset_fp32 || !wrapper->d_scales) {
        cagra_destroy(wrapper);
        return NULL;
    }

    wrapper->stats.index_memory_bytes = dataset_bytes + dataset_fp32_bytes + scales_bytes;

    printf("[CAGRA] Initialized: max_vectors=%d, dim=%d, graph_degree=%d\n",
           max_vectors, dim, wrapper->graph_degree);

    return wrapper;
}

// Build CAGRA index from int8 vectors
int cagra_build_index(
    cagra_wrapper_t* wrapper,
    const int8_t* vectors,
    const float* scales,
    int num_vectors
) {
    if (!wrapper || !vectors || !scales) return -1;

    clock_t start = clock();

    // Copy int8 data to GPU
    size_t data_bytes = num_vectors * wrapper->dim * sizeof(int8_t);
    size_t scales_bytes = num_vectors * sizeof(float);

    cudaMemcpy(wrapper->d_dataset, vectors, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(wrapper->d_scales, scales, scales_bytes, cudaMemcpyHostToDevice);

    // Convert int8 to float32 for CAGRA (CAGRA requires float32)
    int threads = 256;
    int blocks = (num_vectors * wrapper->dim + threads - 1) / threads;

    int8_to_float32_kernel<<<blocks, threads>>>(
        wrapper->d_dataset,
        wrapper->d_scales,
        wrapper->d_dataset_fp32,
        num_vectors,
        wrapper->dim
    );
    cudaDeviceSynchronize();

    // Create DLTensor for dataset
    DLManagedTensor dataset_tensor;
    dataset_tensor.dl_tensor.data = wrapper->d_dataset_fp32;
    dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
    dataset_tensor.dl_tensor.device.device_id = 0;
    dataset_tensor.dl_tensor.ndim = 2;
    dataset_tensor.dl_tensor.dtype.code = kDLFloat;
    dataset_tensor.dl_tensor.dtype.bits = 32;
    dataset_tensor.dl_tensor.dtype.lanes = 1;
    int64_t shape[2] = {num_vectors, wrapper->dim};
    dataset_tensor.dl_tensor.shape = shape;
    dataset_tensor.dl_tensor.strides = NULL;
    dataset_tensor.dl_tensor.byte_offset = 0;

    // Create CAGRA index if not exists
    if (!wrapper->index) {
        if (cuvsCagraIndexCreate(&wrapper->index) != CUVS_SUCCESS) {
            return -1;
        }
    }

    // Build the index
    cuvsError_t err = cuvsCagraBuild(
        wrapper->resources,
        wrapper->index_params,
        &dataset_tensor,
        wrapper->index
    );

    if (err != CUVS_SUCCESS) {
        printf("[CAGRA] Build failed: error code %d\n", err);
        return -1;
    }

    wrapper->num_vectors = num_vectors;

    clock_t end = clock();
    wrapper->stats.build_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    wrapper->stats.num_vectors = num_vectors;
    wrapper->stats.dim = wrapper->dim;

    printf("[CAGRA] Index built: %d vectors in %.2fms (%.0f vecs/sec)\n",
           num_vectors, wrapper->stats.build_time_ms,
           num_vectors / (wrapper->stats.build_time_ms / 1000.0));

    return 0;
}

// Search CAGRA index with int8 query
int cagra_search(
    cagra_wrapper_t* wrapper,
    const int8_t* query,
    float query_scale,
    int k,
    uint32_t* neighbors,
    float* distances
) {
    if (!wrapper || !wrapper->index || !query) return -1;

    clock_t start = clock();

    // Allocate GPU memory for query
    float* d_query_fp32;
    cudaMalloc((void**)&d_query_fp32, wrapper->dim * sizeof(float));

    // Convert int8 query to float32 on GPU
    float* h_query_fp32 = (float*)malloc(wrapper->dim * sizeof(float));
    for (int i = 0; i < wrapper->dim; i++) {
        h_query_fp32[i] = (float)query[i] * query_scale;
    }
    cudaMemcpy(d_query_fp32, h_query_fp32, wrapper->dim * sizeof(float), cudaMemcpyHostToDevice);
    free(h_query_fp32);

    // Allocate GPU memory for results
    uint32_t* d_neighbors;
    float* d_distances;
    cudaMalloc((void**)&d_neighbors, k * sizeof(uint32_t));
    cudaMalloc((void**)&d_distances, k * sizeof(float));

    // Create DLTensors
    DLManagedTensor query_tensor;
    query_tensor.dl_tensor.data = d_query_fp32;
    query_tensor.dl_tensor.device.device_type = kDLCUDA;
    query_tensor.dl_tensor.device.device_id = 0;
    query_tensor.dl_tensor.ndim = 2;
    query_tensor.dl_tensor.dtype.code = kDLFloat;
    query_tensor.dl_tensor.dtype.bits = 32;
    query_tensor.dl_tensor.dtype.lanes = 1;
    int64_t query_shape[2] = {1, wrapper->dim};
    query_tensor.dl_tensor.shape = query_shape;
    query_tensor.dl_tensor.strides = NULL;
    query_tensor.dl_tensor.byte_offset = 0;

    DLManagedTensor neighbors_tensor;
    neighbors_tensor.dl_tensor.data = d_neighbors;
    neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
    neighbors_tensor.dl_tensor.device.device_id = 0;
    neighbors_tensor.dl_tensor.ndim = 2;
    neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
    neighbors_tensor.dl_tensor.dtype.bits = 32;
    neighbors_tensor.dl_tensor.dtype.lanes = 1;
    int64_t neighbors_shape[2] = {1, k};
    neighbors_tensor.dl_tensor.shape = neighbors_shape;
    neighbors_tensor.dl_tensor.strides = NULL;
    neighbors_tensor.dl_tensor.byte_offset = 0;

    DLManagedTensor distances_tensor;
    distances_tensor.dl_tensor.data = d_distances;
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.device.device_id = 0;
    distances_tensor.dl_tensor.ndim = 2;
    distances_tensor.dl_tensor.dtype.code = kDLFloat;
    distances_tensor.dl_tensor.dtype.bits = 32;
    distances_tensor.dl_tensor.dtype.lanes = 1;
    int64_t distances_shape[2] = {1, k};
    distances_tensor.dl_tensor.shape = distances_shape;
    distances_tensor.dl_tensor.strides = NULL;
    distances_tensor.dl_tensor.byte_offset = 0;

    // Perform search
    cuvsFilter filter;
    filter.type = NO_FILTER;
    filter.addr = (uintptr_t)NULL;

    cuvsError_t err = cuvsCagraSearch(
        wrapper->resources,
        wrapper->search_params,
        wrapper->index,
        &query_tensor,
        &neighbors_tensor,
        &distances_tensor,
        filter
    );

    if (err != CUVS_SUCCESS) {
        cudaFree(d_query_fp32);
        cudaFree(d_neighbors);
        cudaFree(d_distances);
        return -1;
    }

    // Copy results back to host
    cudaMemcpy(neighbors, d_neighbors, k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, k * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_query_fp32);
    cudaFree(d_neighbors);
    cudaFree(d_distances);

    clock_t end = clock();
    wrapper->stats.search_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    return 0;
}

// Batch search for multiple queries
int cagra_search_batch(
    cagra_wrapper_t* wrapper,
    const int8_t* queries,
    const float* query_scales,
    int num_queries,
    int k,
    uint32_t* neighbors,
    float* distances
) {
    if (!wrapper || !wrapper->index || !queries) return -1;

    clock_t start = clock();

    // Allocate GPU memory for queries
    size_t queries_bytes = num_queries * wrapper->dim * sizeof(float);
    float* d_queries_fp32;
    cudaMalloc((void**)&d_queries_fp32, queries_bytes);

    // Convert int8 queries to float32
    float* h_queries_fp32 = (float*)malloc(queries_bytes);
    for (int q = 0; q < num_queries; q++) {
        for (int d = 0; d < wrapper->dim; d++) {
            int idx = q * wrapper->dim + d;
            h_queries_fp32[idx] = (float)queries[idx] * query_scales[q];
        }
    }
    cudaMemcpy(d_queries_fp32, h_queries_fp32, queries_bytes, cudaMemcpyHostToDevice);
    free(h_queries_fp32);

    // Allocate GPU memory for results
    uint32_t* d_neighbors;
    float* d_distances;
    cudaMalloc((void**)&d_neighbors, num_queries * k * sizeof(uint32_t));
    cudaMalloc((void**)&d_distances, num_queries * k * sizeof(float));

    // Create batch search params for throughput
    cuvsCagraSearchParams_t batch_params;
    cuvsCagraSearchParamsCreate(&batch_params);
    batch_params->max_queries = num_queries;
    batch_params->algo = MULTI_CTA;  // Better for batch
    batch_params->max_iterations = 64;
    batch_params->team_size = 16;

    // Create DLTensors for batch
    DLManagedTensor queries_tensor;
    queries_tensor.dl_tensor.data = d_queries_fp32;
    queries_tensor.dl_tensor.device.device_type = kDLCUDA;
    queries_tensor.dl_tensor.device.device_id = 0;
    queries_tensor.dl_tensor.ndim = 2;
    queries_tensor.dl_tensor.dtype.code = kDLFloat;
    queries_tensor.dl_tensor.dtype.bits = 32;
    queries_tensor.dl_tensor.dtype.lanes = 1;
    int64_t queries_shape[2] = {num_queries, wrapper->dim};
    queries_tensor.dl_tensor.shape = queries_shape;
    queries_tensor.dl_tensor.strides = NULL;
    queries_tensor.dl_tensor.byte_offset = 0;

    DLManagedTensor neighbors_tensor;
    neighbors_tensor.dl_tensor.data = d_neighbors;
    neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
    neighbors_tensor.dl_tensor.device.device_id = 0;
    neighbors_tensor.dl_tensor.ndim = 2;
    neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
    neighbors_tensor.dl_tensor.dtype.bits = 32;
    neighbors_tensor.dl_tensor.dtype.lanes = 1;
    int64_t neighbors_shape[2] = {num_queries, k};
    neighbors_tensor.dl_tensor.shape = neighbors_shape;
    neighbors_tensor.dl_tensor.strides = NULL;
    neighbors_tensor.dl_tensor.byte_offset = 0;

    DLManagedTensor distances_tensor;
    distances_tensor.dl_tensor.data = d_distances;
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.device.device_id = 0;
    distances_tensor.dl_tensor.ndim = 2;
    distances_tensor.dl_tensor.dtype.code = kDLFloat;
    distances_tensor.dl_tensor.dtype.bits = 32;
    distances_tensor.dl_tensor.dtype.lanes = 1;
    int64_t distances_shape[2] = {num_queries, k};
    distances_tensor.dl_tensor.shape = distances_shape;
    distances_tensor.dl_tensor.strides = NULL;
    distances_tensor.dl_tensor.byte_offset = 0;

    // Perform batch search
    cuvsFilter filter;
    filter.type = NO_FILTER;
    filter.addr = (uintptr_t)NULL;

    cuvsError_t err = cuvsCagraSearch(
        wrapper->resources,
        batch_params,
        wrapper->index,
        &queries_tensor,
        &neighbors_tensor,
        &distances_tensor,
        filter
    );

    cuvsCagraSearchParamsDestroy(batch_params);

    if (err != CUVS_SUCCESS) {
        cudaFree(d_queries_fp32);
        cudaFree(d_neighbors);
        cudaFree(d_distances);
        return -1;
    }

    // Copy results back
    cudaMemcpy(neighbors, d_neighbors, num_queries * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, num_queries * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_queries_fp32);
    cudaFree(d_neighbors);
    cudaFree(d_distances);

    clock_t end = clock();
    double batch_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("[CAGRA] Batch search: %d queries in %.2fms (%.2fms per query, %.0f QPS)\n",
           num_queries, batch_time_ms, batch_time_ms / num_queries,
           num_queries / (batch_time_ms / 1000.0));

    return 0;
}

// Serialize index to file for fast loading
int cagra_serialize_index(cagra_wrapper_t* wrapper, const char* filepath) {
    if (!wrapper || !wrapper->index) return -1;

    clock_t start = clock();

    // CAGRA doesn't have direct serialization in C API yet
    // We'll save the dataset and rebuild on load (still faster than training)

    FILE* file = fopen(filepath, "wb");
    if (!file) return -1;

    // Write header
    fwrite(&wrapper->num_vectors, sizeof(int), 1, file);
    fwrite(&wrapper->dim, sizeof(int), 1, file);
    fwrite(&wrapper->graph_degree, sizeof(int), 1, file);

    // Copy data from GPU and write
    size_t data_bytes = wrapper->num_vectors * wrapper->dim * sizeof(int8_t);
    size_t scales_bytes = wrapper->num_vectors * sizeof(float);

    int8_t* h_dataset = (int8_t*)malloc(data_bytes);
    float* h_scales = (float*)malloc(scales_bytes);

    cudaMemcpy(h_dataset, wrapper->d_dataset, data_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scales, wrapper->d_scales, scales_bytes, cudaMemcpyDeviceToHost);

    fwrite(h_dataset, sizeof(int8_t), wrapper->num_vectors * wrapper->dim, file);
    fwrite(h_scales, sizeof(float), wrapper->num_vectors, file);

    free(h_dataset);
    free(h_scales);
    fclose(file);

    clock_t end = clock();
    wrapper->stats.serialize_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("[CAGRA] Index serialized to %s in %.2fms\n", filepath, wrapper->stats.serialize_time_ms);

    return 0;
}

// Deserialize index from file
int cagra_deserialize_index(cagra_wrapper_t* wrapper, const char* filepath) {
    if (!wrapper) return -1;

    clock_t start = clock();

    FILE* file = fopen(filepath, "rb");
    if (!file) return -1;

    // Read header
    int num_vectors, dim, graph_degree;
    fread(&num_vectors, sizeof(int), 1, file);
    fread(&dim, sizeof(int), 1, file);
    fread(&graph_degree, sizeof(int), 1, file);

    if (dim != wrapper->dim) {
        fclose(file);
        return -1;
    }

    // Read data
    size_t data_bytes = num_vectors * dim * sizeof(int8_t);
    size_t scales_bytes = num_vectors * sizeof(float);

    int8_t* h_dataset = (int8_t*)malloc(data_bytes);
    float* h_scales = (float*)malloc(scales_bytes);

    fread(h_dataset, sizeof(int8_t), num_vectors * dim, file);
    fread(h_scales, sizeof(float), num_vectors, file);
    fclose(file);

    // Rebuild index (fast with CAGRA)
    int result = cagra_build_index(wrapper, h_dataset, h_scales, num_vectors);

    free(h_dataset);
    free(h_scales);

    clock_t end = clock();
    wrapper->stats.deserialize_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("[CAGRA] Index deserialized from %s in %.2fms\n", filepath, wrapper->stats.deserialize_time_ms);

    return result;
}

// Get statistics
void cagra_get_stats(cagra_wrapper_t* wrapper, cagra_stats_t* stats) {
    if (wrapper && stats) {
        *stats = wrapper->stats;
    }
}

// Destroy CAGRA wrapper
void cagra_destroy(cagra_wrapper_t* wrapper) {
    if (!wrapper) return;

    if (wrapper->index) {
        cuvsCagraIndexDestroy(wrapper->index);
    }
    if (wrapper->index_params) {
        cuvsCagraIndexParamsDestroy(wrapper->index_params);
    }
    if (wrapper->search_params) {
        cuvsCagraSearchParamsDestroy(wrapper->search_params);
    }
    if (wrapper->resources) {
        cuvsResourcesDestroy(wrapper->resources);
    }

    if (wrapper->d_dataset) cudaFree(wrapper->d_dataset);
    if (wrapper->d_dataset_fp32) cudaFree(wrapper->d_dataset_fp32);
    if (wrapper->d_scales) cudaFree(wrapper->d_scales);

    free(wrapper);
}

#ifdef __cplusplus
} // extern "C"
#endif
