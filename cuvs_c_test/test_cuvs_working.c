#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>

// Helper to check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Helper to check CUVS errors
#define CUVS_CHECK(call) do { \
    cuvsError_t error = call; \
    if (error == CUVS_ERROR) { \
        fprintf(stderr, "CUVS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

typedef struct {
    float build_time_ms;
    float search_time_ms;
    float throughput_qps;
    float recall_at_k;
    int total_queries;
    int dataset_size;
} BenchmarkResult;

void generate_random_data(float* data, int n, int dim) {
    for (int i = 0; i < n * dim; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }
}

float calculate_recall(int64_t* results, int64_t* ground_truth, int n_queries, int k) {
    int hits = 0;
    for (int q = 0; q < n_queries; q++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (results[q * k + i] == ground_truth[q * k + j]) {
                    hits++;
                    break;
                }
            }
        }
    }
    return (float)hits / (n_queries * k);
}

void brute_force_knn(float* dataset, float* queries, int64_t* results,
                     int n_vectors, int n_queries, int dim, int k) {
    for (int q = 0; q < n_queries; q++) {
        // Calculate distances to all vectors
        float* distances = (float*)malloc(n_vectors * sizeof(float));
        for (int v = 0; v < n_vectors; v++) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = queries[q * dim + d] - dataset[v * dim + d];
                dist += diff * diff;
            }
            distances[v] = dist;
        }

        // Find k nearest
        for (int i = 0; i < k; i++) {
            float min_dist = distances[0];
            int min_idx = 0;
            for (int v = 1; v < n_vectors; v++) {
                if (distances[v] < min_dist) {
                    min_dist = distances[v];
                    min_idx = v;
                }
            }
            results[q * k + i] = min_idx;
            distances[min_idx] = INFINITY;  // Mark as used
        }
        free(distances);
    }
}

BenchmarkResult benchmark_cagra(int n_vectors, int n_queries, int dim, int k) {
    BenchmarkResult result = {0};
    result.dataset_size = n_vectors;
    result.total_queries = n_queries;

    printf("\n📊 Benchmarking CUVS CAGRA:\n");
    printf("  Dataset: %d vectors, %d dimensions\n", n_vectors, dim);
    printf("  Queries: %d, k=%d\n", n_queries, k);

    // Allocate host memory
    float* h_dataset = (float*)malloc(n_vectors * dim * sizeof(float));
    float* h_queries = (float*)malloc(n_queries * dim * sizeof(float));
    int64_t* h_ground_truth = (int64_t*)malloc(n_queries * k * sizeof(int64_t));

    // Generate data
    generate_random_data(h_dataset, n_vectors, dim);
    generate_random_data(h_queries, n_queries, dim);

    // Calculate ground truth
    printf("  Computing ground truth...\n");
    brute_force_knn(h_dataset, h_queries, h_ground_truth, n_vectors, n_queries, dim, k);

    // Allocate device memory
    float *d_dataset, *d_queries;
    int64_t *d_neighbors;
    float *d_distances;

    CUDA_CHECK(cudaMalloc((void**)&d_dataset, n_vectors * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_queries, n_queries * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_neighbors, n_queries * k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_distances, n_queries * k * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_dataset, h_dataset, n_vectors * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, n_queries * dim * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUVS resources
    cuvsResources_t res;
    CUVS_CHECK(cuvsResourcesCreate(&res));

    // Setup dataset tensor (on GPU)
    DLManagedTensor dataset_tensor = {0};
    dataset_tensor.dl_tensor.data = d_dataset;
    dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
    dataset_tensor.dl_tensor.device.device_id = 0;
    dataset_tensor.dl_tensor.ndim = 2;
    int64_t dataset_shape[2] = {n_vectors, dim};
    dataset_tensor.dl_tensor.shape = dataset_shape;
    dataset_tensor.dl_tensor.dtype.code = kDLFloat;
    dataset_tensor.dl_tensor.dtype.bits = 32;
    dataset_tensor.dl_tensor.dtype.lanes = 1;
    dataset_tensor.dl_tensor.strides = NULL;
    dataset_tensor.dl_tensor.byte_offset = 0;

    // Create and configure index params
    cuvsCagraIndexParams_t index_params;
    CUVS_CHECK(cuvsCagraIndexParamsCreate(&index_params));
    index_params->metric = L2Expanded;  // Use L2 distance
    index_params->graph_degree = 64;
    index_params->intermediate_graph_degree = 128;
    index_params->build_algo = NN_DESCENT;

    // Create index
    cuvsCagraIndex_t index;
    CUVS_CHECK(cuvsCagraIndexCreate(&index));

    // Build index and measure time
    printf("  Building CAGRA index...\n");
    clock_t start = clock();
    CUVS_CHECK(cuvsCagraBuild(res, index_params, &dataset_tensor, index));
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_t end = clock();
    result.build_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("  ✅ Index built in %.2f ms\n", result.build_time_ms);

    // Setup query tensor
    DLManagedTensor queries_tensor = {0};
    queries_tensor.dl_tensor.data = d_queries;
    queries_tensor.dl_tensor.device.device_type = kDLCUDA;
    queries_tensor.dl_tensor.device.device_id = 0;
    queries_tensor.dl_tensor.ndim = 2;
    int64_t queries_shape[2] = {n_queries, dim};
    queries_tensor.dl_tensor.shape = queries_shape;
    queries_tensor.dl_tensor.dtype.code = kDLFloat;
    queries_tensor.dl_tensor.dtype.bits = 32;
    queries_tensor.dl_tensor.dtype.lanes = 1;
    queries_tensor.dl_tensor.strides = NULL;
    queries_tensor.dl_tensor.byte_offset = 0;

    // Setup result tensors
    DLManagedTensor neighbors_tensor = {0};
    neighbors_tensor.dl_tensor.data = d_neighbors;
    neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
    neighbors_tensor.dl_tensor.device.device_id = 0;
    neighbors_tensor.dl_tensor.ndim = 2;
    int64_t neighbors_shape[2] = {n_queries, k};
    neighbors_tensor.dl_tensor.shape = neighbors_shape;
    neighbors_tensor.dl_tensor.dtype.code = kDLInt;
    neighbors_tensor.dl_tensor.dtype.bits = 64;
    neighbors_tensor.dl_tensor.dtype.lanes = 1;
    neighbors_tensor.dl_tensor.strides = NULL;
    neighbors_tensor.dl_tensor.byte_offset = 0;

    DLManagedTensor distances_tensor = {0};
    distances_tensor.dl_tensor.data = d_distances;
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.device.device_id = 0;
    distances_tensor.dl_tensor.ndim = 2;
    distances_tensor.dl_tensor.shape = neighbors_shape;
    distances_tensor.dl_tensor.dtype.code = kDLFloat;
    distances_tensor.dl_tensor.dtype.bits = 32;
    distances_tensor.dl_tensor.dtype.lanes = 1;
    distances_tensor.dl_tensor.strides = NULL;
    distances_tensor.dl_tensor.byte_offset = 0;

    // Create search params
    cuvsCagraSearchParams_t search_params;
    CUVS_CHECK(cuvsCagraSearchParamsCreate(&search_params));
    search_params->max_queries = n_queries;
    search_params->itopk_size = k * 10;  // Much larger candidate set
    search_params->max_iterations = 0;
    search_params->algo = AUTO;
    search_params->team_size = 0;
    search_params->search_width = 4;  // Wider search
    search_params->min_iterations = 20;  // More iterations for quality
    search_params->thread_block_size = 0;
    search_params->hashmap_mode = AUTO_HASH;
    search_params->hashmap_min_bitlen = 0;
    search_params->hashmap_max_fill_rate = 0.5;
    search_params->num_random_samplings = 1;
    search_params->rand_xor_mask = 0x128394;

    // Warm up
    cuvsFilter filter = {0, NO_FILTER};
    CUVS_CHECK(cuvsCagraSearch(res, search_params, index, &queries_tensor,
                               &neighbors_tensor, &distances_tensor, filter));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark search
    printf("  Running search benchmark...\n");
    int num_runs = 100;
    start = clock();
    for (int i = 0; i < num_runs; i++) {
        CUVS_CHECK(cuvsCagraSearch(res, search_params, index, &queries_tensor,
                                   &neighbors_tensor, &distances_tensor, filter));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = clock();

    double total_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    result.search_time_ms = total_time_ms / num_runs;
    result.throughput_qps = (n_queries * num_runs * 1000.0) / total_time_ms;

    // Calculate recall
    int64_t* h_results = (int64_t*)malloc(n_queries * k * sizeof(int64_t));
    CUDA_CHECK(cudaMemcpy(h_results, d_neighbors, n_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));
    result.recall_at_k = calculate_recall(h_results, h_ground_truth, n_queries, k);

    printf("  ✅ Search: %.3f ms/query, %.0f QPS\n", result.search_time_ms, result.throughput_qps);
    printf("  ✅ Recall@%d: %.1f%%\n", k, result.recall_at_k * 100);

    // Cleanup
    free(h_results);
    cuvsCagraSearchParamsDestroy(search_params);
    cuvsCagraIndexDestroy(index);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsResourcesDestroy(res);

    cudaFree(d_dataset);
    cudaFree(d_queries);
    cudaFree(d_neighbors);
    cudaFree(d_distances);

    free(h_dataset);
    free(h_queries);
    free(h_ground_truth);

    return result;
}

int main() {
    printf("🚀 CUVS CAGRA Working Benchmark\n");
    printf("================================\n\n");

    // Check CUDA
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%.2f GB, Compute %d.%d)\n",
           prop.name, prop.totalGlobalMem / (1024.0*1024.0*1024.0),
           prop.major, prop.minor);

    // Test configurations
    struct {
        int n_vectors;
        int n_queries;
        int dim;
        int k;
    } test_configs[] = {
        {1000, 10, 128, 10},      // Small
        {10000, 100, 256, 20},    // Medium
        {50000, 100, 512, 50},    // Large
        {100000, 100, 768, 100},  // Extra large
    };

    printf("\n🔬 Running Benchmarks\n");
    printf("====================\n");

    BenchmarkResult* results = (BenchmarkResult*)malloc(sizeof(test_configs)/sizeof(test_configs[0]) * sizeof(BenchmarkResult));

    for (int i = 0; i < sizeof(test_configs)/sizeof(test_configs[0]); i++) {
        results[i] = benchmark_cagra(
            test_configs[i].n_vectors,
            test_configs[i].n_queries,
            test_configs[i].dim,
            test_configs[i].k
        );
    }

    // Print summary
    printf("\n\n📊 BENCHMARK SUMMARY\n");
    printf("====================\n\n");
    printf("Dataset Size | Build (ms) | Search (ms) | QPS      | Recall\n");
    printf("-------------|------------|-------------|----------|--------\n");

    for (int i = 0; i < sizeof(test_configs)/sizeof(test_configs[0]); i++) {
        printf("%12d | %10.2f | %11.3f | %8.0f | %5.1f%%\n",
               results[i].dataset_size,
               results[i].build_time_ms,
               results[i].search_time_ms,
               results[i].throughput_qps,
               results[i].recall_at_k * 100);
    }

    printf("\n✅ Benchmark complete!\n");

    free(results);
    return 0;
}