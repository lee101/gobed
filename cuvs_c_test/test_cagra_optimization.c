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

// Configuration structure
typedef struct {
    const char* name;
    int graph_degree;
    int intermediate_degree;
    int itopk_size;
    int search_width;
    int min_iterations;
    int nprobe_equivalent;  // For comparison with Go implementation
} OptimizationConfig;

// Benchmark result
typedef struct {
    OptimizationConfig config;
    float build_time_ms;
    float search_time_ms;
    float throughput_qps;
    float recall_at_k;
    float exact_match_rate;
} BenchmarkResult;

// Generate random data
void generate_random_data(float* data, int n, int dim) {
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < n * dim; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// Calculate recall
float calculate_recall(int64_t* results, int64_t* ground_truth, int n_queries, int k) {
    int total_hits = 0;
    for (int q = 0; q < n_queries; q++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (results[q * k + i] == ground_truth[q * k + j]) {
                    total_hits++;
                    break;
                }
            }
        }
    }
    return (float)total_hits / (n_queries * k);
}

// Brute force search for ground truth
void brute_force_search(float* dataset, float* queries, int64_t* results,
                       int n_vectors, int n_queries, int dim, int k) {
    for (int q = 0; q < n_queries; q++) {
        float* distances = (float*)malloc(n_vectors * sizeof(float));

        // Calculate all distances
        for (int v = 0; v < n_vectors; v++) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = queries[q * dim + d] - dataset[v * dim + d];
                dist += diff * diff;
            }
            distances[v] = dist;
        }

        // Find k smallest
        for (int i = 0; i < k; i++) {
            float min_dist = INFINITY;
            int min_idx = 0;
            for (int v = 0; v < n_vectors; v++) {
                if (distances[v] < min_dist) {
                    min_dist = distances[v];
                    min_idx = v;
                }
            }
            results[q * k + i] = min_idx;
            distances[min_idx] = INFINITY;
        }

        free(distances);
    }
}

// Run benchmark for a specific configuration
BenchmarkResult run_benchmark(OptimizationConfig config, float* h_dataset, float* h_queries,
                              int64_t* h_ground_truth, int n_vectors, int n_queries, int dim, int k) {
    BenchmarkResult result = {0};
    result.config = config;

    printf("\nTesting: %s\n", config.name);
    printf("  GraphDegree: %d, Itopk: %d, Width: %d, Iterations: %d\n",
           config.graph_degree, config.itopk_size, config.search_width, config.min_iterations);

    // Allocate device memory
    float *d_dataset, *d_queries;
    int64_t *d_neighbors;
    float *d_distances;

    cudaMalloc((void**)&d_dataset, n_vectors * dim * sizeof(float));
    cudaMalloc((void**)&d_queries, n_queries * dim * sizeof(float));
    cudaMalloc((void**)&d_neighbors, n_queries * k * sizeof(int64_t));
    cudaMalloc((void**)&d_distances, n_queries * k * sizeof(float));

    cudaMemcpy(d_dataset, h_dataset, n_vectors * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, n_queries * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Create resources
    cuvsResources_t res;
    cuvsResourcesCreate(&res);

    // Setup dataset tensor
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

    // Create index params
    cuvsCagraIndexParams_t index_params;
    cuvsCagraIndexParamsCreate(&index_params);
    index_params->metric = L2Expanded;
    index_params->graph_degree = config.graph_degree;
    index_params->intermediate_graph_degree = config.intermediate_degree;
    index_params->build_algo = NN_DESCENT;

    // Create index
    cuvsCagraIndex_t index;
    cuvsCagraIndexCreate(&index);

    // Build index
    clock_t start = clock();
    cuvsCagraBuild(res, index_params, &dataset_tensor, index);
    cudaDeviceSynchronize();
    clock_t end = clock();
    result.build_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

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

    // Create search params
    cuvsCagraSearchParams_t search_params;
    cuvsCagraSearchParamsCreate(&search_params);
    search_params->max_queries = n_queries;
    search_params->itopk_size = config.itopk_size;
    search_params->search_width = config.search_width;
    search_params->min_iterations = config.min_iterations;
    search_params->max_iterations = 0;  // Auto
    search_params->algo = AUTO;

    // Warm-up
    cuvsFilter filter = {0, NO_FILTER};
    cuvsCagraSearch(res, search_params, index, &queries_tensor,
                   &neighbors_tensor, &distances_tensor, filter);
    cudaDeviceSynchronize();

    // Benchmark search
    int num_runs = 50;
    start = clock();
    for (int i = 0; i < num_runs; i++) {
        cuvsCagraSearch(res, search_params, index, &queries_tensor,
                       &neighbors_tensor, &distances_tensor, filter);
    }
    cudaDeviceSynchronize();
    end = clock();

    double total_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    result.search_time_ms = total_time_ms / num_runs;
    result.throughput_qps = (n_queries * num_runs * 1000.0) / total_time_ms;

    // Calculate recall
    int64_t* h_results = (int64_t*)malloc(n_queries * k * sizeof(int64_t));
    cudaMemcpy(h_results, d_neighbors, n_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost);
    result.recall_at_k = calculate_recall(h_results, h_ground_truth, n_queries, k);

    // Calculate exact match rate (first result matches ground truth first)
    int exact_matches = 0;
    for (int q = 0; q < n_queries; q++) {
        if (h_results[q * k] == h_ground_truth[q * k]) {
            exact_matches++;
        }
    }
    result.exact_match_rate = (float)exact_matches / n_queries;

    printf("  ✅ Build: %.2f ms, Search: %.3f ms/query, %.0f QPS\n",
           result.build_time_ms, result.search_time_ms, result.throughput_qps);
    printf("  ✅ Recall@%d: %.1f%%, Exact matches: %.1f%%\n",
           k, result.recall_at_k * 100, result.exact_match_rate * 100);

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

    return result;
}

int main() {
    printf("🚀 CAGRA Optimization Benchmark\n");
    printf("===============================\n\n");

    // Check CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%.2f GB, Compute %d.%d)\n\n",
           prop.name, prop.totalGlobalMem / (1024.0*1024.0*1024.0),
           prop.major, prop.minor);

    // Test parameters
    int n_vectors = 5000;
    int n_queries = 100;
    int dim = 512;
    int k = 20;

    printf("Dataset: %d vectors, %d dimensions\n", n_vectors, dim);
    printf("Queries: %d, k=%d\n\n", n_queries, k);

    // Generate test data
    printf("Generating test data...\n");
    float* h_dataset = (float*)malloc(n_vectors * dim * sizeof(float));
    float* h_queries = (float*)malloc(n_queries * dim * sizeof(float));
    int64_t* h_ground_truth = (int64_t*)malloc(n_queries * k * sizeof(int64_t));

    generate_random_data(h_dataset, n_vectors, dim);
    generate_random_data(h_queries, n_queries, dim);

    printf("Computing ground truth...\n");
    brute_force_search(h_dataset, h_queries, h_ground_truth, n_vectors, n_queries, dim, k);

    // Test configurations
    OptimizationConfig configs[] = {
        // Ultra-fast configurations
        {"Ultra-Fast v1", 16, 32, 32, 1, 0, 2},
        {"Ultra-Fast v2", 16, 32, 64, 1, 0, 4},

        // Fast configurations
        {"Fast v1", 32, 64, 64, 1, 2, 8},
        {"Fast v2", 32, 64, 96, 1, 2, 12},

        // Balanced configurations
        {"Balanced v1", 48, 96, 128, 2, 4, 16},
        {"Balanced v2", 64, 128, 160, 2, 4, 20},

        // Current "optimal" from our tests
        {"Current Optimal", 64, 128, 192, 2, 6, 24},

        // Quality-focused
        {"Quality v1", 80, 160, 256, 3, 8, 32},
        {"Quality v2", 96, 192, 320, 4, 10, 40},

        // Ultra-quality
        {"Ultra-Quality", 128, 256, 512, 4, 16, 64},
    };

    int num_configs = sizeof(configs) / sizeof(configs[0]);
    BenchmarkResult* results = (BenchmarkResult*)malloc(num_configs * sizeof(BenchmarkResult));

    printf("\n🔬 Running Benchmarks\n");
    printf("=====================\n");

    for (int i = 0; i < num_configs; i++) {
        results[i] = run_benchmark(configs[i], h_dataset, h_queries, h_ground_truth,
                                   n_vectors, n_queries, dim, k);
    }

    // Print summary
    printf("\n\n📊 BENCHMARK SUMMARY\n");
    printf("====================\n\n");

    printf("Configuration    | Build(ms) | Search(ms) | QPS     | Recall | Exact%%\n");
    printf("-----------------|-----------|------------|---------|--------|-------\n");

    for (int i = 0; i < num_configs; i++) {
        printf("%-16s | %9.2f | %10.3f | %7.0f | %5.1f%% | %5.1f%%\n",
               results[i].config.name,
               results[i].build_time_ms,
               results[i].search_time_ms,
               results[i].throughput_qps,
               results[i].recall_at_k * 100,
               results[i].exact_match_rate * 100);
    }

    // Find best configurations
    printf("\n\n🏆 BEST CONFIGURATIONS\n");
    printf("======================\n");

    // Best for speed with >90% recall
    int best_speed_idx = -1;
    for (int i = 0; i < num_configs; i++) {
        if (results[i].recall_at_k >= 0.9) {
            if (best_speed_idx == -1 || results[i].throughput_qps > results[best_speed_idx].throughput_qps) {
                best_speed_idx = i;
            }
        }
    }

    if (best_speed_idx >= 0) {
        printf("\n⚡ Best Speed (>90%% recall): %s\n", results[best_speed_idx].config.name);
        printf("   %.0f QPS, %.3f ms latency, %.1f%% recall\n",
               results[best_speed_idx].throughput_qps,
               results[best_speed_idx].search_time_ms,
               results[best_speed_idx].recall_at_k * 100);
    }

    // Best recall
    int best_recall_idx = 0;
    for (int i = 1; i < num_configs; i++) {
        if (results[i].recall_at_k > results[best_recall_idx].recall_at_k) {
            best_recall_idx = i;
        }
    }

    printf("\n🎯 Best Recall: %s\n", results[best_recall_idx].config.name);
    printf("   %.1f%% recall, %.1f%% exact matches\n",
           results[best_recall_idx].recall_at_k * 100,
           results[best_recall_idx].exact_match_rate * 100);

    // Best balanced (speed * recall)
    int best_balanced_idx = 0;
    float best_score = 0;
    for (int i = 0; i < num_configs; i++) {
        float score = results[i].recall_at_k * (results[i].throughput_qps / 1000.0);
        if (score > best_score) {
            best_score = score;
            best_balanced_idx = i;
        }
    }

    printf("\n⚖️  Best Balanced: %s\n", results[best_balanced_idx].config.name);
    printf("   %.0f QPS, %.1f%% recall, Score: %.1f\n",
           results[best_balanced_idx].throughput_qps,
           results[best_balanced_idx].recall_at_k * 100,
           best_score);

    printf("\n📋 RECOMMENDATION\n");
    printf("=================\n");
    printf("For production use with Go Custom CAGRA:\n");

    if (best_balanced_idx >= 0) {
        OptimizationConfig* rec = &results[best_balanced_idx].config;
        printf("  GraphDegree: %d\n", rec->graph_degree);
        printf("  IntermediateDegree: %d\n", rec->intermediate_degree);
        printf("  ItopkSize: %d\n", rec->itopk_size);
        printf("  SearchWidth: %d\n", rec->search_width);
        printf("  MinIterations: %d\n", rec->min_iterations);
        printf("  NProbe: %d\n", rec->nprobe_equivalent);
    }

    // Cleanup
    free(h_dataset);
    free(h_queries);
    free(h_ground_truth);
    free(results);

    printf("\n✅ Optimization complete!\n");
    return 0;
}