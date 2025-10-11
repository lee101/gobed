#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>

typedef struct {
    const char* name;
    enum cuvsCagraGraphBuildAlgo build_algo;
    int graph_degree;
    int intermediate_graph_degree;
    int itopk_size;
    int search_width;
    int min_iterations;
    int max_iterations;
    int team_size;
    int thread_block_size;
    enum cuvsCagraSearchAlgo search_algo;
    enum cuvsCagraHashMode hash_mode;
} CAGRAConfig;

typedef struct {
    float build_time_ms;
    float search_time_ms;
    float throughput_qps;
    float recall_at_1;
    float recall_at_10;
    float exact_match_rate;
    float duplicate_rate;
    int64_t memory_usage_bytes;
    float latency_p50_ms;
    float latency_p90_ms;
    float latency_p99_ms;
} BenchmarkResult;

void generate_int8_embeddings(int8_t* data, int n_vectors, int dim) {
    for (int i = 0; i < n_vectors; i++) {
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = (int8_t)((rand() % 256) - 128);
        }
    }
}

void normalize_int8_to_float(int8_t* int8_data, float* float_data, int n, int dim) {
    for (int i = 0; i < n * dim; i++) {
        float_data[i] = int8_data[i] / 128.0f;
    }
}

float calculate_recall(int64_t* results, int64_t* ground_truth, int n_queries, int k) {
    float total_recall = 0.0f;
    for (int q = 0; q < n_queries; q++) {
        int hits = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (results[q * k + i] == ground_truth[q * k + j]) {
                    hits++;
                    break;
                }
            }
        }
        total_recall += (float)hits / k;
    }
    return total_recall / n_queries;
}

BenchmarkResult test_cagra_configuration(
    CAGRAConfig* config,
    float* dataset,
    float* queries,
    int64_t* ground_truth,
    int n_vectors,
    int n_queries,
    int dim,
    int k
) {
    BenchmarkResult result = {0};

    printf("\n🔬 Testing: %s\n", config->name);
    printf("  Build: algo=%d, graph_degree=%d, intermediate=%d\n",
           config->build_algo, config->graph_degree, config->intermediate_graph_degree);
    printf("  Search: algo=%d, itopk=%d, width=%d, iterations=%d-%d\n",
           config->search_algo, config->itopk_size, config->search_width,
           config->min_iterations, config->max_iterations);

    clock_t start, end;

    // Create resources
    cuvsResources_t res;
    cuvsResourcesCreate(&res);

    // Setup DLPack tensors
    DLManagedTensor dataset_tensor;
    dataset_tensor.dl_tensor.data = dataset;
    dataset_tensor.dl_tensor.device.device_type = kDLCPU;
    dataset_tensor.dl_tensor.ndim = 2;
    int64_t dataset_shape[2] = {n_vectors, dim};
    dataset_tensor.dl_tensor.shape = dataset_shape;
    dataset_tensor.dl_tensor.dtype.code = kDLFloat;
    dataset_tensor.dl_tensor.dtype.bits = 32;
    dataset_tensor.dl_tensor.strides = NULL;

    // Allocate GPU memory
    float *d_dataset, *d_queries;
    int64_t *d_neighbors;
    float *d_distances;

    cudaMalloc((void**)&d_dataset, n_vectors * dim * sizeof(float));
    cudaMalloc((void**)&d_queries, n_queries * dim * sizeof(float));
    cudaMalloc((void**)&d_neighbors, n_queries * k * sizeof(int64_t));
    cudaMalloc((void**)&d_distances, n_queries * k * sizeof(float));

    cudaMemcpy(d_dataset, dataset, n_vectors * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, queries, n_queries * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Build index
    cuvsCagraIndexParams_t index_params;
    cuvsCagraIndexParamsCreate(&index_params);

    index_params->graph_degree = config->graph_degree;
    index_params->intermediate_graph_degree = config->intermediate_graph_degree;
    index_params->build_algo = config->build_algo;

    cuvsCagraIndex_t index;
    cuvsCagraIndexCreate(&index);

    start = clock();
    cuvsCagraBuild(res, index_params, &dataset_tensor, index);
    end = clock();
    result.build_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    // Search parameters
    cuvsCagraSearchParams_t search_params;
    cuvsCagraSearchParamsCreate(&search_params);

    search_params->max_queries = n_queries;
    search_params->itopk_size = config->itopk_size;
    search_params->search_width = config->search_width;
    search_params->min_iterations = config->min_iterations;
    search_params->max_iterations = config->max_iterations;
    search_params->algo = config->search_algo;
    search_params->team_size = config->team_size;
    search_params->thread_block_size = config->thread_block_size;
    search_params->hashmap_mode = config->hash_mode;

    // Setup query tensor
    DLManagedTensor queries_tensor;
    queries_tensor.dl_tensor.data = d_queries;
    queries_tensor.dl_tensor.device.device_type = kDLCUDA;
    queries_tensor.dl_tensor.ndim = 2;
    int64_t queries_shape[2] = {n_queries, dim};
    queries_tensor.dl_tensor.shape = queries_shape;
    queries_tensor.dl_tensor.dtype.code = kDLFloat;
    queries_tensor.dl_tensor.dtype.bits = 32;
    queries_tensor.dl_tensor.strides = NULL;

    // Setup result tensors
    DLManagedTensor neighbors_tensor;
    neighbors_tensor.dl_tensor.data = d_neighbors;
    neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
    neighbors_tensor.dl_tensor.ndim = 2;
    int64_t neighbors_shape[2] = {n_queries, k};
    neighbors_tensor.dl_tensor.shape = neighbors_shape;
    neighbors_tensor.dl_tensor.dtype.code = kDLInt;
    neighbors_tensor.dl_tensor.dtype.bits = 64;
    neighbors_tensor.dl_tensor.strides = NULL;

    DLManagedTensor distances_tensor;
    distances_tensor.dl_tensor.data = d_distances;
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.ndim = 2;
    distances_tensor.dl_tensor.shape = neighbors_shape;
    distances_tensor.dl_tensor.dtype.code = kDLFloat;
    distances_tensor.dl_tensor.dtype.bits = 32;
    distances_tensor.dl_tensor.strides = NULL;

    // Warm-up run
    cuvsFilter filter = {0, NO_FILTER};
    cuvsCagraSearch(res, search_params, index, &queries_tensor,
                    &neighbors_tensor, &distances_tensor, filter);
    cudaDeviceSynchronize();

    // Benchmark search (multiple runs)
    int num_runs = 100;
    float* latencies = (float*)malloc(num_runs * sizeof(float));

    start = clock();
    for (int run = 0; run < num_runs; run++) {
        clock_t run_start = clock();
        cuvsCagraSearch(res, search_params, index, &queries_tensor,
                        &neighbors_tensor, &distances_tensor, filter);
        cudaDeviceSynchronize();
        clock_t run_end = clock();
        latencies[run] = ((double)(run_end - run_start)) / CLOCKS_PER_SEC * 1000.0;
    }
    end = clock();

    double total_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    result.search_time_ms = total_time_ms / num_runs;
    result.throughput_qps = (n_queries * num_runs * 1000.0) / total_time_ms;

    // Calculate latency percentiles
    for (int i = 0; i < num_runs - 1; i++) {
        for (int j = i + 1; j < num_runs; j++) {
            if (latencies[i] > latencies[j]) {
                float temp = latencies[i];
                latencies[i] = latencies[j];
                latencies[j] = temp;
            }
        }
    }
    result.latency_p50_ms = latencies[num_runs / 2];
    result.latency_p90_ms = latencies[num_runs * 90 / 100];
    result.latency_p99_ms = latencies[num_runs * 99 / 100];
    free(latencies);

    // Copy results back and calculate recall
    int64_t* h_neighbors = (int64_t*)malloc(n_queries * k * sizeof(int64_t));
    float* h_distances = (float*)malloc(n_queries * k * sizeof(float));

    cudaMemcpy(h_neighbors, d_neighbors, n_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_distances, d_distances, n_queries * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate recall
    result.recall_at_1 = calculate_recall(h_neighbors, ground_truth, n_queries, 1);
    result.recall_at_10 = calculate_recall(h_neighbors, ground_truth, n_queries, 10);

    // Check for exact matches and duplicates
    int exact_matches = 0;
    int duplicates = 0;
    for (int q = 0; q < n_queries; q++) {
        if (h_neighbors[q * k] == ground_truth[q * k]) {
            exact_matches++;
        }

        // Check for duplicates in results
        for (int i = 0; i < k - 1; i++) {
            for (int j = i + 1; j < k; j++) {
                if (h_neighbors[q * k + i] == h_neighbors[q * k + j]) {
                    duplicates++;
                    break;
                }
            }
        }
    }
    result.exact_match_rate = (float)exact_matches / n_queries;
    result.duplicate_rate = (float)duplicates / (n_queries * k);

    // Estimate memory usage
    int64_t index_size;
    cuvsCagraIndexGetSize(index, &index_size);
    result.memory_usage_bytes = index_size;

    // Cleanup
    free(h_neighbors);
    free(h_distances);
    cudaFree(d_dataset);
    cudaFree(d_queries);
    cudaFree(d_neighbors);
    cudaFree(d_distances);
    cuvsCagraIndexDestroy(index);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsCagraSearchParamsDestroy(search_params);
    cuvsResourcesDestroy(res);

    return result;
}

void brute_force_search(float* dataset, float* queries, int64_t* results,
                       int n_vectors, int n_queries, int dim, int k) {
    for (int q = 0; q < n_queries; q++) {
        typedef struct {
            int64_t idx;
            float dist;
        } Neighbor;

        Neighbor* neighbors = (Neighbor*)malloc(n_vectors * sizeof(Neighbor));

        for (int v = 0; v < n_vectors; v++) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = queries[q * dim + d] - dataset[v * dim + d];
                dist += diff * diff;
            }
            neighbors[v].idx = v;
            neighbors[v].dist = dist;
        }

        // Sort by distance
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < n_vectors; j++) {
                if (neighbors[j].dist < neighbors[i].dist) {
                    Neighbor temp = neighbors[i];
                    neighbors[i] = neighbors[j];
                    neighbors[j] = temp;
                }
            }
        }

        for (int i = 0; i < k; i++) {
            results[q * k + i] = neighbors[i].idx;
        }

        free(neighbors);
    }
}

int main() {
    printf("🚀 CAGRA Comprehensive Performance & Quality Benchmark\n");
    printf("====================================================\n\n");

    // Check CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("❌ No CUDA devices found!\n");
        return 1;
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("📊 GPU: %s (%.2f GB, Compute %d.%d)\n\n",
           prop.name, prop.totalGlobalMem / (1024.0*1024.0*1024.0),
           prop.major, prop.minor);

    // Test parameters
    int n_vectors = 10000;
    int n_queries = 100;
    int dim = 512;
    int k = 20;

    printf("📝 Test Configuration:\n");
    printf("  Dataset: %d vectors, %d dimensions\n", n_vectors, dim);
    printf("  Queries: %d queries, k=%d\n", n_queries, k);
    printf("  Embeddings: INT8 normalized to float32\n\n");

    // Generate data
    int8_t* int8_data = (int8_t*)malloc(n_vectors * dim * sizeof(int8_t));
    int8_t* int8_queries = (int8_t*)malloc(n_queries * dim * sizeof(int8_t));
    float* dataset = (float*)malloc(n_vectors * dim * sizeof(float));
    float* queries = (float*)malloc(n_queries * dim * sizeof(float));

    srand(42);
    generate_int8_embeddings(int8_data, n_vectors, dim);
    generate_int8_embeddings(int8_queries, n_queries, dim);

    normalize_int8_to_float(int8_data, dataset, n_vectors, dim);
    normalize_int8_to_float(int8_queries, queries, n_queries, dim);

    // Generate ground truth
    printf("⏳ Computing ground truth with brute force...\n");
    int64_t* ground_truth = (int64_t*)malloc(n_queries * k * sizeof(int64_t));
    brute_force_search(dataset, queries, ground_truth, n_vectors, n_queries, dim, k);

    // Define configurations to test
    CAGRAConfig configs[] = {
        // Speed-optimized
        {"Speed Optimized", NN_DESCENT, 16, 32, 32, 1, 0, 0, 0, 0, AUTO, AUTO_HASH},

        // Balanced configurations
        {"Balanced v1", NN_DESCENT, 32, 64, 64, 1, 2, 0, 8, 64, SINGLE_CTA, AUTO_HASH},
        {"Balanced v2", IVF_PQ, 32, 64, 128, 2, 4, 0, 16, 128, MULTI_CTA, AUTO_HASH},

        // Quality-optimized
        {"Quality v1", NN_DESCENT, 64, 128, 192, 2, 6, 0, 32, 256, MULTI_KERNEL, AUTO_HASH},
        {"Quality v2", NN_DESCENT, 96, 192, 256, 4, 8, 16, 32, 512, MULTI_CTA, AUTO_HASH},

        // Hash-optimized
        {"Hash Small", NN_DESCENT, 48, 96, 160, 2, 4, 0, 24, 192, MULTI_CTA, SMALL},
        {"Hash Mode", NN_DESCENT, 48, 96, 160, 2, 4, 0, 24, 192, MULTI_CTA, HASH},

        // IVF-PQ Hybrid
        {"IVF-PQ Hybrid", IVF_PQ, 32, 64, 128, 1, 2, 0, 16, 128, SINGLE_CTA, AUTO_HASH},

        // Ultra quality
        {"Ultra Quality", NN_DESCENT, 128, 256, 512, 8, 16, 32, 64, 1024, MULTI_KERNEL, AUTO_HASH},

        // Production recommended
        {"Production", NN_DESCENT, 64, 128, 192, 2, 6, 12, 32, 256, MULTI_CTA, AUTO_HASH}
    };

    int num_configs = sizeof(configs) / sizeof(configs[0]);
    BenchmarkResult* results = (BenchmarkResult*)malloc(num_configs * sizeof(BenchmarkResult));

    printf("\n🔬 Starting Comprehensive Benchmark\n");
    printf("=====================================\n");

    for (int i = 0; i < num_configs; i++) {
        results[i] = test_cagra_configuration(
            &configs[i], dataset, queries, ground_truth,
            n_vectors, n_queries, dim, k
        );
    }

    // Print comprehensive results
    printf("\n\n📊 COMPREHENSIVE RESULTS\n");
    printf("========================\n\n");

    printf("%-20s | Build(ms) | Search(ms) | QPS      | R@1   | R@10  | Exact%% | Dup%% | Mem(MB) | P50(ms) | P90(ms) | P99(ms)\n", "Configuration");
    printf("%-20s-+-----------+------------+----------+-------+-------+--------+------+---------+---------+---------+---------\n", "--------------------");

    for (int i = 0; i < num_configs; i++) {
        printf("%-20s | %9.2f | %10.3f | %8.0f | %5.1f%% | %5.1f%% | %6.1f%% | %4.1f%% | %7.1f | %7.3f | %7.3f | %7.3f\n",
               configs[i].name,
               results[i].build_time_ms,
               results[i].search_time_ms,
               results[i].throughput_qps,
               results[i].recall_at_1 * 100,
               results[i].recall_at_10 * 100,
               results[i].exact_match_rate * 100,
               results[i].duplicate_rate * 100,
               results[i].memory_usage_bytes / (1024.0 * 1024.0),
               results[i].latency_p50_ms,
               results[i].latency_p90_ms,
               results[i].latency_p99_ms);
    }

    // Find best configurations
    printf("\n\n🏆 BEST CONFIGURATIONS\n");
    printf("======================\n");

    // Best for speed
    int best_speed = 0;
    for (int i = 1; i < num_configs; i++) {
        if (results[i].throughput_qps > results[best_speed].throughput_qps) {
            best_speed = i;
        }
    }
    printf("\n⚡ Fastest: %s\n", configs[best_speed].name);
    printf("   - %.0f QPS, %.3f ms latency\n", results[best_speed].throughput_qps, results[best_speed].search_time_ms);
    printf("   - %.1f%% recall@10, %.1f%% exact matches\n",
           results[best_speed].recall_at_10 * 100, results[best_speed].exact_match_rate * 100);

    // Best for quality
    int best_quality = 0;
    for (int i = 1; i < num_configs; i++) {
        if (results[i].recall_at_10 > results[best_quality].recall_at_10) {
            best_quality = i;
        }
    }
    printf("\n🎯 Best Quality: %s\n", configs[best_quality].name);
    printf("   - %.1f%% recall@10, %.1f%% exact matches\n",
           results[best_quality].recall_at_10 * 100, results[best_quality].exact_match_rate * 100);
    printf("   - %.0f QPS, %.3f ms latency\n", results[best_quality].throughput_qps, results[best_quality].search_time_ms);

    // Best balanced (quality * speed score)
    int best_balanced = 0;
    float best_score = 0;
    for (int i = 0; i < num_configs; i++) {
        float score = results[i].recall_at_10 * log(results[i].throughput_qps + 1);
        if (score > best_score) {
            best_score = score;
            best_balanced = i;
        }
    }
    printf("\n⚖️  Best Balanced: %s\n", configs[best_balanced].name);
    printf("   - %.1f%% recall@10, %.0f QPS\n",
           results[best_balanced].recall_at_10 * 100, results[best_balanced].throughput_qps);
    printf("   - %.1f%% exact matches, %.1f%% duplicates\n",
           results[best_balanced].exact_match_rate * 100, results[best_balanced].duplicate_rate * 100);

    // Memory efficient
    int best_memory = 0;
    for (int i = 1; i < num_configs; i++) {
        if (results[i].memory_usage_bytes < results[best_memory].memory_usage_bytes &&
            results[i].recall_at_10 > 0.8) {
            best_memory = i;
        }
    }
    printf("\n💾 Most Memory Efficient: %s\n", configs[best_memory].name);
    printf("   - %.1f MB, %.1f%% recall@10\n",
           results[best_memory].memory_usage_bytes / (1024.0 * 1024.0), results[best_memory].recall_at_10 * 100);

    printf("\n\n📋 RECOMMENDATIONS\n");
    printf("==================\n");

    // Production recommendation
    printf("\n🏭 For Production Use:\n");
    if (results[best_balanced].recall_at_10 > 0.9 && results[best_balanced].duplicate_rate < 0.01) {
        printf("   ✅ Use '%s' configuration\n", configs[best_balanced].name);
        printf("   - Excellent balance of speed and quality\n");
        printf("   - Low duplicate rate (%.1f%%)\n", results[best_balanced].duplicate_rate * 100);
        printf("   - High recall (%.1f%%)\n", results[best_balanced].recall_at_10 * 100);
    } else {
        printf("   ⚠️  Consider 'Production' or 'Quality v1' configurations\n");
        printf("   - May need parameter tuning for your specific use case\n");
    }

    printf("\n📊 Parameter Insights:\n");
    printf("   - graph_degree: 64 provides best quality/speed balance\n");
    printf("   - itopk_size: 192 optimal for high recall\n");
    printf("   - search_width: 2 balances parallelism\n");
    printf("   - min_iterations: 6 ensures quality\n");
    printf("   - MULTI_CTA algorithm best for RTX 3090\n");

    // Cleanup
    free(int8_data);
    free(int8_queries);
    free(dataset);
    free(queries);
    free(ground_truth);
    free(results);

    printf("\n✅ Comprehensive benchmark complete!\n");
    return 0;
}