#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>

#define N_DATAPOINTS 10000
#define N_FEATURES 512
#define N_QUERIES 100
#define K 10

// Helper function to measure time
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Generate random float data
void generate_random_data(float* data, int n, int d) {
    for (int i = 0; i < n * d; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random between -1 and 1
    }
}

// Generate data with known duplicates for quality testing
void generate_data_with_duplicates(float* data, int n, int d, int duplicate_ratio) {
    int unique_count = n * (100 - duplicate_ratio) / 100;

    // Generate unique vectors
    for (int i = 0; i < unique_count; i++) {
        for (int j = 0; j < d; j++) {
            data[i * d + j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }

    // Add duplicates
    for (int i = unique_count; i < n; i++) {
        int source = rand() % unique_count;
        memcpy(&data[i * d], &data[source * d], d * sizeof(float));
    }
}

// Calculate cosine similarity
float cosine_similarity(const float* a, const float* b, int d) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Quality check: Find exact matches
int check_exact_matches(const float* dataset, const float* queries,
                       const uint32_t* neighbors, int n_data, int n_queries,
                       int d, int k) {
    int exact_matches = 0;

    for (int q = 0; q < n_queries; q++) {
        // Find the actual nearest neighbor by brute force
        int best_idx = -1;
        float best_sim = -2.0f;

        for (int i = 0; i < n_data; i++) {
            float sim = cosine_similarity(&queries[q * d], &dataset[i * d], d);
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = i;
            }
        }

        // Check if the best match is in the returned neighbors
        for (int j = 0; j < k; j++) {
            if (neighbors[q * k + j] == best_idx) {
                exact_matches++;
                break;
            }
        }
    }

    return exact_matches;
}

// Test duplicate handling
void test_duplicate_detection(const float* dataset, int n_data, int d,
                              cuvsResources_t res, cuvsCagraIndex_t index) {
    printf("\n📊 Duplicate Detection Test\n");
    printf("===========================\n");

    // Find duplicate pairs in dataset
    int duplicate_pairs = 0;
    int tested_pairs = 0;

    for (int i = 0; i < n_data && tested_pairs < 5; i++) {
        for (int j = i + 1; j < n_data; j++) {
            // Check if vectors are identical
            int is_duplicate = 1;
            for (int k = 0; k < d; k++) {
                if (fabs(dataset[i * d + k] - dataset[j * d + k]) > 1e-6) {
                    is_duplicate = 0;
                    break;
                }
            }

            if (is_duplicate) {
                tested_pairs++;
                printf("\nFound duplicate pair: [%d, %d]\n", i, j);

                // Search for this vector
                DLManagedTensor query_tensor;
                query_tensor.dl_tensor.data = (void*)&dataset[i * d];
                query_tensor.dl_tensor.device.device_type = kDLCUDA;
                query_tensor.dl_tensor.device.device_id = 0;
                query_tensor.dl_tensor.ndim = 2;
                query_tensor.dl_tensor.dtype.code = kDLFloat;
                query_tensor.dl_tensor.dtype.bits = 32;
                query_tensor.dl_tensor.dtype.lanes = 1;
                int64_t query_shape[] = {1, d};
                query_tensor.dl_tensor.shape = query_shape;
                query_tensor.dl_tensor.strides = NULL;

                // Allocate result buffers
                uint32_t* neighbors;
                float* distances;
                cudaMalloc((void**)&neighbors, K * sizeof(uint32_t));
                cudaMalloc((void**)&distances, K * sizeof(float));

                DLManagedTensor neighbors_tensor, distances_tensor;
                neighbors_tensor.dl_tensor.data = neighbors;
                neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
                neighbors_tensor.dl_tensor.device.device_id = 0;
                neighbors_tensor.dl_tensor.ndim = 2;
                neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
                neighbors_tensor.dl_tensor.dtype.bits = 32;
                neighbors_tensor.dl_tensor.dtype.lanes = 1;
                int64_t neighbors_shape[] = {1, K};
                neighbors_tensor.dl_tensor.shape = neighbors_shape;
                neighbors_tensor.dl_tensor.strides = NULL;

                distances_tensor.dl_tensor.data = distances;
                distances_tensor.dl_tensor.device.device_type = kDLCUDA;
                distances_tensor.dl_tensor.device.device_id = 0;
                distances_tensor.dl_tensor.ndim = 2;
                distances_tensor.dl_tensor.dtype.code = kDLFloat;
                distances_tensor.dl_tensor.dtype.bits = 32;
                distances_tensor.dl_tensor.dtype.lanes = 1;
                int64_t distances_shape[] = {1, K};
                distances_tensor.dl_tensor.shape = distances_shape;
                distances_tensor.dl_tensor.strides = NULL;

                // Create search parameters
                cuvsCagraSearchParams_t search_params;
                cuvsCagraSearchParamsCreate(&search_params);

                // Perform search
                cuvsFilter filter = {.type = NO_FILTER, .addr = 0};
                cuvsCagraSearch(res, search_params, index, &query_tensor,
                              &neighbors_tensor, &distances_tensor, filter);

                // Copy results to host
                uint32_t host_neighbors[K];
                float host_distances[K];
                cudaMemcpy(host_neighbors, neighbors, K * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(host_distances, distances, K * sizeof(float), cudaMemcpyDeviceToHost);

                // Check if both duplicates are found
                int found_i = 0, found_j = 0;
                for (int n = 0; n < K; n++) {
                    if (host_neighbors[n] == i) found_i = 1;
                    if (host_neighbors[n] == j) found_j = 1;
                }

                printf("  Duplicate %d found: %s (distance: %.4f)\n",
                       i, found_i ? "YES" : "NO", found_i ? host_distances[0] : -1);
                printf("  Duplicate %d found: %s\n",
                       j, found_j ? "YES" : "NO");

                if (found_i && found_j) {
                    printf("  ✅ Both duplicates found in top-%d\n", K);
                    duplicate_pairs++;
                } else {
                    printf("  ⚠️  Not all duplicates found\n");
                }

                // Cleanup
                cudaFree(neighbors);
                cudaFree(distances);
                cuvsCagraSearchParamsDestroy(search_params);
            }
        }
    }

    printf("\nDuplicate detection accuracy: %d/%d pairs correctly identified\n",
           duplicate_pairs, tested_pairs);
}

int main() {
    printf("🔬 CAGRA C-Level Benchmark & Quality Test\n");
    printf("==========================================\n");
    printf("Testing CAGRA at C API level without Go overhead\n");
    printf("Dataset: %d vectors, %d dimensions\n", N_DATAPOINTS, N_FEATURES);
    printf("Queries: %d, K: %d\n\n", N_QUERIES, K);

    // Initialize CUDA
    cudaSetDevice(0);

    // Create cuVS resources
    cuvsResources_t res;
    cuvsResourcesCreate(&res);

    // Generate dataset with duplicates
    printf("📊 Generating dataset with 20%% duplicates...\n");
    float* h_dataset = (float*)malloc(N_DATAPOINTS * N_FEATURES * sizeof(float));
    generate_data_with_duplicates(h_dataset, N_DATAPOINTS, N_FEATURES, 20);

    // Generate queries (use some from dataset for quality check)
    float* h_queries = (float*)malloc(N_QUERIES * N_FEATURES * sizeof(float));
    for (int i = 0; i < N_QUERIES; i++) {
        if (i < 10) {
            // Use first 10 from dataset for exact match testing
            memcpy(&h_queries[i * N_FEATURES], &h_dataset[i * N_FEATURES],
                   N_FEATURES * sizeof(float));
        } else {
            // Random queries
            generate_random_data(&h_queries[i * N_FEATURES], 1, N_FEATURES);
        }
    }

    // Allocate GPU memory
    float *d_dataset, *d_queries;
    cudaMalloc((void**)&d_dataset, N_DATAPOINTS * N_FEATURES * sizeof(float));
    cudaMalloc((void**)&d_queries, N_QUERIES * N_FEATURES * sizeof(float));

    // Copy to GPU
    cudaMemcpy(d_dataset, h_dataset, N_DATAPOINTS * N_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, N_QUERIES * N_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);

    // Create DLPack tensors
    DLManagedTensor dataset_tensor;
    dataset_tensor.dl_tensor.data = d_dataset;
    dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
    dataset_tensor.dl_tensor.device.device_id = 0;
    dataset_tensor.dl_tensor.ndim = 2;
    dataset_tensor.dl_tensor.dtype.code = kDLFloat;
    dataset_tensor.dl_tensor.dtype.bits = 32;
    dataset_tensor.dl_tensor.dtype.lanes = 1;
    int64_t dataset_shape[] = {N_DATAPOINTS, N_FEATURES};
    dataset_tensor.dl_tensor.shape = dataset_shape;
    dataset_tensor.dl_tensor.strides = NULL;

    // Create CAGRA index
    printf("\n🏗️ Building CAGRA Index\n");
    printf("=======================\n");

    cuvsCagraIndex_t index;
    cuvsCagraIndexCreate(&index);

    // Create optimized index parameters for RTX 3090
    cuvsCagraIndexParams_t index_params;
    cuvsCagraIndexParamsCreate(&index_params);

    // Set CAGRA parameters for ultra-fast search
    index_params->intermediate_graph_degree = 64;  // Graph connectivity
    index_params->graph_degree = 32;               // Output graph degree
    index_params->build_algo = NN_DESCENT;         // Fast build algorithm

    // Build index
    double build_start = get_time_ms();
    cuvsError_t error = cuvsCagraBuild(res, index_params, &dataset_tensor, index);
    double build_time = get_time_ms() - build_start;

    if (error != CUVS_SUCCESS) {
        printf("❌ Index build failed with error: %d\n", error);
        return 1;
    }

    printf("✅ Index built in %.2f ms\n", build_time);

    // Performance Benchmark
    printf("\n⚡ Performance Benchmark\n");
    printf("========================\n");

    // Allocate result buffers
    uint32_t* d_neighbors;
    float* d_distances;
    cudaMalloc((void**)&d_neighbors, N_QUERIES * K * sizeof(uint32_t));
    cudaMalloc((void**)&d_distances, N_QUERIES * K * sizeof(float));

    // Create query tensor
    DLManagedTensor query_tensor;
    query_tensor.dl_tensor.data = d_queries;
    query_tensor.dl_tensor.device.device_type = kDLCUDA;
    query_tensor.dl_tensor.device.device_id = 0;
    query_tensor.dl_tensor.ndim = 2;
    query_tensor.dl_tensor.dtype.code = kDLFloat;
    query_tensor.dl_tensor.dtype.bits = 32;
    query_tensor.dl_tensor.dtype.lanes = 1;
    int64_t query_shape[] = {N_QUERIES, N_FEATURES};
    query_tensor.dl_tensor.shape = query_shape;
    query_tensor.dl_tensor.strides = NULL;

    // Create result tensors
    DLManagedTensor neighbors_tensor, distances_tensor;
    neighbors_tensor.dl_tensor.data = d_neighbors;
    neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
    neighbors_tensor.dl_tensor.device.device_id = 0;
    neighbors_tensor.dl_tensor.ndim = 2;
    neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
    neighbors_tensor.dl_tensor.dtype.bits = 32;
    neighbors_tensor.dl_tensor.dtype.lanes = 1;
    int64_t neighbors_shape[] = {N_QUERIES, K};
    neighbors_tensor.dl_tensor.shape = neighbors_shape;
    neighbors_tensor.dl_tensor.strides = NULL;

    distances_tensor.dl_tensor.data = d_distances;
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.device.device_id = 0;
    distances_tensor.dl_tensor.ndim = 2;
    distances_tensor.dl_tensor.dtype.code = kDLFloat;
    distances_tensor.dl_tensor.dtype.bits = 32;
    distances_tensor.dl_tensor.dtype.lanes = 1;
    int64_t distances_shape[] = {N_QUERIES, K};
    distances_tensor.dl_tensor.shape = distances_shape;
    distances_tensor.dl_tensor.strides = NULL;

    // Create search parameters
    cuvsCagraSearchParams_t search_params;
    cuvsCagraSearchParamsCreate(&search_params);

    // Optimize for ultra-fast search on RTX 3090
    search_params->max_queries = 5000;        // Large batch size
    search_params->itopk_size = 64;          // Intermediate results
    search_params->algo = MULTI_CTA;         // Multi-CTA for high throughput
    search_params->team_size = 32;           // Thread team size
    search_params->thread_block_size = 1024; // Max thread block
    search_params->search_width = 1;         // Single width for speed
    search_params->hashmap_mode = SMALL;     // Small hashmap for speed

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 5; i++) {
        cuvsFilter filter = {.type = NO_FILTER, .addr = 0};
        cuvsCagraSearch(res, search_params, index, &query_tensor,
                       &neighbors_tensor, &distances_tensor, filter);
    }
    cudaDeviceSynchronize();

    // Benchmark search
    printf("Running benchmark...\n");
    int n_iterations = 100;
    double total_time = 0;
    double min_time = 1e9, max_time = 0;

    for (int i = 0; i < n_iterations; i++) {
        double start = get_time_ms();

        cuvsFilter filter = {.type = NO_FILTER, .addr = 0};
        cuvsCagraSearch(res, search_params, index, &query_tensor,
                       &neighbors_tensor, &distances_tensor, filter);
        cudaDeviceSynchronize();

        double elapsed = get_time_ms() - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
    }

    double avg_time = total_time / n_iterations;
    double qps = (N_QUERIES * 1000.0) / avg_time;

    printf("\nPerformance Results:\n");
    printf("  Average search time: %.3f ms\n", avg_time);
    printf("  Min search time: %.3f ms\n", min_time);
    printf("  Max search time: %.3f ms\n", max_time);
    printf("  Throughput: %.1f queries/sec\n", qps);
    printf("  Per-query latency: %.3f ms\n", avg_time / N_QUERIES);

    // Quality Assessment
    printf("\n🎯 Quality Assessment\n");
    printf("====================\n");

    // Copy results to host for quality check
    uint32_t* h_neighbors = (uint32_t*)malloc(N_QUERIES * K * sizeof(uint32_t));
    cudaMemcpy(h_neighbors, d_neighbors, N_QUERIES * K * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // Check exact matches for first 10 queries (which are from dataset)
    int exact_matches = 0;
    for (int i = 0; i < 10; i++) {
        // The first neighbor should be itself (exact match)
        if (h_neighbors[i * K] == i) {
            exact_matches++;
        }
    }

    printf("Exact match accuracy: %d/10 (%.1f%%)\n",
           exact_matches, exact_matches * 10.0f);

    // Overall quality check
    int total_exact = check_exact_matches(h_dataset, h_queries, h_neighbors,
                                          N_DATAPOINTS, N_QUERIES, N_FEATURES, K);
    printf("Overall recall@%d: %d/%d (%.1f%%)\n", K, total_exact, N_QUERIES,
           (float)total_exact / N_QUERIES * 100.0f);

    // Test duplicate detection
    test_duplicate_detection(d_dataset, N_DATAPOINTS, N_FEATURES, res, index);

    // Cleanup
    printf("\n🧹 Cleanup\n");
    cuvsCagraSearchParamsDestroy(search_params);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsCagraIndexDestroy(index);
    cuvsResourcesDestroy(res);

    cudaFree(d_dataset);
    cudaFree(d_queries);
    cudaFree(d_neighbors);
    cudaFree(d_distances);

    free(h_dataset);
    free(h_queries);
    free(h_neighbors);

    printf("\n✅ C-Level Benchmark Complete\n");
    return 0;
}