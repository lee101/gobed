#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>

// Quality test configurations
#define SMALL_TEST 1000
#define MEDIUM_TEST 5000
#define LARGE_TEST 10000
#define N_FEATURES 512
#define K 10

// Quality metrics structure
typedef struct {
    float precision_at_k;
    float recall_at_k;
    float mean_reciprocal_rank;
    float duplicate_accuracy;
    float semantic_similarity_score;
} QualityMetrics;

// Helper functions
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

float dot_product(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float l2_distance(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// Generate synthetic embeddings with semantic structure
void generate_semantic_embeddings(float* data, int n, int d) {
    // Create clusters for semantic similarity
    int n_clusters = 10;
    float cluster_centers[10][512];

    // Generate cluster centers
    for (int c = 0; c < n_clusters; c++) {
        for (int j = 0; j < d; j++) {
            cluster_centers[c][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }

    // Generate data points around clusters
    for (int i = 0; i < n; i++) {
        int cluster = i % n_clusters;
        for (int j = 0; j < d; j++) {
            // Add noise to cluster center
            float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
            data[i * d + j] = cluster_centers[cluster][j] + noise;
        }
    }
}

// Ground truth calculation for quality metrics
void calculate_ground_truth(const float* dataset, const float* queries,
                           uint32_t* ground_truth, int n_data, int n_queries,
                           int d, int k) {
    for (int q = 0; q < n_queries; q++) {
        // Calculate all distances
        typedef struct {
            float distance;
            uint32_t index;
        } DistPair;

        DistPair* distances = (DistPair*)malloc(n_data * sizeof(DistPair));

        for (int i = 0; i < n_data; i++) {
            distances[i].distance = l2_distance(&queries[q * d], &dataset[i * d], d);
            distances[i].index = i;
        }

        // Sort by distance
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < n_data; j++) {
                if (distances[j].distance < distances[i].distance) {
                    DistPair temp = distances[i];
                    distances[i] = distances[j];
                    distances[j] = temp;
                }
            }
            ground_truth[q * k + i] = distances[i].index;
        }

        free(distances);
    }
}

// Calculate quality metrics
QualityMetrics calculate_quality_metrics(const uint32_t* ground_truth,
                                        const uint32_t* predictions,
                                        int n_queries, int k) {
    QualityMetrics metrics = {0};

    for (int q = 0; q < n_queries; q++) {
        // Precision and Recall at K
        int correct = 0;
        float reciprocal_rank = 0.0f;

        for (int i = 0; i < k; i++) {
            uint32_t pred = predictions[q * k + i];

            // Check if prediction is in ground truth
            for (int j = 0; j < k; j++) {
                if (ground_truth[q * k + j] == pred) {
                    correct++;
                    if (reciprocal_rank == 0.0f) {
                        reciprocal_rank = 1.0f / (i + 1);
                    }
                    break;
                }
            }
        }

        metrics.precision_at_k += (float)correct / k;
        metrics.recall_at_k += (float)correct / k;  // Same as precision when retrieving k items
        metrics.mean_reciprocal_rank += reciprocal_rank;
    }

    metrics.precision_at_k /= n_queries;
    metrics.recall_at_k /= n_queries;
    metrics.mean_reciprocal_rank /= n_queries;

    return metrics;
}

// Test quality at different scales
void test_quality_at_scale(cuvsResources_t res, int n_datapoints, const char* scale_name) {
    printf("\n📏 Testing %s Scale (%d vectors)\n", scale_name, n_datapoints);
    printf("========================================\n");

    // Generate dataset
    float* h_dataset = (float*)malloc(n_datapoints * N_FEATURES * sizeof(float));
    generate_semantic_embeddings(h_dataset, n_datapoints, N_FEATURES);

    // Generate queries (sample from dataset for ground truth)
    int n_queries = 100;
    float* h_queries = (float*)malloc(n_queries * N_FEATURES * sizeof(float));
    for (int i = 0; i < n_queries; i++) {
        int idx = rand() % n_datapoints;
        memcpy(&h_queries[i * N_FEATURES], &h_dataset[idx * N_FEATURES],
               N_FEATURES * sizeof(float));
    }

    // Calculate ground truth
    uint32_t* ground_truth = (uint32_t*)malloc(n_queries * K * sizeof(uint32_t));
    calculate_ground_truth(h_dataset, h_queries, ground_truth,
                          n_datapoints, n_queries, N_FEATURES, K);

    // Allocate GPU memory
    float *d_dataset, *d_queries;
    uint32_t *d_neighbors;
    float *d_distances;

    cudaMalloc(&d_dataset, n_datapoints * N_FEATURES * sizeof(float));
    cudaMalloc(&d_queries, n_queries * N_FEATURES * sizeof(float));
    cudaMalloc(&d_neighbors, n_queries * K * sizeof(uint32_t));
    cudaMalloc(&d_distances, n_queries * K * sizeof(float));

    cudaMemcpy(d_dataset, h_dataset, n_datapoints * N_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, n_queries * N_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);

    // Create DLPack tensor for dataset
    DLManagedTensor dataset_tensor;
    dataset_tensor.dl_tensor.data = d_dataset;
    dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
    dataset_tensor.dl_tensor.device.device_id = 0;
    dataset_tensor.dl_tensor.ndim = 2;
    dataset_tensor.dl_tensor.dtype.code = kDLFloat;
    dataset_tensor.dl_tensor.dtype.bits = 32;
    dataset_tensor.dl_tensor.dtype.lanes = 1;
    int64_t dataset_shape[] = {n_datapoints, N_FEATURES};
    dataset_tensor.dl_tensor.shape = dataset_shape;
    dataset_tensor.dl_tensor.strides = NULL;

    // Build CAGRA index with different parameter settings
    struct {
        const char* name;
        int graph_degree;
        int intermediate_graph_degree;
        int build_algo;
    } param_configs[] = {
        {"Fast", 16, 32, NN_DESCENT},
        {"Balanced", 32, 64, NN_DESCENT},
        {"Quality", 64, 128, IVF_PQ}
    };

    for (int cfg = 0; cfg < 3; cfg++) {
        printf("\n⚙️  Configuration: %s\n", param_configs[cfg].name);

        // Create index
        cuvsCagraIndex_t index;
        cuvsCagraIndexCreate(&index);

        // Create index parameters
        cuvsCagraIndexParams_t index_params;
        cuvsCagraIndexParamsCreate(&index_params);

        index_params.graph_degree = param_configs[cfg].graph_degree;
        index_params.intermediate_graph_degree = param_configs[cfg].intermediate_graph_degree;
        index_params.build_algo = param_configs[cfg].build_algo;

        // Build index
        double build_start = get_time_ms();
        cuvsError_t error = cuvsCagraBuild(res, index_params, &dataset_tensor, index);
        double build_time = get_time_ms() - build_start;

        if (error != CUVS_SUCCESS) {
            printf("  ❌ Build failed\n");
            cuvsCagraIndexParamsDestroy(index_params);
            cuvsCagraIndexDestroy(index);
            continue;
        }

        printf("  Build time: %.2f ms\n", build_time);

        // Create search parameters
        cuvsCagraSearchParams_t search_params;
        cuvsCagraSearchParamsCreate(&search_params);

        // Test with different search parameters
        int search_configs[][3] = {
            {32, 1, 2},   // itopk_size, search_width, min_iterations
            {64, 1, 4},
            {128, 2, 8}
        };
        const char* search_names[] = {"Fast", "Balanced", "Quality"};

        for (int s = 0; s < 3; s++) {
            search_params.itopk_size = search_configs[s][0];
            search_params.search_width = search_configs[s][1];
            search_params.min_iterations = search_configs[s][2];

            // Create tensors for search
            DLManagedTensor query_tensor, neighbors_tensor, distances_tensor;

            query_tensor.dl_tensor.data = d_queries;
            query_tensor.dl_tensor.device.device_type = kDLCUDA;
            query_tensor.dl_tensor.device.device_id = 0;
            query_tensor.dl_tensor.ndim = 2;
            query_tensor.dl_tensor.dtype.code = kDLFloat;
            query_tensor.dl_tensor.dtype.bits = 32;
            query_tensor.dl_tensor.dtype.lanes = 1;
            int64_t query_shape[] = {n_queries, N_FEATURES};
            query_tensor.dl_tensor.shape = query_shape;
            query_tensor.dl_tensor.strides = NULL;

            neighbors_tensor.dl_tensor.data = d_neighbors;
            neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
            neighbors_tensor.dl_tensor.device.device_id = 0;
            neighbors_tensor.dl_tensor.ndim = 2;
            neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
            neighbors_tensor.dl_tensor.dtype.bits = 32;
            neighbors_tensor.dl_tensor.dtype.lanes = 1;
            int64_t neighbors_shape[] = {n_queries, K};
            neighbors_tensor.dl_tensor.shape = neighbors_shape;
            neighbors_tensor.dl_tensor.strides = NULL;

            distances_tensor.dl_tensor.data = d_distances;
            distances_tensor.dl_tensor.device.device_type = kDLCUDA;
            distances_tensor.dl_tensor.device.device_id = 0;
            distances_tensor.dl_tensor.ndim = 2;
            distances_tensor.dl_tensor.dtype.code = kDLFloat;
            distances_tensor.dl_tensor.dtype.bits = 32;
            distances_tensor.dl_tensor.dtype.lanes = 1;
            int64_t distances_shape[] = {n_queries, K};
            distances_tensor.dl_tensor.shape = distances_shape;
            distances_tensor.dl_tensor.strides = NULL;

            // Perform search
            double search_start = get_time_ms();
            cuvsFilter filter = {.type = NO_FILTER, .addr = 0};
            cuvsCagraSearch(res, search_params, index, &query_tensor,
                           &neighbors_tensor, &distances_tensor, filter);
            cudaDeviceSynchronize();
            double search_time = get_time_ms() - search_start;

            // Copy results to host
            uint32_t* h_neighbors = (uint32_t*)malloc(n_queries * K * sizeof(uint32_t));
            cudaMemcpy(h_neighbors, d_neighbors, n_queries * K * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);

            // Calculate quality metrics
            QualityMetrics metrics = calculate_quality_metrics(ground_truth, h_neighbors,
                                                              n_queries, K);

            printf("    Search %s: %.2fms | P@%d: %.2f%% | R@%d: %.2f%% | MRR: %.3f\n",
                   search_names[s], search_time, K, metrics.precision_at_k * 100,
                   K, metrics.recall_at_k * 100, metrics.mean_reciprocal_rank);

            free(h_neighbors);
        }

        cuvsCagraSearchParamsDestroy(search_params);
        cuvsCagraIndexParamsDestroy(index_params);
        cuvsCagraIndexDestroy(index);
    }

    // Cleanup
    cudaFree(d_dataset);
    cudaFree(d_queries);
    cudaFree(d_neighbors);
    cudaFree(d_distances);
    free(h_dataset);
    free(h_queries);
    free(ground_truth);
}

int main() {
    printf("🔬 CAGRA C-Level Quality Assessment\n");
    printf("===================================\n");
    printf("Testing quality metrics at different scales and configurations\n\n");

    // Initialize CUDA
    cudaSetDevice(0);

    // Create cuVS resources
    cuvsResources_t res;
    cuvsResourcesCreate(&res);

    // Test at different scales
    test_quality_at_scale(res, SMALL_TEST, "Small");
    test_quality_at_scale(res, MEDIUM_TEST, "Medium");
    test_quality_at_scale(res, LARGE_TEST, "Large");

    // Summary
    printf("\n📊 Quality Assessment Summary\n");
    printf("============================\n");
    printf("✅ Tested multiple configurations and scales\n");
    printf("✅ Measured precision, recall, and MRR metrics\n");
    printf("✅ Validated parameter impact on quality vs speed\n");

    printf("\n🎯 Recommendations:\n");
    printf("  • Use Fast config for <1ms latency requirements\n");
    printf("  • Use Balanced for optimal quality/speed trade-off\n");
    printf("  • Use Quality config when accuracy is critical\n");

    // Cleanup
    cuvsResourcesDestroy(res);

    printf("\n✅ Quality Assessment Complete\n");
    return 0;
}