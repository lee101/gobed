#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/brute_force.h>
#include <dlpack/dlpack.h>

#define DIM 384
#define N_VECTORS 100
#define N_QUERIES 10
#define K 10

float normalize_to_int8_range(float val) {
    return fmaxf(-1.0f, fminf(1.0f, val));
}

void generate_distinct_embeddings(int8_t* embeddings, int n_vectors, int dim) {
    srand(42);
    for (int i = 0; i < n_vectors; i++) {
        // Generate distinct patterns for each vector
        for (int d = 0; d < dim; d++) {
            float val = sin(i * 0.1f + d * 0.05f) + cos(i * 0.2f - d * 0.03f);
            val = normalize_to_int8_range(val);
            embeddings[i * dim + d] = (int8_t)(val * 127.0f);
        }
    }
}

void generate_query_embeddings(int8_t* queries, int n_queries, int dim) {
    srand(123);
    const char* query_themes[] = {
        "peace", "relaxing", "war", "stress", "happy",
        "sad", "technical", "artistic", "fast", "slow"
    };

    for (int q = 0; q < n_queries; q++) {
        printf("Query %d theme: %s\n", q, query_themes[q]);
        for (int d = 0; d < dim; d++) {
            // Different pattern for each query theme
            float val = sin(q * 0.3f + d * 0.07f) * cos(q * 0.5f - d * 0.04f);
            val = normalize_to_int8_range(val);
            queries[q * dim + d] = (int8_t)(val * 127.0f);
        }
    }
}

void print_results(const char* test_name, uint32_t* neighbors, float* distances,
                   int n_queries, int k) {
    printf("\n=== %s ===\n", test_name);

    // Check for duplicate issue: are all queries returning same results?
    int duplicate_issue = 1;
    for (int q = 1; q < n_queries; q++) {
        for (int i = 0; i < k; i++) {
            if (neighbors[q * k + i] != neighbors[i]) {
                duplicate_issue = 0;
                break;
            }
        }
        if (!duplicate_issue) break;
    }

    if (duplicate_issue) {
        printf("❌ CRITICAL BUG: All queries return identical results!\n");
    } else {
        printf("✓ Different queries return different results\n");
    }

    // Show first 3 queries' results
    for (int q = 0; q < 3 && q < n_queries; q++) {
        printf("\nQuery %d top-5 results: ", q);
        for (int i = 0; i < 5 && i < k; i++) {
            printf("%u(%.3f) ", neighbors[q * k + i], distances[q * k + i]);
        }
        printf("\n");
    }

    // Check diversity within each query's results
    for (int q = 0; q < n_queries; q++) {
        int unique_count = 0;
        int seen[N_VECTORS] = {0};
        for (int i = 0; i < k; i++) {
            uint32_t idx = neighbors[q * k + i];
            if (idx < N_VECTORS && !seen[idx]) {
                seen[idx] = 1;
                unique_count++;
            }
        }
        if (unique_count < k) {
            printf("Query %d: Only %d unique results out of %d (duplicates!)\n",
                   q, unique_count, k);
        }
    }
}

int test_cagra_search() {
    printf("Testing CAGRA for duplicate bug...\n");

    // Allocate host memory
    int8_t* h_dataset = (int8_t*)malloc(N_VECTORS * DIM * sizeof(int8_t));
    int8_t* h_queries = (int8_t*)malloc(N_QUERIES * DIM * sizeof(int8_t));
    uint32_t* h_neighbors = (uint32_t*)malloc(N_QUERIES * K * sizeof(uint32_t));
    float* h_distances = (float*)malloc(N_QUERIES * K * sizeof(float));

    // Generate distinct test data
    generate_distinct_embeddings(h_dataset, N_VECTORS, DIM);
    generate_query_embeddings(h_queries, N_QUERIES, DIM);

    // Allocate device memory
    int8_t *d_dataset, *d_queries;
    uint32_t *d_neighbors;
    float *d_distances;

    cudaMalloc((void**)&d_dataset, N_VECTORS * DIM * sizeof(int8_t));
    cudaMalloc((void**)&d_queries, N_QUERIES * DIM * sizeof(int8_t));
    cudaMalloc((void**)&d_neighbors, N_QUERIES * K * sizeof(uint32_t));
    cudaMalloc((void**)&d_distances, N_QUERIES * K * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_dataset, h_dataset, N_VECTORS * DIM * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, N_QUERIES * DIM * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Create cuvsResources
    cuvsResources_t res;
    cuvsError_t status = cuvsResourcesCreate(&res);
    if (status != CUVS_SUCCESS) {
        printf("Failed to create resources: %d\n", status);
        return 1;
    }

    // Create DLManagedTensors
    DLManagedTensor dataset_tensor = {
        .dl_tensor = {
            .data = d_dataset,
            .device = {.device_type = kDLCUDA, .device_id = 0},
            .ndim = 2,
            .dtype = {.code = kDLInt, .bits = 8, .lanes = 1},
            .shape = (int64_t[]){N_VECTORS, DIM},
            .strides = NULL,
            .byte_offset = 0
        },
        .manager_ctx = NULL,
        .deleter = NULL
    };

    DLManagedTensor queries_tensor = {
        .dl_tensor = {
            .data = d_queries,
            .device = {.device_type = kDLCUDA, .device_id = 0},
            .ndim = 2,
            .dtype = {.code = kDLInt, .bits = 8, .lanes = 1},
            .shape = (int64_t[]){N_QUERIES, DIM},
            .strides = NULL,
            .byte_offset = 0
        },
        .manager_ctx = NULL,
        .deleter = NULL
    };

    DLManagedTensor neighbors_tensor = {
        .dl_tensor = {
            .data = d_neighbors,
            .device = {.device_type = kDLCUDA, .device_id = 0},
            .ndim = 2,
            .dtype = {.code = kDLUInt, .bits = 32, .lanes = 1},
            .shape = (int64_t[]){N_QUERIES, K},
            .strides = NULL,
            .byte_offset = 0
        },
        .manager_ctx = NULL,
        .deleter = NULL
    };

    DLManagedTensor distances_tensor = {
        .dl_tensor = {
            .data = d_distances,
            .device = {.device_type = kDLCUDA, .device_id = 0},
            .ndim = 2,
            .dtype = {.code = kDLFloat, .bits = 32, .lanes = 1},
            .shape = (int64_t[]){N_QUERIES, K},
            .strides = NULL,
            .byte_offset = 0
        },
        .manager_ctx = NULL,
        .deleter = NULL
    };

    // Test different search parameters
    struct {
        const char* name;
        uint32_t itopk_size;
        uint32_t search_width;
        uint32_t min_iterations;
        uint32_t max_iterations;
        uint32_t algo;
        uint32_t team_size;
    } test_configs[] = {
        {"Default", 64, 1, 0, 0, 0, 0},
        {"High Quality", 256, 3, 8, 16, 0, 0},
        {"Fast", 32, 1, 0, 0, 1, 8},
        {"Balanced", 128, 2, 4, 8, 0, 0}
    };

    for (int cfg = 0; cfg < 4; cfg++) {
        printf("\n--- Testing config: %s ---\n", test_configs[cfg].name);

        // Create and build index
        cuvsCagraIndex_t index;
        cuvsCagraIndexParams_t build_params;
        status = cuvsCagraIndexParamsCreate(&build_params);

        status = cuvsCagraBuild(res, build_params, &dataset_tensor, index);
        if (status != CUVS_SUCCESS) {
            printf("Failed to build index: %d\n", status);
            continue;
        }

        // Create search params
        cuvsCagraSearchParams_t search_params;
        status = cuvsCagraSearchParamsCreate(&search_params);

        // Set search parameters
        search_params->itopk_size = test_configs[cfg].itopk_size;
        search_params->search_width = test_configs[cfg].search_width;
        search_params->min_iterations = test_configs[cfg].min_iterations;
        search_params->max_iterations = test_configs[cfg].max_iterations;
        search_params->algo = test_configs[cfg].algo;
        search_params->team_size = test_configs[cfg].team_size;

        // Perform search
        cuvsFilter filter = {0, NO_FILTER};
        status = cuvsCagraSearch(res, search_params, index,
                                &queries_tensor, &neighbors_tensor, &distances_tensor, filter);

        if (status != CUVS_SUCCESS) {
            printf("Search failed: %d\n", status);
        } else {
            // Copy results back to host
            cudaMemcpy(h_neighbors, d_neighbors, N_QUERIES * K * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
            cudaMemcpy(h_distances, d_distances, N_QUERIES * K * sizeof(float),
                      cudaMemcpyDeviceToHost);

            // Analyze results
            print_results(test_configs[cfg].name, h_neighbors, h_distances, N_QUERIES, K);
        }

        // Cleanup
        cuvsCagraIndexDestroy(index);
        cuvsCagraIndexParamsDestroy(build_params);
        cuvsCagraSearchParamsDestroy(search_params);
    }

    // Test brute force for comparison
    printf("\n--- Testing Brute Force (ground truth) ---\n");

    // Build brute force index
    cuvsBruteForceIndex_t bf_index;
    status = cuvsBruteForceIndexCreate(&bf_index);
    status = cuvsBruteForceBuild(res, &dataset_tensor, L2Expanded, 0.0f, bf_index);

    // Reset results
    cudaMemset(d_neighbors, 0, N_QUERIES * K * sizeof(uint32_t));
    cudaMemset(d_distances, 0, N_QUERIES * K * sizeof(float));

    cuvsFilter bf_filter = {0, NO_FILTER};
    status = cuvsBruteForceSearch(res, bf_index, &queries_tensor,
                                  &neighbors_tensor, &distances_tensor, bf_filter);

    if (status == CUVS_SUCCESS) {
        cudaMemcpy(h_neighbors, d_neighbors, N_QUERIES * K * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(h_distances, d_distances, N_QUERIES * K * sizeof(float),
                  cudaMemcpyDeviceToHost);
        print_results("Brute Force", h_neighbors, h_distances, N_QUERIES, K);
    }

    // Cleanup brute force index
    cuvsBruteForceIndexDestroy(bf_index);

    // Cleanup
    cuvsResourcesDestroy(res);
    cudaFree(d_dataset);
    cudaFree(d_queries);
    cudaFree(d_neighbors);
    cudaFree(d_distances);
    free(h_dataset);
    free(h_queries);
    free(h_neighbors);
    free(h_distances);

    return 0;
}

int main() {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);

    return test_cagra_search();
}