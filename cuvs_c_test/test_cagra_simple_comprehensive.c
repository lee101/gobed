#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>

int main() {
    printf("🚀 CAGRA Simple Comprehensive Test\n");
    printf("===================================\n\n");

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
    int n_vectors = 1000;
    int n_queries = 10;
    int dim = 128;
    int k = 10;

    printf("📝 Test Configuration:\n");
    printf("  Dataset: %d vectors, %d dimensions\n", n_vectors, dim);
    printf("  Queries: %d queries, k=%d\n\n", n_queries, k);

    // Generate simple float data
    float* dataset = (float*)malloc(n_vectors * dim * sizeof(float));
    float* queries = (float*)malloc(n_queries * dim * sizeof(float));

    srand(42);
    for (int i = 0; i < n_vectors * dim; i++) {
        dataset[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < n_queries * dim; i++) {
        queries[i] = (float)(rand() % 100) / 100.0f;
    }

    // Create resources
    printf("Creating resources...\n");
    cuvsResources_t res;
    cuvsError_t res_status = cuvsResourcesCreate(&res);
    if (res_status != 1) {
        printf("❌ Failed to create resources: %d\n", res_status);
        return 1;
    }

    // Setup dataset tensor
    DLManagedTensor dataset_tensor;
    dataset_tensor.dl_tensor.data = dataset;
    dataset_tensor.dl_tensor.device.device_type = kDLCPU;
    dataset_tensor.dl_tensor.ndim = 2;
    int64_t dataset_shape[2] = {n_vectors, dim};
    dataset_tensor.dl_tensor.shape = dataset_shape;
    dataset_tensor.dl_tensor.dtype.code = kDLFloat;
    dataset_tensor.dl_tensor.dtype.bits = 32;
    dataset_tensor.dl_tensor.strides = NULL;

    // Create index params
    printf("Creating index params...\n");
    cuvsCagraIndexParams_t index_params;
    cuvsError_t params_status = cuvsCagraIndexParamsCreate(&index_params);
    if (params_status != 1) {
        printf("❌ Failed to create index params: %d\n", params_status);
        cuvsResourcesDestroy(res);
        return 1;
    }

    // Set basic parameters
    index_params->graph_degree = 32;
    index_params->intermediate_graph_degree = 64;
    index_params->build_algo = NN_DESCENT;

    // Create index
    printf("Creating index...\n");
    cuvsCagraIndex_t index;
    cuvsError_t index_status = cuvsCagraIndexCreate(&index);
    if (index_status != 1) {
        printf("❌ Failed to create index: %d\n", index_status);
        cuvsCagraIndexParamsDestroy(index_params);
        cuvsResourcesDestroy(res);
        return 1;
    }

    // Build index
    printf("Building index...\n");
    cuvsError_t build_status = cuvsCagraBuild(res, index_params, &dataset_tensor, index);
    if (build_status != 1) {  // 0 is success
        printf("❌ Failed to build index: %d\n", build_status);
        cuvsCagraIndexDestroy(index);
        cuvsCagraIndexParamsDestroy(index_params);
        cuvsResourcesDestroy(res);
        return 1;
    }
    printf("✅ Index built successfully!\n");

    // Allocate GPU memory for queries and results
    float *d_queries;
    int64_t *d_neighbors;
    float *d_distances;

    cudaMalloc((void**)&d_queries, n_queries * dim * sizeof(float));
    cudaMalloc((void**)&d_neighbors, n_queries * k * sizeof(int64_t));
    cudaMalloc((void**)&d_distances, n_queries * k * sizeof(float));

    cudaMemcpy(d_queries, queries, n_queries * dim * sizeof(float), cudaMemcpyHostToDevice);

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

    // Create search params
    printf("Creating search params...\n");
    cuvsCagraSearchParams_t search_params;
    cuvsError_t search_params_status = cuvsCagraSearchParamsCreate(&search_params);
    if (search_params_status != 1) {
        printf("❌ Failed to create search params: %d\n", search_params_status);
        cudaFree(d_queries);
        cudaFree(d_neighbors);
        cudaFree(d_distances);
        cuvsCagraIndexDestroy(index);
        cuvsCagraIndexParamsDestroy(index_params);
        cuvsResourcesDestroy(res);
        return 1;
    }

    search_params->max_queries = n_queries;
    search_params->itopk_size = 64;
    search_params->search_width = 1;
    search_params->min_iterations = 0;
    search_params->max_iterations = 0;
    search_params->algo = AUTO;

    // Test different configurations
    printf("\n🔬 Testing Different Configurations\n");
    printf("=====================================\n");

    struct {
        const char* name;
        int itopk_size;
        int search_width;
        int min_iterations;
    } configs[] = {
        {"Fast", 32, 1, 0},
        {"Balanced", 64, 2, 2},
        {"Quality", 128, 2, 4},
        {"Ultra", 256, 4, 8}
    };

    for (int i = 0; i < 4; i++) {
        printf("\nTesting %s configuration...\n", configs[i].name);
        search_params->itopk_size = configs[i].itopk_size;
        search_params->search_width = configs[i].search_width;
        search_params->min_iterations = configs[i].min_iterations;

        // Perform search
        cuvsFilter filter = {0, NO_FILTER};
        clock_t start = clock();
        cuvsError_t search_status = cuvsCagraSearch(res, search_params, index, &queries_tensor,
                                                    &neighbors_tensor, &distances_tensor, filter);
        cudaDeviceSynchronize();
        clock_t end = clock();

        if (search_status != 1) {
            printf("  ❌ Search failed: %d\n", search_status);
        } else {
            double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
            printf("  ✅ Search time: %.3f ms\n", time_ms);
            printf("  Throughput: %.0f QPS\n", (n_queries * 1000.0) / time_ms);
        }
    }

    // Get index size
    printf("\n📊 Index Memory Usage\n");
    printf("===================\n");
    int64_t index_size;
    cuvsError_t size_status = cuvsCagraIndexGetSize(index, &index_size);
    if (size_status == 1) {
        printf("Index size: %.2f MB\n", index_size / (1024.0 * 1024.0));
    } else {
        printf("❌ Failed to get index size: %d\n", size_status);
    }

    // Cleanup
    printf("\n🧹 Cleaning up...\n");
    cudaFree(d_queries);
    cudaFree(d_neighbors);
    cudaFree(d_distances);
    free(dataset);
    free(queries);
    cuvsCagraSearchParamsDestroy(search_params);
    cuvsCagraIndexDestroy(index);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsResourcesDestroy(res);

    printf("✅ Test complete!\n");
    return 0;
}