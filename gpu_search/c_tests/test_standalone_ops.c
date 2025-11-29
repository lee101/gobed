// test_standalone_ops.c - Test the LibTorch-free CUDA operations
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "cuda_ops/cuda_ops.h"

void print_result(const char* test_name, cuda_op_result_t result) {
    printf("[%s] %s\n",
           result == CUDA_OP_SUCCESS ? "PASS" : "FAIL",
           test_name);
    if (result != CUDA_OP_SUCCESS) {
        printf("  Error code: %d\n", result);
    }
}

int main() {
    printf("=== LibTorch-Free CUDA Operations Test ===\n\n");

    // Test CUDA device detection
    int device_count;
    cuda_op_result_t result = cuda_get_device_count(&device_count);
    print_result("CUDA device detection", result);
    if (result == CUDA_OP_SUCCESS) {
        printf("  Found %d CUDA device(s)\n", device_count);
    }

    // Test CUDA capabilities
    int compute_major, compute_minor, cuda_version;
    result = check_cuda_capabilities(&compute_major, &compute_minor, &cuda_version);
    print_result("CUDA capability check", result);
    if (result == CUDA_OP_SUCCESS) {
        printf("  Compute capability: %d.%d\n", compute_major, compute_minor);
        printf("  CUDA runtime version: %d\n", cuda_version);
    }

    // Test memory allocation
    void* gpu_ptr = NULL;
    size_t test_size = 1024 * sizeof(float);
    result = cuda_malloc(&gpu_ptr, test_size);
    print_result("CUDA memory allocation", result);

    if (result == CUDA_OP_SUCCESS) {
        // Test memory operations
        float* host_data = (float*)malloc(test_size);
        for (int i = 0; i < 1024; i++) {
            host_data[i] = (float)i;
        }

        result = cuda_memcpy_h2d(gpu_ptr, host_data, test_size);
        print_result("Host to device memcpy", result);

        float* result_data = (float*)malloc(test_size);
        result = cuda_memcpy_d2h(result_data, gpu_ptr, test_size);
        print_result("Device to host memcpy", result);

        // Verify data
        int errors = 0;
        for (int i = 0; i < 1024; i++) {
            if (result_data[i] != host_data[i]) {
                errors++;
            }
        }
        printf("  Memory copy verification: %d errors\n", errors);

        free(host_data);
        free(result_data);
        cuda_free(gpu_ptr);
    }

    // Test INT8 dot product operation
    printf("\n=== INT8 Dot Product Test ===\n");

    const int N = 1024;
    const int D = 512;

    // Allocate host memory
    int8_t* query_host = (int8_t*)malloc(D * sizeof(int8_t));
    int8_t* db_host = (int8_t*)malloc(N * D * sizeof(int8_t));
    int32_t* result_host = (int32_t*)malloc(N * sizeof(int32_t));

    // Initialize test data
    srand((unsigned int)time(NULL));
    for (int i = 0; i < D; i++) {
        query_host[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < N * D; i++) {
        db_host[i] = (int8_t)(rand() % 256 - 128);
    }

    // Allocate device memory
    int8_t* query_gpu;
    int8_t* db_gpu;
    int32_t* result_gpu;

    result = cuda_malloc((void**)&query_gpu, D * sizeof(int8_t));
    if (result != CUDA_OP_SUCCESS) {
        printf("Failed to allocate query GPU memory\n");
        goto cleanup;
    }

    result = cuda_malloc((void**)&db_gpu, N * D * sizeof(int8_t));
    if (result != CUDA_OP_SUCCESS) {
        printf("Failed to allocate database GPU memory\n");
        goto cleanup;
    }

    result = cuda_malloc((void**)&result_gpu, N * sizeof(int32_t));
    if (result != CUDA_OP_SUCCESS) {
        printf("Failed to allocate result GPU memory\n");
        goto cleanup;
    }

    // Copy data to device
    result = cuda_memcpy_h2d(query_gpu, query_host, D * sizeof(int8_t));
    if (result != CUDA_OP_SUCCESS) {
        printf("Failed to copy query to device\n");
        goto cleanup;
    }

    result = cuda_memcpy_h2d(db_gpu, db_host, N * D * sizeof(int8_t));
    if (result != CUDA_OP_SUCCESS) {
        printf("Failed to copy database to device\n");
        goto cleanup;
    }

    // Test single query operation
    clock_t start = clock();
    result = i8dot512_scores(query_gpu, db_gpu, result_gpu, N);
    cuda_synchronize();
    clock_t end = clock();

    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    print_result("INT8 dot product (GPU)", result);
    printf("  GPU time: %.2f ms\n", gpu_time);

    if (result == CUDA_OP_SUCCESS) {
        // Copy results back
        result = cuda_memcpy_d2h(result_host, result_gpu, N * sizeof(int32_t));
        if (result == CUDA_OP_SUCCESS) {
            // Verify with CPU computation
            start = clock();
            int32_t cpu_result = 0;
            for (int i = 0; i < D; i++) {
                cpu_result += (int32_t)query_host[i] * (int32_t)db_host[i];  // First vector
            }
            end = clock();

            double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
            printf("  CPU time (single vector): %.2f ms\n", cpu_time);
            printf("  GPU result[0]: %d, CPU result: %d\n", result_host[0], cpu_result);
            printf("  Match: %s\n", result_host[0] == cpu_result ? "YES" : "NO");
        }
    }

    // Test batch operation
    printf("\n=== Batch INT8 Dot Product Test ===\n");
    const int B = 8;
    int8_t* queries_host = (int8_t*)malloc(B * D * sizeof(int8_t));
    int32_t* batch_result_host = (int32_t*)malloc(B * N * sizeof(int32_t));

    // Initialize batch queries
    for (int i = 0; i < B * D; i++) {
        queries_host[i] = (int8_t)(rand() % 256 - 128);
    }

    int8_t* queries_gpu;
    int32_t* batch_result_gpu;

    result = cuda_malloc((void**)&queries_gpu, B * D * sizeof(int8_t));
    if (result != CUDA_OP_SUCCESS) {
        printf("Failed to allocate batch queries GPU memory\n");
        goto cleanup;
    }

    result = cuda_malloc((void**)&batch_result_gpu, B * N * sizeof(int32_t));
    if (result != CUDA_OP_SUCCESS) {
        printf("Failed to allocate batch result GPU memory\n");
        goto cleanup;
    }

    result = cuda_memcpy_h2d(queries_gpu, queries_host, B * D * sizeof(int8_t));
    if (result != CUDA_OP_SUCCESS) {
        printf("Failed to copy batch queries to device\n");
        goto cleanup;
    }

    start = clock();
    result = i8dot512_batch(queries_gpu, db_gpu, batch_result_gpu, B, N);
    cuda_synchronize();
    end = clock();

    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    print_result("Batch INT8 dot product (GPU)", result);
    printf("  GPU time (%d queries): %.2f ms\n", B, gpu_time);
    printf("  Throughput: %.2f queries/ms\n", (double)B / gpu_time);

    if (result == CUDA_OP_SUCCESS) {
        result = cuda_memcpy_d2h(batch_result_host, batch_result_gpu, B * N * sizeof(int32_t));
        if (result == CUDA_OP_SUCCESS) {
            printf("  Batch results sample: %d, %d, %d\n",
                   batch_result_host[0], batch_result_host[1], batch_result_host[2]);
        }
    }

    printf("\n=== Performance Summary ===\n");
    if (gpu_time > 0) {
        printf("Single query throughput: %.2f vectors/ms\n", (double)N / gpu_time);
        printf("Batch throughput: %.2f query-vectors/ms\n", (double)(B * N) / gpu_time);
    }

cleanup:
    // Cleanup
    if (query_host) free(query_host);
    if (db_host) free(db_host);
    if (result_host) free(result_host);
    if (queries_host) free(queries_host);
    if (batch_result_host) free(batch_result_host);

    if (query_gpu) cuda_free(query_gpu);
    if (db_gpu) cuda_free(db_gpu);
    if (result_gpu) cuda_free(result_gpu);
    if (queries_gpu) cuda_free(queries_gpu);
    if (batch_result_gpu) cuda_free(batch_result_gpu);

    printf("\n=== Test Complete ===\n");
    return 0;
}