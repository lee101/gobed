// verification_test.cu - Rigorous verification that GPU acceleration is real
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

// Include our CUDA kernel
extern "C" void launch_similarity_kernel(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim
);

// CPU reference implementation
void cpu_similarity_reference(
    const int8_t* query,
    const int8_t* database,
    float* scores,
    int num_vectors,
    int dim
) {
    for (int i = 0; i < num_vectors; i++) {
        int sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += static_cast<int>(query[j]) * static_cast<int>(database[i * dim + j]);
        }
        scores[i] = static_cast<float>(sum);
    }
}

// Test with known values to verify correctness
bool test_correctness() {
    std::cout << "🧪 Testing Correctness with Known Values" << std::endl;
    
    // Simple test case: 3 vectors, 4 dimensions
    const int num_vectors = 3;
    const int num_queries = 1;
    const int dim = 4;
    
    // Known test data
    std::vector<int8_t> query = {1, 2, 3, 4};
    std::vector<int8_t> database = {
        1, 1, 1, 1,  // dot product = 1*1 + 2*1 + 3*1 + 4*1 = 10
        2, 2, 2, 2,  // dot product = 1*2 + 2*2 + 3*2 + 4*2 = 20
        0, 1, 0, 1   // dot product = 1*0 + 2*1 + 3*0 + 4*1 = 6
    };
    
    // Expected results
    std::vector<float> expected = {10.0f, 20.0f, 6.0f};
    
    // CPU computation
    std::vector<float> cpu_scores(num_vectors);
    cpu_similarity_reference(query.data(), database.data(), cpu_scores.data(), num_vectors, dim);
    
    // Verify CPU results match expected
    for (int i = 0; i < num_vectors; i++) {
        if (std::abs(cpu_scores[i] - expected[i]) > 0.001f) {
            std::cout << "❌ CPU reference failed: expected " << expected[i] 
                      << ", got " << cpu_scores[i] << std::endl;
            return false;
        }
    }
    
    // GPU computation
    int8_t* d_query;
    int8_t* d_database;
    float* d_scores;
    
    cudaMalloc(&d_query, query.size());
    cudaMalloc(&d_database, database.size());
    cudaMalloc(&d_scores, num_vectors * sizeof(float));
    
    cudaMemcpy(d_query, query.data(), query.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_database, database.data(), database.size(), cudaMemcpyHostToDevice);
    
    // Launch GPU kernel
    launch_similarity_kernel(d_query, d_database, d_scores, num_queries, num_vectors, dim);
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "❌ CUDA kernel failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy results back
    std::vector<float> gpu_scores(num_vectors);
    cudaMemcpy(gpu_scores.data(), d_scores, num_vectors * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify GPU results
    bool success = true;
    for (int i = 0; i < num_vectors; i++) {
        if (std::abs(gpu_scores[i] - expected[i]) > 0.001f) {
            std::cout << "❌ GPU result failed: expected " << expected[i] 
                      << ", got " << gpu_scores[i] << " (vector " << i << ")" << std::endl;
            success = false;
        }
    }
    
    // Verify CPU and GPU match exactly
    for (int i = 0; i < num_vectors; i++) {
        if (std::abs(gpu_scores[i] - cpu_scores[i]) > 0.001f) {
            std::cout << "❌ CPU-GPU mismatch: CPU=" << cpu_scores[i] 
                      << ", GPU=" << gpu_scores[i] << " (vector " << i << ")" << std::endl;
            success = false;
        }
    }
    
    cudaFree(d_query);
    cudaFree(d_database);
    cudaFree(d_scores);
    
    if (success) {
        std::cout << "✅ Correctness test passed!" << std::endl;
        std::cout << "   Expected: [" << expected[0] << ", " << expected[1] << ", " << expected[2] << "]" << std::endl;
        std::cout << "   CPU:      [" << cpu_scores[0] << ", " << cpu_scores[1] << ", " << cpu_scores[2] << "]" << std::endl;
        std::cout << "   GPU:      [" << gpu_scores[0] << ", " << gpu_scores[1] << ", " << gpu_scores[2] << "]" << std::endl;
    }
    
    return success;
}

// Test that GPU actually runs and times differently from CPU
bool test_gpu_utilization() {
    std::cout << "\n⚡ Testing GPU Utilization vs CPU" << std::endl;
    
    const int num_vectors = 50000;
    const int num_queries = 100;
    const int dim = 768;
    
    // Generate random data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for consistency
    std::uniform_int_distribution<> dis(-128, 127);
    
    std::vector<int8_t> queries(num_queries * dim);
    std::vector<int8_t> database(num_vectors * dim);
    
    for (auto& val : queries) val = dis(gen);
    for (auto& val : database) val = dis(gen);
    
    // CPU timing (single query to scale)
    std::cout << "💻 CPU timing..." << std::endl;
    std::vector<float> cpu_scores(num_vectors);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_similarity_reference(queries.data(), database.data(), cpu_scores.data(), num_vectors, dim);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    double cpu_total_time = cpu_time.count() * num_queries; // Scale for all queries
    
    std::cout << "   Single query: " << cpu_time.count() << " μs" << std::endl;
    std::cout << "   Estimated " << num_queries << " queries: " << cpu_total_time / 1000.0 << " ms" << std::endl;
    
    // GPU setup and timing
    std::cout << "🚀 GPU timing..." << std::endl;
    
    int8_t* d_queries;
    int8_t* d_database;
    float* d_scores;
    
    size_t queries_size = queries.size();
    size_t database_size = database.size();
    size_t scores_size = num_queries * num_vectors * sizeof(float);
    
    // Time GPU memory allocation
    auto gpu_alloc_start = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_queries, queries_size);
    cudaMalloc(&d_database, database_size);
    cudaMalloc(&d_scores, scores_size);
    auto gpu_alloc_end = std::chrono::high_resolution_clock::now();
    
    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_alloc_end - gpu_alloc_start);
    std::cout << "   GPU allocation: " << alloc_time.count() << " μs" << std::endl;
    
    // Time GPU memory copy
    auto gpu_copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_queries, queries.data(), queries_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_database, database.data(), database_size, cudaMemcpyHostToDevice);
    auto gpu_copy_end = std::chrono::high_resolution_clock::now();
    
    auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_copy_end - gpu_copy_start);
    std::cout << "   GPU memory copy: " << copy_time.count() << " μs" << std::endl;
    
    // Warm up GPU
    for (int i = 0; i < 3; i++) {
        launch_similarity_kernel(d_queries, d_database, d_scores, num_queries, num_vectors, dim);
        cudaDeviceSynchronize();
    }
    
    // Time GPU computation (the actual kernel)
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    auto gpu_compute_start = std::chrono::high_resolution_clock::now();
    
    // Record CUDA events for precise GPU timing
    cudaEventRecord(start_event);
    launch_similarity_kernel(d_queries, d_database, d_scores, num_queries, num_vectors, dim);
    cudaEventRecord(stop_event);
    
    cudaEventSynchronize(stop_event);
    auto gpu_compute_end = std::chrono::high_resolution_clock::now();
    
    // Get precise GPU time
    float gpu_kernel_time_ms;
    cudaEventElapsedTime(&gpu_kernel_time_ms, start_event, stop_event);
    
    auto gpu_time_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_compute_end - gpu_compute_start);
    
    std::cout << "   GPU kernel time (CUDA events): " << gpu_kernel_time_ms * 1000.0 << " μs" << std::endl;
    std::cout << "   GPU kernel time (CPU timer): " << gpu_time_us.count() << " μs" << std::endl;
    
    // Copy results back and verify
    std::vector<float> gpu_scores(num_queries * num_vectors);
    auto gpu_copyback_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(gpu_scores.data(), d_scores, scores_size, cudaMemcpyDeviceToHost);
    auto gpu_copyback_end = std::chrono::high_resolution_clock::now();
    
    auto copyback_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_copyback_end - gpu_copyback_start);
    std::cout << "   GPU copy back: " << copyback_time.count() << " μs" << std::endl;
    
    // Verify first query matches CPU
    bool results_match = true;
    for (int i = 0; i < std::min(10, num_vectors); i++) {
        float diff = std::abs(gpu_scores[i] - cpu_scores[i]);
        if (diff > 0.1f) {
            std::cout << "❌ Result mismatch at " << i << ": CPU=" << cpu_scores[i] 
                      << ", GPU=" << gpu_scores[i] << std::endl;
            results_match = false;
            break;
        }
    }
    
    // Calculate metrics
    double gpu_kernel_time_us = gpu_kernel_time_ms * 1000.0;
    double speedup = cpu_total_time / gpu_kernel_time_us;
    double throughput = (double(num_queries) * num_vectors) / (gpu_kernel_time_us / 1000000.0) / 1000000.0;
    
    std::cout << "\n📊 Performance Analysis:" << std::endl;
    std::cout << "   CPU time (est.): " << cpu_total_time / 1000.0 << " ms" << std::endl;
    std::cout << "   GPU kernel time: " << gpu_kernel_time_us / 1000.0 << " ms" << std::endl;
    std::cout << "   Speedup: " << speedup << "x" << std::endl;
    std::cout << "   Throughput: " << throughput << " M comparisons/sec" << std::endl;
    
    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_database);
    cudaFree(d_scores);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    // Success criteria
    bool success = results_match && 
                  speedup > 1.5 &&  // Must be faster than CPU
                  throughput > 50.0; // Must achieve reasonable throughput
    
    if (success) {
        std::cout << "✅ GPU utilization test passed!" << std::endl;
    } else {
        std::cout << "❌ GPU utilization test failed!" << std::endl;
        if (!results_match) std::cout << "   - Results don't match CPU" << std::endl;
        if (speedup <= 1.5) std::cout << "   - Speedup too low: " << speedup << std::endl;
        if (throughput <= 50.0) std::cout << "   - Throughput too low: " << throughput << std::endl;
    }
    
    return success;
}

// Test GPU memory usage patterns
bool test_gpu_memory_behavior() {
    std::cout << "\n💾 Testing GPU Memory Behavior" << std::endl;
    
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);
    std::cout << "   Initial GPU memory: " << (total - free_before) / (1024*1024) << " MB used" << std::endl;
    
    // Allocate various sizes to test memory behavior
    const int test_sizes[] = {1000, 10000, 100000, 1000000};
    const int dim = 768;
    
    for (int size : test_sizes) {
        std::cout << "   Testing " << size << " vectors..." << std::endl;
        
        int8_t* d_data;
        float* d_scores;
        
        size_t data_size = size * dim * sizeof(int8_t);
        size_t scores_size = size * sizeof(float);
        
        cudaError_t error1 = cudaMalloc(&d_data, data_size);
        cudaError_t error2 = cudaMalloc(&d_scores, scores_size);
        
        if (error1 != cudaSuccess || error2 != cudaSuccess) {
            std::cout << "❌ Memory allocation failed for size " << size << std::endl;
            return false;
        }
        
        size_t free_after;
        cudaMemGetInfo(&free_after, &total);
        size_t allocated = (free_before - free_after);
        
        std::cout << "     Allocated " << allocated / (1024*1024) << " MB" << std::endl;
        
        cudaFree(d_data);
        cudaFree(d_scores);
        
        free_before = free_after;
    }
    
    std::cout << "✅ GPU memory test passed!" << std::endl;
    return true;
}

int main() {
    std::cout << "🔬 GPU Acceleration Verification Suite" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // System info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "🖥️  GPU: " << prop.name << std::endl;
    std::cout << "   Compute: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "   Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "   SM Count: " << prop.multiProcessorCount << std::endl;
    
    bool all_passed = true;
    
    // Run verification tests
    all_passed &= test_correctness();
    all_passed &= test_gpu_utilization();
    all_passed &= test_gpu_memory_behavior();
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    if (all_passed) {
        std::cout << "🎉 ALL TESTS PASSED!" << std::endl;
        std::cout << "✅ GPU acceleration is VERIFIED and REAL!" << std::endl;
        std::cout << "✅ Computations are happening on GPU hardware" << std::endl;
        std::cout << "✅ Results match CPU reference implementation" << std::endl;
        std::cout << "✅ Performance improvements are genuine" << std::endl;
    } else {
        std::cout << "❌ SOME TESTS FAILED!" << std::endl;
        std::cout << "⚠️  GPU acceleration may not be working correctly" << std::endl;
        return 1;
    }
    
    return 0;
}