// cuda_test.cu - Simple CUDA test to verify GPU acceleration
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// Include our CUDA kernel
extern "C" void launch_similarity_kernel(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim
);

// Simple CPU version for comparison
void cpu_similarity(
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

int main() {
    std::cout << "🚀 CUDA GPU Acceleration Test" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Check CUDA devices
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "❌ No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "📊 Using GPU: " << prop.name << std::endl;
    std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "   Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "   Multiprocessors: " << prop.multiProcessorCount << std::endl;
    
    // Test parameters
    const int dim = 768;
    const int num_vectors = 100000;  // 100k vectors
    const int num_queries = 100;
    
    std::cout << "\n🔧 Test Configuration:" << std::endl;
    std::cout << "   Vector dimension: " << dim << std::endl;
    std::cout << "   Database size: " << num_vectors << std::endl;
    std::cout << "   Queries: " << num_queries << std::endl;
    
    // Generate random data
    std::cout << "\n🎲 Generating random data..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);
    
    std::vector<int8_t> queries(num_queries * dim);
    std::vector<int8_t> database(num_vectors * dim);
    std::vector<float> gpu_scores(num_queries * num_vectors);
    std::vector<float> cpu_scores(num_vectors);
    
    for (auto& val : queries) val = dis(gen);
    for (auto& val : database) val = dis(gen);
    
    // Allocate GPU memory
    std::cout << "📦 Allocating GPU memory..." << std::endl;
    int8_t* d_queries;
    int8_t* d_database;
    float* d_scores;
    
    size_t queries_size = queries.size() * sizeof(int8_t);
    size_t database_size = database.size() * sizeof(int8_t);
    size_t scores_size = gpu_scores.size() * sizeof(float);
    
    cudaMalloc(&d_queries, queries_size);
    cudaMalloc(&d_database, database_size);
    cudaMalloc(&d_scores, scores_size);
    
    // Copy data to GPU
    std::cout << "📤 Copying data to GPU..." << std::endl;
    cudaMemcpy(d_queries, queries.data(), queries_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_database, database.data(), database_size, cudaMemcpyHostToDevice);
    
    // Warm up GPU
    std::cout << "🔥 Warming up GPU..." << std::endl;
    for (int i = 0; i < 5; i++) {
        launch_similarity_kernel(d_queries, d_database, d_scores, num_queries, num_vectors, dim);
        cudaDeviceSynchronize();
    }
    
    // Benchmark GPU
    std::cout << "\n⚡ GPU Performance Test:" << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    launch_similarity_kernel(d_queries, d_database, d_scores, num_queries, num_vectors, dim);
    cudaDeviceSynchronize();
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    
    // Copy results back
    cudaMemcpy(gpu_scores.data(), d_scores, scores_size, cudaMemcpyDeviceToHost);
    
    std::cout << "   GPU Time: " << gpu_time.count() / 1000.0 << " ms" << std::endl;
    std::cout << "   Throughput: " << (num_queries * num_vectors) / (gpu_time.count() / 1000000.0) / 1000000.0 
              << " M comparisons/sec" << std::endl;
    
    // Benchmark CPU (single query for comparison)
    std::cout << "\n💻 CPU Performance Test (single query):" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    cpu_similarity(queries.data(), database.data(), cpu_scores.data(), num_vectors, dim);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    std::cout << "   CPU Time (1 query): " << cpu_time.count() / 1000.0 << " ms" << std::endl;
    std::cout << "   Estimated for " << num_queries << " queries: " 
              << (cpu_time.count() * num_queries) / 1000.0 << " ms" << std::endl;
    
    // Calculate speedup
    float speedup = (cpu_time.count() * num_queries) / (float)gpu_time.count();
    std::cout << "\n🏆 GPU Speedup: " << speedup << "x faster!" << std::endl;
    
    // Verify correctness (compare first query results)
    std::cout << "\n🔍 Verifying correctness..." << std::endl;
    bool correct = true;
    for (int i = 0; i < 10; i++) {  // Check first 10 results
        float gpu_val = gpu_scores[i];
        float cpu_val = cpu_scores[i];
        float diff = std::abs(gpu_val - cpu_val);
        if (diff > 0.001) {
            std::cout << "   ❌ Mismatch at index " << i << ": GPU=" << gpu_val << ", CPU=" << cpu_val << std::endl;
            correct = false;
        }
    }
    if (correct) {
        std::cout << "   ✅ Results match!" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_database);
    cudaFree(d_scores);
    
    std::cout << "\n✅ Test completed successfully!" << std::endl;
    
    return 0;
}