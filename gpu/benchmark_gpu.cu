// benchmark_gpu.cu - Comprehensive GPU performance benchmark
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

extern "C" void launch_similarity_kernel(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim
);

void run_benchmark(int dim, int num_vectors, int num_queries) {
    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);
    
    std::vector<int8_t> queries(num_queries * dim);
    std::vector<int8_t> database(num_vectors * dim);
    std::vector<float> scores(num_queries * num_vectors);
    
    for (auto& val : queries) val = dis(gen);
    for (auto& val : database) val = dis(gen);
    
    // Allocate GPU memory
    int8_t* d_queries;
    int8_t* d_database;
    float* d_scores;
    
    cudaMalloc(&d_queries, queries.size() * sizeof(int8_t));
    cudaMalloc(&d_database, database.size() * sizeof(int8_t));
    cudaMalloc(&d_scores, scores.size() * sizeof(float));
    
    // Copy to GPU
    cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_database, database.data(), database.size() * sizeof(int8_t), cudaMemcpyHostToDevice);
    
    // Warm up
    for (int i = 0; i < 3; i++) {
        launch_similarity_kernel(d_queries, d_database, d_scores, num_queries, num_vectors, dim);
        cudaDeviceSynchronize();
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    launch_similarity_kernel(d_queries, d_database, d_scores, num_queries, num_vectors, dim);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate metrics
    double time_ms = duration.count() / 1000.0;
    double comparisons = (double)num_queries * num_vectors;
    double throughput = comparisons / (duration.count() / 1000000.0) / 1000000.0;  // M comparisons/sec
    double qps = num_queries / (duration.count() / 1000000.0);  // Queries per second
    
    std::cout << std::setw(10) << num_vectors 
              << std::setw(10) << num_queries
              << std::setw(10) << dim
              << std::setw(12) << std::fixed << std::setprecision(2) << time_ms
              << std::setw(15) << std::fixed << std::setprecision(1) << throughput
              << std::setw(12) << std::fixed << std::setprecision(0) << qps
              << std::endl;
    
    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_database);
    cudaFree(d_scores);
}

int main() {
    std::cout << "🚀 GPU Performance Benchmark Suite" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Check GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "📊 GPU: " << prop.name << std::endl;
    std::cout << "   SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "   Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "   Shared Memory/Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << std::endl;
    
    std::cout << "🏁 Running benchmarks..." << std::endl;
    std::cout << std::setw(10) << "Vectors" 
              << std::setw(10) << "Queries"
              << std::setw(10) << "Dim"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "QPS"
              << std::endl;
    std::cout << std::setw(10) << "" 
              << std::setw(10) << ""
              << std::setw(10) << ""
              << std::setw(12) << ""
              << std::setw(15) << "(M comp/s)"
              << std::setw(12) << ""
              << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
    
    // Different database sizes
    run_benchmark(768, 1000, 100);
    run_benchmark(768, 10000, 100);
    run_benchmark(768, 100000, 100);
    run_benchmark(768, 1000000, 10);
    
    std::cout << std::endl;
    
    // Different query batch sizes
    std::cout << "📊 Query Batch Size Impact (100k vectors, dim=768):" << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
    run_benchmark(768, 100000, 1);
    run_benchmark(768, 100000, 10);
    run_benchmark(768, 100000, 100);
    run_benchmark(768, 100000, 1000);
    
    std::cout << std::endl;
    
    // Different dimensions
    std::cout << "📏 Dimension Impact (100k vectors, 100 queries):" << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
    run_benchmark(128, 100000, 100);
    run_benchmark(256, 100000, 100);
    run_benchmark(512, 100000, 100);
    run_benchmark(768, 100000, 100);
    run_benchmark(1024, 100000, 100);
    
    std::cout << std::endl;
    std::cout << "✅ Benchmark completed!" << std::endl;
    
    // Memory usage estimate
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\n💾 GPU Memory: " << (total_mem - free_mem) / (1024*1024) << " MB used / " 
              << total_mem / (1024*1024) << " MB total" << std::endl;
    
    return 0;
}