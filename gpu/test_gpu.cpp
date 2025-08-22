// test_gpu.cpp - Test program for GPU acceleration
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include "torch_cgo_wrapper.h"

void generate_random_vectors(int8_t* data, int num_vectors, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);
    
    for (int i = 0; i < num_vectors * dim; ++i) {
        data[i] = dis(gen);
    }
}

void print_results(const SearchResult& result) {
    std::cout << "\n📊 Top-" << result.count << " Results:" << std::endl;
    for (int i = 0; i < result.count && i < 5; ++i) {
        std::cout << "   [" << i << "] ID: " << result.ids[i] 
                  << ", Score: " << result.scores[i] << std::endl;
    }
}

int main() {
    std::cout << "🚀 GPU Acceleration Test Program" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Check CUDA availability
    if (torch_cuda_is_available()) {
        std::cout << "✅ CUDA is available!" << std::endl;
        std::cout << "📊 Number of CUDA devices: " << torch_cuda_device_count() << std::endl;
    } else {
        std::cout << "❌ CUDA is not available" << std::endl;
    }
    
    std::cout << "📚 LibTorch version: " << torch_get_version() << std::endl;
    
    // Test parameters
    const int vector_dim = 768;      // Typical embedding dimension
    const int num_vectors = 10000;   // Database size
    const int num_queries = 100;     // Number of test queries
    const int k = 10;                // Top-k results
    
    std::cout << "\n🔧 Test Configuration:" << std::endl;
    std::cout << "   Vector dimension: " << vector_dim << std::endl;
    std::cout << "   Database size: " << num_vectors << std::endl;
    std::cout << "   Number of queries: " << num_queries << std::endl;
    std::cout << "   Top-k: " << k << std::endl;
    
    // Create indexer with GPU
    IndexConfig config;
    config.vector_dim = vector_dim;
    config.num_subquantizers = 32;
    config.codebook_size = 256;
    config.ivf_clusters = 100;
    config.probe_lists = 10;
    config.rerank_k = 100;
    config.device_id = 0;  // Use GPU 0
    
    std::cout << "\n🏗️ Creating GPU indexer..." << std::endl;
    TorchIndexerHandle indexer = torch_indexer_create(config);
    if (!indexer) {
        std::cerr << "❌ Failed to create indexer" << std::endl;
        return 1;
    }
    
    // Generate random vectors
    std::cout << "🎲 Generating random vectors..." << std::endl;
    std::vector<int8_t> database(num_vectors * vector_dim);
    std::vector<int8_t> queries(num_queries * vector_dim);
    
    generate_random_vectors(database.data(), num_vectors, vector_dim);
    generate_random_vectors(queries.data(), num_queries, vector_dim);
    
    // Train indexer
    std::cout << "🎓 Training indexer..." << std::endl;
    if (!torch_indexer_train(indexer, database.data(), num_vectors, vector_dim)) {
        std::cerr << "❌ Failed to train indexer" << std::endl;
        torch_indexer_destroy(indexer);
        return 1;
    }
    
    // Add vectors to index
    std::cout << "📚 Adding vectors to index..." << std::endl;
    auto start_add = std::chrono::high_resolution_clock::now();
    
    if (!torch_indexer_add_vectors(indexer, database.data(), num_vectors, vector_dim)) {
        std::cerr << "❌ Failed to add vectors" << std::endl;
        torch_indexer_destroy(indexer);
        return 1;
    }
    
    auto end_add = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_add - start_add);
    std::cout << "✅ Vectors added in " << add_time.count() << " ms" << std::endl;
    
    // Get index stats
    IndexStats stats = torch_indexer_get_stats(indexer);
    std::cout << "\n📊 Index Statistics:" << std::endl;
    std::cout << "   Vectors: " << stats.num_vectors << std::endl;
    std::cout << "   Dimension: " << stats.vector_dim << std::endl;
    std::cout << "   Trained: " << (stats.is_trained ? "Yes" : "No") << std::endl;
    std::cout << "   Built: " << (stats.index_built ? "Yes" : "No") << std::endl;
    if (stats.gpu_memory_mb > 0) {
        std::cout << "   GPU Memory: " << stats.gpu_memory_mb << " MB" << std::endl;
    }
    
    // Benchmark search performance
    std::cout << "\n🎯 Running search benchmark..." << std::endl;
    
    // Warm-up
    for (int i = 0; i < 10; ++i) {
        SearchResult result = torch_indexer_search(indexer, queries.data(), vector_dim, k);
        torch_search_result_free(&result);
    }
    
    // Actual benchmark
    auto start_search = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_queries; ++i) {
        SearchResult result = torch_indexer_search(
            indexer, 
            queries.data() + i * vector_dim, 
            vector_dim, 
            k
        );
        
        if (i == 0) {
            print_results(result);
        }
        
        torch_search_result_free(&result);
    }
    
    auto end_search = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search);
    
    double avg_time_us = search_time.count() / (double)num_queries;
    double qps = 1000000.0 / avg_time_us;
    
    std::cout << "\n🏆 Performance Results:" << std::endl;
    std::cout << "   Total time: " << search_time.count() / 1000.0 << " ms" << std::endl;
    std::cout << "   Average per query: " << avg_time_us << " μs" << std::endl;
    std::cout << "   Queries per second: " << qps << " QPS" << std::endl;
    
    // Cleanup
    std::cout << "\n🧹 Cleaning up..." << std::endl;
    torch_indexer_destroy(indexer);
    
    std::cout << "✅ Test completed successfully!" << std::endl;
    
    return 0;
}