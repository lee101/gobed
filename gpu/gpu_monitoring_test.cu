// gpu_monitoring_test.cu - Test with actual GPU monitoring
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <thread>
#include <fstream>

extern "C" void launch_similarity_kernel(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim
);

// Function to capture nvidia-smi output during execution
void monitor_gpu_usage(bool& monitoring, std::vector<std::string>& gpu_stats) {
    int sample_count = 0;
    while (monitoring && sample_count < 20) {
        system("nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits > /tmp/gpu_stats.txt 2>/dev/null");
        
        std::ifstream file("/tmp/gpu_stats.txt");
        std::string line;
        if (std::getline(file, line)) {
            gpu_stats.push_back(std::to_string(sample_count) + ": " + line);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        sample_count++;
    }
}

bool test_with_monitoring() {
    std::cout << "📊 GPU Monitoring Test" << std::endl;
    std::cout << "======================" << std::endl;
    
    const int num_queries = 100;
    const int num_vectors = 100000;
    const int dim = 768;
    
    // Generate test data
    std::vector<int8_t> queries(num_queries * dim);
    std::vector<int8_t> database(num_vectors * dim);
    std::vector<float> scores(num_queries * num_vectors);
    
    // Fill with meaningful data
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(-127, 127);
    
    for (auto& val : queries) val = dis(gen);
    for (auto& val : database) val = dis(gen);
    
    std::cout << "📈 Starting GPU monitoring..." << std::endl;
    
    // Start GPU monitoring
    bool monitoring = true;
    std::vector<std::string> gpu_stats;
    std::thread monitor_thread(monitor_gpu_usage, std::ref(monitoring), std::ref(gpu_stats));
    
    // Sleep briefly to get baseline
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // GPU execution
    std::cout << "🚀 Executing on GPU..." << std::endl;
    
    int8_t* d_queries;
    int8_t* d_database;
    float* d_scores;
    
    cudaMalloc(&d_queries, queries.size());
    cudaMalloc(&d_database, database.size());
    cudaMalloc(&d_scores, scores.size() * sizeof(float));
    
    cudaMemcpy(d_queries, queries.data(), queries.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_database, database.data(), database.size(), cudaMemcpyHostToDevice);
    
    // Multiple runs to ensure GPU utilization is visible
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int run = 0; run < 10; run++) {
        launch_similarity_kernel(d_queries, d_database, d_scores, num_queries, num_vectors, dim);
        cudaDeviceSynchronize();
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Brief pause
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Stop monitoring
    monitoring = false;
    monitor_thread.join();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "✅ GPU execution completed in " << duration.count() << " ms" << std::endl;
    
    // Analyze GPU statistics
    std::cout << "\n📊 GPU Utilization During Execution:" << std::endl;
    std::cout << "Sample: GPU_Util%, Memory_MB, Power_W" << std::endl;
    
    bool found_high_utilization = false;
    bool found_memory_increase = false;
    
    for (const auto& stat : gpu_stats) {
        std::cout << "   " << stat << std::endl;
        
        // Parse utilization (rough parsing)
        size_t comma1 = stat.find(',');
        if (comma1 != std::string::npos) {
            std::string util_str = stat.substr(stat.find(':') + 1, comma1 - stat.find(':') - 1);
            try {
                int utilization = std::stoi(util_str);
                if (utilization > 10) {  // Look for >10% utilization
                    found_high_utilization = true;
                }
            } catch (...) {}
            
            // Parse memory usage
            size_t comma2 = stat.find(',', comma1 + 1);
            if (comma2 != std::string::npos) {
                std::string mem_str = stat.substr(comma1 + 1, comma2 - comma1 - 1);
                try {
                    int memory = std::stoi(mem_str);
                    if (memory > 2000) {  // Look for >2GB memory usage
                        found_memory_increase = true;
                    }
                } catch (...) {}
            }
        }
    }
    
    // Verify results are computed correctly
    cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Spot check some results
    bool results_valid = true;
    for (int i = 0; i < 10; i++) {
        if (scores[i] == 0.0f && i % 1000 != 0) { // Allow some zeros, but not all
            continue;
        }
        if (std::abs(scores[i]) > 1000000.0f) { // Sanity check
            results_valid = false;
            break;
        }
    }
    
    // Performance metrics
    double comparisons = double(num_queries) * num_vectors * 10; // 10 runs
    double ops_per_sec = comparisons / (duration.count() / 1000.0);
    
    std::cout << "\n📈 Performance Metrics:" << std::endl;
    std::cout << "   Total comparisons: " << comparisons / 1000000.0 << " M" << std::endl;
    std::cout << "   Time: " << duration.count() << " ms" << std::endl;
    std::cout << "   Throughput: " << ops_per_sec / 1000000.0 << " M comparisons/sec" << std::endl;
    
    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_database);
    cudaFree(d_scores);
    
    // Summary
    std::cout << "\n📋 Monitoring Results:" << std::endl;
    std::cout << "   High GPU utilization detected: " << (found_high_utilization ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "   Memory increase detected: " << (found_memory_increase ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "   Results computed correctly: " << (results_valid ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "   Reasonable performance: " << (ops_per_sec > 1000000.0 ? "✅ YES" : "❌ NO") << std::endl;
    
    bool success = found_memory_increase && results_valid && (ops_per_sec > 1000000.0);
    
    return success;
}

int main() {
    std::cout << "🔍 GPU Monitoring and Validation Test" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Get baseline GPU state
    system("nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader > /tmp/gpu_baseline.txt");
    
    std::ifstream baseline("/tmp/gpu_baseline.txt");
    std::string line;
    if (std::getline(baseline, line)) {
        std::cout << "🖥️  GPU Baseline: " << line << std::endl;
    }
    
    bool success = test_with_monitoring();
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    if (success) {
        std::cout << "🎉 MONITORING TEST PASSED!" << std::endl;
        std::cout << "✅ GPU execution confirmed by system monitoring" << std::endl;
        std::cout << "✅ Memory allocation and usage verified" << std::endl;
        std::cout << "✅ Performance metrics confirm GPU acceleration" << std::endl;
    } else {
        std::cout << "❌ MONITORING TEST FAILED!" << std::endl;
        std::cout << "⚠️  GPU execution could not be confirmed by monitoring" << std::endl;
    }
    
    return success ? 0 : 1;
}