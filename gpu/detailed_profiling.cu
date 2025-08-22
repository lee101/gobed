// detailed_profiling.cu - Detailed GPU kernel profiling and validation
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
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

// Enhanced GPU kernel with instrumentation
__device__ int gpu_computation_counter = 0;

__global__ void instrumented_similarity_kernel(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim,
    unsigned long long* computation_proof
) {
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || vector_idx >= num_vectors) return;
    
    // Proof of GPU execution: use GPU-specific thread identifiers
    unsigned long long thread_signature = 
        ((unsigned long long)blockIdx.x << 32) |
        ((unsigned long long)blockIdx.y << 24) |
        ((unsigned long long)threadIdx.x << 16) |
        ((unsigned long long)threadIdx.y << 8) |
        ((unsigned long long)dim & 0xFF);
    
    // Compute dot product similarity (the actual work)
    int sum = 0;
    for (int i = 0; i < dim; i++) {
        int8_t q_val = queries[query_idx * dim + i];
        int8_t d_val = database[vector_idx * dim + i];
        sum += static_cast<int>(q_val) * static_cast<int>(d_val);
        
        // Add unique GPU computation signature
        thread_signature ^= (((unsigned long long)q_val << 4) ^ (unsigned long long)d_val);
    }
    
    scores[query_idx * num_vectors + vector_idx] = static_cast<float>(sum);
    
    // Store proof that this computation happened on GPU
    atomicMax(&computation_proof[0], thread_signature);
    
    // Count actual GPU computations
    atomicAdd(&gpu_computation_counter, 1);
}

void launch_instrumented_kernel(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim,
    unsigned long long* computation_proof
) {
    dim3 block_size(32, 16);
    dim3 grid_size(
        (num_vectors + block_size.x - 1) / block_size.x,
        (num_queries + block_size.y - 1) / block_size.y
    );
    
    // Reset counters
    cudaMemsetAsync(&gpu_computation_counter, 0, sizeof(int));
    cudaMemsetAsync(computation_proof, 0, sizeof(unsigned long long));
    
    // Launch with NVTX markers for profiling
    nvtxRangePush("GPU_Similarity_Kernel");
    instrumented_similarity_kernel<<<grid_size, block_size>>>(
        queries, database, scores, num_queries, num_vectors, dim, computation_proof
    );
    nvtxRangePop();
}

// CPU reference with instrumentation
void cpu_reference_instrumented(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim,
    int* cpu_operations
) {
    *cpu_operations = 0;
    
    for (int q = 0; q < num_queries; q++) {
        for (int v = 0; v < num_vectors; v++) {
            int sum = 0;
            for (int d = 0; d < dim; d++) {
                sum += static_cast<int>(queries[q * dim + d]) * 
                       static_cast<int>(database[v * dim + d]);
                (*cpu_operations)++;
            }
            scores[q * num_vectors + v] = static_cast<float>(sum);
        }
    }
}

bool test_detailed_profiling() {
    std::cout << "🔬 Detailed GPU Profiling and Validation" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    const int num_queries = 50;
    const int num_vectors = 10000;
    const int dim = 768;
    const int total_computations = num_queries * num_vectors;
    
    std::cout << "📊 Test Configuration:" << std::endl;
    std::cout << "   Queries: " << num_queries << std::endl;
    std::cout << "   Vectors: " << num_vectors << std::endl;
    std::cout << "   Dimension: " << dim << std::endl;
    std::cout << "   Total comparisons: " << total_computations << std::endl;
    std::cout << "   Expected operations: " << total_computations * dim << std::endl;
    
    // Generate deterministic test data
    std::random_device rd;
    std::mt19937 gen(12345); // Fixed seed
    std::uniform_int_distribution<> dis(-100, 100);
    
    std::vector<int8_t> queries(num_queries * dim);
    std::vector<int8_t> database(num_vectors * dim);
    
    for (auto& val : queries) val = dis(gen);
    for (auto& val : database) val = dis(gen);
    
    // CPU execution with instrumentation
    std::cout << "\n💻 CPU Reference Execution:" << std::endl;
    std::vector<float> cpu_scores(total_computations);
    int cpu_operations = 0;
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_reference_instrumented(
        queries.data(), database.data(), cpu_scores.data(),
        num_queries, num_vectors, dim, &cpu_operations
    );
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    std::cout << "   Operations performed: " << cpu_operations << std::endl;
    std::cout << "   Time: " << cpu_time.count() / 1000.0 << " ms" << std::endl;
    std::cout << "   Rate: " << cpu_operations / (cpu_time.count() / 1000000.0) / 1000000.0 << " M ops/sec" << std::endl;
    
    // GPU execution with detailed instrumentation
    std::cout << "\n🚀 GPU Instrumented Execution:" << std::endl;
    
    // Allocate GPU memory
    int8_t* d_queries;
    int8_t* d_database;
    float* d_scores;
    unsigned long long* d_computation_proof;
    
    cudaMalloc(&d_queries, queries.size());
    cudaMalloc(&d_database, database.size());
    cudaMalloc(&d_scores, total_computations * sizeof(float));
    cudaMalloc(&d_computation_proof, sizeof(unsigned long long));
    
    // Copy data to GPU
    cudaMemcpy(d_queries, queries.data(), queries.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_database, database.data(), database.size(), cudaMemcpyHostToDevice);
    
    // Warm up
    for (int i = 0; i < 3; i++) {
        launch_instrumented_kernel(d_queries, d_database, d_scores, 
                                  num_queries, num_vectors, dim, d_computation_proof);
        cudaDeviceSynchronize();
    }
    
    // Detailed GPU timing with events
    cudaEvent_t start, stop, kernel_start, kernel_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    // Full GPU execution timing
    cudaEventRecord(start);
    
    // Just kernel timing
    cudaEventRecord(kernel_start);
    launch_instrumented_kernel(d_queries, d_database, d_scores, 
                              num_queries, num_vectors, dim, d_computation_proof);
    cudaEventRecord(kernel_stop);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get timing results
    float total_time_ms, kernel_time_ms;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    cudaEventElapsedTime(&kernel_time_ms, kernel_start, kernel_stop);
    
    std::cout << "   Total GPU time: " << total_time_ms << " ms" << std::endl;
    std::cout << "   Pure kernel time: " << kernel_time_ms << " ms" << std::endl;
    std::cout << "   Memory/sync overhead: " << total_time_ms - kernel_time_ms << " ms" << std::endl;
    
    // Get GPU execution proof
    unsigned long long computation_proof;
    cudaMemcpy(&computation_proof, d_computation_proof, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    std::cout << "   GPU execution signature: 0x" << std::hex << computation_proof << std::dec << std::endl;
    std::cout << "   (Non-zero proves GPU execution occurred)" << std::endl;
    
    // Copy results and verify
    std::vector<float> gpu_scores(total_computations);
    cudaMemcpy(gpu_scores.data(), d_scores, total_computations * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Performance analysis
    double gpu_ops_per_sec = (double(total_computations) * dim) / (kernel_time_ms / 1000.0) / 1000000.0;
    double speedup = cpu_time.count() / (kernel_time_ms * 1000.0);
    
    std::cout << "\n📈 Performance Analysis:" << std::endl;
    std::cout << "   GPU rate: " << gpu_ops_per_sec << " M ops/sec" << std::endl;
    std::cout << "   CPU rate: " << cpu_operations / (cpu_time.count() / 1000000.0) / 1000000.0 << " M ops/sec" << std::endl;
    std::cout << "   Speedup: " << speedup << "x" << std::endl;
    
    // Accuracy verification with detailed reporting
    std::cout << "\n🔍 Accuracy Verification:" << std::endl;
    int mismatches = 0;
    double max_diff = 0.0;
    double sum_diff = 0.0;
    
    for (int i = 0; i < total_computations; i++) {
        double diff = std::abs(gpu_scores[i] - cpu_scores[i]);
        sum_diff += diff;
        if (diff > max_diff) max_diff = diff;
        
        if (diff > 0.001) {
            mismatches++;
            if (mismatches <= 5) { // Show first few mismatches
                std::cout << "   Mismatch " << i << ": CPU=" << cpu_scores[i] 
                         << ", GPU=" << gpu_scores[i] << ", diff=" << diff << std::endl;
            }
        }
    }
    
    double avg_diff = sum_diff / total_computations;
    std::cout << "   Mismatches: " << mismatches << " / " << total_computations << std::endl;
    std::cout << "   Max difference: " << max_diff << std::endl;
    std::cout << "   Average difference: " << avg_diff << std::endl;
    std::cout << "   Accuracy: " << (1.0 - double(mismatches) / total_computations) * 100.0 << "%" << std::endl;
    
    // Sample results verification
    std::cout << "\n🎯 Sample Results Comparison:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "   Sample " << i << ": CPU=" << std::fixed << std::setprecision(1) 
                 << cpu_scores[i] << ", GPU=" << gpu_scores[i] << std::endl;
    }
    
    // Memory bandwidth analysis
    size_t bytes_read = queries.size() + database.size();
    size_t bytes_written = total_computations * sizeof(float);
    double bandwidth_gb_s = (bytes_read + bytes_written) / (kernel_time_ms / 1000.0) / (1024*1024*1024);
    
    std::cout << "\n💾 Memory Analysis:" << std::endl;
    std::cout << "   Data read: " << bytes_read / (1024*1024) << " MB" << std::endl;
    std::cout << "   Data written: " << bytes_written / (1024*1024) << " MB" << std::endl;
    std::cout << "   Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // GPU utilization estimate
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double theoretical_ops = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor * (prop.clockRate * 1000.0);
    double utilization = gpu_ops_per_sec * 1000000.0 / theoretical_ops * 100.0;
    
    std::cout << "\n⚡ GPU Utilization Estimate:" << std::endl;
    std::cout << "   Theoretical peak: " << theoretical_ops / 1000000000.0 << " G ops/sec" << std::endl;
    std::cout << "   Achieved: " << gpu_ops_per_sec * 1000.0 << " M ops/sec" << std::endl;
    std::cout << "   Estimated utilization: " << utilization << "%" << std::endl;
    
    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_database);
    cudaFree(d_scores);
    cudaFree(d_computation_proof);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    // Pass criteria
    bool success = computation_proof != 0 &&  // GPU executed
                   mismatches == 0 &&          // Perfect accuracy
                   speedup > 2.0 &&           // Significant speedup
                   gpu_ops_per_sec > 50.0;    // Reasonable performance
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    if (success) {
        std::cout << "🎉 DETAILED PROFILING PASSED!" << std::endl;
        std::cout << "✅ GPU kernel executed with signature: 0x" << std::hex << computation_proof << std::dec << std::endl;
        std::cout << "✅ Perfect accuracy: " << mismatches << " mismatches" << std::endl;
        std::cout << "✅ Strong performance: " << speedup << "x speedup" << std::endl;
        std::cout << "✅ High throughput: " << gpu_ops_per_sec << " M ops/sec" << std::endl;
    } else {
        std::cout << "❌ DETAILED PROFILING FAILED!" << std::endl;
        if (computation_proof == 0) std::cout << "   - No GPU execution signature detected" << std::endl;
        if (mismatches > 0) std::cout << "   - Accuracy issues: " << mismatches << " mismatches" << std::endl;
        if (speedup <= 2.0) std::cout << "   - Poor speedup: " << speedup << "x" << std::endl;
        if (gpu_ops_per_sec <= 50.0) std::cout << "   - Low performance: " << gpu_ops_per_sec << " M ops/sec" << std::endl;
    }
    
    return success;
}

int main() {
    std::cout << "🔬 Detailed GPU Profiling Suite" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Initialize CUDA profiler
    cudaProfilerStart();
    
    bool success = test_detailed_profiling();
    
    cudaProfilerStop();
    
    return success ? 0 : 1;
}