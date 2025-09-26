// cuda_similarity_optimized.cu - Memory-optimized CUDA similarity search
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>

// Improved kernel with vectorized loads
__global__ void compute_similarity_int8_vectorized(
    const int8_t* queries,      
    const int8_t* database,     
    float* scores,              
    int num_queries,
    int num_vectors, 
    int dim
) {
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || vector_idx >= num_vectors) return;
    
    const int8_t* query_ptr = queries + query_idx * dim;
    const int8_t* vector_ptr = database + vector_idx * dim;
    
    int sum = 0;
    
    // Vectorized loading for better memory bandwidth
    int i = 0;
    if (dim >= 4) {
        for (i = 0; i <= dim - 4; i += 4) {
            // Load 4 bytes at once when aligned
            int32_t query_4 = *reinterpret_cast<const int32_t*>(query_ptr + i);
            int32_t vector_4 = *reinterpret_cast<const int32_t*>(vector_ptr + i);
            
            // Unpack and multiply
            int8_t q0 = query_4 & 0xFF;
            int8_t q1 = (query_4 >> 8) & 0xFF;
            int8_t q2 = (query_4 >> 16) & 0xFF;
            int8_t q3 = (query_4 >> 24) & 0xFF;
            
            int8_t v0 = vector_4 & 0xFF;
            int8_t v1 = (vector_4 >> 8) & 0xFF;
            int8_t v2 = (vector_4 >> 16) & 0xFF;
            int8_t v3 = (vector_4 >> 24) & 0xFF;
            
            sum += q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3;
        }
    }
    
    // Handle remaining elements
    for (; i < dim; i++) {
        sum += static_cast<int>(query_ptr[i]) * static_cast<int>(vector_ptr[i]);
    }
    
    scores[query_idx * num_vectors + vector_idx] = static_cast<float>(sum);
}

// Shared memory tiled kernel for better cache utilization
__global__ void compute_similarity_int8_tiled(
    const int8_t* queries,      
    const int8_t* database,     
    float* scores,              
    int num_queries,
    int num_vectors, 
    int dim
) {
    const int TILE_SIZE = 32;
    __shared__ int8_t shared_query[TILE_SIZE * 384];  // Assume max dim 384
    
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Collaborative loading of query to shared memory
    if (threadIdx.x == 0 && query_idx < num_queries) {
        for (int d = 0; d < dim; d++) {
            shared_query[threadIdx.y * dim + d] = queries[query_idx * dim + d];
        }
    }
    
    __syncthreads();
    
    if (query_idx >= num_queries || vector_idx >= num_vectors) return;
    
    // Compute dot product using shared memory for query
    int sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += static_cast<int>(shared_query[threadIdx.y * dim + i]) * 
               static_cast<int>(database[vector_idx * dim + i]);
    }
    
    scores[query_idx * num_vectors + vector_idx] = static_cast<float>(sum);
}

class OptimizedCudaSimilaritySearch {
private:
    // Persistent GPU memory pools
    int8_t* d_database;
    int8_t* d_query_pool;
    float* d_scores_pool;
    int* d_indices_pool;
    float* d_top_scores_pool;
    
    // CUDA streams for async operations
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    
    // Configuration
    int num_vectors;
    int vector_dim;
    int max_queries;
    bool initialized;
    
    // Performance tracking
    std::vector<float> query_times;
    
public:
    OptimizedCudaSimilaritySearch() : 
        d_database(nullptr), d_query_pool(nullptr), d_scores_pool(nullptr),
        d_indices_pool(nullptr), d_top_scores_pool(nullptr),
        initialized(false), max_queries(1000) {}
    
    ~OptimizedCudaSimilaritySearch() {
        cleanup();
    }
    
    void cleanup() {
        if (d_database) cudaFree(d_database);
        if (d_query_pool) cudaFree(d_query_pool);
        if (d_scores_pool) cudaFree(d_scores_pool);
        if (d_indices_pool) cudaFree(d_indices_pool);
        if (d_top_scores_pool) cudaFree(d_top_scores_pool);
        
        if (initialized) {
            cudaStreamDestroy(compute_stream);
            cudaStreamDestroy(transfer_stream);
        }
        
        d_database = d_query_pool = d_scores_pool = nullptr;
        d_indices_pool = nullptr;
        d_top_scores_pool = nullptr;
        initialized = false;
    }
    
    bool initialize(int vectors, int dim, int max_batch_queries = 1000) {
        cleanup();  // Clean up any existing allocation
        
        num_vectors = vectors;
        vector_dim = dim;
        max_queries = max_batch_queries;
        
        // Create CUDA streams
        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&transfer_stream);
        
        // Allocate persistent GPU memory pools
        size_t database_size = num_vectors * vector_dim * sizeof(int8_t);
        size_t query_pool_size = max_queries * vector_dim * sizeof(int8_t);
        size_t scores_pool_size = max_queries * num_vectors * sizeof(float);
        size_t indices_pool_size = max_queries * 50 * sizeof(int);  // Assume max k=50
        size_t top_scores_pool_size = max_queries * 50 * sizeof(float);
        
        cudaError_t error;
        
        error = cudaMalloc(&d_database, database_size);
        if (error != cudaSuccess) goto allocation_error;
        
        error = cudaMalloc(&d_query_pool, query_pool_size);
        if (error != cudaSuccess) goto allocation_error;
        
        error = cudaMalloc(&d_scores_pool, scores_pool_size);
        if (error != cudaSuccess) goto allocation_error;
        
        error = cudaMalloc(&d_indices_pool, indices_pool_size);
        if (error != cudaSuccess) goto allocation_error;
        
        error = cudaMalloc(&d_top_scores_pool, top_scores_pool_size);
        if (error != cudaSuccess) goto allocation_error;
        
        initialized = true;
        
        std::cout << "🚀 Optimized CUDA similarity search initialized:" << std::endl;
        std::cout << "   Vectors: " << vectors << " x " << dim << "D" << std::endl;
        std::cout << "   Max batch queries: " << max_batch_queries << std::endl;
        std::cout << "   Database memory: " << database_size / 1e6 << " MB" << std::endl;
        std::cout << "   Total GPU memory: " << (database_size + query_pool_size + scores_pool_size + 
                     indices_pool_size + top_scores_pool_size) / 1e6 << " MB" << std::endl;
        return true;
        
    allocation_error:
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(error) << std::endl;
        cleanup();
        return false;
    }
    
    bool add_vectors(const std::vector<int8_t>& vectors) {
        if (!initialized) return false;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t data_size = vectors.size() * sizeof(int8_t);
        
        // Async transfer on dedicated stream
        cudaError_t error = cudaMemcpyAsync(d_database, vectors.data(), data_size, 
                                           cudaMemcpyHostToDevice, transfer_stream);
        
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy vectors to GPU: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // Synchronize transfer stream
        cudaStreamSynchronize(transfer_stream);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "✅ Added " << vectors.size() / vector_dim << " vectors to GPU in " 
                  << duration.count() / 1000.0 << " ms" << std::endl;
        return true;
    }
    
    bool batch_search_optimized(const std::vector<int8_t>& queries, int k, 
                               std::vector<int>& top_indices, std::vector<float>& top_scores,
                               bool use_tiled_kernel = false) {
        if (!initialized) return false;
        
        int num_queries = queries.size() / vector_dim;
        if (num_queries > max_queries) {
            std::cerr << "Too many queries: " << num_queries << " > " << max_queries << std::endl;
            return false;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Async query transfer
        size_t queries_size = queries.size() * sizeof(int8_t);
        cudaMemcpyAsync(d_query_pool, queries.data(), queries_size, 
                       cudaMemcpyHostToDevice, transfer_stream);
        
        // Configure CUDA kernel launch parameters
        dim3 block_size(32, 16);  // 512 threads per block
        dim3 grid_size(
            (num_vectors + block_size.x - 1) / block_size.x,
            (num_queries + block_size.y - 1) / block_size.y
        );
        
        // Wait for transfer, then launch compute
        cudaStreamWaitEvent(compute_stream, 0);
        
        // Launch optimized similarity computation kernel
        if (use_tiled_kernel) {
            compute_similarity_int8_tiled<<<grid_size, block_size, 0, compute_stream>>>(
                d_query_pool, d_database, d_scores_pool, 
                num_queries, num_vectors, vector_dim
            );
        } else {
            compute_similarity_int8_vectorized<<<grid_size, block_size, 0, compute_stream>>>(
                d_query_pool, d_database, d_scores_pool, 
                num_queries, num_vectors, vector_dim
            );
        }
        
        // Launch top-k selection kernel on same stream
        dim3 topk_block_size(256);
        dim3 topk_grid_size((num_queries + topk_block_size.x - 1) / topk_block_size.x);
        
        extern __global__ void select_topk(const float*, int*, float*, int, int, int);
        select_topk<<<topk_grid_size, topk_block_size, 0, compute_stream>>>(
            d_scores_pool, d_indices_pool, d_top_scores_pool,
            num_queries, num_vectors, k
        );
        
        // Async result transfer back to host
        top_indices.resize(num_queries * k);
        top_scores.resize(num_queries * k);
        
        cudaMemcpyAsync(top_indices.data(), d_indices_pool, 
                       num_queries * k * sizeof(int), 
                       cudaMemcpyDeviceToHost, compute_stream);
        cudaMemcpyAsync(top_scores.data(), d_top_scores_pool, 
                       num_queries * k * sizeof(float), 
                       cudaMemcpyDeviceToHost, compute_stream);
        
        // Wait for all operations to complete
        cudaStreamSynchronize(compute_stream);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        float query_time_ms = duration.count() / 1000.0;
        query_times.push_back(query_time_ms);
        
        std::cout << "🔍 Optimized batch search: " << num_queries << " queries in " 
                  << query_time_ms << "ms (" 
                  << (query_time_ms / num_queries) << "ms/query, "
                  << (num_queries * 1000.0 / query_time_ms) << " qps)" << std::endl;
        
        return true;
    }
    
    void print_performance_stats() {
        if (query_times.empty()) return;
        
        float total_time = 0;
        float min_time = query_times[0];
        float max_time = query_times[0];
        
        for (float time : query_times) {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        float avg_time = total_time / query_times.size();
        
        std::cout << "\n📊 Performance Statistics:" << std::endl;
        std::cout << "   Total queries: " << query_times.size() << std::endl;
        std::cout << "   Average time: " << avg_time << " ms" << std::endl;
        std::cout << "   Min time: " << min_time << " ms" << std::endl;
        std::cout << "   Max time: " << max_time << " ms" << std::endl;
        std::cout << "   Average QPS: " << (1000.0 / avg_time) << std::endl;
    }
    
    void print_memory_usage() {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t used_mem = total_mem - free_mem;
        
        std::cout << "💾 GPU Memory Usage: " 
                  << (used_mem / 1024 / 1024) << " / " 
                  << (total_mem / 1024 / 1024) << " MB ("
                  << (100.0 * used_mem / total_mem) << "%)" << std::endl;
    }
};

// Updated test function
extern "C" {
    void test_optimized_cuda_similarity() {
        std::cout << "🚀 Testing Optimized CUDA Similarity Search" << std::endl;
        
        // Test parameters
        const int num_vectors = 1000000;
        const int vector_dim = 384;
        const int num_queries = 100;
        const int k = 10;
        const int batch_size = 50;
        
        // Initialize optimized search
        OptimizedCudaSimilaritySearch search;
        if (!search.initialize(num_vectors, vector_dim, num_queries)) {
            std::cerr << "Failed to initialize optimized CUDA similarity search" << std::endl;
            return;
        }
        
        // Generate test data
        std::vector<int8_t> database(num_vectors * vector_dim);
        for (int i = 0; i < database.size(); i++) {
            database[i] = static_cast<int8_t>(rand() % 256 - 128);
        }
        
        // Add vectors to search index
        if (!search.add_vectors(database)) {
            std::cerr << "Failed to add vectors to search index" << std::endl;
            return;
        }
        
        search.print_memory_usage();
        
        // Test multiple batches to measure performance consistency
        std::vector<int8_t> queries(num_queries * vector_dim);
        for (int i = 0; i < queries.size(); i++) {
            queries[i] = static_cast<int8_t>(rand() % 256 - 128);
        }
        
        std::cout << "\n🧪 Running performance tests..." << std::endl;
        
        // Test standard kernel
        for (int batch = 0; batch < 5; batch++) {
            std::vector<int> top_indices;
            std::vector<float> top_scores;
            
            search.batch_search_optimized(queries, k, top_indices, top_scores, false);
        }
        
        // Test tiled kernel
        std::cout << "\n🧪 Testing tiled kernel..." << std::endl;
        for (int batch = 0; batch < 5; batch++) {
            std::vector<int> top_indices;
            std::vector<float> top_scores;
            
            search.batch_search_optimized(queries, k, top_indices, top_scores, true);
        }
        
        search.print_performance_stats();
        search.print_memory_usage();
        std::cout << "✅ Optimized CUDA similarity search test completed!" << std::endl;
    }
}