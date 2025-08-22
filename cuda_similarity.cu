// cuda_similarity.cu - Direct CUDA similarity search
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>

// CUDA kernel for int8 dot product similarity
__global__ void compute_similarity_int8(
    const int8_t* queries,      // [num_queries, dim]
    const int8_t* database,     // [num_vectors, dim] 
    float* scores,              // [num_queries, num_vectors]
    int num_queries,
    int num_vectors, 
    int dim
) {
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || vector_idx >= num_vectors) return;
    
    // Compute dot product
    int sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += static_cast<int>(queries[query_idx * dim + i]) * 
               static_cast<int>(database[vector_idx * dim + i]);
    }
    
    scores[query_idx * num_vectors + vector_idx] = static_cast<float>(sum);
}

// Optimized kernel using shared memory
__global__ void compute_similarity_int8_optimized(
    const int8_t* queries,      
    const int8_t* database,     
    float* scores,              
    int num_queries,
    int num_vectors, 
    int dim
) {
    extern __shared__ int8_t shared_mem[];
    
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || vector_idx >= num_vectors) return;
    
    // Use shared memory for better memory access patterns
    int8_t* shared_query = shared_mem;
    int8_t* shared_vector = shared_mem + blockDim.y * dim;
    
    // Load query to shared memory
    if (threadIdx.x == 0 && query_idx < num_queries) {
        for (int i = 0; i < dim; i++) {
            shared_query[threadIdx.y * dim + i] = queries[query_idx * dim + i];
        }
    }
    
    // Load vector to shared memory  
    if (threadIdx.y == 0 && vector_idx < num_vectors) {
        for (int i = 0; i < dim; i++) {
            shared_vector[threadIdx.x * dim + i] = database[vector_idx * dim + i];
        }
    }
    
    __syncthreads();
    
    // Compute dot product from shared memory
    int sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += static_cast<int>(shared_query[threadIdx.y * dim + i]) * 
               static_cast<int>(shared_vector[threadIdx.x * dim + i]);
    }
    
    scores[query_idx * num_vectors + vector_idx] = static_cast<float>(sum);
}

// Top-K selection kernel
__global__ void select_topk(
    const float* scores,        // [num_queries, num_vectors]
    int* top_indices,          // [num_queries, k]
    float* top_scores,         // [num_queries, k] 
    int num_queries,
    int num_vectors,
    int k
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;
    
    const float* query_scores = scores + query_idx * num_vectors;
    int* query_indices = top_indices + query_idx * k;
    float* query_top_scores = top_scores + query_idx * k;
    
    // Initialize with first k elements
    for (int i = 0; i < k; i++) {
        query_indices[i] = i;
        query_top_scores[i] = query_scores[i];
    }
    
    // Find k largest elements using selection sort
    for (int vec = k; vec < num_vectors; vec++) {
        float current_score = query_scores[vec];
        
        // Find minimum in current top-k
        int min_idx = 0;
        for (int i = 1; i < k; i++) {
            if (query_top_scores[i] < query_top_scores[min_idx]) {
                min_idx = i;
            }
        }
        
        // Replace if current score is larger
        if (current_score > query_top_scores[min_idx]) {
            query_top_scores[min_idx] = current_score;
            query_indices[min_idx] = vec;
        }
    }
    
    // Sort the top-k results
    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            if (query_top_scores[i] < query_top_scores[j]) {
                // Swap scores
                float temp_score = query_top_scores[i];
                query_top_scores[i] = query_top_scores[j];
                query_top_scores[j] = temp_score;
                
                // Swap indices
                int temp_idx = query_indices[i];
                query_indices[i] = query_indices[j];
                query_indices[j] = temp_idx;
            }
        }
    }
}

class CudaSimilaritySearch {
private:
    int8_t* d_database;
    int num_vectors;
    int vector_dim;
    bool initialized;
    
public:
    CudaSimilaritySearch() : d_database(nullptr), initialized(false) {}
    
    ~CudaSimilaritySearch() {
        if (d_database) {
            cudaFree(d_database);
        }
    }
    
    bool initialize(int vectors, int dim) {
        num_vectors = vectors;
        vector_dim = dim;
        
        // Allocate GPU memory for database
        size_t database_size = num_vectors * vector_dim * sizeof(int8_t);
        cudaError_t error = cudaMalloc(&d_database, database_size);
        
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        initialized = true;
        std::cout << "✅ CUDA similarity search initialized: " 
                  << vectors << " vectors, " << dim << "D" << std::endl;
        return true;
    }
    
    bool add_vectors(const std::vector<int8_t>& vectors) {
        if (!initialized) return false;
        
        size_t data_size = vectors.size() * sizeof(int8_t);
        cudaError_t error = cudaMemcpy(d_database, vectors.data(), data_size, cudaMemcpyHostToDevice);
        
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy vectors to GPU: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        std::cout << "✅ Copied " << vectors.size() / vector_dim << " vectors to GPU" << std::endl;
        return true;
    }
    
    bool search(const std::vector<int8_t>& queries, int k, 
                std::vector<int>& top_indices, std::vector<float>& top_scores) {
        if (!initialized) return false;
        
        int num_queries = queries.size() / vector_dim;
        auto start = std::chrono::high_resolution_clock::now();
        
        // Allocate GPU memory for queries
        int8_t* d_queries;
        size_t queries_size = queries.size() * sizeof(int8_t);
        cudaMalloc(&d_queries, queries_size);
        cudaMemcpy(d_queries, queries.data(), queries_size, cudaMemcpyHostToDevice);
        
        // Allocate GPU memory for scores
        float* d_scores;
        size_t scores_size = num_queries * num_vectors * sizeof(float);
        cudaMalloc(&d_scores, scores_size);
        
        // Allocate GPU memory for results
        int* d_top_indices;
        float* d_top_scores;
        cudaMalloc(&d_top_indices, num_queries * k * sizeof(int));
        cudaMalloc(&d_top_scores, num_queries * k * sizeof(float));
        
        // Configure CUDA kernel launch parameters
        dim3 block_size(32, 16);  // 512 threads per block
        dim3 grid_size(
            (num_vectors + block_size.x - 1) / block_size.x,
            (num_queries + block_size.y - 1) / block_size.y
        );
        
        // Launch similarity computation kernel
        compute_similarity_int8<<<grid_size, block_size>>>(
            d_queries, d_database, d_scores, 
            num_queries, num_vectors, vector_dim
        );
        
        // Check for kernel launch errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // Launch top-k selection kernel
        dim3 topk_block_size(256);
        dim3 topk_grid_size((num_queries + topk_block_size.x - 1) / topk_block_size.x);
        
        select_topk<<<topk_grid_size, topk_block_size>>>(
            d_scores, d_top_indices, d_top_scores,
            num_queries, num_vectors, k
        );
        
        // Wait for kernels to complete
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Copy results back to host
        top_indices.resize(num_queries * k);
        top_scores.resize(num_queries * k);
        
        cudaMemcpy(top_indices.data(), d_top_indices, num_queries * k * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(top_scores.data(), d_top_scores, num_queries * k * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_queries);
        cudaFree(d_scores);
        cudaFree(d_top_indices);
        cudaFree(d_top_scores);
        
        std::cout << "🔍 GPU search completed: " << num_queries << " queries in " 
                  << duration.count() << "μs (" 
                  << (duration.count() / num_queries) << "μs/query)" << std::endl;
        
        return true;
    }
    
    void print_gpu_memory_usage() {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t used_mem = total_mem - free_mem;
        
        std::cout << "💾 GPU Memory: " 
                  << (used_mem / 1024 / 1024) << " / " 
                  << (total_mem / 1024 / 1024) << " MB" << std::endl;
    }
};

// Test function
extern "C" {
    void test_cuda_similarity_search() {
        std::cout << "🚀 Testing Direct CUDA Similarity Search" << std::endl;
        
        // Test parameters
        const int num_vectors = 10000;
        const int vector_dim = 512;
        const int num_queries = 100;
        const int k = 10;
        
        // Initialize CUDA similarity search
        CudaSimilaritySearch search;
        if (!search.initialize(num_vectors, vector_dim)) {
            std::cerr << "Failed to initialize CUDA similarity search" << std::endl;
            return;
        }
        
        // Generate test data
        std::vector<int8_t> database(num_vectors * vector_dim);
        std::vector<int8_t> queries(num_queries * vector_dim);
        
        // Fill with random data
        for (int i = 0; i < database.size(); i++) {
            database[i] = static_cast<int8_t>(rand() % 256 - 128);
        }
        for (int i = 0; i < queries.size(); i++) {
            queries[i] = static_cast<int8_t>(rand() % 256 - 128);
        }
        
        std::cout << "📊 Generated " << num_vectors << " database vectors, " 
                  << num_queries << " query vectors (" << vector_dim << "D)" << std::endl;
        
        // Add vectors to search index
        if (!search.add_vectors(database)) {
            std::cerr << "Failed to add vectors to search index" << std::endl;
            return;
        }
        
        search.print_gpu_memory_usage();
        
        // Perform search
        std::vector<int> top_indices;
        std::vector<float> top_scores;
        
        if (!search.search(queries, k, top_indices, top_scores)) {
            std::cerr << "Search failed" << std::endl;
            return;
        }
        
        // Print sample results
        std::cout << "\n📋 Sample Results:" << std::endl;
        for (int q = 0; q < std::min(3, num_queries); q++) {
            std::cout << "Query " << q << " top results:" << std::endl;
            for (int i = 0; i < k; i++) {
                int idx = q * k + i;
                std::cout << "  " << (i + 1) << ". Vector " << top_indices[idx] 
                         << " (score: " << top_scores[idx] << ")" << std::endl;
            }
        }
        
        search.print_gpu_memory_usage();
        std::cout << "✅ CUDA similarity search test completed!" << std::endl;
    }
}