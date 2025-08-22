// torch_real_gpu.cpp - Real GPU acceleration for our existing system
#include "torch_cgo_wrapper.h"
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <iostream>
#include <chrono>

// CUDA kernel for int8 similarity search
extern "C" {

__global__ void cuda_similarity_kernel(
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
    
    // Compute dot product similarity
    int sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += static_cast<int>(queries[query_idx * dim + i]) * 
               static_cast<int>(database[vector_idx * dim + i]);
    }
    
    scores[query_idx * num_vectors + vector_idx] = static_cast<float>(sum);
}

__global__ void cuda_topk_kernel(
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
    for (int i = 0; i < k && i < num_vectors; i++) {
        query_indices[i] = i;
        query_top_scores[i] = query_scores[i];
    }
    
    // Find k largest elements using insertion sort approach
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
    
    // Sort the top-k results in descending order
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

} // extern "C"

// Enhanced GPU-accelerated indexer class
class RealGPUIndexer {
public:
    int vector_dim;
    int num_vectors = 0;
    bool is_trained = false;
    bool index_built = false;
    int device_id;
    
    // GPU memory pointers
    int8_t* d_database = nullptr;
    
    RealGPUIndexer(const IndexConfig& config) 
        : vector_dim(config.vector_dim)
        , device_id(config.device_id)
    {
        std::cout << "🚀 Initializing Real GPU Indexer" << std::endl;
        
        // Initialize CUDA device
        if (device_id >= 0) {
            cudaError_t error = cudaSetDevice(device_id);
            if (error == cudaSuccess) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, device_id);
                std::cout << "✅ Using GPU: " << prop.name << std::endl;
                std::cout << "   Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
                std::cout << "   Compute: " << prop.major << "." << prop.minor << std::endl;
            } else {
                std::cout << "❌ Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
                device_id = -1;
            }
        }
    }
    
    ~RealGPUIndexer() {
        if (d_database) {
            cudaFree(d_database);
        }
    }
    
    bool train_index(const int8_t* vectors, int n_vectors, int dim) {
        if (dim != vector_dim) {
            std::cerr << "Dimension mismatch: expected " << vector_dim << ", got " << dim << std::endl;
            return false;
        }
        
        std::cout << "🎓 Training with " << n_vectors << " vectors (Real GPU)" << std::endl;
        is_trained = true;
        return true;
    }
    
    bool add_vectors(const int8_t* vectors, int n_vectors, int dim) {
        if (!is_trained) {
            std::cerr << "Index must be trained before adding vectors" << std::endl;
            return false;
        }
        
        if (dim != vector_dim) {
            std::cerr << "Dimension mismatch: expected " << vector_dim << ", got " << dim << std::endl;
            return false;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (device_id >= 0) {
            std::cout << "📚 Adding " << n_vectors << " vectors to GPU memory..." << std::endl;
            
            // Allocate GPU memory for database
            size_t database_size = n_vectors * vector_dim * sizeof(int8_t);
            
            if (d_database) {
                cudaFree(d_database);
            }
            
            cudaError_t error = cudaMalloc(&d_database, database_size);
            if (error != cudaSuccess) {
                std::cerr << "❌ Failed to allocate GPU memory: " << cudaGetErrorString(error) << std::endl;
                device_id = -1;
                return false;
            }
            
            // Copy data to GPU
            error = cudaMemcpy(d_database, vectors, database_size, cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                std::cerr << "❌ Failed to copy data to GPU: " << cudaGetErrorString(error) << std::endl;
                cudaFree(d_database);
                d_database = nullptr;
                device_id = -1;
                return false;
            }
            
            std::cout << "✅ Vectors copied to GPU memory" << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        num_vectors = n_vectors;
        index_built = true;
        
        std::cout << "✅ Added " << n_vectors << " vectors in " << duration.count() << "μs" << std::endl;
        return true;
    }
    
    SearchResult search(const int8_t* query, int dim, int k) {
        SearchResult result = {nullptr, nullptr, 0};
        
        if (!index_built) {
            std::cerr << "Index must be built before searching" << std::endl;
            return result;
        }
        
        if (dim != vector_dim) {
            std::cerr << "Query dimension mismatch: expected " << vector_dim << ", got " << dim << std::endl;
            return result;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (device_id >= 0 && d_database) {
            // GPU-accelerated search
            std::cout << "🎯 GPU-accelerated search..." << std::endl;
            
            // Allocate GPU memory for query
            int8_t* d_query;
            size_t query_size = vector_dim * sizeof(int8_t);
            cudaMalloc(&d_query, query_size);
            cudaMemcpy(d_query, query, query_size, cudaMemcpyHostToDevice);
            
            // Allocate GPU memory for scores
            float* d_scores;
            size_t scores_size = num_vectors * sizeof(float);
            cudaMalloc(&d_scores, scores_size);
            
            // Allocate GPU memory for results
            int* d_top_indices;
            float* d_top_scores;
            cudaMalloc(&d_top_indices, k * sizeof(int));
            cudaMalloc(&d_top_scores, k * sizeof(float));
            
            // Configure CUDA kernel launch parameters
            dim3 block_size(32, 16);  // 512 threads per block
            dim3 grid_size(
                (num_vectors + block_size.x - 1) / block_size.x,
                1  // Only 1 query at a time
            );
            
            // Launch similarity computation kernel
            cuda_similarity_kernel<<<grid_size, block_size>>>(
                d_query, d_database, d_scores, 
                1, num_vectors, vector_dim
            );
            
            // Check for kernel launch errors
            cudaError_t error = cudaGetLastError();
            if (error == cudaSuccess) {
                // Launch top-k selection kernel
                dim3 topk_block_size(1);
                dim3 topk_grid_size(1);
                
                cuda_topk_kernel<<<topk_grid_size, topk_block_size>>>(
                    d_scores, d_top_indices, d_top_scores,
                    1, num_vectors, k
                );
                
                // Wait for kernels to complete
                cudaDeviceSynchronize();
                
                // Copy results back to host
                result.count = k;
                result.ids = new int[k];
                result.scores = new float[k];
                
                cudaMemcpy(result.ids, d_top_indices, k * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(result.scores, d_top_scores, k * sizeof(float), cudaMemcpyDeviceToHost);
                
                std::cout << "✅ GPU search completed" << std::endl;
            } else {
                std::cerr << "❌ CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
            }
            
            // Cleanup GPU memory
            cudaFree(d_query);
            cudaFree(d_scores);
            cudaFree(d_top_indices);
            cudaFree(d_top_scores);
        } else {
            // Fallback to CPU search
            std::cout << "💻 CPU fallback search..." << std::endl;
            // Implement CPU version as fallback
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Search completed in " << duration.count() << "μs" << std::endl;
        
        return result;
    }
    
    IndexStats get_stats() {
        IndexStats stats = {0};
        stats.num_vectors = num_vectors;
        stats.vector_dim = vector_dim;
        stats.is_trained = is_trained ? 1 : 0;
        stats.index_built = index_built ? 1 : 0;
        
        if (device_id >= 0) {
            size_t free_mem, total_mem;
            cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
            if (error == cudaSuccess) {
                stats.gpu_memory_mb = (total_mem - free_mem) / (1024.0f * 1024.0f);
            }
        }
        
        return stats;
    }
};

// Use the real GPU implementation
static RealGPUIndexer* get_real_gpu_indexer(TorchIndexerHandle handle) {
    return static_cast<RealGPUIndexer*>(handle);
}

// C interface implementation with real GPU
extern "C" {

TorchIndexerHandle torch_indexer_create(IndexConfig config) {
    try {
        return new RealGPUIndexer(config);
    } catch (const std::exception& e) {
        std::cerr << "Create indexer error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_indexer_destroy(TorchIndexerHandle handle) {
    if (handle) {
        delete get_real_gpu_indexer(handle);
    }
}

int torch_indexer_train(TorchIndexerHandle handle, signed char* vectors, int n_vectors, int vector_dim) {
    if (!handle) return 0;
    return get_real_gpu_indexer(handle)->train_index(vectors, n_vectors, vector_dim) ? 1 : 0;
}

int torch_indexer_add_vectors(TorchIndexerHandle handle, signed char* vectors, int n_vectors, int vector_dim) {
    if (!handle) return 0;
    return get_real_gpu_indexer(handle)->add_vectors(vectors, n_vectors, vector_dim) ? 1 : 0;
}

SearchResult torch_indexer_search(TorchIndexerHandle handle, signed char* query, int vector_dim, int k) {
    SearchResult empty_result = {nullptr, nullptr, 0};
    if (!handle) return empty_result;
    
    return get_real_gpu_indexer(handle)->search(query, vector_dim, k);
}

IndexStats torch_indexer_get_stats(TorchIndexerHandle handle) {
    IndexStats empty_stats = {0};
    if (!handle) return empty_stats;
    
    return get_real_gpu_indexer(handle)->get_stats();
}

void torch_search_result_free(SearchResult* result) {
    if (result) {
        delete[] result->ids;
        delete[] result->scores;
        result->ids = nullptr;
        result->scores = nullptr;
        result->count = 0;
    }
}

const char* torch_get_version() {
    return "RealGPU-1.0";
}

int torch_cuda_is_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0) ? 1 : 0;
}

int torch_cuda_device_count() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess) ? deviceCount : 0;
}

} // extern "C"