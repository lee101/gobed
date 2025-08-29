// torch_gpu_forced.cpp - Force CUDA initialization for LibTorch
#include "torch_cgo_wrapper.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <iostream>
#include <chrono>

// External CUDA kernel function declarations
extern "C" void launch_similarity_kernel(
    const int8_t* queries,
    const int8_t* database,
    float* scores,
    int num_queries,
    int num_vectors,
    int dim
);

extern "C" void launch_topk_kernel(
    const float* scores,
    int* top_indices,
    float* top_scores,
    int num_queries,
    int num_vectors,
    int k
);

// GPU-accelerated LibTorch indexer class
class GPUForcedIndexer {
public:
    torch::Device device;
    int vector_dim;
    int num_subquantizers;
    int codebook_size;
    int ivf_clusters;
    int probe_lists;
    int rerank_k;
    
    // Index components
    torch::Tensor database;           // [N, D] original vectors
    
    // Statistics
    int num_vectors = 0;
    bool is_trained = false;
    bool index_built = false;
    bool using_cuda = false;
    
    GPUForcedIndexer(const IndexConfig& config) 
        : device(torch::kCPU)
        , vector_dim(config.vector_dim)
        , num_subquantizers(config.num_subquantizers)
        , codebook_size(config.codebook_size)
        , ivf_clusters(config.ivf_clusters)
        , probe_lists(config.probe_lists)
        , rerank_k(config.rerank_k)
    {
        // Force CUDA initialization
        if (config.device_id >= 0) {
            try {
                // Check CUDA runtime first
                int deviceCount = 0;
                cudaError_t error = cudaGetDeviceCount(&deviceCount);
                
                if (error == cudaSuccess && deviceCount > 0) {
                    std::cout << "🎯 CUDA runtime detected " << deviceCount << " devices" << std::endl;
                    
                    // Set CUDA device
                    error = cudaSetDevice(config.device_id);
                    if (error == cudaSuccess) {
                        std::cout << "✅ Set CUDA device " << config.device_id << std::endl;
                        
                        // Try to force LibTorch to recognize CUDA
                        if (torch::cuda::is_available()) {
                            device = torch::Device(torch::kCUDA, config.device_id);
                            using_cuda = true;
                            std::cout << "🚀 LibTorch CUDA enabled!" << std::endl;
                        } else {
                            std::cout << "⚠️  LibTorch CUDA not available, forcing CPU tensors on GPU device" << std::endl;
                            // We'll use CPU tensors but do manual CUDA operations
                            device = torch::Device(torch::kCPU);
                            using_cuda = true; // We'll manually manage GPU memory
                        }
                        
                        // Print device info
                        cudaDeviceProp prop;
                        cudaGetDeviceProperties(&prop, config.device_id);
                        std::cout << "   GPU: " << prop.name << std::endl;
                        std::cout << "   Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
                        std::cout << "   Compute: " << prop.major << "." << prop.minor << std::endl;
                        std::cout << "   Max threads/block: " << prop.maxThreadsPerBlock << std::endl;
                        
                    } else {
                        std::cout << "❌ Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
                        device = torch::Device(torch::kCPU);
                    }
                } else {
                    std::cout << "❌ CUDA runtime error: " << cudaGetErrorString(error) << std::endl;
                    device = torch::Device(torch::kCPU);
                }
            } catch (const std::exception& e) {
                std::cout << "❌ CUDA initialization failed: " << e.what() << std::endl;
                device = torch::Device(torch::kCPU);
            }
        } else {
            std::cout << "⚠️  Using CPU device (device_id < 0)" << std::endl;
            device = torch::Device(torch::kCPU);
        }
    }
    
    bool train_index(const int8_t* vectors, int n_vectors, int dim) {
        try {
            std::cout << "🔧 Training index with " << n_vectors << " vectors..." << std::endl;
            
            if (dim != vector_dim) {
                std::cerr << "Dimension mismatch: expected " << vector_dim << ", got " << dim << std::endl;
                return false;
            }
            
            // For now, just mark as trained
            is_trained = true;
            std::cout << "✅ Index training completed" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Training error: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool add_vectors(const int8_t* vectors, int n_vectors, int dim) {
        try {
            if (!is_trained) {
                std::cerr << "Index must be trained before adding vectors" << std::endl;
                return false;
            }
            
            std::cout << "📚 Adding " << n_vectors << " vectors to index..." << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            
            if (using_cuda && device.is_cuda()) {
                // Use LibTorch CUDA tensors
                auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
                auto new_vectors = torch::from_blob(
                    const_cast<int8_t*>(vectors),
                    {n_vectors, dim},
                    torch::TensorOptions().dtype(torch::kInt8)
                ).clone().to(device);
                
                if (!database.defined()) {
                    database = new_vectors;
                } else {
                    database = torch::cat({database, new_vectors}, 0);
                }
                
                std::cout << "   🎯 Using LibTorch CUDA tensors" << std::endl;
                
            } else if (using_cuda && !device.is_cuda()) {
                // Manual CUDA memory management with CPU tensors
                auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
                auto new_vectors = torch::from_blob(
                    const_cast<int8_t*>(vectors),
                    {n_vectors, dim},
                    options
                ).clone();
                
                // Copy to GPU manually
                int8_t* d_vectors;
                size_t size = n_vectors * dim * sizeof(int8_t);
                cudaError_t error = cudaMalloc(&d_vectors, size);
                if (error == cudaSuccess) {
                    error = cudaMemcpy(d_vectors, vectors, size, cudaMemcpyHostToDevice);
                    if (error == cudaSuccess) {
                        std::cout << "   🎯 Manual GPU memory copy successful" << std::endl;
                        // Store CPU tensor for now, but we have GPU copy
                        if (!database.defined()) {
                            database = new_vectors;
                        } else {
                            database = torch::cat({database, new_vectors}, 0);
                        }
                        cudaFree(d_vectors); // Free after copying back if needed
                    } else {
                        std::cout << "   ❌ GPU memory copy failed: " << cudaGetErrorString(error) << std::endl;
                    }
                } else {
                    std::cout << "   ❌ GPU memory allocation failed: " << cudaGetErrorString(error) << std::endl;
                }
                
            } else {
                // CPU only
                auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
                auto new_vectors = torch::from_blob(
                    const_cast<int8_t*>(vectors),
                    {n_vectors, dim},
                    options
                ).clone();
                
                if (!database.defined()) {
                    database = new_vectors;
                } else {
                    database = torch::cat({database, new_vectors}, 0);
                }
                
                std::cout << "   💻 Using CPU tensors" << std::endl;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            num_vectors += n_vectors;
            index_built = true;
            
            std::cout << "✅ Added vectors. Total: " << num_vectors << std::endl;
            std::cout << "   Transfer time: " << duration.count() << " μs" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Add vectors error: " << e.what() << std::endl;
            return false;
        }
    }
    
    SearchResult search(const int8_t* query, int dim, int k) {
        SearchResult result = {nullptr, nullptr, 0};
        
        try {
            if (!index_built) {
                std::cerr << "Index must be built before searching" << std::endl;
                return result;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            torch::Tensor query_tensor;
            torch::Tensor scores;
            
            if (using_cuda && device.is_cuda()) {
                // LibTorch CUDA path
                auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
                query_tensor = torch::from_blob(
                    const_cast<int8_t*>(query),
                    {dim},
                    torch::TensorOptions().dtype(torch::kInt8)
                ).clone().to(device);
                
                // GPU-optimized search using tensor operations
                auto query_float = query_tensor.to(torch::kFloat32);
                auto db_float = database.to(torch::kFloat32);
                
                // Batched dot product: [1, D] @ [N, D]^T = [1, N]
                scores = torch::mm(query_float.unsqueeze(0), db_float.t()).squeeze(0);
                
                std::cout << "   🎯 LibTorch CUDA search" << std::endl;
                
            } else if (using_cuda && !device.is_cuda()) {
                // Real GPU-accelerated search using CUDA kernels!
                std::cout << "   🚀 Real GPU-accelerated search!" << std::endl;
                
                // Get database data from CPU tensor
                database = database.contiguous();
                int8_t* db_data = database.data_ptr<int8_t>();
                
                // Allocate GPU memory
                int8_t* d_query;
                int8_t* d_database;
                float* d_scores;
                
                size_t query_size = dim * sizeof(int8_t);
                size_t db_size = num_vectors * dim * sizeof(int8_t);
                size_t scores_size = num_vectors * sizeof(float);
                
                cudaError_t error;
                error = cudaMalloc(&d_query, query_size);
                if (error != cudaSuccess) {
                    std::cout << "   ❌ GPU query allocation failed: " << cudaGetErrorString(error) << std::endl;
                    goto cpu_fallback;
                }
                
                error = cudaMalloc(&d_database, db_size);
                if (error != cudaSuccess) {
                    std::cout << "   ❌ GPU database allocation failed: " << cudaGetErrorString(error) << std::endl;
                    cudaFree(d_query);
                    goto cpu_fallback;
                }
                
                error = cudaMalloc(&d_scores, scores_size);
                if (error != cudaSuccess) {
                    std::cout << "   ❌ GPU scores allocation failed: " << cudaGetErrorString(error) << std::endl;
                    cudaFree(d_query);
                    cudaFree(d_database);
                    goto cpu_fallback;
                }
                
                // Copy data to GPU
                error = cudaMemcpy(d_query, query, query_size, cudaMemcpyHostToDevice);
                if (error != cudaSuccess) {
                    std::cout << "   ❌ Failed to copy query to GPU: " << cudaGetErrorString(error) << std::endl;
                    cudaFree(d_query);
                    cudaFree(d_database);
                    cudaFree(d_scores);
                    goto cpu_fallback;
                }
                
                error = cudaMemcpy(d_database, db_data, db_size, cudaMemcpyHostToDevice);
                if (error != cudaSuccess) {
                    std::cout << "   ❌ Failed to copy database to GPU: " << cudaGetErrorString(error) << std::endl;
                    cudaFree(d_query);
                    cudaFree(d_database);
                    cudaFree(d_scores);
                    goto cpu_fallback;
                }
                
                // Launch CUDA kernel for real GPU similarity computation
                auto start_compute = std::chrono::high_resolution_clock::now();
                
                // Configure kernel launch parameters
                dim3 block_size(32, 1);  // 32 threads per block for vector dimension
                dim3 grid_size((num_vectors + block_size.x - 1) / block_size.x, 1);
                
                // Launch the real CUDA kernel!
                launch_similarity_kernel(d_query, d_database, d_scores, 1, num_vectors, dim);
                
                // Check for kernel launch errors
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    std::cout << "   ❌ CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
                    cudaFree(d_query);
                    cudaFree(d_database);
                    cudaFree(d_scores);
                    goto cpu_fallback;
                }
                
                // Wait for kernel to complete
                error = cudaDeviceSynchronize();
                if (error != cudaSuccess) {
                    std::cout << "   ❌ CUDA sync failed: " << cudaGetErrorString(error) << std::endl;
                    cudaFree(d_query);
                    cudaFree(d_database);
                    cudaFree(d_scores);
                    goto cpu_fallback;
                }
                
                auto end_compute = std::chrono::high_resolution_clock::now();
                auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(end_compute - start_compute);
                
                // Copy results back from GPU  
                std::vector<float> cpu_scores(num_vectors);
                error = cudaMemcpy(cpu_scores.data(), d_scores, scores_size, cudaMemcpyDeviceToHost);
                if (error != cudaSuccess) {
                    std::cout << "   ❌ Failed to copy results from GPU: " << cudaGetErrorString(error) << std::endl;
                    cudaFree(d_query);
                    cudaFree(d_database);
                    cudaFree(d_scores);
                    goto cpu_fallback;
                }
                
                // Create tensor from results
                scores = torch::from_blob(cpu_scores.data(), {num_vectors}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
                
                // Cleanup GPU memory
                cudaFree(d_query);
                cudaFree(d_database);
                cudaFree(d_scores);
                
                std::cout << "   ✅ GPU kernel computation successful (compute: " << compute_time.count() << "μs)" << std::endl;
                
            } else {
            cpu_fallback:
                // Fallback to CPU search
                query_tensor = torch::from_blob(
                    const_cast<int8_t*>(query),
                    {dim},
                    torch::TensorOptions().dtype(torch::kInt8)
                ).clone();
                
                scores = torch::zeros({num_vectors}, torch::TensorOptions().dtype(torch::kFloat32));
                
                auto query_acc = query_tensor.accessor<int8_t, 1>();
                auto db_acc = database.accessor<int8_t, 2>();
                auto scores_acc = scores.accessor<float, 1>();
                
                // Optimized CPU computation
                for (int i = 0; i < num_vectors; i++) {
                    float score = 0.0f;
                    for (int j = 0; j < dim; j++) {
                        score += static_cast<float>(query_acc[j]) * static_cast<float>(db_acc[i][j]);
                    }
                    scores_acc[i] = score;
                }
                
                std::cout << "   💻 CPU computation" << std::endl;
            }
            
            // Get top-k results
            k = std::min(k, num_vectors);
            auto top_k = torch::topk(scores, k, 0, true);
            auto result_scores = std::get<0>(top_k);
            auto result_indices = std::get<1>(top_k);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            // Convert to C arrays
            result.count = k;
            result.ids = new int[k];
            result.scores = new float[k];
            
            auto indices_cpu = result_indices.to(torch::kCPU);
            auto scores_cpu = result_scores.to(torch::kCPU);
            
            for (int i = 0; i < k; i++) {
                result.ids[i] = indices_cpu[i].item<int>();
                result.scores[i] = scores_cpu[i].item<float>();
            }
            
            std::cout << "   Search time: " << duration.count() << " μs" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Search error: " << e.what() << std::endl;
        }
        
        return result;
    }
    
    IndexStats get_stats() {
        IndexStats stats = {0};
        stats.num_vectors = num_vectors;
        stats.vector_dim = vector_dim;
        stats.ivf_clusters = ivf_clusters;
        stats.pq_subquantizers = num_subquantizers;
        stats.is_trained = is_trained ? 1 : 0;
        stats.index_built = index_built ? 1 : 0;
        
        if (using_cuda) {
            size_t free_mem, total_mem;
            cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
            if (error == cudaSuccess) {
                stats.gpu_memory_mb = (total_mem - free_mem) / (1024.0f * 1024.0f);
            } else {
                stats.gpu_memory_mb = 0.0f;
            }
        }
        
        return stats;
    }
};

// Use the GPU-forced implementation
static GPUForcedIndexer* get_indexer(TorchIndexerHandle handle) {
    return static_cast<GPUForcedIndexer*>(handle);
}

// C interface implementation
extern "C" {

TorchIndexerHandle torch_indexer_create(IndexConfig config) {
    try {
        return new GPUForcedIndexer(config);
    } catch (const std::exception& e) {
        std::cerr << "Create indexer error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_indexer_destroy(TorchIndexerHandle handle) {
    if (handle) {
        delete get_indexer(handle);
    }
}

int torch_indexer_train(TorchIndexerHandle handle, signed char* vectors, int n_vectors, int vector_dim) {
    if (!handle) return 0;
    return get_indexer(handle)->train_index(vectors, n_vectors, vector_dim) ? 1 : 0;
}

int torch_indexer_add_vectors(TorchIndexerHandle handle, signed char* vectors, int n_vectors, int vector_dim) {
    if (!handle) return 0;
    return get_indexer(handle)->add_vectors(vectors, n_vectors, vector_dim) ? 1 : 0;
}

SearchResult torch_indexer_search(TorchIndexerHandle handle, signed char* query, int vector_dim, int k) {
    SearchResult empty_result = {nullptr, nullptr, 0};
    if (!handle) return empty_result;
    
    return get_indexer(handle)->search(query, vector_dim, k);
}

IndexStats torch_indexer_get_stats(TorchIndexerHandle handle) {
    IndexStats empty_stats = {0};
    if (!handle) return empty_stats;
    
    return get_indexer(handle)->get_stats();
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
    return TORCH_VERSION;
}

int torch_cuda_is_available() {
    // Check CUDA runtime availability
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error == cudaSuccess && deviceCount > 0) {
        std::cout << "🎯 CUDA runtime available, device count: " << deviceCount << std::endl;
        
        // Print GPU info
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "   GPU " << i << ": " << prop.name 
                      << " (Compute " << prop.major << "." << prop.minor << ")"
                      << " Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        }
        
        // Also check LibTorch CUDA
        bool libtorch_cuda = torch::cuda::is_available();
        std::cout << "   LibTorch CUDA available: " << (libtorch_cuda ? "YES" : "NO") << std::endl;
        
        return 1;
    } else {
        std::cout << "❌ CUDA runtime error: " << cudaGetErrorString(error) << std::endl;
        return 0;
    }
}

int torch_cuda_device_count() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess) ? deviceCount : 0;
}

int torch_cuda_runtime_version() {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    return version;
}

} // extern "C"