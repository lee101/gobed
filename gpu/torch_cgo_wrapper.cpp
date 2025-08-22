// torch_cgo_wrapper.cpp - C++ implementation for Go CGO LibTorch integration
#include "torch_cgo_wrapper.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>

// Forward declarations for custom CUDA operations
namespace torch {
namespace ops {
namespace gobed_ann {
    torch::Tensor i8dot512_scores(const torch::Tensor& q, const torch::Tensor& db);
    torch::Tensor build_pq_lut(const torch::Tensor& q_rot, const torch::Tensor& cb);
    torch::Tensor adc_scan(const torch::Tensor& codes, const torch::Tensor& lut);
}
}
}

// Internal LibTorch indexer class
class LibTorchIndexer {
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
    torch::Tensor ivf_centroids;      // [num_clusters, D] IVF centroids  
    torch::Tensor pq_codebooks;       // [M, K, D/M] PQ codebooks
    torch::Tensor quantized_codes;    // [N, M] quantized codes
    torch::Tensor ivf_lists;          // List assignments for each vector
    
    // Statistics
    int num_vectors = 0;
    bool is_trained = false;
    bool index_built = false;
    
    LibTorchIndexer(const IndexConfig& config) 
        : device(torch::kCPU)
        , vector_dim(config.vector_dim)
        , num_subquantizers(config.num_subquantizers)
        , codebook_size(config.codebook_size)
        , ivf_clusters(config.ivf_clusters)
        , probe_lists(config.probe_lists)
        , rerank_k(config.rerank_k)
    {
        if (torch::cuda::is_available() && config.device_id >= 0) {
            device = torch::Device(torch::kCUDA, config.device_id);
            std::cout << "🎯 Using CUDA device " << config.device_id << std::endl;
        } else {
            device = torch::Device(torch::kCPU);
            std::cout << "⚠️  Using CPU device" << std::endl;
        }
    }
    
    bool train_index(const int8_t* vectors, int n_vectors, int dim) {
        try {
            std::cout << "🔧 Training index with " << n_vectors << " vectors..." << std::endl;
            
            if (dim != vector_dim) {
                std::cerr << "Dimension mismatch: expected " << vector_dim << ", got " << dim << std::endl;
                return false;
            }
            
            // Create training tensor
            auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
            auto training_vectors = torch::from_blob(
                const_cast<int8_t*>(vectors), 
                {n_vectors, dim}, 
                options
            ).clone(); // Clone to ensure memory safety
            
            // 1. Train IVF centroids
            std::cout << "   Training IVF centroids..." << std::endl;
            ivf_centroids = train_ivf_centroids(training_vectors);
            
            // 2. Train PQ codebooks  
            std::cout << "   Training PQ codebooks..." << std::endl;
            pq_codebooks = train_pq_codebooks(training_vectors);
            
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
            
            auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
            auto new_vectors = torch::from_blob(
                const_cast<int8_t*>(vectors),
                {n_vectors, dim},
                options
            ).clone();
            
            // Store original vectors for reranking
            if (!database.defined()) {
                database = new_vectors;
            } else {
                database = torch::cat({database, new_vectors}, 0);
            }
            
            // Assign to IVF lists
            auto ivf_assignments = assign_to_ivf_lists(new_vectors);
            if (!ivf_lists.defined()) {
                ivf_lists = ivf_assignments;
            } else {
                // Adjust indices for concatenation
                auto adjusted_assignments = ivf_assignments + num_vectors;
                ivf_lists = torch::cat({ivf_lists, adjusted_assignments}, 0);
            }
            
            // Quantize using PQ
            auto new_codes = quantize_vectors(new_vectors);
            if (!quantized_codes.defined()) {
                quantized_codes = new_codes;
            } else {
                quantized_codes = torch::cat({quantized_codes, new_codes}, 0);
            }
            
            num_vectors += n_vectors;
            index_built = true;
            
            std::cout << "✅ Added vectors. Total: " << num_vectors << std::endl;
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
            
            auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
            auto query_tensor = torch::from_blob(
                const_cast<int8_t*>(query),
                {dim},
                options
            ).clone();
            
            // Stage 1: IVF probe to get candidate lists
            auto candidate_ids = probe_ivf_lists(query_tensor);
            
            if (candidate_ids.numel() == 0) {
                return result; // Empty result
            }
            
            // Stage 2: PQ-based scoring of candidates
            auto pq_scores = score_with_pq(query_tensor, candidate_ids);
            
            // Stage 3: Select top candidates for reranking
            int rerank_count = std::min(rerank_k, static_cast<int>(candidate_ids.numel()));
            auto top_indices = std::get<1>(torch::topk(pq_scores, rerank_count, 0, true));
            auto rerank_candidates = candidate_ids.index_select(0, top_indices);
            
            // Stage 4: Exact reranking using original vectors
            auto final_scores = exact_rerank(query_tensor, rerank_candidates);
            
            // Return top-k results
            k = std::min(k, static_cast<int>(rerank_candidates.numel()));
            auto top_k = torch::topk(final_scores, k, 0, true);
            auto result_scores = std::get<0>(top_k);
            auto result_indices = std::get<1>(top_k);
            auto result_ids = rerank_candidates.index_select(0, result_indices);
            
            // Convert to C arrays
            result.count = k;
            result.ids = new int[k];
            result.scores = new float[k];
            
            auto ids_cpu = result_ids.to(torch::kCPU);
            auto scores_cpu = result_scores.to(torch::kCPU);
            
            for (int i = 0; i < k; i++) {
                result.ids[i] = ids_cpu[i].item<int>();
                result.scores[i] = scores_cpu[i].item<float>();
            }
            
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
        
        if (torch::cuda::is_available()) {
            // Simple memory estimation
            stats.gpu_memory_mb = 100.0f; // Placeholder
        }
        
        return stats;
    }
    
private:
    torch::Tensor train_ivf_centroids(const torch::Tensor& vectors) {
        int n = vectors.size(0);
        int d = vectors.size(1);
        int k = ivf_clusters;
        
        // Initialize centroids randomly
        auto indices = torch::randperm(n, device).slice(0, 0, k);
        auto centroids = vectors.index_select(0, indices).to(torch::kFloat);
        
        // K-means iterations
        for (int iter = 0; iter < 20; iter++) {
            // Assign vectors to closest centroids  
            auto distances = torch::zeros({n, k}, device);
            for (int i = 0; i < k; i++) {
                auto centroid_int8 = centroids[i].round().clamp(-128, 127).to(torch::kInt8);
                auto scores = torch::ops::gobed_ann::i8dot512_scores(centroid_int8, vectors);
                distances.slice(1, i, i+1) = -scores.unsqueeze(1); // Convert similarity to distance
            }
            
            auto assignments = distances.argmin(1);
            
            // Update centroids
            auto new_centroids = torch::zeros_like(centroids);
            for (int i = 0; i < k; i++) {
                auto mask = assignments == i;
                auto count = mask.sum().item<int>();
                if (count > 0) {
                    new_centroids[i] = vectors.masked_select(mask.unsqueeze(1).expand_as(vectors))
                                             .view({count, d}).to(torch::kFloat).mean(0);
                } else {
                    new_centroids[i] = centroids[i]; // Keep old centroid if no assignments
                }
            }
            centroids = new_centroids;
        }
        
        return centroids.round().clamp(-128, 127).to(torch::kInt8);
    }
    
    torch::Tensor train_pq_codebooks(const torch::Tensor& vectors) {
        int n = vectors.size(0);
        int d = vectors.size(1);
        int m = num_subquantizers;
        int k = codebook_size;
        int subvec_dim = d / m;
        
        auto codebooks = torch::zeros({m, k, subvec_dim}, 
                                    torch::TensorOptions().dtype(torch::kInt8).device(device));
        
        // Train each subquantizer independently
        for (int i = 0; i < m; i++) {
            int start_idx = i * subvec_dim;
            int end_idx = (i + 1) * subvec_dim;
            auto subvectors = vectors.slice(1, start_idx, end_idx).to(torch::kFloat);
            
            // K-means for this subquantizer
            auto indices = torch::randperm(n, device).slice(0, 0, k);
            auto centroids = subvectors.index_select(0, indices);
            
            for (int iter = 0; iter < 10; iter++) {
                // Assign to closest centroids
                auto distances = torch::cdist(subvectors, centroids);
                auto assignments = distances.argmin(1);
                
                // Update centroids
                auto new_centroids = torch::zeros_like(centroids);
                for (int j = 0; j < k; j++) {
                    auto mask = assignments == j;
                    auto count = mask.sum().item<int>();
                    if (count > 0) {
                        new_centroids[j] = subvectors.masked_select(mask.unsqueeze(1).expand_as(subvectors))
                                                   .view({count, subvec_dim}).mean(0);
                    } else {
                        new_centroids[j] = centroids[j];
                    }
                }
                centroids = new_centroids;
            }
            
            codebooks[i] = centroids.round().clamp(-128, 127).to(torch::kInt8);
        }
        
        return codebooks;
    }
    
    torch::Tensor assign_to_ivf_lists(const torch::Tensor& vectors) {
        int n = vectors.size(0);
        auto assignments = torch::zeros({n}, torch::TensorOptions().dtype(torch::kLong).device(device));
        
        auto best_scores = torch::full({n}, -std::numeric_limits<float>::infinity(), 
                                     torch::TensorOptions().dtype(torch::kFloat).device(device));
        
        for (int i = 0; i < ivf_clusters; i++) {
            auto scores = torch::ops::gobed_ann::i8dot512_scores(ivf_centroids[i], vectors);
            auto mask = scores > best_scores;
            assignments.masked_fill_(mask, i);
            best_scores = torch::max(best_scores, scores);
        }
        
        return assignments;
    }
    
    torch::Tensor quantize_vectors(const torch::Tensor& vectors) {
        int n = vectors.size(0);
        int d = vectors.size(1);
        int m = num_subquantizers;
        int k = codebook_size;
        int subvec_dim = d / m;
        
        auto codes = torch::zeros({n, m}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
        
        for (int i = 0; i < m; i++) {
            int start_idx = i * subvec_dim;
            int end_idx = (i + 1) * subvec_dim;
            auto subvectors = vectors.slice(1, start_idx, end_idx);
            
            auto codebook = pq_codebooks[i]; // [K, subvec_dim]
            
            auto best_codes = torch::zeros({n}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
            auto best_scores = torch::full({n}, -std::numeric_limits<float>::infinity(),
                                         torch::TensorOptions().dtype(torch::kFloat).device(device));
            
            for (int j = 0; j < k; j++) {
                auto codeword = codebook[j];
                auto scores = (subvectors.to(torch::kInt) * codeword.to(torch::kInt)).sum(1).to(torch::kFloat);
                auto mask = scores > best_scores;
                best_codes.masked_fill_(mask, j);
                best_scores = torch::max(best_scores, scores);
            }
            
            codes.slice(1, i, i+1) = best_codes.unsqueeze(1);
        }
        
        return codes;
    }
    
    torch::Tensor probe_ivf_lists(const torch::Tensor& query) {
        auto scores = torch::zeros({ivf_clusters}, device);
        for (int i = 0; i < ivf_clusters; i++) {
            auto score = torch::ops::gobed_ann::i8dot512_scores(ivf_centroids[i].unsqueeze(0), query.unsqueeze(0));
            scores[i] = score[0];
        }
        
        int probe_count = std::min(probe_lists, ivf_clusters);
        auto top_lists = std::get<1>(torch::topk(scores, probe_count, 0, true));
        
        std::vector<torch::Tensor> candidates;
        for (int i = 0; i < probe_count; i++) {
            int list_id = top_lists[i].item<int>();
            auto mask = ivf_lists == list_id;
            auto indices = torch::nonzero(mask).squeeze(1);
            if (indices.numel() > 0) {
                candidates.push_back(indices);
            }
        }
        
        if (candidates.empty()) {
            return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(device));
        }
        
        return torch::cat(candidates, 0);
    }
    
    torch::Tensor score_with_pq(const torch::Tensor& query, const torch::Tensor& candidate_ids) {
        // Build PQ lookup table
        auto lut = torch::ops::gobed_ann::build_pq_lut(query, pq_codebooks);
        
        // Get codes for candidates
        auto candidate_codes = quantized_codes.index_select(0, candidate_ids);
        
        // Compute ADC scores
        auto scores = torch::ops::gobed_ann::adc_scan(lut, candidate_codes);
        
        return scores;
    }
    
    torch::Tensor exact_rerank(const torch::Tensor& query, const torch::Tensor& candidate_ids) {
        auto candidates = database.index_select(0, candidate_ids);
        auto scores = torch::ops::gobed_ann::i8dot512_scores(query, candidates);
        return scores;
    }
};

// C interface implementation
extern "C" {

TorchIndexerHandle torch_indexer_create(IndexConfig config) {
    try {
        return new LibTorchIndexer(config);
    } catch (const std::exception& e) {
        std::cerr << "Create indexer error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_indexer_destroy(TorchIndexerHandle handle) {
    if (handle) {
        delete static_cast<LibTorchIndexer*>(handle);
    }
}

int torch_indexer_train(TorchIndexerHandle handle, signed char* vectors, int n_vectors, int vector_dim) {
    if (!handle) return 0;
    auto* indexer = static_cast<LibTorchIndexer*>(handle);
    return indexer->train_index(vectors, n_vectors, vector_dim) ? 1 : 0;
}

int torch_indexer_add_vectors(TorchIndexerHandle handle, signed char* vectors, int n_vectors, int vector_dim) {
    if (!handle) return 0;
    auto* indexer = static_cast<LibTorchIndexer*>(handle);
    return indexer->add_vectors(vectors, n_vectors, vector_dim) ? 1 : 0;
}

SearchResult torch_indexer_search(TorchIndexerHandle handle, signed char* query, int vector_dim, int k) {
    SearchResult empty_result = {nullptr, nullptr, 0};
    if (!handle) return empty_result;
    
    auto* indexer = static_cast<LibTorchIndexer*>(handle);
    return indexer->search(query, vector_dim, k);
}

IndexStats torch_indexer_get_stats(TorchIndexerHandle handle) {
    IndexStats empty_stats = {0};
    if (!handle) return empty_stats;
    
    auto* indexer = static_cast<LibTorchIndexer*>(handle);
    return indexer->get_stats();
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
    return torch::cuda::is_available() ? 1 : 0;
}

int torch_cuda_device_count() {
    return torch::cuda::device_count();
}

} // extern "C"