// Persistent CUDA index with concurrent operations using streams
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

// Persistent index with stream-based concurrency
struct PersistentIndex {
    // Document storage
    int8_t* d_documents;      // All document embeddings
    int* d_doc_ids;          // Document IDs (for tracking)
    bool* d_doc_valid;       // Valid flags for documents

    // Index metadata
    int max_docs;
    int dim;
    std::atomic<int> active_docs;
    std::atomic<int> next_slot;  // Next free slot for insertion

    // Stream management for concurrent operations
    cudaStream_t index_stream;    // For indexing operations
    cudaStream_t search_stream;   // For search operations
    cudaStream_t update_stream;   // For updates/deletions

    // Host-side tracking
    std::unordered_map<int, int> doc_to_slot;  // docID -> slot in GPU array
    std::mutex index_mutex;

    // Memory pools for operations
    float* d_temp_scores;     // Temporary scores for search
    int* d_temp_indices;      // Temporary indices for search

    // CUBLAS handle for optimized operations
    cublasHandle_t cublas_handle;
};

extern "C" {

// Kernel for parallel int8 dot product with validity check
__global__ void persistent_search_kernel(
    const int8_t* query,
    const int8_t* documents,
    const bool* valid_flags,
    float* scores,
    int num_docs,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_docs) return;

    // Skip invalid documents
    if (!valid_flags[idx]) {
        scores[idx] = -1e9f;  // Very negative score
        return;
    }

    const int8_t* doc = documents + idx * dim;
    int32_t dot = 0;

    // Vectorized dot product using warp-level primitives
    #pragma unroll 8
    for (int i = 0; i < dim; i += 8) {
        // Process 8 elements at once
        int4 query_vals = *reinterpret_cast<const int4*>(query + i);
        int4 doc_vals = *reinterpret_cast<const int4*>(doc + i);

        // Unpack and compute
        dot += ((char*)&query_vals)[0] * ((char*)&doc_vals)[0];
        dot += ((char*)&query_vals)[1] * ((char*)&doc_vals)[1];
        dot += ((char*)&query_vals)[2] * ((char*)&doc_vals)[2];
        dot += ((char*)&query_vals)[3] * ((char*)&doc_vals)[3];
        dot += ((char*)&query_vals)[4] * ((char*)&doc_vals)[4];
        dot += ((char*)&query_vals)[5] * ((char*)&doc_vals)[5];
        dot += ((char*)&query_vals)[6] * ((char*)&doc_vals)[6];
        dot += ((char*)&query_vals)[7] * ((char*)&doc_vals)[7];
    }

    scores[idx] = (float)dot;
}

// Kernel for batch document updates
__global__ void batch_update_kernel(
    int8_t* documents,
    bool* valid_flags,
    const int8_t* new_docs,
    const int* slots,
    int num_updates,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_updates) return;

    int slot = slots[idx];

    // Copy new document to slot
    int8_t* dest = documents + slot * dim;
    const int8_t* src = new_docs + idx * dim;

    // Vectorized copy
    for (int i = threadIdx.y; i < dim; i += blockDim.y) {
        dest[i] = src[i];
    }

    // Mark as valid
    if (threadIdx.y == 0) {
        valid_flags[slot] = true;
    }
}

// Kernel for marking documents as deleted
__global__ void mark_deleted_kernel(
    bool* valid_flags,
    const int* slots,
    int num_deletes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_deletes) return;

    valid_flags[slots[idx]] = false;
}

// Kernel for compaction (defragmentation)
__global__ void compact_documents_kernel(
    int8_t* documents,
    bool* valid_flags,
    int* compaction_map,
    int num_docs,
    int dim
) {
    // This would compact valid documents to remove gaps
    // Implementation depends on specific requirements
}

// Create persistent index
void* cuda_persistent_index_create(int max_docs, int dim) {
    PersistentIndex* idx = new PersistentIndex();

    idx->max_docs = max_docs;
    idx->dim = dim;
    idx->active_docs = 0;
    idx->next_slot = 0;

    // Allocate GPU memory
    size_t doc_size = max_docs * dim * sizeof(int8_t);
    cudaMalloc(&idx->d_documents, doc_size);
    cudaMalloc(&idx->d_doc_ids, max_docs * sizeof(int));
    cudaMalloc(&idx->d_doc_valid, max_docs * sizeof(bool));

    // Initialize validity flags to false
    cudaMemset(idx->d_doc_valid, 0, max_docs * sizeof(bool));

    // Allocate temporary buffers
    cudaMalloc(&idx->d_temp_scores, max_docs * sizeof(float));
    cudaMalloc(&idx->d_temp_indices, max_docs * sizeof(int));

    // Create streams for concurrent operations
    cudaStreamCreate(&idx->index_stream);
    cudaStreamCreate(&idx->search_stream);
    cudaStreamCreate(&idx->update_stream);

    // Create CUBLAS handle
    cublasCreate(&idx->cublas_handle);
    cublasSetStream(idx->cublas_handle, idx->search_stream);

    return idx;
}

// Destroy persistent index
void cuda_persistent_index_destroy(void* handle) {
    PersistentIndex* idx = (PersistentIndex*)handle;

    cudaFree(idx->d_documents);
    cudaFree(idx->d_doc_ids);
    cudaFree(idx->d_doc_valid);
    cudaFree(idx->d_temp_scores);
    cudaFree(idx->d_temp_indices);

    cudaStreamDestroy(idx->index_stream);
    cudaStreamDestroy(idx->search_stream);
    cudaStreamDestroy(idx->update_stream);

    cublasDestroy(idx->cublas_handle);

    delete idx;
}

// Update index with new documents (can run concurrently with search)
int cuda_persistent_index_update(void* handle, const int8_t* vectors, int* doc_ids, int num_docs) {
    PersistentIndex* idx = (PersistentIndex*)handle;

    // Lock for slot allocation
    std::lock_guard<std::mutex> lock(idx->index_mutex);

    // Allocate slots for new documents
    std::vector<int> slots;
    slots.reserve(num_docs);

    for (int i = 0; i < num_docs; i++) {
        int doc_id = doc_ids[i];

        // Check if document already exists
        auto it = idx->doc_to_slot.find(doc_id);
        int slot;

        if (it != idx->doc_to_slot.end()) {
            // Update existing
            slot = it->second;
        } else {
            // Allocate new slot
            slot = idx->next_slot.fetch_add(1);
            if (slot >= idx->max_docs) {
                // Need compaction or resize
                idx->next_slot = idx->max_docs;
                return i; // Return number successfully added
            }
            idx->doc_to_slot[doc_id] = slot;
            idx->active_docs++;
        }

        slots.push_back(slot);
    }

    // Copy slots to device
    int* d_slots;
    cudaMallocAsync(&d_slots, slots.size() * sizeof(int), idx->index_stream);
    cudaMemcpyAsync(d_slots, slots.data(), slots.size() * sizeof(int),
                    cudaMemcpyHostToDevice, idx->index_stream);

    // Copy documents to device
    int8_t* d_new_docs;
    size_t doc_size = num_docs * idx->dim * sizeof(int8_t);
    cudaMallocAsync(&d_new_docs, doc_size, idx->index_stream);
    cudaMemcpyAsync(d_new_docs, vectors, doc_size,
                    cudaMemcpyHostToDevice, idx->index_stream);

    // Launch update kernel on index stream
    dim3 block(32, 8);
    dim3 grid((num_docs + 31) / 32, 1);

    batch_update_kernel<<<grid, block, 0, idx->index_stream>>>(
        idx->d_documents,
        idx->d_doc_valid,
        d_new_docs,
        d_slots,
        num_docs,
        idx->dim
    );

    // Async cleanup
    cudaFreeAsync(d_slots, idx->index_stream);
    cudaFreeAsync(d_new_docs, idx->index_stream);

    // Note: Stream will complete asynchronously
    return num_docs;
}

// Remove documents from index
int cuda_persistent_index_remove(void* handle, int* doc_ids, int num_docs) {
    PersistentIndex* idx = (PersistentIndex*)handle;

    std::lock_guard<std::mutex> lock(idx->index_mutex);

    std::vector<int> slots_to_delete;

    for (int i = 0; i < num_docs; i++) {
        auto it = idx->doc_to_slot.find(doc_ids[i]);
        if (it != idx->doc_to_slot.end()) {
            slots_to_delete.push_back(it->second);
            idx->doc_to_slot.erase(it);
            idx->active_docs--;
        }
    }

    if (!slots_to_delete.empty()) {
        // Copy slots to device
        int* d_slots;
        cudaMallocAsync(&d_slots, slots_to_delete.size() * sizeof(int), idx->update_stream);
        cudaMemcpyAsync(d_slots, slots_to_delete.data(),
                        slots_to_delete.size() * sizeof(int),
                        cudaMemcpyHostToDevice, idx->update_stream);

        // Mark as deleted
        int threads = 256;
        int blocks = (slots_to_delete.size() + threads - 1) / threads;

        mark_deleted_kernel<<<blocks, threads, 0, idx->update_stream>>>(
            idx->d_doc_valid,
            d_slots,
            slots_to_delete.size()
        );

        cudaFreeAsync(d_slots, idx->update_stream);
    }

    return slots_to_delete.size();
}

// Search index (can run concurrently with updates)
int cuda_persistent_index_search(void* handle, const int8_t* query, int k, int* out_indices, float* out_scores) {
    PersistentIndex* idx = (PersistentIndex*)handle;

    int num_docs = idx->next_slot.load();
    if (num_docs == 0) return 0;

    // Copy query to device on search stream
    int8_t* d_query;
    cudaMallocAsync(&d_query, idx->dim * sizeof(int8_t), idx->search_stream);
    cudaMemcpyAsync(d_query, query, idx->dim * sizeof(int8_t),
                    cudaMemcpyHostToDevice, idx->search_stream);

    // Launch search kernel on search stream
    int threads = 256;
    int blocks = (num_docs + threads - 1) / threads;

    persistent_search_kernel<<<blocks, threads, 0, idx->search_stream>>>(
        d_query,
        idx->d_documents,
        idx->d_doc_valid,
        idx->d_temp_scores,
        num_docs,
        idx->dim
    );

    // Use Thrust to find top-k on search stream
    thrust::device_ptr<float> scores_ptr(idx->d_temp_scores);
    thrust::device_ptr<int> indices_ptr(idx->d_temp_indices);

    // Initialize indices
    thrust::sequence(thrust::cuda::par.on(idx->search_stream),
                     indices_ptr, indices_ptr + num_docs);

    // Sort by scores (descending)
    thrust::sort_by_key(thrust::cuda::par.on(idx->search_stream),
                        scores_ptr, scores_ptr + num_docs,
                        indices_ptr,
                        thrust::greater<float>());

    // Copy top-k results
    int result_count = min(k, idx->active_docs.load());
    cudaMemcpyAsync(out_indices, idx->d_temp_indices,
                    result_count * sizeof(int),
                    cudaMemcpyDeviceToHost, idx->search_stream);
    cudaMemcpyAsync(out_scores, idx->d_temp_scores,
                    result_count * sizeof(float),
                    cudaMemcpyDeviceToHost, idx->search_stream);

    // Sync search stream
    cudaStreamSynchronize(idx->search_stream);

    // Cleanup
    cudaFreeAsync(d_query, idx->search_stream);

    return result_count;
}

// Get index statistics
int cuda_persistent_index_get_stats(void* handle, int* active_docs, int* capacity, float* gpu_memory_mb) {
    PersistentIndex* idx = (PersistentIndex*)handle;

    *active_docs = idx->active_docs.load();
    *capacity = idx->max_docs;

    // Calculate GPU memory usage
    size_t total_bytes =
        idx->max_docs * idx->dim * sizeof(int8_t) +  // documents
        idx->max_docs * sizeof(int) +                // doc_ids
        idx->max_docs * sizeof(bool) +               // valid flags
        idx->max_docs * sizeof(float) +              // temp scores
        idx->max_docs * sizeof(int);                 // temp indices

    *gpu_memory_mb = (float)total_bytes / (1024.0f * 1024.0f);

    return 0;
}

// Compact index to remove gaps (maintenance operation)
int cuda_persistent_index_compact(void* handle) {
    PersistentIndex* idx = (PersistentIndex*)handle;

    // This would reorganize documents to remove gaps
    // Best done during low activity periods
    // Implementation depends on specific requirements

    return 0;
}

} // extern "C"