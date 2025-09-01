// GPU Memory Pool Implementation for Ultra-Fast Operations
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

extern "C" {

// GPU Memory Block
typedef struct {
    void* ptr;
    size_t size;
    int in_use;
} gpu_memory_block_t;

// GPU Memory Pool
typedef struct {
    gpu_memory_block_t* blocks;
    int num_blocks;
    int max_blocks;
    size_t total_allocated;
    size_t total_available;
} gpu_memory_pool_t;

// Create memory pool with initial size in MB
gpu_memory_pool_t* gpu_memory_pool_create(size_t initial_size_mb) {
    gpu_memory_pool_t* pool = (gpu_memory_pool_t*)malloc(sizeof(gpu_memory_pool_t));
    if (!pool) return nullptr;
    
    pool->max_blocks = 1024;  // Max 1024 memory blocks
    pool->blocks = (gpu_memory_block_t*)calloc(pool->max_blocks, sizeof(gpu_memory_block_t));
    pool->num_blocks = 0;
    pool->total_allocated = 0;
    pool->total_available = initial_size_mb * 1024 * 1024;
    
    // Pre-allocate initial GPU memory
    void* initial_mem;
    size_t initial_bytes = initial_size_mb * 1024 * 1024;
    cudaError_t err = cudaMalloc(&initial_mem, initial_bytes);
    
    if (err == cudaSuccess && pool->num_blocks < pool->max_blocks) {
        pool->blocks[0].ptr = initial_mem;
        pool->blocks[0].size = initial_bytes;
        pool->blocks[0].in_use = 0;
        pool->num_blocks = 1;
        pool->total_allocated = initial_bytes;
    }
    
    return pool;
}

// Destroy memory pool
void gpu_memory_pool_destroy(gpu_memory_pool_t* pool) {
    if (!pool) return;
    
    // Free all GPU memory blocks
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i].ptr) {
            cudaFree(pool->blocks[i].ptr);
        }
    }
    
    free(pool->blocks);
    free(pool);
}

// Allocate memory from pool
void* gpu_memory_pool_alloc(gpu_memory_pool_t* pool, size_t size) {
    if (!pool) return nullptr;
    
    // First try to find existing free block
    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->blocks[i].in_use && pool->blocks[i].size >= size) {
            pool->blocks[i].in_use = 1;
            pool->total_available -= size;
            return pool->blocks[i].ptr;
        }
    }
    
    // Allocate new block if no suitable free block found
    if (pool->num_blocks < pool->max_blocks) {
        void* new_mem;
        cudaError_t err = cudaMalloc(&new_mem, size);
        if (err == cudaSuccess) {
            pool->blocks[pool->num_blocks].ptr = new_mem;
            pool->blocks[pool->num_blocks].size = size;
            pool->blocks[pool->num_blocks].in_use = 1;
            pool->num_blocks++;
            pool->total_allocated += size;
            return new_mem;
        }
    }
    
    return nullptr;
}

// Free memory back to pool
void gpu_memory_pool_free(gpu_memory_pool_t* pool, void* ptr) {
    if (!pool || !ptr) return;
    
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i].ptr == ptr) {
            pool->blocks[i].in_use = 0;
            pool->total_available += pool->blocks[i].size;
            return;
        }
    }
}

// Get current memory usage
size_t gpu_memory_pool_get_usage(gpu_memory_pool_t* pool) {
    if (!pool) return 0;
    
    size_t used = 0;
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i].in_use) {
            used += pool->blocks[i].size;
        }
    }
    return used;
}

// Get available memory
size_t gpu_memory_pool_get_available(gpu_memory_pool_t* pool) {
    if (!pool) return 0;
    return pool->total_available;
}

// Get total GPU VRAM in MB
size_t gpu_get_total_vram_mb(void) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    return prop.totalGlobalMem / (1024 * 1024);
}

// Get available GPU VRAM in MB
size_t gpu_get_available_vram_mb(void) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem / (1024 * 1024);
}

// Defragment memory pool
int gpu_memory_defragment(gpu_memory_pool_t* pool) {
    if (!pool) return 0;
    
    // Simple defragmentation: mark all unused blocks as available
    int freed_blocks = 0;
    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->blocks[i].in_use && pool->blocks[i].ptr) {
            // Could coalesce adjacent blocks here for better efficiency
            freed_blocks++;
        }
    }
    
    return freed_blocks;
}

} // extern "C"