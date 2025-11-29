// +build gpu

package gobed

/*
#cgo CFLAGS: -I./gpu
#cgo LDFLAGS: -L./gpu -lgpu_memory -L/usr/local/cuda-12.9/lib64 -lcudart -Wl,-rpath,./gpu -Wl,-rpath,/usr/local/cuda-12.9/lib64
#include <stdlib.h>

// GPU Memory Management C API
typedef struct {
    void* ptr;
    size_t size;
    int in_use;
} gpu_memory_block_t;

typedef struct {
    gpu_memory_block_t* blocks;
    int num_blocks;
    int max_blocks;
    size_t total_allocated;
    size_t total_available;
} gpu_memory_pool_t;

// Memory pool operations
gpu_memory_pool_t* gpu_memory_pool_create(size_t initial_size_mb);
void gpu_memory_pool_destroy(gpu_memory_pool_t* pool);
void* gpu_memory_pool_alloc(gpu_memory_pool_t* pool, size_t size);
void gpu_memory_pool_free(gpu_memory_pool_t* pool, void* ptr);
size_t gpu_memory_pool_get_usage(gpu_memory_pool_t* pool);
size_t gpu_memory_pool_get_available(gpu_memory_pool_t* pool);

// VRAM utilities
size_t gpu_get_total_vram_mb(void);
size_t gpu_get_available_vram_mb(void);
int gpu_memory_defragment(gpu_memory_pool_t* pool);
*/
import "C"
import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// GPUMemoryPool provides optimized GPU memory management for bulk operations
type GPUMemoryPool struct {
	handle    *C.gpu_memory_pool_t
	mutex     sync.RWMutex
	maxSizeMB int64
	
	// Statistics
	allocations   int64
	deallocations int64
	peakUsage     int64
	
	// Memory blocks tracking
	activeBlocks map[unsafe.Pointer]*MemoryBlock
	blockPool    sync.Pool
}

// MemoryBlock represents a managed GPU memory block
type MemoryBlock struct {
	Ptr      unsafe.Pointer
	Size     int64
	InUse    bool
	Created  time.Time
	LastUsed time.Time
}

// NewGPUMemoryPool creates a new GPU memory pool
func NewGPUMemoryPool(maxSizeMB int) *GPUMemoryPool {
	handle := C.gpu_memory_pool_create(C.size_t(maxSizeMB))
	if handle == nil {
		log.Printf("Warning: Failed to create GPU memory pool, using fallback")
		return nil
	}
	
	pool := &GPUMemoryPool{
		handle:       handle,
		maxSizeMB:    int64(maxSizeMB),
		activeBlocks: make(map[unsafe.Pointer]*MemoryBlock),
	}
	
	// Initialize block pool for recycling
	pool.blockPool = sync.Pool{
		New: func() interface{} {
			return &MemoryBlock{}
		},
	}
	
	runtime.SetFinalizer(pool, (*GPUMemoryPool).destroy)
	
	log.Printf("GPU Memory Pool created: %d MB capacity", maxSizeMB)
	return pool
}

// Alloc allocates GPU memory with automatic pool management
func (p *GPUMemoryPool) Alloc(sizeBytes int64) (unsafe.Pointer, error) {
	if p.handle == nil {
		return nil, fmt.Errorf("memory pool not initialized")
	}
	
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	// Try to allocate from pool
	ptr := C.gpu_memory_pool_alloc(p.handle, C.size_t(sizeBytes))
	if ptr == nil {
		return nil, fmt.Errorf("failed to allocate %d bytes from GPU memory pool", sizeBytes)
	}
	
	// Track allocation
	block := p.blockPool.Get().(*MemoryBlock)
	block.Ptr = ptr
	block.Size = sizeBytes
	block.InUse = true
	block.Created = time.Now()
	block.LastUsed = time.Now()
	
	p.activeBlocks[ptr] = block
	p.allocations++
	
	// Update peak usage
	currentUsage := p.getCurrentUsage()
	if currentUsage > p.peakUsage {
		p.peakUsage = currentUsage
	}
	
	return ptr, nil
}

// Free releases GPU memory back to the pool
func (p *GPUMemoryPool) Free(ptr unsafe.Pointer) error {
	if p.handle == nil || ptr == nil {
		return fmt.Errorf("invalid memory pool or pointer")
	}
	
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	// Find and remove block
	block, exists := p.activeBlocks[ptr]
	if !exists {
		return fmt.Errorf("attempt to free unknown memory block")
	}
	
	// Free memory
	C.gpu_memory_pool_free(p.handle, ptr)
	
	// Return block to pool
	block.InUse = false
	delete(p.activeBlocks, ptr)
	p.blockPool.Put(block)
	p.deallocations++
	
	return nil
}

// AllocBatch allocates multiple blocks efficiently
func (p *GPUMemoryPool) AllocBatch(sizes []int64) ([]unsafe.Pointer, error) {
	if p.handle == nil {
		return nil, fmt.Errorf("memory pool not initialized")
	}
	
	ptrs := make([]unsafe.Pointer, len(sizes))
	var err error
	
	// Allocate all blocks
	for i, size := range sizes {
		ptrs[i], err = p.Alloc(size)
		if err != nil {
			// Clean up partial allocation
			for j := 0; j < i; j++ {
				p.Free(ptrs[j])
			}
			return nil, fmt.Errorf("batch allocation failed at index %d: %w", i, err)
		}
	}
	
	return ptrs, nil
}

// FreeBatch frees multiple blocks efficiently
func (p *GPUMemoryPool) FreeBatch(ptrs []unsafe.Pointer) error {
	var firstError error
	
	for i, ptr := range ptrs {
		if err := p.Free(ptr); err != nil && firstError == nil {
			firstError = fmt.Errorf("batch free failed at index %d: %w", i, err)
		}
	}
	
	return firstError
}

// GetUsage returns current memory pool usage statistics
func (p *GPUMemoryPool) GetUsage() MemoryUsage {
	if p.handle == nil {
		return MemoryUsage{}
	}
	
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	used := int64(C.gpu_memory_pool_get_usage(p.handle))
	available := int64(C.gpu_memory_pool_get_available(p.handle))
	
	return MemoryUsage{
		TotalMB:       p.maxSizeMB,
		UsedMB:        used / (1024 * 1024),
		AvailableMB:   available / (1024 * 1024),
		PeakMB:        p.peakUsage / (1024 * 1024),
		Allocations:   p.allocations,
		Deallocations: p.deallocations,
		ActiveBlocks:  int64(len(p.activeBlocks)),
	}
}

// getCurrentUsage returns current usage in bytes (must hold mutex)
func (p *GPUMemoryPool) getCurrentUsage() int64 {
	return int64(C.gpu_memory_pool_get_usage(p.handle))
}

// Defragment performs memory defragmentation to reduce fragmentation
func (p *GPUMemoryPool) Defragment() error {
	if p.handle == nil {
		return fmt.Errorf("memory pool not initialized")
	}
	
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	result := C.gpu_memory_defragment(p.handle)
	if result == 0 {
		return fmt.Errorf("memory defragmentation failed")
	}
	
	log.Printf("GPU memory defragmentation completed")
	return nil
}

// AutoDefragment starts automatic defragmentation based on fragmentation level
func (p *GPUMemoryPool) AutoDefragment(fragmentationThreshold float64) {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		
		for range ticker.C {
			usage := p.GetUsage()
			fragmentation := float64(usage.UsedMB) / float64(usage.TotalMB)
			
			if fragmentation > fragmentationThreshold {
				if err := p.Defragment(); err != nil {
					log.Printf("Auto-defragmentation failed: %v", err)
				}
			}
		}
	}()
}

// GetOptimalBatchSize calculates optimal batch size based on available memory
func (p *GPUMemoryPool) GetOptimalBatchSize(vectorDim, bytesPerVector int) int {
	usage := p.GetUsage()
	availableBytes := usage.AvailableMB * 1024 * 1024
	
	// Reserve 20% for overhead
	usableBytes := int64(float64(availableBytes) * 0.8)
	
	// Calculate batch size
	batchSize := int(usableBytes) / bytesPerVector
	
	// Apply reasonable bounds
	if batchSize < 100 {
		batchSize = 100
	}
	if batchSize > 10000 {
		batchSize = 10000
	}
	
	return batchSize
}

// WaitForAvailableMemory waits until sufficient memory becomes available
func (p *GPUMemoryPool) WaitForAvailableMemory(requiredMB int64, timeoutSec int) error {
	timeout := time.After(time.Duration(timeoutSec) * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-timeout:
			return fmt.Errorf("timeout waiting for %d MB of GPU memory", requiredMB)
		case <-ticker.C:
			usage := p.GetUsage()
			if usage.AvailableMB >= requiredMB {
				return nil
			}
		}
	}
}

// destroy cleans up GPU resources
func (p *GPUMemoryPool) destroy() {
	if p.handle != nil {
		C.gpu_memory_pool_destroy(p.handle)
		p.handle = nil
	}
}

// Cleanup explicitly releases all resources
func (p *GPUMemoryPool) Cleanup() {
	p.destroy()
}

// MemoryUsage contains memory pool statistics
type MemoryUsage struct {
	TotalMB       int64 `json:"total_mb"`
	UsedMB        int64 `json:"used_mb"`
	AvailableMB   int64 `json:"available_mb"`
	PeakMB        int64 `json:"peak_mb"`
	Allocations   int64 `json:"allocations"`
	Deallocations int64 `json:"deallocations"`
	ActiveBlocks  int64 `json:"active_blocks"`
}

// GetAvailableVRAM returns available GPU VRAM in MB
func GetAvailableVRAM() int64 {
	return int64(C.gpu_get_available_vram_mb())
}

// GetTotalVRAM returns total GPU VRAM in MB
func GetTotalVRAM() int64 {
	return int64(C.gpu_get_total_vram_mb())
}

// GPUMemoryInfo provides comprehensive GPU memory information
type GPUMemoryInfo struct {
	TotalVRAM     int64   `json:"total_vram_mb"`
	AvailableVRAM int64   `json:"available_vram_mb"`
	UsedVRAM      int64   `json:"used_vram_mb"`
	Utilization   float64 `json:"utilization_pct"`
}

// GetGPUMemoryInfo returns detailed GPU memory information
func GetGPUMemoryInfo() GPUMemoryInfo {
	total := GetTotalVRAM()
	available := GetAvailableVRAM()
	used := total - available
	
	var utilization float64
	if total > 0 {
		utilization = float64(used) / float64(total) * 100
	}
	
	return GPUMemoryInfo{
		TotalVRAM:     total,
		AvailableVRAM: available,
		UsedVRAM:      used,
		Utilization:   utilization,
	}
}

// OptimalBatchSizeCalculator helps determine optimal batch sizes for different operations
type OptimalBatchSizeCalculator struct {
	availableVRAM int64
	overheadPct   float64
}

// NewBatchSizeCalculator creates a batch size calculator
func NewBatchSizeCalculator() *OptimalBatchSizeCalculator {
	return &OptimalBatchSizeCalculator{
		availableVRAM: GetAvailableVRAM(),
		overheadPct:   0.2, // 20% overhead
	}
}

// CalculateForEmbedding calculates optimal batch size for embedding operations
func (c *OptimalBatchSizeCalculator) CalculateForEmbedding(vectorDim, embedDim int) int {
	// Estimate memory per vector: input tokens + embedding output + intermediate calculations
	bytesPerVector := vectorDim*4 + embedDim*4 + embedDim*4 // rough estimate
	
	usableVRAM := int64(float64(c.availableVRAM) * (1.0 - c.overheadPct)) * 1024 * 1024
	batchSize := int(usableVRAM / int64(bytesPerVector))
	
	// Apply bounds
	if batchSize < 32 {
		batchSize = 32
	}
	if batchSize > 8192 {
		batchSize = 8192
	}
	
	return batchSize
}

// CalculateForIndexing calculates optimal batch size for indexing operations
func (c *OptimalBatchSizeCalculator) CalculateForIndexing(vectorDim, numClusters int) int {
	// Estimate memory: vectors + cluster assignments + distances
	bytesPerVector := vectorDim*1 + 4 + numClusters*4 // int8 vector + assignment + distances
	
	usableVRAM := int64(float64(c.availableVRAM) * (1.0 - c.overheadPct)) * 1024 * 1024
	batchSize := int(usableVRAM / int64(bytesPerVector))
	
	// Apply bounds
	if batchSize < 100 {
		batchSize = 100
	}
	if batchSize > 10000 {
		batchSize = 10000
	}
	
	return batchSize
}

// MonitorMemoryPressure monitors GPU memory pressure and provides warnings
func MonitorMemoryPressure(thresholdPct float64, callback func(float64)) {
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		
		for range ticker.C {
			info := GetGPUMemoryInfo()
			if info.Utilization > thresholdPct {
				callback(info.Utilization)
			}
		}
	}()
}