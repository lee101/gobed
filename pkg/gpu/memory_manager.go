package gobed

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

/*
#cgo CFLAGS: -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include -I./gpu
#cgo LDFLAGS: -L./gpu -ltorch_cgo_wrapper -L/usr/local/cuda/lib64 -lcudart
#include "torch_cgo_wrapper.h"
#include <cuda_runtime_api.h>
#include <stdlib.h>

// GPU memory management functions
static size_t cuda_get_free_memory() {
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        return free_mem;
    }
    return 0;
}

static size_t cuda_get_total_memory() {
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        return total_mem;
    }
    return 0;
}
*/
import "C"

// GPUMemoryManager manages CUDA memory for high-performance indexing and search
type GPUMemoryManager struct {
	deviceID    int
	totalMemory uint64
	
	// Memory pools for different object types
	vectorPool     *GPUBlockPool
	queryPool      *GPUBlockPool
	resultPool     *GPUBlockPool
	
	// Memory tracking
	allocatedBytes uint64
	peakUsage      uint64
	allocationCount uint64
	
	// Configuration
	maxMemoryUsage uint64 // Maximum memory to use (bytes)
	reserveMemory  uint64 // Memory to keep free for system
	
	// Synchronization
	mutex sync.RWMutex
}

// GPUBlockPool manages a pool of GPU memory blocks
type GPUBlockPool struct {
	blockSize    uint64
	maxBlocks    int
	freeBlocks   []unsafe.Pointer
	allocBlocks  map[unsafe.Pointer]bool
	totalAlloc   uint64
	mutex        sync.Mutex
	allocCount   uint64
	freeCount    uint64
}

// GPUMemoryConfig configures the GPU memory manager
type GPUMemoryConfig struct {
	DeviceID              int
	MaxMemoryUsagePercent float64 // Percentage of GPU memory to use
	VectorPoolBlockSize   uint64  // Size of vector memory blocks
	QueryPoolBlockSize    uint64  // Size of query memory blocks  
	ResultPoolBlockSize   uint64  // Size of result memory blocks
	MaxVectorBlocks       int     // Maximum vector blocks
	MaxQueryBlocks        int     // Maximum query blocks
	MaxResultBlocks       int     // Maximum result blocks
	ReserveMemoryMB       uint64  // Memory to reserve for system (MB)
}

// DefaultGPUMemoryConfig returns optimized GPU memory configuration
func DefaultGPUMemoryConfig() GPUMemoryConfig {
	return GPUMemoryConfig{
		DeviceID:              0,
		MaxMemoryUsagePercent: 0.85, // Use 85% of GPU memory
		VectorPoolBlockSize:   1024 * 1024 * 4,  // 4MB blocks for vectors
		QueryPoolBlockSize:    1024 * 512,       // 512KB blocks for queries
		ResultPoolBlockSize:   1024 * 256,       // 256KB blocks for results
		MaxVectorBlocks:       1000,              // Up to 4GB for vectors
		MaxQueryBlocks:        100,               // Up to 50MB for queries
		MaxResultBlocks:       100,               // Up to 25MB for results
		ReserveMemoryMB:       1024,              // Reserve 1GB for system
	}
}

// NewGPUMemoryManager creates a new GPU memory manager
func NewGPUMemoryManager(config GPUMemoryConfig) (*GPUMemoryManager, error) {
	if !IsCUDAAvailable() {
		return nil, fmt.Errorf("CUDA is not available")
	}

	// Get GPU memory info
	totalMemory := uint64(C.cuda_get_total_memory())
	if totalMemory == 0 {
		return nil, fmt.Errorf("failed to get GPU memory info")
	}

	maxUsage := uint64(float64(totalMemory) * config.MaxMemoryUsagePercent)
	reserveMemory := config.ReserveMemoryMB * 1024 * 1024

	manager := &GPUMemoryManager{
		deviceID:       config.DeviceID,
		totalMemory:    totalMemory,
		maxMemoryUsage: maxUsage,
		reserveMemory:  reserveMemory,
	}

	// Initialize memory pools
	var err error
	
	manager.vectorPool, err = newGPUBlockPool(config.VectorPoolBlockSize, config.MaxVectorBlocks)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector pool: %w", err)
	}

	manager.queryPool, err = newGPUBlockPool(config.QueryPoolBlockSize, config.MaxQueryBlocks)
	if err != nil {
		manager.vectorPool.destroy()
		return nil, fmt.Errorf("failed to create query pool: %w", err)
	}

	manager.resultPool, err = newGPUBlockPool(config.ResultPoolBlockSize, config.MaxResultBlocks)
	if err != nil {
		manager.vectorPool.destroy()
		manager.queryPool.destroy()
		return nil, fmt.Errorf("failed to create result pool: %w", err)
	}

	// Set finalizer
	runtime.SetFinalizer(manager, (*GPUMemoryManager).destroy)

	return manager, nil
}

// newGPUBlockPool creates a new GPU memory pool
func newGPUBlockPool(blockSize uint64, maxBlocks int) (*GPUBlockPool, error) {
	pool := &GPUBlockPool{
		blockSize:   blockSize,
		maxBlocks:   maxBlocks,
		freeBlocks:  make([]unsafe.Pointer, 0, maxBlocks),
		allocBlocks: make(map[unsafe.Pointer]bool, maxBlocks),
	}

	// Pre-allocate more blocks for better performance
	initialBlocks := maxBlocks / 2
	if initialBlocks > 20 {
		initialBlocks = 20
	}

	for i := 0; i < initialBlocks; i++ {
		block, err := pool.allocateBlock()
		if err != nil {
			pool.destroy()
			return nil, err
		}
		pool.freeBlocks = append(pool.freeBlocks, block)
	}

	return pool, nil
}

// allocateBlock allocates a single GPU memory block
func (pool *GPUBlockPool) allocateBlock() (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	
	// Allocate GPU memory using CUDA runtime
	result := C.cudaMalloc((*unsafe.Pointer)(&ptr), C.size_t(pool.blockSize))
	if result != C.cudaSuccess {
		return nil, fmt.Errorf("GPU memory allocation failed: %d", result)
	}

	atomic.AddUint64(&pool.totalAlloc, pool.blockSize)
	atomic.AddUint64(&pool.allocCount, 1)
	
	return ptr, nil
}

// GetBlock gets a memory block from the pool
func (pool *GPUBlockPool) GetBlock() (unsafe.Pointer, error) {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()

	// Try to reuse a free block
	if len(pool.freeBlocks) > 0 {
		block := pool.freeBlocks[len(pool.freeBlocks)-1]
		pool.freeBlocks = pool.freeBlocks[:len(pool.freeBlocks)-1]
		pool.allocBlocks[block] = true
		return block, nil
	}

	// Allocate new block if under limit
	if len(pool.allocBlocks) < pool.maxBlocks {
		block, err := pool.allocateBlock()
		if err != nil {
			return nil, err
		}
		pool.allocBlocks[block] = true
		return block, nil
	}

	return nil, fmt.Errorf("memory pool exhausted (max blocks: %d)", pool.maxBlocks)
}

// ReturnBlock returns a memory block to the pool
func (pool *GPUBlockPool) ReturnBlock(block unsafe.Pointer) {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()

	if !pool.allocBlocks[block] {
		return // Not from this pool
	}

	delete(pool.allocBlocks, block)
	pool.freeBlocks = append(pool.freeBlocks, block)
	atomic.AddUint64(&pool.freeCount, 1)
}

// destroy cleans up the memory pool
func (pool *GPUBlockPool) destroy() {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()

	// Free all allocated blocks
	for block := range pool.allocBlocks {
		C.cudaFree(block)
	}
	
	for _, block := range pool.freeBlocks {
		C.cudaFree(block)
	}

	pool.allocBlocks = nil
	pool.freeBlocks = nil
}

// AllocateVectorMemory allocates GPU memory for vectors
func (m *GPUMemoryManager) AllocateVectorMemory() (unsafe.Pointer, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.checkMemoryLimits(m.vectorPool.blockSize) {
		return nil, fmt.Errorf("GPU memory limit exceeded")
	}

	block, err := m.vectorPool.GetBlock()
	if err != nil {
		return nil, err
	}

	atomic.AddUint64(&m.allocatedBytes, m.vectorPool.blockSize)
	m.updatePeakUsage()
	atomic.AddUint64(&m.allocationCount, 1)

	return block, nil
}

// FreeVectorMemory frees GPU vector memory
func (m *GPUMemoryManager) FreeVectorMemory(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	m.vectorPool.ReturnBlock(ptr)
	atomic.AddUint64(&m.allocatedBytes, ^uint64(m.vectorPool.blockSize-1))
}

// AllocateQueryMemory allocates GPU memory for queries
func (m *GPUMemoryManager) AllocateQueryMemory() (unsafe.Pointer, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.checkMemoryLimits(m.queryPool.blockSize) {
		return nil, fmt.Errorf("GPU memory limit exceeded")
	}

	block, err := m.queryPool.GetBlock()
	if err != nil {
		return nil, err
	}

	atomic.AddUint64(&m.allocatedBytes, m.queryPool.blockSize)
	m.updatePeakUsage()
	atomic.AddUint64(&m.allocationCount, 1)

	return block, nil
}

// FreeQueryMemory frees GPU query memory
func (m *GPUMemoryManager) FreeQueryMemory(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	m.queryPool.ReturnBlock(ptr)
	atomic.AddUint64(&m.allocatedBytes, ^uint64(m.queryPool.blockSize-1))
}

// AllocateResultMemory allocates GPU memory for results
func (m *GPUMemoryManager) AllocateResultMemory() (unsafe.Pointer, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.checkMemoryLimits(m.resultPool.blockSize) {
		return nil, fmt.Errorf("GPU memory limit exceeded")
	}

	block, err := m.resultPool.GetBlock()
	if err != nil {
		return nil, err
	}

	atomic.AddUint64(&m.allocatedBytes, m.resultPool.blockSize)
	m.updatePeakUsage()
	atomic.AddUint64(&m.allocationCount, 1)

	return block, nil
}

// FreeResultMemory frees GPU result memory
func (m *GPUMemoryManager) FreeResultMemory(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	m.resultPool.ReturnBlock(ptr)
	atomic.AddUint64(&m.allocatedBytes, ^uint64(m.resultPool.blockSize-1))
}

// checkMemoryLimits checks if allocation would exceed limits
func (m *GPUMemoryManager) checkMemoryLimits(size uint64) bool {
	currentUsage := atomic.LoadUint64(&m.allocatedBytes)
	freeMemory := uint64(C.cuda_get_free_memory())

	// Check against our configured limits
	if currentUsage+size > m.maxMemoryUsage {
		return false
	}

	// Check against actual GPU memory
	if size+m.reserveMemory > freeMemory {
		return false
	}

	return true
}

// updatePeakUsage updates peak memory usage tracking
func (m *GPUMemoryManager) updatePeakUsage() {
	currentUsage := atomic.LoadUint64(&m.allocatedBytes)
	for {
		peak := atomic.LoadUint64(&m.peakUsage)
		if currentUsage <= peak {
			break
		}
		if atomic.CompareAndSwapUint64(&m.peakUsage, peak, currentUsage) {
			break
		}
	}
}

// GetMemoryStats returns detailed memory statistics
func (m *GPUMemoryManager) GetMemoryStats() GPUMemoryStats {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	return GPUMemoryStats{
		TotalMemoryGB:     float64(m.totalMemory) / 1024 / 1024 / 1024,
		FreeMemoryGB:      float64(C.cuda_get_free_memory()) / 1024 / 1024 / 1024,
		AllocatedGB:       float64(atomic.LoadUint64(&m.allocatedBytes)) / 1024 / 1024 / 1024,
		PeakUsageGB:       float64(atomic.LoadUint64(&m.peakUsage)) / 1024 / 1024 / 1024,
		MaxUsageGB:        float64(m.maxMemoryUsage) / 1024 / 1024 / 1024,
		AllocationCount:   atomic.LoadUint64(&m.allocationCount),
		VectorPoolStats:   m.getPoolStats(m.vectorPool, "vectors"),
		QueryPoolStats:    m.getPoolStats(m.queryPool, "queries"),
		ResultPoolStats:   m.getPoolStats(m.resultPool, "results"),
	}
}

// GPUMemoryStats provides detailed GPU memory statistics
type GPUMemoryStats struct {
	TotalMemoryGB     float64
	FreeMemoryGB      float64
	AllocatedGB       float64
	PeakUsageGB       float64
	MaxUsageGB        float64
	AllocationCount   uint64
	VectorPoolStats   PoolStats
	QueryPoolStats    PoolStats
	ResultPoolStats   PoolStats
}

// PoolStats provides statistics for a memory pool
type PoolStats struct {
	Name          string
	BlockSizeMB   float64
	MaxBlocks     int
	AllocBlocks   int
	FreeBlocks    int
	TotalAllocGB  float64
	AllocCount    uint64
	FreeCount     uint64
}

// getPoolStats returns statistics for a memory pool
func (m *GPUMemoryManager) getPoolStats(pool *GPUBlockPool, name string) PoolStats {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()

	return PoolStats{
		Name:          name,
		BlockSizeMB:   float64(pool.blockSize) / 1024 / 1024,
		MaxBlocks:     pool.maxBlocks,
		AllocBlocks:   len(pool.allocBlocks),
		FreeBlocks:    len(pool.freeBlocks),
		TotalAllocGB:  float64(atomic.LoadUint64(&pool.totalAlloc)) / 1024 / 1024 / 1024,
		AllocCount:    atomic.LoadUint64(&pool.allocCount),
		FreeCount:     atomic.LoadUint64(&pool.freeCount),
	}
}

// ForceGarbageCollection forces GPU memory garbage collection
func (m *GPUMemoryManager) ForceGarbageCollection() {
	// Clear free blocks to reduce memory footprint
	m.vectorPool.mutex.Lock()
	for _, block := range m.vectorPool.freeBlocks {
		C.cudaFree(block)
	}
	m.vectorPool.freeBlocks = m.vectorPool.freeBlocks[:0]
	m.vectorPool.mutex.Unlock()

	m.queryPool.mutex.Lock()
	for _, block := range m.queryPool.freeBlocks {
		C.cudaFree(block)
	}
	m.queryPool.freeBlocks = m.queryPool.freeBlocks[:0]
	m.queryPool.mutex.Unlock()

	m.resultPool.mutex.Lock()
	for _, block := range m.resultPool.freeBlocks {
		C.cudaFree(block)
	}
	m.resultPool.freeBlocks = m.resultPool.freeBlocks[:0]
	m.resultPool.mutex.Unlock()

	// Force CUDA context synchronization
	C.cudaDeviceSynchronize()
}

// StartMemoryMonitor starts a background goroutine to monitor GPU memory usage
func (m *GPUMemoryManager) StartMemoryMonitor(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for range ticker.C {
			stats := m.GetMemoryStats()
			
			// Log memory usage if high
			usagePercent := stats.AllocatedGB / stats.TotalMemoryGB * 100
			if usagePercent > 70 {
				fmt.Printf("High GPU memory usage: %.1f%% (%.2fGB/%.2fGB)\n",
					usagePercent, stats.AllocatedGB, stats.TotalMemoryGB)
			}

			// Trigger garbage collection if memory is getting low
			if stats.FreeMemoryGB < 1.0 { // Less than 1GB free
				fmt.Printf("Triggering GPU memory garbage collection\n")
				m.ForceGarbageCollection()
			}
		}
	}()
}

// Close gracefully shuts down the memory manager
func (m *GPUMemoryManager) Close() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.vectorPool.destroy()
	m.queryPool.destroy()
	m.resultPool.destroy()

	runtime.SetFinalizer(m, nil)
	
	fmt.Printf("🧠 GPU Memory Manager closed\n")
	return nil
}

// destroy is called by finalizer
func (m *GPUMemoryManager) destroy() {
	m.Close()
}