//go:build !gpu || !cgo
// +build !gpu !cgo

package gobed

import (
	"fmt"
	"time"
	"unsafe"
)

type GPUMemoryManager struct {
	deviceID    int
	totalMemory uint64
}

type GPUBlockPool struct {
	blockSize uint64
	maxBlocks int
}

type GPUMemoryConfig struct {
	DeviceID              int
	MaxMemoryUsagePercent float64
	VectorPoolBlockSize   uint64
	QueryPoolBlockSize    uint64
	ResultPoolBlockSize   uint64
	MaxVectorBlocks       int
	MaxQueryBlocks        int
	MaxResultBlocks       int
	ReserveMemoryMB       uint64
}

func DefaultGPUMemoryConfig() GPUMemoryConfig {
	return GPUMemoryConfig{
		DeviceID:              0,
		MaxMemoryUsagePercent: 0.85,
		VectorPoolBlockSize:   1024 * 1024 * 4,
		QueryPoolBlockSize:    1024 * 512,
		ResultPoolBlockSize:   1024 * 256,
		MaxVectorBlocks:       1000,
		MaxQueryBlocks:        100,
		MaxResultBlocks:       100,
		ReserveMemoryMB:       1024,
	}
}

func NewGPUMemoryManager(config GPUMemoryConfig) (*GPUMemoryManager, error) {
	return nil, fmt.Errorf("GPU support not compiled in - build with -tags gpu")
}

func (m *GPUMemoryManager) AllocateVectorMemory() (unsafe.Pointer, error) {
	return nil, fmt.Errorf("GPU support not compiled in")
}

func (m *GPUMemoryManager) FreeVectorMemory(ptr unsafe.Pointer) {}

func (m *GPUMemoryManager) AllocateQueryMemory() (unsafe.Pointer, error) {
	return nil, fmt.Errorf("GPU support not compiled in")
}

func (m *GPUMemoryManager) FreeQueryMemory(ptr unsafe.Pointer) {}

func (m *GPUMemoryManager) AllocateResultMemory() (unsafe.Pointer, error) {
	return nil, fmt.Errorf("GPU support not compiled in")
}

func (m *GPUMemoryManager) FreeResultMemory(ptr unsafe.Pointer) {}

type GPUMemoryStats struct {
	TotalMemoryGB   float64
	FreeMemoryGB    float64
	AllocatedGB     float64
	PeakUsageGB     float64
	MaxUsageGB      float64
	AllocationCount uint64
	VectorPoolStats PoolStats
	QueryPoolStats  PoolStats
	ResultPoolStats PoolStats
}

type PoolStats struct {
	Name         string
	BlockSizeMB  float64
	MaxBlocks    int
	AllocBlocks  int
	FreeBlocks   int
	TotalAllocGB float64
	AllocCount   uint64
	FreeCount    uint64
}

func (m *GPUMemoryManager) GetMemoryStats() GPUMemoryStats {
	return GPUMemoryStats{}
}

func (m *GPUMemoryManager) ForceGarbageCollection() {}

func (m *GPUMemoryManager) StartMemoryMonitor(interval time.Duration) {}

func (m *GPUMemoryManager) Close() error {
	return nil
}
