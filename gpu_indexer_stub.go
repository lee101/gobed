// +build !gpu

package gobed

import (
	"fmt"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// GPUIndexer stub for non-GPU builds
type GPUIndexer struct {
	vectorDim int
	deviceID  int
	isReady   bool
}

// IndexConfig stub
type IndexConfig struct {
	VectorDim        int
	NumSubquantizers int
	CodebookSize     int
	IVFClusters      int
	ProbeLists       int
	RerankK          int
	DeviceID         int
}

// NewGPUIndexer returns error for non-GPU builds
func NewGPUIndexer(config IndexConfig) (*GPUIndexer, error) {
	return nil, fmt.Errorf("GPU support not compiled in - build with -tags gpu")
}

// Initialize returns error for non-GPU builds
func (g *GPUIndexer) Initialize() error {
	return fmt.Errorf("GPU support not compiled in - build with -tags gpu")
}

// IndexVectors returns error for non-GPU builds
func (g *GPUIndexer) IndexVectors(vectors []simd.Vec512, scales []float32) error {
	return fmt.Errorf("GPU support not compiled in - build with -tags gpu")
}

// Search returns error for non-GPU builds
func (g *GPUIndexer) Search(query simd.Vec512, scale float32, k int) ([]int, []float32, error) {
	return nil, nil, fmt.Errorf("GPU support not compiled in - build with -tags gpu")
}

// BatchSearch returns error for non-GPU builds
func (g *GPUIndexer) BatchSearch(queries [][]int8, k int) ([][]SearchResult, error) {
	return nil, fmt.Errorf("GPU support not compiled in - build with -tags gpu")
}

// AddVectors returns error for non-GPU builds
func (g *GPUIndexer) AddVectors(vectors [][]int8) error {
	return fmt.Errorf("GPU support not compiled in - build with -tags gpu")
}

// Close is a no-op for non-GPU builds
func (g *GPUIndexer) Close() error {
	// No-op
	return nil
}

// IsReady returns false for non-GPU builds
func (g *GPUIndexer) IsReady() bool {
	return false
}

// IndexStats provides indexer statistics
type IndexStats struct {
	NumVectors      int
	VectorDim       int
	IVFClusters     int
	PQSubquantizers int
	GPUMemoryMB     float32
	IsTrained       bool
	IndexBuilt      bool
}

// GetStats returns empty stats for non-GPU builds
func (g *GPUIndexer) GetStats() IndexStats {
	return IndexStats{
		NumVectors:  0,
		GPUMemoryMB: 0,
		IsTrained:   false,
		IndexBuilt:  false,
	}
}

// IsCUDAAvailable returns false for non-GPU builds
func IsCUDAAvailable() bool {
	return false
}

// GetCUDAVersion returns empty string for non-GPU builds
func GetCUDAVersion() string {
	return "N/A (GPU support not compiled in)"
}

// GetCUDADeviceCount returns 0 for non-GPU builds
func GetCUDADeviceCount() int {
	return 0
}

// DefaultGPUConfig returns a default configuration for GPU indexing (stub)
func DefaultGPUConfig() IndexConfig {
	return IndexConfig{
		VectorDim:        512,  // Updated to match 512 dimension
		NumSubquantizers: 8,
		CodebookSize:     256,
		IVFClusters:      1024,
		ProbeLists:       64,
		RerankK:          1000,
		DeviceID:         0,
	}
}

// TrainIndex returns error for non-GPU builds
func (g *GPUIndexer) TrainIndex(vectors [][]int8) error {
	return fmt.Errorf("GPU support not compiled in - build with -tags gpu")
}

// GetMemoryUsage returns 0 for non-GPU builds
func (g *GPUIndexer) GetMemoryUsage() uint64 {
	return 0
}