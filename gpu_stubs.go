// +build !gpu,!cuda

package gobed

import "time"

// GPU stubs for CPU-only builds

// GPUBatchProcessor is a stub for CPU builds
type GPUBatchProcessor struct {
	enabled bool
}

// BatchResult represents the result of batch processing
type BatchResult struct {
	Embeddings [][]float32
	Time       time.Duration
}

// GPUStats represents GPU statistics
type GPUStats struct {
	DeviceID     int
	MemoryUsed   int64
	MemoryTotal  int64
	Utilization  float32
	Temperature  float32
}

// NewGPUBatchProcessor returns a stub for CPU builds
func NewGPUBatchProcessor(model *EmbeddingModel, cache *TokenPatternCache) *GPUBatchProcessor {
	return &GPUBatchProcessor{enabled: false}
}

// ProcessBatch processes a batch (CPU fallback)
func (g *GPUBatchProcessor) ProcessBatch(texts []string) ([]*EmbedInt8Result, error) {
	// Return nil to indicate CPU processing should be used
	return nil, nil
}

// GetStats returns empty stats for CPU builds
func (g *GPUBatchProcessor) GetStats() *GPUStats {
	return &GPUStats{}
}

// IsCUDAAvailable always returns false for CPU builds
func IsCUDAAvailable() bool {
	return false
}