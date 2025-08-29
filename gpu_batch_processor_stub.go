// +build !gpu

package gobed

import (
	"fmt"
)

// GPUBatchProcessor stub for non-GPU builds
type GPUBatchProcessor struct {
	model *EmbeddingModel
	cache *TokenPatternCache
}

// NewGPUBatchProcessor returns a stub for non-GPU builds
func NewGPUBatchProcessor(model *EmbeddingModel, cache *TokenPatternCache) *GPUBatchProcessor {
	return &GPUBatchProcessor{
		model: model,
		cache: cache,
	}
}

// ProcessBatch returns error for non-GPU builds
func (p *GPUBatchProcessor) ProcessBatch(texts []string) ([]*EmbedInt8Result, error) {
	// Fallback to sequential CPU processing
	results := make([]*EmbedInt8Result, len(texts))
	for i, text := range texts {
		result, err := p.model.EmbedInt8(text)
		if err != nil {
			return nil, fmt.Errorf("failed to embed text %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// Shutdown is a no-op for non-GPU builds
func (p *GPUBatchProcessor) Shutdown() {
	// No-op
}

// GetMetrics returns empty metrics for non-GPU builds
func (p *GPUBatchProcessor) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"gpu_enabled": false,
		"message": "GPU batch processing not available",
	}
}

// GetStats returns empty stats for non-GPU builds
func (p *GPUBatchProcessor) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"gpu_enabled": false,
		"batch_size": 0,
		"processed": 0,
	}
}