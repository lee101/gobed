package search

import gobed "github.com/lee101/gobed"

// Type aliases to reuse optimized implementations from the root gobed package.
type (
	EmbeddingModel       = gobed.EmbeddingModel
	EmbedInt8Result      = gobed.EmbedInt8Result
	ObjectPool           = gobed.ObjectPool
	BatchProcessor       = gobed.BatchProcessor
	MemoryOptimizedCache = gobed.MemoryOptimizedCache
	SearchResult         = gobed.SearchResult
)

var (
	NewObjectPool           = gobed.NewObjectPool
	NewBatchProcessor       = gobed.NewBatchProcessor
	NewMemoryOptimizedCache = gobed.NewMemoryOptimizedCache
)

// IsCUDAAvailable defers to the core runtime detection from the gobed package.
func IsCUDAAvailable() bool {
	return gobed.IsCUDAAvailable()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
