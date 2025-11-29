package search

import gobed "github.com/lee101/gobed"

// Re-export the high level search engine API from the root gobed package so
// consumers depending on github.com/lee101/gobed/pkg/search get the same
// production-ready GPU-optimized implementation.
type (
	SearchEngine  = gobed.SearchEngine
	SearchConfig  = gobed.SearchConfig
	SearchOptions = gobed.SearchOptions
	IndexRequest  = gobed.IndexRequest
	IndexResponse = gobed.IndexResponse
	IndexingStats = gobed.IndexingStats
)

var (
	DefaultSearchConfig       = gobed.DefaultSearchConfig
	AsyncSearchConfig         = gobed.AsyncSearchConfig
	GPUSearchConfig           = gobed.GPUSearchConfig
	AutoOptimizedSearchConfig = gobed.AutoOptimizedSearchConfig
	NewAutoSearchEngine       = gobed.NewAutoSearchEngine
	FastSearchEngine          = gobed.FastSearchEngine
)

func NewSearchEngine(model *EmbeddingModel) *SearchEngine {
	return gobed.NewSearchEngine(model)
}

func NewSearchEngineWithConfig(model *EmbeddingModel, config SearchConfig) *SearchEngine {
	return gobed.NewSearchEngineWithConfig(model, config)
}

func NewAsyncSearchEngine(model *EmbeddingModel) *SearchEngine {
	return gobed.NewAsyncSearchEngine(model)
}

func NewGPUSearchEngine(model *EmbeddingModel) *SearchEngine {
	return gobed.NewGPUSearchEngine(model)
}

func NewCAGRASearchEngine(model *EmbeddingModel) *SearchEngine {
	return gobed.NewCAGRASearchEngine(model)
}
