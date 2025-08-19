package gobed

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/gobed/ann/search"
	"github.com/gobed/ann/simd"
)

// SearchEngine provides a high-level API for vector search
type SearchEngine struct {
	model       *EmbeddingModel
	index       *search.Engine
	documents   map[int]string // ID to text mapping
	config      SearchConfig
	mu          sync.RWMutex
	initialized bool
}

// SearchConfig configures the search engine
type SearchConfig struct {
	// Automatic mode - let the engine choose optimal settings
	AutoMode bool

	// Manual configuration (when AutoMode is false)
	MaxExactSearchSize int  // Use exact search below this size (default: 50000)
	NumClusters        int  // Number of IVF clusters (default: auto)
	SearchClusters     int  // Number of clusters to search (default: auto)
	UseCompression     bool // Use PQ compression for large datasets (default: auto)
	UseGraphRouting    bool // Use HNSW for centroid routing (default: auto)
	CandidatesToRerank int  // Number of candidates to rerank (default: auto)
}

// DefaultSearchConfig returns optimized default configuration
func DefaultSearchConfig() SearchConfig {
	return SearchConfig{
		AutoMode:           true,
		MaxExactSearchSize: 5000,  // Bias toward speed - use approximate search early
	}
}

// NewSearchEngine creates a new search engine
func NewSearchEngine(model *EmbeddingModel) *SearchEngine {
	return NewSearchEngineWithConfig(model, DefaultSearchConfig())
}

// NewSearchEngineWithConfig creates a search engine with custom configuration
func NewSearchEngineWithConfig(model *EmbeddingModel, config SearchConfig) *SearchEngine {
	return &SearchEngine{
		model:     model,
		documents: make(map[int]string),
		config:    config,
	}
}

// Index adds and indexes a single text with auto-generated ID
func (se *SearchEngine) Index(text string) (int, error) {
	se.mu.Lock()
	defer se.mu.Unlock()

	id := len(se.documents)
	return id, se.indexWithID(id, text)
}

// IndexWithID adds and indexes a text with a specific ID
func (se *SearchEngine) IndexWithID(id int, text string) error {
	se.mu.Lock()
	defer se.mu.Unlock()

	return se.indexWithID(id, text)
}

// IndexBatch efficiently indexes multiple texts
func (se *SearchEngine) IndexBatch(texts []string) ([]int, error) {
	ids := make([]int, len(texts))
	for i := range texts {
		ids[i] = len(se.documents) + i
	}
	
	err := se.IndexBatchWithIDs(ids, texts)
	return ids, err
}

// IndexBatchWithIDs efficiently indexes multiple texts with specific IDs
func (se *SearchEngine) IndexBatchWithIDs(ids []int, texts []string) error {
	if len(ids) != len(texts) {
		return fmt.Errorf("ids and texts must have the same length")
	}

	se.mu.Lock()
	defer se.mu.Unlock()

	// Generate embeddings
	vectors := make([]simd.Vec512, len(texts))
	scales := make([]float32, len(texts))

	for i, text := range texts {
		embedding, err := se.model.EmbedInt8(text)
		if err != nil {
			return fmt.Errorf("failed to embed text %d: %v", i, err)
		}

		copy(vectors[i][:], embedding.Vector)
		scales[i] = embedding.Scale
		se.documents[ids[i]] = text
	}

	// Initialize index if needed
	if !se.initialized {
		err := se.initializeIndex(len(se.documents) + len(texts))
		if err != nil {
			return fmt.Errorf("failed to initialize index: %v", err)
		}
	}

	// Add to index
	return se.index.AddBatch(vectors, scales, ids)
}

// Search performs semantic search and returns top K results
func (se *SearchEngine) Search(query string, k int) ([]SearchResult, error) {
	return se.SearchWithOptions(query, SearchOptions{TopK: k})
}

// SearchOptions provides advanced search options
type SearchOptions struct {
	TopK           int     // Number of results to return
	MinSimilarity  float32 // Minimum similarity threshold (0-1)
	MaxDistance    float32 // Maximum distance threshold
	IncludeVectors bool    // Include embedding vectors in results
}

// SearchWithOptions performs search with advanced options
func (se *SearchEngine) SearchWithOptions(query string, opts SearchOptions) ([]SearchResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}

	se.mu.RLock()
	defer se.mu.RUnlock()

	if !se.initialized {
		return nil, fmt.Errorf("index not initialized - add documents first")
	}

	// Generate query embedding
	embedding, err := se.model.EmbedInt8(query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %v", err)
	}

	var vec simd.Vec512
	copy(vec[:], embedding.Vector)

	// Perform search
	results, err := se.index.Search(&vec, opts.TopK)
	if err != nil {
		return nil, err
	}

	// Convert and filter results
	searchResults := make([]SearchResult, 0, len(results))
	for _, r := range results {
		similarity := r.Score / (127.0 * 127.0 * 512) // Normalize to [0,1]
		
		// Apply filters
		if opts.MinSimilarity > 0 && similarity < opts.MinSimilarity {
			continue
		}
		if opts.MaxDistance > 0 && r.Distance > opts.MaxDistance {
			continue
		}

		searchResults = append(searchResults, SearchResult{
			ID:         r.ID,
			Text:       se.documents[r.ID],
			Similarity: similarity,
			Distance:   r.Distance,
		})
	}

	return searchResults, nil
}

// FindSimilar finds documents similar to a given document ID
func (se *SearchEngine) FindSimilar(documentID int, k int) ([]SearchResult, error) {
	se.mu.RLock()
	text, exists := se.documents[documentID]
	se.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("document with ID %d not found", documentID)
	}

	// Search using the document's text
	results, err := se.Search(text, k+1) // +1 to exclude self
	if err != nil {
		return nil, err
	}

	// Filter out the query document itself
	filtered := make([]SearchResult, 0, k)
	for _, r := range results {
		if r.ID != documentID {
			filtered = append(filtered, r)
			if len(filtered) >= k {
				break
			}
		}
	}

	return filtered, nil
}

// Size returns the number of indexed documents
func (se *SearchEngine) Size() int {
	se.mu.RLock()
	defer se.mu.RUnlock()
	return len(se.documents)
}

// Clear removes all indexed documents
func (se *SearchEngine) Clear() {
	se.mu.Lock()
	defer se.mu.Unlock()

	se.documents = make(map[int]string)
	se.index = nil
	se.initialized = false
}

// GetDocument retrieves a document by ID
func (se *SearchEngine) GetDocument(id int) (string, bool) {
	se.mu.RLock()
	defer se.mu.RUnlock()
	text, exists := se.documents[id]
	return text, exists
}

// GetAllDocuments returns all indexed documents
func (se *SearchEngine) GetAllDocuments() map[int]string {
	se.mu.RLock()
	defer se.mu.RUnlock()
	
	// Return a copy to prevent external modifications
	docs := make(map[int]string, len(se.documents))
	for id, text := range se.documents {
		docs[id] = text
	}
	return docs
}

// Stats returns search engine statistics
func (se *SearchEngine) Stats() SearchEngineStats {
	se.mu.RLock()
	defer se.mu.RUnlock()

	stats := SearchEngineStats{
		NumDocuments: len(se.documents),
		Initialized:  se.initialized,
	}

	if se.initialized && se.index != nil {
		engineStats := se.index.Stats()
		stats.IndexType = engineStats.IndexType
		stats.MemoryUsageMB = float64(engineStats.MemoryUsage) / 1024 / 1024
		stats.IndexDetails = map[string]interface{}{
			"nLists":      engineStats.NLists,
			"pqEnabled":   engineStats.PQEnabled,
			"hnswEnabled": engineStats.HNSWEnabled,
		}
	}

	return stats
}

// SearchEngineStats contains engine statistics
type SearchEngineStats struct {
	NumDocuments  int
	IndexType     string
	MemoryUsageMB float64
	Initialized   bool
	IndexDetails  map[string]interface{}
}

// Private methods

func (se *SearchEngine) indexWithID(id int, text string) error {
	// Generate embedding
	embedding, err := se.model.EmbedInt8(text)
	if err != nil {
		return fmt.Errorf("failed to embed text: %v", err)
	}

	// Store document
	se.documents[id] = text

	// Initialize index if needed
	if !se.initialized {
		err := se.initializeIndex(1000) // Start with estimated size
		if err != nil {
			return err
		}
	}

	// Add to index
	var vec simd.Vec512
	copy(vec[:], embedding.Vector)
	return se.index.Add(vec, embedding.Scale, id)
}

func (se *SearchEngine) initializeIndex(estimatedSize int) error {
	config := se.generateIndexConfig(estimatedSize)
	se.index = search.NewEngine(config)
	
	// If we need IVF, we must train first
	if estimatedSize > config.MaxFlatSize && config.NList > 0 {
		// Generate training samples (can be synthetic for speed)
		trainSize := min(estimatedSize/10, 10000)
		trainVectors := make([]simd.Vec512, trainSize)
		trainScales := make([]float32, trainSize)
		
		// Generate random training vectors
		for i := 0; i < trainSize; i++ {
			for j := 0; j < 512; j++ {
				trainVectors[i][j] = int8(rand.Intn(256) - 128)
			}
			trainScales[i] = 1.0
		}
		
		// Train the index
		err := se.index.Train(trainVectors, trainScales)
		if err != nil {
			return fmt.Errorf("failed to train index: %v", err)
		}
	}
	
	se.initialized = true
	return nil
}

func (se *SearchEngine) generateIndexConfig(estimatedSize int) search.Config {
	if !se.config.AutoMode {
		// Use manual configuration
		return search.Config{
			MaxFlatSize: se.config.MaxExactSearchSize,
			NList:       se.config.NumClusters,
			NProbe:      se.config.SearchClusters,
			M:           64,
			NBits:       8,
			HNSWEnabled: se.config.UseGraphRouting,
			RerankSize:  se.config.CandidatesToRerank,
			UseParallel: true,
		}
	}

	// Auto mode - aggressively choose approximate methods for speed
	if estimatedSize <= 5000 {
		// Very small dataset - exact search is still fast
		return search.Config{
			MaxFlatSize: 10000,
			UseParallel: true,
		}
	} else if estimatedSize <= 20000 {
		// Small dataset - simple IVF with small clusters
		return search.Config{
			MaxFlatSize: 1000,
			NList:       int(float64(estimatedSize) / 50), // ~50 vectors per cluster
			NProbe:      4,  // Search only 4 clusters for speed
			HNSWEnabled: false,
			RerankSize:  64,  // Small rerank for speed
			UseParallel: true,
		}
	} else if estimatedSize <= 100000 {
		// Medium dataset - IVF with moderate settings
		return search.Config{
			MaxFlatSize: 1000,
			NList:       int(float64(estimatedSize) / 100), // ~100 vectors per cluster
			NProbe:      8,   // Still fast with 8 probes
			M:           32,  // Light PQ compression
			NBits:       8,
			HNSWEnabled: false,  // Skip HNSW for simplicity
			RerankSize:  100,
			UseParallel: true,
		}
	} else if estimatedSize <= 500000 {
		// Large dataset - IVF with HNSW routing
		return search.Config{
			MaxFlatSize: 1000,
			NList:       min(4096, int(float64(estimatedSize) / 200)), // ~200 vectors per cluster
			NProbe:      12,
			M:           64,
			NBits:       8,
			HNSWEnabled: true,  // Use HNSW for faster routing
			HNSWM:       8,     // Small graph for speed
			HNSWEfC:     100,   // Faster construction
			RerankSize:  150,
			UseParallel: true,
		}
	} else {
		// Very large dataset - maximum speed optimization
		return search.Config{
			MaxFlatSize: 1000,
			NList:       8192,  // Fixed large number of clusters
			NProbe:      16,    // Still relatively few probes
			M:           64,
			NBits:       6,     // Aggressive compression
			HNSWEnabled: true,
			HNSWM:       16,
			HNSWEfC:     200,
			RerankSize:  200,   // Moderate reranking
			UseParallel: true,
		}
	}
}

// Optimize rebuilds the index with optimal parameters for current data
func (se *SearchEngine) Optimize() error {
	se.mu.Lock()
	defer se.mu.Unlock()

	if len(se.documents) == 0 {
		return fmt.Errorf("no documents to optimize")
	}

	fmt.Printf("Optimizing index for %d documents...\n", len(se.documents))
	start := time.Now()

	// Collect all current data
	ids := make([]int, 0, len(se.documents))
	texts := make([]string, 0, len(se.documents))
	for id, text := range se.documents {
		ids = append(ids, id)
		texts = append(texts, text)
	}

	// Generate embeddings
	vectors := make([]simd.Vec512, len(texts))
	scales := make([]float32, len(texts))

	for i, text := range texts {
		embedding, err := se.model.EmbedInt8(text)
		if err != nil {
			return fmt.Errorf("failed to embed text during optimization: %v", err)
		}
		copy(vectors[i][:], embedding.Vector)
		scales[i] = embedding.Scale
	}

	// Recreate index with optimal config
	config := se.generateIndexConfig(len(se.documents))
	se.index = search.NewEngine(config)

	// Train if applicable
	if len(vectors) > config.MaxFlatSize && config.NList > 0 {
		trainSize := min(len(vectors), 100000)
		err := se.index.Train(vectors[:trainSize], scales[:trainSize])
		if err != nil {
			return fmt.Errorf("failed to train optimized index: %v", err)
		}
	}

	// Re-add all vectors
	err := se.index.AddBatch(vectors, scales, ids)
	if err != nil {
		return fmt.Errorf("failed to rebuild index: %v", err)
	}

	se.initialized = true
	fmt.Printf("Optimization completed in %v\n", time.Since(start))

	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}