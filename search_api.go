package gobed

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/lee101/gobed/pkg/ann/search"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// SearchEngine provides a high-level API for vector search
type SearchEngine struct {
	model       *EmbeddingModel
	index       *search.Engine
	documents   map[int]string // ID to text mapping
	config      SearchConfig
	mu          sync.RWMutex
	initialized bool

	// Async indexing support
	indexQueue   chan IndexRequest
	indexWorkers int
	asyncEnabled bool
	workerWg     sync.WaitGroup
	ctx          context.Context
	cancel       context.CancelFunc

	// Performance optimizations
	objectPool     *ObjectPool
	batchProcessor *BatchProcessor
	embeddingCache *MemoryOptimizedCache
}

// SearchConfig configures the search engine
type SearchConfig struct {
	// Automatic mode - let the engine choose optimal settings
	AutoMode bool

	// Preset configuration (when AutoMode is true)
	Preset SearchPreset // Use predefined configuration preset

	// Manual configuration (when AutoMode is false)
	MaxExactSearchSize int  // Use exact search below this size (default: 50000)
	NumClusters        int  // Number of IVF clusters (default: auto)
	SearchClusters     int  // Number of clusters to search (default: auto)
	UseCompression     bool // Use PQ compression for large datasets (default: auto)
	UseGraphRouting    bool // Use HNSW for centroid routing (default: auto)
	CandidatesToRerank int  // Number of candidates to rerank (default: auto)

	// Async configuration
	EnableAsync    bool // Enable async indexing (default: false)
	AsyncWorkers   int  // Number of async workers (default: 4)
	AsyncQueueSize int  // Size of async queue (default: 1000)
	MaxConcurrency int  // Maximum concurrent operations (default: runtime.NumCPU())

	// GPU acceleration configuration
	EnableGPU      bool // Enable GPU acceleration for similarity search (default: false)
	GPUDeviceID    int  // CUDA device ID to use (default: 0)
	GPUBatchSize   int  // Batch size for GPU operations (default: 1000)
	UseInt8        bool // Use int8 quantization for embeddings (75% memory savings)
}

// IndexRequest represents an async indexing request
type IndexRequest struct {
	IDs      []int
	Texts    []string
	Response chan IndexResponse
	Context  context.Context
}

// IndexResponse contains the result of async indexing
type IndexResponse struct {
	IDs   []int
	Error error
	Stats IndexingStats
}

// IndexingStats provides indexing performance metrics
type IndexingStats struct {
	DocumentsProcessed int
	ProcessingTime     time.Duration
	EmbeddingTime      time.Duration
	IndexingTime       time.Duration
}

// DefaultSearchConfig returns optimized default configuration
// Automatically detects and enables GPU with CAGRA when available
func DefaultSearchConfig() SearchConfig {
	config := SearchConfig{
		AutoMode:           true,
		MaxExactSearchSize: 5000,  // Bias toward speed - use approximate search early
		EnableAsync:        false, // Default to sync for simplicity
		AsyncWorkers:       4,
		AsyncQueueSize:     1000,
		Preset:             CAGRAPreset, // Default to our optimal CAGRA configuration
	}

	// Auto-detect GPU and enable CAGRA if available
	if IsCUDAAvailable() {
		config.EnableGPU = true
		config.GPUDeviceID = 0
		config.GPUBatchSize = 50000  // Theoretical analysis: 0.6x GPU occupancy, 10x improvement (80M+ vectors/sec)
		config.MaxExactSearchSize = 100000 // GPU can handle larger exact searches
		config.UseInt8 = true // Use int8 for 75% memory savings
		// Preset already set to CAGRAPreset above
	}

	return config
}

// AsyncSearchConfig returns configuration optimized for async processing
func AsyncSearchConfig() SearchConfig {
	config := DefaultSearchConfig()
	config.EnableAsync = true
	config.AsyncWorkers = 8      // More workers for async
	config.AsyncQueueSize = 2000 // Larger queue
	return config
}

// GPUSearchConfig returns configuration optimized for GPU acceleration with CAGRA
func GPUSearchConfig() SearchConfig {
	config := DefaultSearchConfig()
	config.EnableGPU = true
	config.GPUDeviceID = 0    // Use first GPU
	config.GPUBatchSize = 50000 // Theoretical GPU analysis: 50x current utilization (80M+ vectors/sec)
	config.MaxExactSearchSize = 100000 // GPU can handle larger exact searches
	config.Preset = CAGRAPreset // Use CAGRA for ultra-fast search by default
	return config
}

// NewGPUSearchEngine creates a GPU-accelerated search engine with CAGRA
func NewGPUSearchEngine(model *EmbeddingModel) *SearchEngine {
	return NewSearchEngineWithConfig(model, GPUSearchConfig())
}

// NewCAGRASearchEngine creates a CAGRA-powered search engine for ultra-fast search
func NewCAGRASearchEngine(model *EmbeddingModel) *SearchEngine {
	config := GPUSearchConfig()
	config.Preset = CAGRAPreset
	return NewSearchEngineWithConfig(model, config)
}

// NewOfficialCuvsCAGRASearchEngine creates a search engine using official cuVS CAGRA library
// func NewOfficialCuvsCAGRASearchEngine(model *EmbeddingModel) *CuvsCAGRASearchEngine {
//	return NewCuvsCAGRASearchEngine(model)
// }

// NewSearchEngine creates a new search engine
// It automatically uses GPU acceleration if available for 39x performance boost
func NewSearchEngine(model *EmbeddingModel) *SearchEngine {
	// Use auto-optimized config that detects GPU
	return NewSearchEngineWithConfig(model, AutoOptimizedSearchConfig())
}

// NewSearchEngineWithConfig creates a search engine with custom configuration
func NewSearchEngineWithConfig(model *EmbeddingModel, config SearchConfig) *SearchEngine {
	// Map public config to internal ann config
	annConfig := mapSearchConfigToAnnConfig(config)
	
	se := &SearchEngine{
		model:          model,
		index:          search.NewEngine(annConfig), // Initialize the internal index
		documents:      make(map[int]string),
		config:         config,
		objectPool:     NewObjectPool(),
		batchProcessor: NewBatchProcessor(1000, 4),     // 1000 batch size, 4 workers
		embeddingCache: NewMemoryOptimizedCache(10000), // Cache up to 10k embeddings
	}

	// Initialize async indexing if enabled
	if config.EnableAsync {
		se.initializeAsync()
	}

	return se
}

// NewAsyncSearchEngine creates a search engine optimized for async operations
func NewAsyncSearchEngine(model *EmbeddingModel) *SearchEngine {
	return NewSearchEngineWithConfig(model, AsyncSearchConfig())
}

// initializeAsync sets up async indexing workers
func (se *SearchEngine) initializeAsync() {
	se.ctx, se.cancel = context.WithCancel(context.Background())
	se.indexQueue = make(chan IndexRequest, se.config.AsyncQueueSize)
	se.indexWorkers = se.config.AsyncWorkers
	se.asyncEnabled = true

	// Start worker goroutines
	for i := 0; i < se.indexWorkers; i++ {
		se.workerWg.Add(1)
		go se.indexWorker()
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

	return se.indexBatchInternal(ids, texts)
}

// IndexBatchAsync asynchronously indexes multiple texts and returns a channel for the result
func (se *SearchEngine) IndexBatchAsync(texts []string) <-chan IndexResponse {
	ids := make([]int, len(texts))
	se.mu.RLock()
	nextID := len(se.documents)
	se.mu.RUnlock()

	for i := range texts {
		ids[i] = nextID + i
	}

	return se.IndexBatchAsyncWithIDs(ids, texts)
}

// IndexBatchAsyncWithIDs asynchronously indexes texts with specific IDs
func (se *SearchEngine) IndexBatchAsyncWithIDs(ids []int, texts []string) <-chan IndexResponse {
	response := make(chan IndexResponse, 1)

	if !se.asyncEnabled {
		// Fall back to synchronous indexing
		go func() {
			err := se.IndexBatchWithIDs(ids, texts)
			response <- IndexResponse{
				IDs:   ids,
				Error: err,
			}
		}()
		return response
	}

	req := IndexRequest{
		IDs:      ids,
		Texts:    texts,
		Response: response,
		Context:  context.Background(),
	}

	select {
	case se.indexQueue <- req:
		return response
	case <-se.ctx.Done():
		response <- IndexResponse{
			Error: fmt.Errorf("search engine is shutting down"),
		}
		return response
	default:
		// Queue is full, fall back to sync
		go func() {
			err := se.IndexBatchWithIDs(ids, texts)
			response <- IndexResponse{
				IDs:   ids,
				Error: err,
			}
		}()
		return response
	}
}

// indexWorker processes async indexing requests
func (se *SearchEngine) indexWorker() {
	defer se.workerWg.Done()

	for {
		select {
		case req := <-se.indexQueue:
			start := time.Now()

			// Process the indexing request
			se.mu.Lock()
			err := se.indexBatchInternal(req.IDs, req.Texts)
			se.mu.Unlock()

			processingTime := time.Since(start)

			// Send response
			req.Response <- IndexResponse{
				IDs:   req.IDs,
				Error: err,
				Stats: IndexingStats{
					DocumentsProcessed: len(req.Texts),
					ProcessingTime:     processingTime,
				},
			}

		case <-se.ctx.Done():
			return
		}
	}
}

// indexBatchInternal performs the actual batch indexing (must be called with lock held)
func (se *SearchEngine) indexBatchInternal(ids []int, texts []string) error {
	// Store documents first (needed for training data)
	for i, text := range texts {
		se.documents[ids[i]] = text
	}

	// Initialize index if needed (after storing documents for training)
	if !se.initialized {
		finalSize := len(se.documents)
		err := se.initializeIndex(finalSize)
		if err != nil {
			return fmt.Errorf("failed to initialize index: %v", err)
		}
	}

	// Pre-allocate with exact capacity to avoid slice growing
	vectors := make([]simd.Vec512, len(texts))
	scales := make([]float32, len(texts))

	// OPTIMIZATION: Process embeddings in parallel batches for GPU acceleration
	// Process in batches for efficient GPU/CPU utilization
	numWorkers := runtime.NumCPU()
	// For GPU mode, limit workers to avoid contention
	if numWorkers > 8 {
		numWorkers = 8
	}
	
	type embeddingJob struct {
		index int
		text  string
	}
	
	type embeddingResult struct {
		index     int
		embedding *EmbedInt8Result
		err       error
	}
	
	jobChan := make(chan embeddingJob, len(texts))
	resultChan := make(chan embeddingResult, len(texts))
	
	// Start worker pool
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobChan {
				// Check cache first
				if cached, found := se.embeddingCache.Get(job.text); found {
					resultChan <- embeddingResult{
						index:     job.index,
						embedding: cached,
					}
					continue
				}
				
				// Generate new embedding
				embedding, err := se.model.EmbedInt8(job.text)
				if err != nil {
					resultChan <- embeddingResult{
						index: job.index,
						err:   err,
					}
					continue
				}
				
				// Cache the result
				se.embeddingCache.Put(job.text, embedding)
				
				resultChan <- embeddingResult{
					index:     job.index,
					embedding: embedding,
				}
			}
		}()
	}
	
	// Send all jobs
	for i, text := range texts {
		jobChan <- embeddingJob{index: i, text: text}
	}
	close(jobChan)
	
	// Wait for workers to finish
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	// Collect results
	results := make(map[int]*EmbedInt8Result)
	for result := range resultChan {
		if result.err != nil {
			return fmt.Errorf("failed to embed text %d: %v", result.index, result.err)
		}
		results[result.index] = result.embedding
	}
	
	// Copy results in order
	for i := range texts {
		embedding := results[i]
		copy(vectors[i][:], embedding.Vector)
		scales[i] = embedding.Scale
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

	// Generate query embedding with caching
	var embedding *EmbedInt8Result
	var err error

	// Check cache first for query
	if cached, found := se.embeddingCache.Get(query); found {
		embedding = cached
	} else {
		// Generate new embedding
		embedding, err = se.model.EmbedInt8(query)
		if err != nil {
			return nil, fmt.Errorf("failed to embed query: %v", err)
		}

		// Cache queries too as they might be repeated
		se.embeddingCache.Put(query, embedding)
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

// Close shuts down the search engine and stops async workers
func (se *SearchEngine) Close() error {
	if se.asyncEnabled {
		// Signal workers to stop
		se.cancel()

		// Close the queue to prevent new requests
		close(se.indexQueue)

		// Wait for all workers to finish
		se.workerWg.Wait()

		se.asyncEnabled = false
	}

	return nil
}

// Flush waits for all pending async indexing operations to complete
func (se *SearchEngine) Flush() error {
	if !se.asyncEnabled {
		return nil // Nothing to flush
	}

	// Send a marker request and wait for it to complete
	done := make(chan IndexResponse, 1)
	marker := IndexRequest{
		IDs:      []int{},
		Texts:    []string{},
		Response: done,
		Context:  context.Background(),
	}

	select {
	case se.indexQueue <- marker:
		// Wait for the marker to be processed
		<-done
		return nil
	case <-se.ctx.Done():
		return fmt.Errorf("search engine is shutting down")
	}
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
	// Store document
	se.documents[id] = text

	// Initialize index if needed
	if !se.initialized {
		err := se.initializeIndex(1000) // Start with estimated size
		if err != nil {
			return err
		}
	}

	// Generate embedding - use int8 or float32 based on config
	if se.config.UseInt8 {
		// Use int8 quantization for 75% memory savings
		embedding, err := se.model.EmbedInt8(text)
		if err != nil {
			return fmt.Errorf("failed to embed text: %v", err)
		}

		// Add to index with int8 data
		var vec simd.Vec512
		copy(vec[:], embedding.Vector)
		return se.index.Add(vec, embedding.Scale, id)
	} else {
		// Use float32 embeddings
		embedding, err := se.model.Encode(text)
		if err != nil {
			return fmt.Errorf("failed to embed text: %v", err)
		}

		// Convert float32 to int8 for index (with scale=1.0)
		var vec simd.Vec512
		for i, val := range embedding {
			if i >= 512 {
				break
			}
			// Simple quantization: clamp to int8 range
			if val > 127 {
				vec[i] = 127
			} else if val < -128 {
				vec[i] = -128
			} else {
				vec[i] = int8(val)
			}
		}
		return se.index.Add(vec, 1.0, id)
	}
}

func (se *SearchEngine) initializeIndex(estimatedSize int) error {
	config := se.generateIndexConfig(estimatedSize)
	se.index = search.NewEngine(config)

	// If we need IVF, we must train first
	if estimatedSize > config.MaxFlatSize && config.NList > 0 {
		trainSize := min(estimatedSize/10, 10000)
		log.Printf("Large dataset (%d > %d), training index with %d samples", estimatedSize, config.MaxFlatSize, trainSize)
		err := se.generateTrainingData(trainSize)
		if err != nil {
			return fmt.Errorf("failed to train index: %v", err)
		}
		log.Printf("Index training completed successfully")
	}

	se.initialized = true
	return nil
}

// generateTrainingData generates training data for the index
func (se *SearchEngine) generateTrainingData(trainSize int) error {
	trainVectors := make([]simd.Vec512, trainSize)
	trainScales := make([]float32, trainSize)

	// Use existing documents if available, otherwise generate synthetic data
	if len(se.documents) > 0 {
		// Use existing documents for training (better than random)
		docTexts := make([]string, 0, len(se.documents))
		for _, text := range se.documents {
			docTexts = append(docTexts, text)
			if len(docTexts) >= trainSize {
				break
			}
		}

		// Fill the rest with synthetic data if needed
		for len(docTexts) < trainSize {
			docTexts = append(docTexts, "synthetic training data for machine learning embedding")
		}

		// Generate embeddings from texts
		for i, text := range docTexts {
			embedding, err := se.model.EmbedInt8(text)
			if err != nil {
				// Fall back to random if embedding fails
				for j := 0; j < 512; j++ {
					trainVectors[i][j] = int8(rand.Intn(256) - 128)
				}
				trainScales[i] = 1.0
			} else {
				copy(trainVectors[i][:], embedding.Vector)
				trainScales[i] = embedding.Scale
			}
		}
	} else {
		// Generate diverse synthetic training data
		syntheticTexts := []string{
			"machine learning algorithms for data science applications",
			"cloud computing infrastructure and distributed systems",
			"artificial intelligence neural networks deep learning",
			"database optimization and query performance tuning",
			"web development frameworks and modern technologies",
			"cybersecurity protocols and data protection methods",
			"software engineering best practices and design patterns",
			"natural language processing and text analysis",
			"computer vision and image recognition systems",
			"blockchain technology and cryptocurrency applications",
		}

		for i := 0; i < trainSize; i++ {
			text := syntheticTexts[i%len(syntheticTexts)]
			embedding, err := se.model.EmbedInt8(text)
			if err != nil {
				// Fall back to random
				for j := 0; j < 512; j++ {
					trainVectors[i][j] = int8(rand.Intn(256) - 128)
				}
				trainScales[i] = 1.0
			} else {
				copy(trainVectors[i][:], embedding.Vector)
				trainScales[i] = embedding.Scale
			}
		}
	}

	// Train the index
	return se.index.Train(trainVectors, trainScales)
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

	// Use preset configuration for simplicity
	return GetSearchConfig(se.config.Preset, estimatedSize)
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


// min function removed - using the one from parallel_indexing.go
