// gpu_search_engine.go - GPU-accelerated search engine for Go
package main

// #cgo CFLAGS: -I./
// #cgo LDFLAGS: -L./ -ltorch_cgo_wrapper -L/home/lee/code/gobed/libtorch/lib -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10 -L/usr/local/cuda-12.0/lib64 -lcudart -lcublas -Wl,-rpath,. -Wl,-rpath,/home/lee/code/gobed/libtorch/lib -Wl,-rpath,/usr/local/cuda-12.0/lib64
// #include "torch_cgo_wrapper.h"
import "C"
import (
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/lee101/gobed"
)

// GPUSearchEngine provides GPU-accelerated vector search
type GPUSearchEngine struct {
	model       *gobed.EmbeddingModel
	indexer     C.TorchIndexerHandle
	documents   map[int]string
	config      GPUSearchConfig
	mu          sync.RWMutex
	initialized bool
	embeddings  [][]int8 // Store embeddings for GPU access
}

// GPUSearchConfig configures the GPU search engine
type GPUSearchConfig struct {
	// GPU settings
	EnableGPU    bool // Enable GPU acceleration
	DeviceID     int  // CUDA device ID (default: 0)
	BatchSize    int  // Batch size for GPU operations (default: 1000)
	
	// Index settings
	VectorDim        int // Vector dimension (default: 1024)
	NumSubQuantizers int // PQ subquantizers (default: 32)
	CodebookSize     int // Codebook size (default: 256)
	IVFClusters      int // IVF clusters (default: 100)
	ProbeLists       int // Probe lists (default: 10)
	RerankK          int // Rerank candidates (default: 100)
}

// DefaultGPUSearchConfig returns default GPU configuration
func DefaultGPUSearchConfig() GPUSearchConfig {
	return GPUSearchConfig{
		EnableGPU:        true,
		DeviceID:         0,
		BatchSize:        1000,
		VectorDim:        1024,
		NumSubQuantizers: 32,
		CodebookSize:     256,
		IVFClusters:      100,
		ProbeLists:       10,
		RerankK:          100,
	}
}

// NewGPUSearchEngine creates a new GPU-accelerated search engine
func NewGPUSearchEngine(model *gobed.EmbeddingModel) *GPUSearchEngine {
	return NewGPUSearchEngineWithConfig(model, DefaultGPUSearchConfig())
}

// NewGPUSearchEngineWithConfig creates a GPU search engine with custom config
func NewGPUSearchEngineWithConfig(model *gobed.EmbeddingModel, config GPUSearchConfig) *GPUSearchEngine {
	return &GPUSearchEngine{
		model:     model,
		documents: make(map[int]string),
		config:    config,
		embeddings: make([][]int8, 0),
	}
}

// GPUSearchResult represents a search result from GPU search
type GPUSearchResult struct {
	ID         int     `json:"id"`
	Text       string  `json:"text"`
	Score      float32 `json:"score"`
	Distance   float32 `json:"distance"`
}

// Index adds and indexes a single text with auto-generated ID
func (gse *GPUSearchEngine) Index(text string) (int, error) {
	gse.mu.Lock()
	defer gse.mu.Unlock()

	id := len(gse.documents)
	return id, gse.indexWithID(id, text)
}

// IndexWithID adds and indexes a text with a specific ID
func (gse *GPUSearchEngine) IndexWithID(id int, text string) error {
	gse.mu.Lock()
	defer gse.mu.Unlock()

	return gse.indexWithID(id, text)
}

// IndexBatch efficiently indexes multiple texts using GPU acceleration
func (gse *GPUSearchEngine) IndexBatch(texts []string) ([]int, error) {
	gse.mu.Lock()
	defer gse.mu.Unlock()

	if !gse.config.EnableGPU {
		return nil, fmt.Errorf("GPU acceleration is disabled")
	}

	// Check CUDA availability
	if C.torch_cuda_is_available() == 0 {
		return nil, fmt.Errorf("CUDA is not available")
	}

	fmt.Printf(" GPU batch indexing %d documents...\n", len(texts))
	start := time.Now()

	// Generate embeddings for all texts
	embeddings := make([][]int8, len(texts))
	ids := make([]int, len(texts))
	
	for i, text := range texts {
		// Generate embedding
		embedding, err := gse.model.EmbedInt8(text)
		if err != nil {
			return nil, fmt.Errorf("failed to embed text %d: %v", i, err)
		}

		// Convert to int8 and store
		int8Embedding := make([]int8, len(embedding.Vector))
		for j, val := range embedding.Vector {
			int8Embedding[j] = int8(val)
		}
		embeddings[i] = int8Embedding

		// Store document
		id := len(gse.documents) + i
		ids[i] = id
		gse.documents[id] = text
	}

	embedTime := time.Since(start)
	fmt.Printf(" Generated embeddings in %v\n", embedTime)

	// Initialize GPU indexer if needed
	if !gse.initialized {
		err := gse.initializeGPUIndex(len(texts))
		if err != nil {
			return nil, fmt.Errorf("failed to initialize GPU index: %v", err)
		}
	}

	// Add embeddings to GPU index
	indexStart := time.Now()
	err := gse.addEmbeddingsToIndex(embeddings)
	if err != nil {
		return nil, fmt.Errorf("failed to add embeddings to GPU index: %v", err)
	}

	indexTime := time.Since(indexStart)
	totalTime := time.Since(start)

	fmt.Printf(" GPU indexing completed:\n")
	fmt.Printf("   Documents: %d\n", len(texts))
	fmt.Printf("   Embedding time: %v\n", embedTime)
	fmt.Printf("   GPU indexing time: %v\n", indexTime)
	fmt.Printf("   Total time: %v\n", totalTime)
	fmt.Printf("   Speed: %.0f docs/sec\n", float64(len(texts))/totalTime.Seconds())

	return ids, nil
}

// Search performs GPU-accelerated similarity search
func (gse *GPUSearchEngine) Search(query string, k int) ([]GPUSearchResult, error) {
	gse.mu.RLock()
	defer gse.mu.RUnlock()

	if !gse.initialized {
		return nil, fmt.Errorf("index not initialized - add documents first")
	}

	if !gse.config.EnableGPU {
		return nil, fmt.Errorf("GPU acceleration is disabled")
	}

	fmt.Printf(" GPU search: \"%s\" (k=%d)\n", query, k)
	start := time.Now()

	// Generate query embedding
	embedding, err := gse.model.EmbedInt8(query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %v", err)
	}

	// Convert to int8
	queryInt8 := make([]C.schar, len(embedding.Vector))
	for i, val := range embedding.Vector {
		queryInt8[i] = C.schar(val)
	}

	// Perform GPU search
	searchResult := C.torch_indexer_search(
		gse.indexer,
		&queryInt8[0],
		C.int(len(queryInt8)),
		C.int(k),
	)

	searchTime := time.Since(start)

	if searchResult.count == 0 {
		return []GPUSearchResult{}, nil
	}

	// Convert results
	results := make([]GPUSearchResult, searchResult.count)
	
	// Convert C arrays to Go slices
	ids := (*[1 << 20]C.int)(unsafe.Pointer(searchResult.ids))[:searchResult.count:searchResult.count]
	scores := (*[1 << 20]C.float)(unsafe.Pointer(searchResult.scores))[:searchResult.count:searchResult.count]

	for i := 0; i < int(searchResult.count); i++ {
		id := int(ids[i])
		score := float32(scores[i])
		
		results[i] = GPUSearchResult{
			ID:       id,
			Text:     gse.documents[id],
			Score:    score,
			Distance: 1.0 - score, // Convert similarity to distance
		}
	}

	// Free result memory
	C.torch_search_result_free(&searchResult)

	fmt.Printf(" GPU search completed in %v\n", searchTime)
	fmt.Printf("   Results: %d/%d\n", len(results), k)
	if len(results) > 0 {
		fmt.Printf("   Top score: %.4f\n", results[0].Score)
	}

	return results, nil
}

// Size returns the number of indexed documents
func (gse *GPUSearchEngine) Size() int {
	gse.mu.RLock()
	defer gse.mu.RUnlock()
	return len(gse.documents)
}

// Close releases GPU resources
func (gse *GPUSearchEngine) Close() error {
	gse.mu.Lock()
	defer gse.mu.Unlock()

	if gse.indexer != nil {
		C.torch_indexer_destroy(gse.indexer)
		gse.indexer = nil
	}

	fmt.Println(" GPU resources released")
	return nil
}

// GetStats returns GPU indexer statistics
func (gse *GPUSearchEngine) GetStats() map[string]interface{} {
	gse.mu.RLock()
	defer gse.mu.RUnlock()

	if !gse.initialized {
		return map[string]interface{}{
			"documents": 0,
			"gpu_enabled": gse.config.EnableGPU,
			"initialized": false,
		}
	}

	stats := C.torch_indexer_get_stats(gse.indexer)
	
	return map[string]interface{}{
		"documents":     int(stats.num_vectors),
		"dimension":     int(stats.vector_dim),
		"gpu_memory_mb": float32(stats.gpu_memory_mb),
		"gpu_enabled":   gse.config.EnableGPU,
		"device_id":     gse.config.DeviceID,
		"initialized":   gse.initialized,
	}
}

// Internal methods

func (gse *GPUSearchEngine) indexWithID(id int, text string) error {
	// Generate embedding
	embedding, err := gse.model.EmbedInt8(text)
	if err != nil {
		return fmt.Errorf("failed to embed text: %v", err)
	}

	// Store document
	gse.documents[id] = text

	// Convert and store embedding
	int8Embedding := make([]int8, len(embedding.Vector))
	for i, val := range embedding.Vector {
		int8Embedding[i] = int8(val)
	}

	// Extend embeddings slice if needed
	for len(gse.embeddings) <= id {
		gse.embeddings = append(gse.embeddings, nil)
	}
	gse.embeddings[id] = int8Embedding

	// Initialize index if this is the first document
	if !gse.initialized {
		return gse.initializeGPUIndex(1)
	}

	return nil
}

func (gse *GPUSearchEngine) initializeGPUIndex(estimatedSize int) error {
	if gse.initialized {
		return nil
	}

	fmt.Printf("🏗 Initializing GPU index (estimated size: %d)...\n", estimatedSize)

	// Create GPU indexer config
	config := C.IndexConfig{
		vector_dim:        C.int(gse.config.VectorDim),
		num_subquantizers: C.int(gse.config.NumSubQuantizers),
		codebook_size:     C.int(gse.config.CodebookSize),
		ivf_clusters:      C.int(gse.config.IVFClusters),
		probe_lists:       C.int(gse.config.ProbeLists),
		rerank_k:          C.int(gse.config.RerankK),
		device_id:         C.int(gse.config.DeviceID),
	}

	// Create indexer
	gse.indexer = C.torch_indexer_create(config)
	if gse.indexer == nil {
		return fmt.Errorf("failed to create GPU indexer")
	}

	fmt.Printf(" GPU indexer created (device: %d)\n", gse.config.DeviceID)
	gse.initialized = true

	return nil
}

func (gse *GPUSearchEngine) addEmbeddingsToIndex(embeddings [][]int8) error {
	if len(embeddings) == 0 {
		return nil
	}

	vectorDim := len(embeddings[0])
	numVectors := len(embeddings)

	// Flatten embeddings for C interface
	flatEmbeddings := make([]C.schar, numVectors*vectorDim)
	for i, emb := range embeddings {
		for j, val := range emb {
			flatEmbeddings[i*vectorDim+j] = C.schar(val)
		}
	}

	// Train indexer if needed
	result := C.torch_indexer_train(
		gse.indexer,
		&flatEmbeddings[0],
		C.int(numVectors),
		C.int(vectorDim),
	)

	if result == 0 {
		return fmt.Errorf("failed to train GPU indexer")
	}

	// Add vectors to index
	result = C.torch_indexer_add_vectors(
		gse.indexer,
		&flatEmbeddings[0],
		C.int(numVectors),
		C.int(vectorDim),
	)

	if result == 0 {
		return fmt.Errorf("failed to add vectors to GPU index")
	}

	// Store embeddings
	gse.embeddings = append(gse.embeddings, embeddings...)

	return nil
}