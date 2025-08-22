package gobed

import (
	"fmt"
	"time"

	"github.com/lee101/gobed/ann/search"
	"github.com/lee101/gobed/ann/simd"
)

// VectorIndex provides high-performance vector search capabilities
type VectorIndex struct {
	engine *search.Engine
	model  *EmbeddingModel
	config VectorIndexConfig
	// bulkIndexer *BulkGPUIndexer // Optional bulk GPU indexer (disabled for now)
}

// VectorIndexConfig configures the vector index
type VectorIndexConfig struct {
	// Index configuration
	MaxFlatSize int  // Use flat index below this size (default: 50000)
	NList       int  // Number of IVF clusters (default: 4096)
	NProbe      int  // Number of clusters to search (default: 8)
	UsePQ       bool // Use product quantization (default: true for >100k)
	UseHNSW     bool // Use HNSW for routing (default: true)

	// Search configuration
	RerankSize  int  // Number of candidates to rerank (default: 128)
	UseParallel bool // Use parallel search (default: true)

	// Bulk indexing configuration
	EnableBulkGPU bool // Enable GPU bulk indexing for large datasets
	BulkBatchSize int  // Batch size for GPU bulk indexing (default: 5000)
}

// DefaultVectorIndexConfig returns default configuration
func DefaultVectorIndexConfig() VectorIndexConfig {
	return VectorIndexConfig{
		MaxFlatSize:   5000, // Use approximate search early for speed
		NList:         1024, // Moderate number of clusters
		NProbe:        8,    // Few probes for low latency
		UsePQ:         true, // Enable compression for large datasets
		UseHNSW:       true, // Use graph routing when beneficial
		RerankSize:    100,  // Smaller rerank for speed
		UseParallel:   true,
		EnableBulkGPU: true, // Enable GPU bulk indexing for large datasets
		BulkBatchSize: 5000, // Process 5k documents at once on GPU
	}
}

// NewVectorIndex creates a new vector index
func NewVectorIndex(model *EmbeddingModel, config VectorIndexConfig) *VectorIndex {
	// Convert to search engine config
	engineConfig := search.Config{
		MaxFlatSize: config.MaxFlatSize,
		NList:       config.NList,
		NProbe:      config.NProbe,
		RerankSize:  config.RerankSize,
		UseParallel: config.UseParallel,
		HNSWEnabled: config.UseHNSW,
	}

	if config.UsePQ {
		engineConfig.M = 64
		engineConfig.NBits = 8
	}

	idx := &VectorIndex{
		engine: search.NewEngine(engineConfig),
		model:  model,
		config: config,
	}

	// GPU indexer disabled for now - would be initialized here
	if config.EnableBulkGPU {
		fmt.Printf("⚠️  Bulk GPU indexing not yet implemented\n")
	}

	return idx
}

// Document represents a document to index
type Document struct {
	ID   int
	Text string
}

// SearchResult represents a search result with similarity score
type SearchResult struct {
	ID         int
	Text       string
	Similarity float32
	Distance   float32
}

// AddDocument adds a document to the index
func (idx *VectorIndex) AddDocument(doc Document) error {
	// Generate embedding
	embedding, err := idx.model.EmbedInt8(doc.Text)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %v", err)
	}

	// Convert to simd.Vec512
	var vec simd.Vec512
	copy(vec[:], embedding.Vector)

	// Add to index
	return idx.engine.Add(vec, embedding.Scale, doc.ID)
}

// AddDocuments adds multiple documents efficiently
func (idx *VectorIndex) AddDocuments(docs []Document) error {
	// GPU indexing disabled for now

	// Fall back to standard CPU indexing
	vectors := make([]simd.Vec512, len(docs))
	scales := make([]float32, len(docs))
	ids := make([]int, len(docs))

	// Generate embeddings in parallel
	for i, doc := range docs {
		embedding, err := idx.model.EmbedInt8(doc.Text)
		if err != nil {
			return fmt.Errorf("failed to generate embedding for doc %d: %v", doc.ID, err)
		}

		copy(vectors[i][:], embedding.Vector)
		scales[i] = embedding.Scale
		ids[i] = doc.ID
	}

	// Add batch to index
	return idx.engine.AddBatch(vectors, scales, ids)
}

// AddDocumentsBulkGPU forces GPU bulk indexing regardless of size
func (idx *VectorIndex) AddDocumentsBulkGPU(docs []Document) error {
	return fmt.Errorf("bulk GPU indexing not yet implemented")
}

// AddDocumentsWithMonitoring adds documents with real-time GPU monitoring
func (idx *VectorIndex) AddDocumentsWithMonitoring(docs []Document) (<-chan interface{}, error) {
	return nil, fmt.Errorf("bulk GPU indexing not yet implemented")
}

// Search performs similarity search
func (idx *VectorIndex) Search(query string, k int) ([]SearchResult, error) {
	// Generate query embedding
	embedding, err := idx.model.EmbedInt8(query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %v", err)
	}

	// Convert to simd.Vec512
	var vec simd.Vec512
	copy(vec[:], embedding.Vector)

	// Search
	results, err := idx.engine.Search(&vec, k)
	if err != nil {
		return nil, err
	}

	// Convert results
	searchResults := make([]SearchResult, len(results))
	for i, r := range results {
		searchResults[i] = SearchResult{
			ID:         r.ID,
			Similarity: r.Score / (127.0 * 127.0 * 512), // Normalize to [0,1]
			Distance:   r.Distance,
		}
	}

	return searchResults, nil
}

// Train trains the index on sample data for better performance
func (idx *VectorIndex) Train(texts []string) error {
	if len(texts) == 0 {
		return nil
	}

	fmt.Printf("Training index on %d samples...\n", len(texts))
	start := time.Now()

	// Generate embeddings for training
	vectors := make([]simd.Vec512, len(texts))
	scales := make([]float32, len(texts))

	for i, text := range texts {
		embedding, err := idx.model.EmbedInt8(text)
		if err != nil {
			return fmt.Errorf("failed to generate training embedding: %v", err)
		}

		copy(vectors[i][:], embedding.Vector)
		scales[i] = embedding.Scale
	}

	// Train the engine
	err := idx.engine.Train(vectors, scales)
	if err != nil {
		return err
	}

	fmt.Printf("Training completed in %v\n", time.Since(start))
	return nil
}

// Size returns the number of indexed documents
func (idx *VectorIndex) Size() int {
	return idx.engine.Stats().Size
}

// Stats returns index statistics
func (idx *VectorIndex) Stats() VectorIndexStats {
	engineStats := idx.engine.Stats()

	return VectorIndexStats{
		Size:        engineStats.Size,
		IndexType:   engineStats.IndexType,
		MemoryUsage: engineStats.MemoryUsage,
		NLists:      engineStats.NLists,
		PQEnabled:   engineStats.PQEnabled,
		HNSWEnabled: engineStats.HNSWEnabled,
	}
}

// VectorIndexStats contains index statistics
type VectorIndexStats struct {
	Size        int
	IndexType   string
	MemoryUsage int64
	NLists      int
	PQEnabled   bool
	HNSWEnabled bool
}

// EmbedInt8Result represents an int8 quantized embedding
type EmbedInt8Result struct {
	Vector []int8
	Scale  float32
}

// EmbedInt8 generates int8 quantized embeddings
func (m *EmbeddingModel) EmbedInt8(text string) (*EmbedInt8Result, error) {
	// First get float32 embedding
	embedding, err := m.Encode(text)
	if err != nil {
		return nil, err
	}

	// Quantize to int8
	quantized, scale := quantizeEmbedding(embedding)

	return &EmbedInt8Result{
		Vector: quantized,
		Scale:  scale,
	}, nil
}

// quantizeEmbedding converts float32 embedding to int8 with scale
func quantizeEmbedding(embedding []float32) ([]int8, float32) {
	// Find min and max for quantization
	minVal := embedding[0]
	maxVal := embedding[0]

	for _, v := range embedding[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute scale for symmetric quantization
	absMax := maxVal
	if -minVal > absMax {
		absMax = -minVal
	}

	scale := absMax / 127.0
	if scale == 0 {
		scale = 1.0
	}

	// Quantize
	quantized := make([]int8, len(embedding))
	for i, v := range embedding {
		q := int32(v / scale)
		if q < -128 {
			q = -128
		} else if q > 127 {
			q = 127
		}
		quantized[i] = int8(q)
	}

	return quantized, scale
}

// Example usage function
func ExampleVectorSearch() {
	// Load model
	model, err := LoadModel()
	if err != nil {
		panic(err)
	}

	// Create index with custom config
	config := DefaultVectorIndexConfig()
	config.NProbe = 16 // Higher recall

	index := NewVectorIndex(model, config)

	// Add documents
	docs := []Document{
		{ID: 1, Text: "The cat sat on the mat"},
		{ID: 2, Text: "Dogs are loyal companions"},
		{ID: 3, Text: "Machine learning is fascinating"},
		{ID: 4, Text: "The weather is nice today"},
		{ID: 5, Text: "Cats and dogs are pets"},
	}

	err = index.AddDocuments(docs)
	if err != nil {
		panic(err)
	}

	// Search
	results, err := index.Search("feline animals", 3)
	if err != nil {
		panic(err)
	}

	fmt.Println("Search results:")
	for _, r := range results {
		fmt.Printf("ID: %d, Similarity: %.3f\n", r.ID, r.Similarity)
	}

	// Print stats
	stats := index.Stats()
	fmt.Printf("\nIndex stats: %+v\n", stats)
}
