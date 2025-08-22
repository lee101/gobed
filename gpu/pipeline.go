// pipeline.go - GPU-accelerated pipeline for Gobed
package gpu

import (
	"fmt"
	"net/http"
	"runtime"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

// Pipeline provides end-to-end GPU acceleration from text to search
type Pipeline struct {
	embedder     *gobed.EmbeddingModel
	searchClient *SearchClient
	database     [][]int8
	texts        []string
	mu           sync.RWMutex
	config       Config
}

// Config configures the GPU pipeline
type Config struct {
	ModelPath      string
	GPUServerURL   string
	BatchSize      int
	UseGPUIndexing bool
	PreloadGPU     bool
	MaxVectors     int
	GPUOnlyMode    bool // If true, clear CPU memory after GPU upload
}

// NewPipeline creates a new GPU-accelerated pipeline
func NewPipeline(config Config) (*Pipeline, error) {
	// Initialize embedder
	embedder, err := gobed.LoadModel()
	if err != nil {
		return nil, fmt.Errorf("failed to create embedder: %w", err)
	}

	// Initialize GPU search client
	searchClient := &SearchClient{
		BaseURL: config.GPUServerURL,
		Client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}

	// Check GPU server health
	health, err := searchClient.Health()
	if err != nil {
		return nil, fmt.Errorf("GPU server not available: %w", err)
	}

	pipeline := &Pipeline{
		embedder:     embedder,
		searchClient: searchClient,
		database:     make([][]int8, 0, config.MaxVectors),
		texts:        make([]string, 0, config.MaxVectors),
		config:       config,
	}

	// Preload database to GPU if requested
	if config.PreloadGPU && health.DatabaseSize > 0 {
		fmt.Printf("GPU server already has %d vectors loaded\n", health.DatabaseSize)
	}

	return pipeline, nil
}

// IndexTexts processes texts to embeddings and stores them on GPU
func (p *Pipeline) IndexTexts(texts []string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	batchSize := p.config.BatchSize
	if batchSize == 0 {
		batchSize = 128 // Increased default for better GPU utilization
	}

	allEmbeddings := make([][]int8, 0, len(texts))

	// Process in batches using GPU embedding
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]

		// Use GPU embedding if available, fallback to CPU
		if p.config.UseGPUIndexing {
			embedResp, err := p.searchClient.EmbedTexts(batch)
			if err != nil {
				fmt.Printf("GPU embedding failed, falling back to CPU: %v\n", err)
				// Fallback to CPU embedding
				int8Embeddings, err := p.embedBatchCPU(batch)
				if err != nil {
					return fmt.Errorf("CPU embedding fallback failed: %w", err)
				}
				allEmbeddings = append(allEmbeddings, int8Embeddings...)
			} else {
				fmt.Printf("GPU embedded %d texts in %.1fms (%.0f texts/sec)\n",
					embedResp.Count, embedResp.EmbedTimeMs, embedResp.TextsPerSec)
				allEmbeddings = append(allEmbeddings, embedResp.Embeddings...)
			}
		} else {
			// CPU embedding
			int8Embeddings, err := p.embedBatchCPU(batch)
			if err != nil {
				return fmt.Errorf("CPU embedding failed: %w", err)
			}
			allEmbeddings = append(allEmbeddings, int8Embeddings...)
		}
	}

	// Store texts
	p.texts = append(p.texts, texts...)
	p.database = append(p.database, allEmbeddings...)

	// Upload to GPU
	loadResp, err := p.searchClient.LoadDatabase(p.database)
	if err != nil {
		return fmt.Errorf("failed to load to GPU: %w", err)
	}

	fmt.Printf("Indexed %d texts to GPU (%.1f MB)\n",
		len(texts), loadResp.MemoryMB)

	// Clear CPU memory if GPU-only mode is enabled
	if p.config.GPUOnlyMode {
		fmt.Printf("🗑️  Clearing CPU embeddings (GPU-only mode)\n")
		// Clear the database slice but keep texts for search results
		p.database = nil
		// Force garbage collection to free memory immediately
		runtime.GC()
		fmt.Printf("✅ CPU memory freed, using only GPU storage\n")
	}

	return nil
}

// Search performs GPU-accelerated similarity search
func (p *Pipeline) Search(query string, k int) ([]Result, error) {
	// Embed query using GPU if available
	var queryInt8 []int8
	if p.config.UseGPUIndexing {
		embedResp, err := p.searchClient.EmbedTexts([]string{query})
		if err != nil {
			fmt.Printf("GPU query embedding failed, falling back to CPU: %v\n", err)
			// Fallback to CPU embedding
			queryEmb, err := p.embedder.Encode(query)
			if err != nil {
				return nil, fmt.Errorf("failed to embed query: %w", err)
			}
			queryInt8 = Float32ToInt8(queryEmb)
		} else {
			if len(embedResp.Embeddings) > 0 {
				queryInt8 = embedResp.Embeddings[0]
			} else {
				return nil, fmt.Errorf("GPU embedding returned no results")
			}
		}
	} else {
		// CPU embedding
		queryEmb, err := p.embedder.Encode(query)
		if err != nil {
			return nil, fmt.Errorf("failed to embed query: %w", err)
		}
		queryInt8 = Float32ToInt8(queryEmb)
	}

	// Search on GPU
	searchResp, err := p.searchClient.Search(queryInt8, k)
	if err != nil {
		return nil, fmt.Errorf("GPU search failed: %w", err)
	}

	// Build results
	p.mu.RLock()
	defer p.mu.RUnlock()

	results := make([]Result, len(searchResp.IDs))
	for i, id := range searchResp.IDs {
		if id < len(p.texts) {
			results[i] = Result{
				Text:  p.texts[id],
				Score: searchResp.Scores[i],
				ID:    id,
			}
		}
	}

	return results, nil
}

// BatchSearch performs batch GPU search
func (p *Pipeline) BatchSearch(queries []string, k int) ([][]Result, error) {
	// Embed all queries using GPU if available
	var queryEmbeddings [][]int8
	if p.config.UseGPUIndexing {
		embedResp, err := p.searchClient.EmbedTexts(queries)
		if err != nil {
			fmt.Printf("GPU batch query embedding failed, falling back to CPU: %v\n", err)
			// Fallback to CPU embedding
			queryEmbeddings = make([][]int8, len(queries))
			for i, query := range queries {
				emb, err := p.embedder.Encode(query)
				if err != nil {
					return nil, fmt.Errorf("failed to embed query %d: %w", i, err)
				}
				queryEmbeddings[i] = Float32ToInt8(emb)
			}
		} else {
			queryEmbeddings = embedResp.Embeddings
		}
	} else {
		// CPU embedding
		queryEmbeddings = make([][]int8, len(queries))
		for i, query := range queries {
			emb, err := p.embedder.Encode(query)
			if err != nil {
				return nil, fmt.Errorf("failed to embed query %d: %w", i, err)
			}
			queryEmbeddings[i] = Float32ToInt8(emb)
		}
	}

	// Batch search on GPU
	batchResp, err := p.searchClient.BatchSearch(queryEmbeddings, k)
	if err != nil {
		return nil, fmt.Errorf("GPU batch search failed: %w", err)
	}

	// Build results
	p.mu.RLock()
	defer p.mu.RUnlock()

	allResults := make([][]Result, len(queries))
	for i := range queries {
		results := make([]Result, len(batchResp.BatchIDs[i]))
		for j, id := range batchResp.BatchIDs[i] {
			if id < len(p.texts) {
				results[j] = Result{
					Text:  p.texts[id],
					Score: batchResp.BatchScores[i][j],
					ID:    id,
				}
			}
		}
		allResults[i] = results
	}

	return allResults, nil
}

// StreamingIndex adds texts to GPU as they arrive
func (p *Pipeline) StreamingIndex(textChan <-chan string, batchSize int) error {
	batch := make([]string, 0, batchSize)
	ticker := time.NewTicker(1 * time.Second) // Flush every second
	defer ticker.Stop()

	for {
		select {
		case text, ok := <-textChan:
			if !ok {
				// Channel closed, index remaining batch
				if len(batch) > 0 {
					return p.IndexTexts(batch)
				}
				return nil
			}

			batch = append(batch, text)

			// Index when batch is full
			if len(batch) >= batchSize {
				if err := p.IndexTexts(batch); err != nil {
					return err
				}
				batch = batch[:0] // Reset batch
			}

		case <-ticker.C:
			// Flush partial batch periodically
			if len(batch) > 0 {
				if err := p.IndexTexts(batch); err != nil {
					return err
				}
				batch = batch[:0]
			}
		}
	}
}

// GetStats returns pipeline statistics
func (p *Pipeline) GetStats() (*Stats, error) {
	p.mu.RLock()
	numTexts := len(p.texts)
	numEmbeddings := len(p.database)
	p.mu.RUnlock()

	// Calculate CPU memory usage
	cpuMemoryMB := 0.0
	if p.database != nil {
		// Each int8 embedding is 512 bytes, plus slice overhead
		cpuMemoryMB = float64(numEmbeddings*512) / 1e6
	}

	// Get GPU stats
	health, err := p.searchClient.Health()
	if err != nil {
		return nil, err
	}

	// Get benchmark results
	benchmark, err := p.searchClient.Benchmark()
	if err != nil {
		return nil, err
	}

	return &Stats{
		NumTexts:       numTexts,
		NumEmbeddings:  numEmbeddings,
		GPUDevice:      health.Device,
		GPUMemoryMB:    benchmark.Database.MemoryMB,
		CPUMemoryMB:    cpuMemoryMB,
		SingleQueryMS:  benchmark.SingleQuery.AvgLatencyMs,
		SingleQueryQPS: benchmark.SingleQuery.QPS,
		BatchQPS:       benchmark.Batch.QPS,
	}, nil
}

// Stats holds pipeline statistics
type Stats struct {
	NumTexts       int
	NumEmbeddings  int
	GPUDevice      string
	GPUMemoryMB    float64
	CPUMemoryMB    float64
	SingleQueryMS  float64
	SingleQueryQPS float64
	BatchQPS       float64
}

// Result represents a search result
type Result struct {
	Text  string
	Score float32
	ID    int
}

// embedBatchCPU performs CPU embedding for fallback
func (p *Pipeline) embedBatchCPU(batch []string) ([][]int8, error) {
	int8Embeddings := make([][]int8, len(batch))
	for j, text := range batch {
		emb, err := p.embedder.Encode(text)
		if err != nil {
			return nil, fmt.Errorf("failed to encode text %d: %w", j, err)
		}
		int8Embeddings[j] = Float32ToInt8(emb)
	}
	return int8Embeddings, nil
}

// Float32ToInt8 converts float32 embeddings to int8 with scaling
func Float32ToInt8(input []float32) []int8 {
	output := make([]int8, len(input))

	// Find max absolute value for scaling
	maxAbs := float32(0)
	for _, v := range input {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}

	// Scale to int8 range
	scale := float32(127.0) / maxAbs
	for i, v := range input {
		scaled := v * scale
		if scaled > 127 {
			output[i] = 127
		} else if scaled < -128 {
			output[i] = -128
		} else {
			output[i] = int8(scaled)
		}
	}

	return output
}
