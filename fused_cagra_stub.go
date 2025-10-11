//go:build !fusedcagra || !cgo
// +build !fusedcagra !cgo

package gobed

import (
	"fmt"
	"log"
	"sort"
	"sync"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// FusedCAGRAConfig configures the fused CAGRA engine.
type FusedCAGRAConfig struct {
	VocabSize   int
	EmbedDim    int
	MaxVectors  int
	TopK        int
	GraphDegree int
}

// DefaultFusedCAGRAConfig returns the default configuration used by production builds.
func DefaultFusedCAGRAConfig() FusedCAGRAConfig {
	return FusedCAGRAConfig{
		VocabSize:   250000,
		EmbedDim:    512,
		MaxVectors:  1000000,
		TopK:        10,
		GraphDegree: 32,
	}
}

// FusedCAGRAAvailable reports whether the CUDA implementation is linked in.
func FusedCAGRAAvailable() bool { return false }

// FusedCAGRAEngine provides a CPU fallback that mimics the fused GPU pipeline when CUDA libraries are unavailable.
type FusedCAGRAEngine struct {
	mutex         sync.RWMutex
	vocabSize     int
	embedDim      int
	numVectors    int
	topK          int
	searchCount   int64
	avgSearchTime time.Duration
	isBuilt       bool

	embedWeights []int8
	embedScales  []float32
	dbVectors    []simd.Vec512
	dbScales     []float32
}

// NewFusedCAGRAEngine creates a CPU-backed fused engine when GPU kernels are not available.
func NewFusedCAGRAEngine(config FusedCAGRAConfig) (*FusedCAGRAEngine, error) {
	if config.VocabSize <= 0 || config.EmbedDim <= 0 {
		return nil, fmt.Errorf("invalid fused CAGRA config: vocab=%d embed=%d", config.VocabSize, config.EmbedDim)
	}

	engine := &FusedCAGRAEngine{
		vocabSize: config.VocabSize,
		embedDim:  config.EmbedDim,
		topK:      config.TopK,
	}

	log.Printf("FusedCAGRAEngine: GPU kernels unavailable, running CPU fallback (top_k=%d)", config.TopK)

	return engine, nil
}

// BuildIndex loads quantized embeddings and prepares the CPU search buffers.
func (engine *FusedCAGRAEngine) BuildIndex(
	embedWeights []int8,
	embedScales []float32,
	database []simd.Vec512,
	dbScales []float32,
) error {
	engine.mutex.Lock()
	defer engine.mutex.Unlock()

	if len(embedWeights) != engine.vocabSize*engine.embedDim {
		return fmt.Errorf("embedding weights size mismatch: expected %d, got %d",
			engine.vocabSize*engine.embedDim, len(embedWeights))
	}
	if len(embedScales) != engine.vocabSize {
		return fmt.Errorf("embedding scales size mismatch: expected %d, got %d",
			engine.vocabSize, len(embedScales))
	}
	if len(database) == 0 {
		return fmt.Errorf("no database vectors provided")
	}
	if len(dbScales) != len(database) {
		return fmt.Errorf("database scales mismatch: expected %d, got %d",
			len(database), len(dbScales))
	}

	engine.embedWeights = append(engine.embedWeights[:0], embedWeights...)
	engine.embedScales = append(engine.embedScales[:0], embedScales...)

	engine.dbVectors = append(engine.dbVectors[:0], database...)
	engine.dbScales = append(engine.dbScales[:0], dbScales...)

	engine.numVectors = len(database)
	engine.isBuilt = true

	log.Printf("FusedCAGRAEngine: built CPU fallback index (%d vectors)", engine.numVectors)
	return nil
}

// SearchBatch executes a CPU implementation of the fused pipeline.
func (engine *FusedCAGRAEngine) SearchBatch(tokenBatch [][]uint16, maxTokens int) ([][]SearchResult, error) {
	engine.mutex.RLock()
	defer engine.mutex.RUnlock()

	if !engine.isBuilt {
		return nil, fmt.Errorf("index not built")
	}
	if len(tokenBatch) == 0 {
		return [][]SearchResult{}, nil
	}

	start := time.Now()
	results := make([][]SearchResult, len(tokenBatch))

	for i, tokens := range tokenBatch {
		if len(tokens) > maxTokens {
			tokens = tokens[:maxTokens]
		}
		query := engine.buildQueryEmbedding(tokens)
		results[i] = engine.searchQuery(query)
	}

	elapsed := time.Since(start)
	avgPerQuery := elapsed / time.Duration(len(tokenBatch))

	engine.searchCount += int64(len(tokenBatch))
	if engine.searchCount <= int64(len(tokenBatch)) {
		engine.avgSearchTime = avgPerQuery
	} else {
		const alpha = 0.1
		engine.avgSearchTime = time.Duration(
			float64(engine.avgSearchTime)*(1-alpha) + float64(avgPerQuery)*alpha,
		)
	}

	log.Printf("FusedCAGRAEngine: CPU batch search %d queries in %v (%.3fms/query)",
		len(tokenBatch), elapsed, float64(avgPerQuery.Microseconds())/1000.0)

	return results, nil
}

// Search runs a single-query search through the CPU fallback.
func (engine *FusedCAGRAEngine) Search(tokens []uint16) ([]SearchResult, error) {
	results, err := engine.SearchBatch([][]uint16{tokens}, len(tokens))
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return []SearchResult{}, nil
	}
	return results[0], nil
}

// GetStats returns statistics collected during CPU fallback execution.
func (engine *FusedCAGRAEngine) GetStats() FusedCAGRAStats {
	engine.mutex.RLock()
	defer engine.mutex.RUnlock()

	return FusedCAGRAStats{
		VocabSize:       engine.vocabSize,
		EmbedDim:        engine.embedDim,
		NumVectors:      engine.numVectors,
		TopK:            engine.topK,
		AvgSearchTimeMs: float64(engine.avgSearchTime.Microseconds()) / 1000.0,
		SearchCount:     engine.searchCount,
		IsBuilt:         engine.isBuilt,
	}
}

// Close releases any CPU resources (no-op for fallback mode).
func (engine *FusedCAGRAEngine) Close() {}

// FusedCAGRAStats mirrors the GPU implementation statistics payload.
type FusedCAGRAStats struct {
	VocabSize       int
	EmbedDim        int
	NumVectors      int
	TopK            int
	AvgSearchTimeMs float64
	SearchCount     int64
	IsBuilt         bool
}

func (engine *FusedCAGRAEngine) buildQueryEmbedding(tokens []uint16) []float32 {
	query := make([]float32, engine.embedDim)
	if len(tokens) == 0 {
		return query
	}

	valid := 0
	for _, token := range tokens {
		if int(token) >= engine.vocabSize {
			continue
		}
		offset := int(token) * engine.embedDim
		scale := engine.embedScales[int(token)]
		for j := 0; j < engine.embedDim; j++ {
			query[j] += float32(engine.embedWeights[offset+j]) * scale
		}
		valid++
	}

	if valid == 0 {
		for i := range query {
			query[i] = 0
		}
		return query
	}

	inv := 1.0 / float32(valid)
	for i := range query {
		query[i] *= inv
	}
	return query
}

func (engine *FusedCAGRAEngine) searchQuery(query []float32) []SearchResult {
	topK := engine.topK
	if topK <= 0 {
		topK = 10
	}
	candidates := make([]SearchResult, len(engine.dbVectors))
	for i := range engine.dbVectors {
		score := float32(0)
		scale := engine.dbScales[i]
		vec := engine.dbVectors[i]
		for j := 0; j < engine.embedDim; j++ {
			score += query[j] * float32(vec[j]) * scale
		}
		candidates[i] = SearchResult{
			ID:         i,
			Similarity: score,
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Similarity > candidates[j].Similarity
	})

	if len(candidates) > topK {
		candidates = candidates[:topK]
	}
	return candidates
}
