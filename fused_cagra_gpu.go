//go:build cgo && fusedcagra
// +build cgo,fusedcagra

package gobed

/*
#cgo CFLAGS: -I./
#cgo LDFLAGS: -L./ -lfused_cagra -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -ldl -lstdc++ -lm
#include <stdlib.h>
#include <stdint.h>

// C interface declarations
void* create_fused_context(
    int8_t* embed_weights,
    float* embed_scales_raw,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales_raw,
    int num_vectors,
    int top_k);

void build_cagra_graph(void* context, int degree);

void fused_search(
    void* context,
    uint16_t* token_batch,
    int* token_lengths,
    int batch_size,
    int max_tokens,
    float* output_distances,
    int* output_indices);

void destroy_fused_context(void* context);
*/
import "C"
import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// FusedCAGRAAvailable reports whether the fused CUDA implementation is compiled in.
func FusedCAGRAAvailable() bool { return true }

// FusedCAGRAEngine provides ultra-fast fused embedding generation and CAGRA search
type FusedCAGRAEngine struct {
	context unsafe.Pointer
	mutex   sync.RWMutex

	// Configuration
	vocabSize  int
	embedDim   int
	numVectors int
	topK       int

	// Performance tracking
	searchCount   int64
	avgSearchTime time.Duration

	// State
	isBuilt bool
}

// FusedCAGRAConfig configures the fused CAGRA engine
type FusedCAGRAConfig struct {
	VocabSize   int // Vocabulary size for embedding weights
	EmbedDim    int // Embedding dimension (512 for gobed)
	MaxVectors  int // Maximum database vectors
	TopK        int // Number of results to return
	GraphDegree int // CAGRA graph degree (32 for fast)
}

// DefaultFusedCAGRAConfig returns optimized configuration for ultra-fast search
func DefaultFusedCAGRAConfig() FusedCAGRAConfig {
	return FusedCAGRAConfig{
		VocabSize:   250000,  // Typical vocab size
		EmbedDim:    512,     // gobed dimension
		MaxVectors:  1000000, // Support 1M vectors
		TopK:        10,      // Default top-k
		GraphDegree: 32,      // Fast graph degree
	}
}

// NewFusedCAGRAEngine creates a new fused CAGRA engine
func NewFusedCAGRAEngine(config FusedCAGRAConfig) (*FusedCAGRAEngine, error) {
	engine := &FusedCAGRAEngine{
		vocabSize:  config.VocabSize,
		embedDim:   config.EmbedDim,
		numVectors: 0, // Will be set when data is loaded
		topK:       config.TopK,
	}

	// Set finalizer for cleanup
	runtime.SetFinalizer(engine, (*FusedCAGRAEngine).destroy)

	log.Printf("🚀 Fused CAGRA Engine created: vocab=%d, dim=%d, top_k=%d",
		config.VocabSize, config.EmbedDim, config.TopK)

	return engine, nil
}

// BuildIndex builds the fused CAGRA index from embeddings and database vectors
func (engine *FusedCAGRAEngine) BuildIndex(
	embedWeights []int8, // Flattened embedding weights [vocab_size * embed_dim]
	embedScales []float32, // Embedding quantization scales [vocab_size]
	database []simd.Vec512, // Database vectors [num_vectors]
	dbScales []float32, // Database quantization scales [num_vectors]
) error {
	engine.mutex.Lock()
	defer engine.mutex.Unlock()

	numVectors := len(database)
	if numVectors == 0 {
		return fmt.Errorf("no database vectors provided")
	}

	if len(embedScales) != engine.vocabSize {
		return fmt.Errorf("embedding scales mismatch: expected %d, got %d",
			engine.vocabSize, len(embedScales))
	}

	if len(dbScales) != numVectors {
		return fmt.Errorf("database scales mismatch: expected %d, got %d",
			numVectors, len(dbScales))
	}

	log.Printf("🔨 Building Fused CAGRA index: %d vectors, %d vocab", numVectors, engine.vocabSize)
	start := time.Now()

	// Flatten database vectors
	flatDatabase := make([]int8, numVectors*engine.embedDim)
	for i, vec := range database {
		for j := 0; j < engine.embedDim; j++ {
			flatDatabase[i*engine.embedDim+j] = vec[j]
		}
	}

	// Create fused context
	engine.context = C.create_fused_context(
		(*C.int8_t)(unsafe.Pointer(&embedWeights[0])),
		(*C.float)(unsafe.Pointer(&embedScales[0])),
		C.int(engine.vocabSize),
		C.int(engine.embedDim),
		(*C.int8_t)(unsafe.Pointer(&flatDatabase[0])),
		(*C.float)(unsafe.Pointer(&dbScales[0])),
		C.int(numVectors),
		C.int(engine.topK),
	)

	if engine.context == nil {
		return fmt.Errorf("failed to create fused CAGRA context")
	}

	// Build CAGRA graph
	C.build_cagra_graph(engine.context, C.int(32)) // Fixed degree for now

	engine.numVectors = numVectors
	engine.isBuilt = true

	buildTime := time.Since(start)
	buildThroughput := float64(numVectors) / buildTime.Seconds()

	log.Printf("✅ Fused CAGRA index built: %v (%.0f vectors/sec)", buildTime, buildThroughput)

	return nil
}

// SearchBatch performs batch search using fused embedding generation and CAGRA search
func (engine *FusedCAGRAEngine) SearchBatch(
	tokenBatch [][]uint16, // Batch of tokenized queries [batch_size][max_tokens]
	maxTokens int, // Maximum tokens per query
) ([][]SearchResult, error) {
	engine.mutex.RLock()
	defer engine.mutex.RUnlock()

	if !engine.isBuilt {
		return nil, fmt.Errorf("index not built")
	}

	batchSize := len(tokenBatch)
	if batchSize == 0 {
		return [][]SearchResult{}, nil
	}

	start := time.Now()

	// Flatten token batch for C interface
	flatTokens := make([]uint16, batchSize*maxTokens)
	tokenLengths := make([]int, batchSize)

	for i, tokens := range tokenBatch {
		tokenLengths[i] = len(tokens)
		for j, token := range tokens {
			if j < maxTokens {
				flatTokens[i*maxTokens+j] = token
			}
		}
	}

	effectiveTopK := engine.topK
	if effectiveTopK <= 0 || effectiveTopK > engine.numVectors {
		effectiveTopK = engine.numVectors
		if effectiveTopK == 0 {
			return make([][]SearchResult, batchSize), nil
		}
	}

	// Allocate output buffers
	outputDistances := make([]float32, batchSize*effectiveTopK)
	outputIndices32 := make([]int32, batchSize*effectiveTopK)

	// Call fused search kernel
	// Convert token lengths to int32 for C API
	tokenLengths32 := make([]int32, batchSize)
	for i := range tokenLengths {
		tokenLengths32[i] = int32(tokenLengths[i])
	}

	C.fused_search(
		engine.context,
		(*C.uint16_t)(unsafe.Pointer(&flatTokens[0])),
		(*C.int)(unsafe.Pointer(&tokenLengths32[0])),
		C.int(batchSize),
		C.int(maxTokens),
		(*C.float)(unsafe.Pointer(&outputDistances[0])),
		(*C.int)(unsafe.Pointer(&outputIndices32[0])),
	)

	searchTime := time.Since(start)
	avgPerQuery := searchTime / time.Duration(batchSize)
	qps := float64(batchSize) / searchTime.Seconds()

	// Update performance tracking
	engine.searchCount += int64(batchSize)
	if engine.searchCount <= int64(batchSize) {
		engine.avgSearchTime = avgPerQuery
	} else {
		// Exponential moving average
		alpha := 0.1
		engine.avgSearchTime = time.Duration(
			float64(engine.avgSearchTime)*(1-alpha) + float64(avgPerQuery)*alpha,
		)
	}

	log.Printf("⚡ Fused CAGRA batch: %d queries in %v (%.3fms/query, %.0f QPS)",
		batchSize, searchTime, float64(avgPerQuery.Microseconds())/1000.0, qps)

	// Convert results
	results := make([][]SearchResult, batchSize)
	for i := 0; i < batchSize; i++ {
		row := make([]SearchResult, 0, effectiveTopK)
		seen := make(map[int]struct{}, effectiveTopK)
		for j := 0; j < effectiveTopK; j++ {
			idx := i*effectiveTopK + j
			id := int(outputIndices32[idx])
			if id < 0 || id >= engine.numVectors {
				continue
			}
			if _, exists := seen[id]; exists {
				continue
			}
			seen[id] = struct{}{}
			row = append(row, SearchResult{
				ID:         id,
				Similarity: outputDistances[idx],
				Distance:   -outputDistances[idx],
			})
		}
		results[i] = row
	}

	return results, nil
}

// Search performs single query search
func (engine *FusedCAGRAEngine) Search(tokens []uint16) ([]SearchResult, error) {
	results, err := engine.SearchBatch([][]uint16{tokens}, len(tokens)+10)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return []SearchResult{}, nil
	}
	return results[0], nil
}

// GetStats returns performance statistics
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

// FusedCAGRAStats contains performance statistics
type FusedCAGRAStats struct {
	VocabSize       int
	EmbedDim        int
	NumVectors      int
	TopK            int
	AvgSearchTimeMs float64
	SearchCount     int64
	IsBuilt         bool
}

// destroy cleans up resources
func (engine *FusedCAGRAEngine) destroy() {
	if engine.context != nil {
		C.destroy_fused_context(engine.context)
		engine.context = nil
	}
}

// Close explicitly releases resources
func (engine *FusedCAGRAEngine) Close() {
	engine.destroy()
}
