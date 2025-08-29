// +build gpu

package gobed

/*
#cgo CFLAGS: -I./gpu
#cgo LDFLAGS: -L./gpu -lpure_cuda_indexer -L/usr/local/cuda-12.0/lib64 -lcudart -lcublas -Wl,-rpath,./gpu -Wl,-rpath,/usr/local/cuda-12.0/lib64
#include "pure_cuda_indexer.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// GPUIndexer provides CUDA-accelerated indexing and search using pure CUDA
type GPUIndexer struct {
	handle       unsafe.Pointer
	config       IndexConfig
	mutex        sync.RWMutex
	vectorDim    int
	vocabSize    int
	embedDim     int
	deviceID     int
	isReady      bool
	embeddings   []float32
	numVectors   int
}

// IndexConfig configures the GPU indexer
type IndexConfig struct {
	VectorDim        int
	VocabSize        int  // For token embeddings
	EmbedDim         int  // Embedding dimension
	DeviceID         int
}

// DefaultGPUConfig returns an optimal configuration for GPU indexing
func DefaultGPUConfig() IndexConfig {
	return IndexConfig{
		VectorDim: 512,   // truncated/pooled dimension for efficiency
		VocabSize: 30522, // BERT vocab size
		EmbedDim:  1024,  // original model embedding dimension
		DeviceID:  0,     // Use first GPU
	}
}

// DefaultGPUConfig512 returns config optimized for 512 token truncation
func DefaultGPUConfig512() IndexConfig {
	return IndexConfig{
		VectorDim: 512,   // Reduced for faster processing
		VocabSize: 30522, // BERT vocab size
		EmbedDim:  512,   // Reduced embedding dimension
		DeviceID:  0,     // Use first GPU
	}
}

// IsCUDAAvailable checks if CUDA is available
func IsCUDAAvailable() bool {
	return int(C.cuda_is_available()) != 0
}

// GetCUDAMemoryUsage returns current GPU memory usage in bytes
func GetCUDAMemoryUsage() uint64 {
	return uint64(C.cuda_get_memory_usage())
}

// NewGPUIndexer creates a new pure CUDA-accelerated indexer
func NewGPUIndexer(config IndexConfig) (*GPUIndexer, error) {
	// Check CUDA availability
	if !IsCUDAAvailable() {
		return nil, fmt.Errorf("CUDA is not available")
	}

	// Create indexer handle
	handle := C.cuda_index_create(
		C.int(config.VectorDim),
		C.int(config.VocabSize),
		C.int(config.EmbedDim),
	)
	if handle == nil {
		return nil, fmt.Errorf("failed to create pure CUDA indexer")
	}

	indexer := &GPUIndexer{
		handle:     handle,
		config:     config,
		vectorDim:  config.VectorDim,
		vocabSize:  config.VocabSize,
		embedDim:   config.EmbedDim,
		deviceID:   config.DeviceID,
		numVectors: 0,
	}

	// Set finalizer to ensure cleanup
	runtime.SetFinalizer(indexer, (*GPUIndexer).destroy)

	// Indexer created successfully
	return indexer, nil
}

// LoadEmbeddings loads the token embedding table to GPU
func (g *GPUIndexer) LoadEmbeddings(embeddings []float32) error {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if len(embeddings) != g.vocabSize*g.embedDim {
		return fmt.Errorf("embeddings size mismatch: expected %d, got %d",
			g.vocabSize*g.embedDim, len(embeddings))
	}

	result := C.cuda_load_embeddings(
		g.handle,
		(*C.float)(unsafe.Pointer(&embeddings[0])),
	)

	if result == 0 {
		return fmt.Errorf("failed to load embeddings to GPU")
	}

	g.embeddings = embeddings
	// Embeddings loaded to GPU
	return nil
}

// LoadEmbeddingsInt8 loads quantized int8 embeddings to GPU for faster processing
// This reduces memory usage by 75% and improves performance through:
// - 4x less memory bandwidth usage
// - Better cache utilization
// - Vectorized int8 operations
func (g *GPUIndexer) LoadEmbeddingsInt8(embeddings []int8, scales []float32) error {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if len(embeddings) != g.vocabSize*g.embedDim {
		return fmt.Errorf("int8 embeddings size mismatch: expected %d, got %d",
			g.vocabSize*g.embedDim, len(embeddings))
	}

	if len(scales) != g.vocabSize {
		return fmt.Errorf("scales size mismatch: expected %d, got %d",
			g.vocabSize, len(scales))
	}

	result := C.cuda_load_embeddings_int8(
		g.handle,
		(*C.schar)(unsafe.Pointer(&embeddings[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
	)

	if result == 0 {
		return fmt.Errorf("failed to load int8 embeddings to GPU")
	}

	// Int8 embeddings loaded to GPU
	return nil
}

// AddVectors adds int8 quantized vectors to the index
func (g *GPUIndexer) AddVectors(vectors [][]int8, scales []float32) error {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if len(vectors) == 0 {
		return fmt.Errorf("no vectors to add")
	}

	if len(scales) != len(vectors) {
		return fmt.Errorf("scales count mismatch: %d vectors, %d scales", len(vectors), len(scales))
	}

	// Flatten vectors for C interface
	numVectors := len(vectors)
	flatVectors := make([]int8, numVectors*g.vectorDim)
	
	for i, vec := range vectors {
		if len(vec) != g.vectorDim {
			return fmt.Errorf("vector dimension mismatch at index %d: expected %d, got %d",
				i, g.vectorDim, len(vec))
		}
		copy(flatVectors[i*g.vectorDim:(i+1)*g.vectorDim], vec)
	}

	result := C.cuda_index_add(
		g.handle,
		(*C.schar)(unsafe.Pointer(&flatVectors[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int(numVectors),
	)

	if result == 0 {
		return fmt.Errorf("failed to add vectors to GPU index")
	}

	g.numVectors += numVectors
	// Vectors added to GPU index
	return nil
}

// SearchWithTokens searches using token IDs (generates embedding on GPU)
func (g *GPUIndexer) SearchWithTokens(tokenIDs []int32, k int) ([]int32, []float32, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if len(tokenIDs) == 0 {
		return nil, nil, fmt.Errorf("no tokens provided")
	}

	if g.numVectors == 0 {
		return nil, nil, fmt.Errorf("index is empty")
	}

	// Allocate result buffers
	resultIndices := make([]int32, k)
	resultScores := make([]float32, k)

	result := C.cuda_search_with_tokens(
		g.handle,
		(*C.int)(unsafe.Pointer(&tokenIDs[0])),
		C.int(len(tokenIDs)),
		(*C.int)(unsafe.Pointer(&resultIndices[0])),
		(*C.float)(unsafe.Pointer(&resultScores[0])),
		C.int(k),
	)

	if result == 0 {
		return nil, nil, fmt.Errorf("GPU search failed")
	}

	return resultIndices, resultScores, nil
}

// SearchWithEmbedding searches using pre-computed int8 embedding
func (g *GPUIndexer) SearchWithEmbedding(query []int8, queryScale float32, k int) ([]int32, []float32, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if len(query) != g.vectorDim {
		return nil, nil, fmt.Errorf("query dimension mismatch: expected %d, got %d",
			g.vectorDim, len(query))
	}

	if g.numVectors == 0 {
		return nil, nil, fmt.Errorf("index is empty")
	}

	// Allocate result buffers
	resultIndices := make([]int32, k)
	resultScores := make([]float32, k)

	result := C.cuda_search_with_embedding(
		g.handle,
		(*C.schar)(unsafe.Pointer(&query[0])),
		C.float(queryScale),
		(*C.int)(unsafe.Pointer(&resultIndices[0])),
		(*C.float)(unsafe.Pointer(&resultScores[0])),
		C.int(k),
	)

	if result == 0 {
		return nil, nil, fmt.Errorf("GPU search failed")
	}

	return resultIndices, resultScores, nil
}

// Search performs k-nearest neighbor search (wrapper for compatibility)
func (g *GPUIndexer) Search(query []int8, queryScale float32, k int) ([]int32, []float32, error) {
	return g.SearchWithEmbedding(query, queryScale, k)
}

// GetMemoryUsage returns current GPU memory usage
func (g *GPUIndexer) GetMemoryUsage() uint64 {
	return GetCUDAMemoryUsage()
}

// SetMaxTokens sets the maximum number of tokens to process (truncation)
func (g *GPUIndexer) SetMaxTokens(maxTokens int) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	C.cuda_set_max_tokens(g.handle, C.int(maxTokens))
	// Max tokens updated
}

// BulkIndexTokens indexes multiple token sequences in batch on GPU
// This is optimized for bulk indexing - keeps everything on GPU
func (g *GPUIndexer) BulkIndexTokens(tokenSequences [][]int32, seqLengths []int) (int, error) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	if len(tokenSequences) == 0 {
		return 0, fmt.Errorf("no token sequences provided")
	}
	
	if len(seqLengths) != len(tokenSequences) {
		return 0, fmt.Errorf("sequence lengths mismatch: %d sequences, %d lengths", 
			len(tokenSequences), len(seqLengths))
	}
	
	batchSize := len(tokenSequences)
	
	// Find max sequence length
	maxSeqLen := 0
	for _, seq := range tokenSequences {
		if len(seq) > maxSeqLen {
			maxSeqLen = len(seq)
		}
	}
	
	// Flatten token sequences into a single array
	flatTokens := make([]int32, batchSize*maxSeqLen)
	for i, seq := range tokenSequences {
		for j, token := range seq {
			flatTokens[i*maxSeqLen+j] = token
		}
	}
	
	// Convert to C types
	cSeqLengths := make([]C.int, len(seqLengths))
	for i, l := range seqLengths {
		cSeqLengths[i] = C.int(l)
	}
	
	result := C.cuda_bulk_index_tokens(
		g.handle,
		(*C.int)(unsafe.Pointer(&flatTokens[0])),
		(*C.int)(unsafe.Pointer(&cSeqLengths[0])),
		C.int(batchSize),
		C.int(maxSeqLen),
	)
	
	if result == 0 {
		return 0, fmt.Errorf("bulk indexing failed on GPU")
	}
	
	g.numVectors += int(result)
	return int(result), nil
}

// destroy cleans up GPU resources
func (g *GPUIndexer) destroy() {
	if g.handle != nil {
		C.cuda_index_destroy(g.handle)
		g.handle = nil
		// CUDA indexer destroyed
	}
}

// BatchSearch performs k-nearest neighbor search for multiple queries
func (g *GPUIndexer) BatchSearch(queries [][]int8, k int) ([][]SearchResult, error) {
	if len(queries) == 0 {
		return nil, fmt.Errorf("no queries provided")
	}

	// Validate all queries have correct dimension
	for i, query := range queries {
		if len(query) != g.vectorDim {
			return nil, fmt.Errorf("query %d dimension mismatch: expected %d, got %d",
				i, g.vectorDim, len(query))
		}
	}

	// Check if index is empty
	g.mutex.RLock()
	if g.numVectors == 0 {
		g.mutex.RUnlock()
		return nil, fmt.Errorf("index is empty")
	}
	g.mutex.RUnlock()

	// Process queries in parallel for better GPU utilization
	results := make([][]SearchResult, len(queries))
	var wg sync.WaitGroup
	errChan := make(chan error, len(queries))

	// Process each query
	for i, query := range queries {
		wg.Add(1)
		go func(idx int, q []int8) {
			defer wg.Done()

			// Perform search with default scale of 1.0
			indices, scores, err := g.SearchWithEmbedding(q, 1.0, k)
			if err != nil {
				errChan <- fmt.Errorf("query %d failed: %w", idx, err)
				return
			}

			// Convert to SearchResult format
			queryResults := make([]SearchResult, len(indices))
			for j := range indices {
				queryResults[j] = SearchResult{
					ID:         int(indices[j]),
					Similarity: scores[j],
				}
			}
			results[idx] = queryResults
		}(i, query)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// Close explicitly releases GPU resources
func (g *GPUIndexer) Close() {
	g.destroy()
}