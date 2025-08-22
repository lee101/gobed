package gobed

/*
#cgo LDFLAGS: -L./gpu -ltorch_cgo_wrapper
#include "./gpu/torch_cgo_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/lee101/gobed/ann/simd"
)

// GPUIndexer provides CUDA-accelerated indexing and search
type GPUIndexer struct {
	handle    C.TorchIndexerHandle
	config    IndexConfig
	mutex     sync.RWMutex
	vectorDim int
	deviceID  int
	isReady   bool
}

// IndexConfig configures the GPU indexer
type IndexConfig struct {
	VectorDim        int
	NumSubquantizers int
	CodebookSize     int
	IVFClusters      int
	ProbeLists       int
	RerankK          int
	DeviceID         int
}

// DefaultGPUConfig returns an optimal configuration for GPU indexing
func DefaultGPUConfig() IndexConfig {
	return IndexConfig{
		VectorDim:        512, // Standard embedding dimension
		NumSubquantizers: 8,   // Product quantization subvectors
		CodebookSize:     256, // Codebook size per subquantizer
		IVFClusters:      1024, // IVF clusters for coarse quantization
		ProbeLists:       64,   // Number of lists to probe during search
		RerankK:          1000, // Number of candidates to rerank
		DeviceID:         0,    // Use first GPU
	}
}

// NewGPUIndexer creates a new CUDA-accelerated indexer
func NewGPUIndexer(config IndexConfig) (*GPUIndexer, error) {
	// Check CUDA availability
	if int(C.torch_cuda_is_available()) == 0 {
		return nil, fmt.Errorf("CUDA is not available")
	}

	deviceCount := int(C.torch_cuda_device_count())
	if config.DeviceID >= deviceCount {
		return nil, fmt.Errorf("invalid device ID %d, only %d devices available", config.DeviceID, deviceCount)
	}

	// Create C config
	cConfig := C.IndexConfig{
		vector_dim:         C.int(config.VectorDim),
		num_subquantizers:  C.int(config.NumSubquantizers),
		codebook_size:      C.int(config.CodebookSize),
		ivf_clusters:       C.int(config.IVFClusters),
		probe_lists:        C.int(config.ProbeLists),
		rerank_k:          C.int(config.RerankK),
		device_id:         C.int(config.DeviceID),
	}

	// Create indexer handle
	handle := C.torch_indexer_create(cConfig)
	if handle == nil {
		return nil, fmt.Errorf("failed to create GPU indexer")
	}

	indexer := &GPUIndexer{
		handle:    handle,
		config:    config,
		vectorDim: config.VectorDim,
		deviceID:  config.DeviceID,
	}

	// Set finalizer to ensure cleanup
	runtime.SetFinalizer(indexer, (*GPUIndexer).destroy)

	fmt.Printf("🚀 GPU Indexer created (Device: %d, VectorDim: %d)\n", 
		config.DeviceID, config.VectorDim)
	return indexer, nil
}

// TrainIndex trains the indexer with a set of vectors (CUDA-accelerated)
func (g *GPUIndexer) TrainIndex(vectors [][]int8) error {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	if len(vectors[0]) != g.vectorDim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", 
			g.vectorDim, len(vectors[0]))
	}

	// Flatten vectors for C interface
	numVectors := len(vectors)
	flatVectors := make([]int8, numVectors*g.vectorDim)
	
	for i, vec := range vectors {
		copy(flatVectors[i*g.vectorDim:(i+1)*g.vectorDim], vec)
	}

	// Train using CUDA kernels
	result := C.torch_indexer_train(
		g.handle,
		(*C.schar)(unsafe.Pointer(&flatVectors[0])),
		C.int(numVectors),
		C.int(g.vectorDim),
	)

	if result == 0 {
		return fmt.Errorf("GPU training failed")
	}

	fmt.Printf("✅ GPU Index trained with %d vectors\n", numVectors)
	return nil
}

// AddVectors adds vectors to the index (CUDA-accelerated batch processing)
func (g *GPUIndexer) AddVectors(vectors [][]int8) error {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if len(vectors) == 0 {
		return fmt.Errorf("no vectors to add")
	}

	// Process in optimal GPU batch sizes
	batchSize := g.getOptimalBatchSize()
	
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		batch := vectors[i:end]
		if err := g.addVectorBatch(batch); err != nil {
			return fmt.Errorf("failed to add batch %d-%d: %w", i, end, err)
		}
	}

	g.isReady = true
	fmt.Printf("✅ Added %d vectors to GPU index\n", len(vectors))
	return nil
}

// addVectorBatch adds a batch of vectors using CUDA
func (g *GPUIndexer) addVectorBatch(vectors [][]int8) error {
	numVectors := len(vectors)
	flatVectors := make([]int8, numVectors*g.vectorDim)
	
	for i, vec := range vectors {
		if len(vec) != g.vectorDim {
			return fmt.Errorf("vector dimension mismatch at index %d", i)
		}
		copy(flatVectors[i*g.vectorDim:(i+1)*g.vectorDim], vec)
	}

	result := C.torch_indexer_add_vectors(
		g.handle,
		(*C.schar)(unsafe.Pointer(&flatVectors[0])),
		C.int(numVectors),
		C.int(g.vectorDim),
	)

	if result == 0 {
		return fmt.Errorf("GPU vector addition failed")
	}

	return nil
}

// Search performs CUDA-accelerated similarity search  
func (g *GPUIndexer) Search(query []int8, k int) ([]SearchResult, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if !g.isReady {
		return nil, fmt.Errorf("index not ready - no vectors added")
	}

	if len(query) != g.vectorDim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", 
			g.vectorDim, len(query))
	}

	// Perform GPU-accelerated search
	result := C.torch_indexer_search(
		g.handle,
		(*C.schar)(unsafe.Pointer(&query[0])),
		C.int(g.vectorDim),
		C.int(k),
	)

	if result.count == 0 {
		return []SearchResult{}, nil
	}

	// Convert C results to Go
	results := make([]SearchResult, int(result.count))
	
	// Access C arrays safely
	ids := (*[1 << 30]C.int)(unsafe.Pointer(result.ids))[:result.count:result.count]
	scores := (*[1 << 30]C.float)(unsafe.Pointer(result.scores))[:result.count:result.count]

	for i := 0; i < int(result.count); i++ {
		results[i] = SearchResult{
			ID:         int(ids[i]),
			Similarity: float32(scores[i]),
		}
	}

	// Free C memory
	C.torch_search_result_free(&result)

	return results, nil
}

// BatchSearch performs CUDA-accelerated batch similarity search
func (g *GPUIndexer) BatchSearch(queries [][]int8, k int) ([][]SearchResult, error) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	if !g.isReady {
		return nil, fmt.Errorf("index not ready")
	}

	if len(queries) == 0 {
		return [][]SearchResult{}, nil
	}

	// Process queries in parallel on GPU
	results := make([][]SearchResult, len(queries))
	var wg sync.WaitGroup
	var mu sync.Mutex
	errors := make([]error, len(queries))

	// Optimal GPU batch processing
	batchSize := g.getOptimalQueryBatchSize()
	
	for i := 0; i < len(queries); i += batchSize {
		wg.Add(1)
		go func(startIdx int) {
			defer wg.Done()
			
			endIdx := startIdx + batchSize
			if endIdx > len(queries) {
				endIdx = len(queries)
			}

			for j := startIdx; j < endIdx; j++ {
				queryResults, err := g.Search(queries[j], k)
				
				mu.Lock()
				results[j] = queryResults
				errors[j] = err
				mu.Unlock()
			}
		}(i)
	}

	wg.Wait()

	// Check for errors
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("query %d failed: %w", i, err)
		}
	}

	return results, nil
}

// GetStats returns GPU indexer statistics
func (g *GPUIndexer) GetStats() IndexStats {
	g.mutex.RLock()
	defer g.mutex.RUnlock()

	stats := C.torch_indexer_get_stats(g.handle)
	
	return IndexStats{
		NumVectors:       int(stats.num_vectors),
		VectorDim:        int(stats.vector_dim),
		IVFClusters:      int(stats.ivf_clusters),
		PQSubquantizers:  int(stats.pq_subquantizers),
		GPUMemoryMB:      float32(stats.gpu_memory_mb),
		IsTrained:        stats.is_trained != 0,
		IndexBuilt:       stats.index_built != 0,
	}
}

// IndexStats provides indexer statistics
type IndexStats struct {
	NumVectors      int
	VectorDim       int
	IVFClusters     int
	PQSubquantizers int
	GPUMemoryMB     float32
	IsTrained       bool
	IndexBuilt      bool
}

// getOptimalBatchSize determines optimal batch size for GPU processing
func (g *GPUIndexer) getOptimalBatchSize() int {
	// RTX 3080 with 16GB can handle large batches
	// Adjust based on vector dimension and available memory
	baseSize := 1024
	
	// Scale down for larger dimensions
	if g.vectorDim > 512 {
		baseSize /= 2
	}
	if g.vectorDim > 1024 {
		baseSize /= 2
	}

	return baseSize
}

// getOptimalQueryBatchSize determines optimal query batch size
func (g *GPUIndexer) getOptimalQueryBatchSize() int {
	// For queries, we can process more in parallel
	return 64
}

// Close gracefully closes the GPU indexer
func (g *GPUIndexer) Close() error {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if g.handle != nil {
		C.torch_indexer_destroy(g.handle)
		g.handle = nil
		runtime.SetFinalizer(g, nil)
	}

	return nil
}

// destroy is called by finalizer
func (g *GPUIndexer) destroy() {
	g.Close()
}

// AddEmbeddingToGPU adds an int8 embedding to the GPU index
func (g *GPUIndexer) AddEmbeddingToGPU(embedding *EmbedInt8Result, id int) error {
	if embedding == nil || len(embedding.Vector) != g.vectorDim {
		return fmt.Errorf("invalid embedding")
	}

	// Convert to single vector slice and add
	vectors := [][]int8{embedding.Vector}
	return g.AddVectors(vectors)
}

// SearchWithSIMDVector performs search using SIMD-optimized vector
func (g *GPUIndexer) SearchWithSIMDVector(vec *simd.Vec512, k int) ([]SearchResult, error) {
	// Convert SIMD vector to int8 slice
	query := make([]int8, len(vec))
	copy(query, vec[:])
	
	return g.Search(query, k)
}

// GetCUDAVersion returns the CUDA version string
func GetCUDAVersion() string {
	version := C.torch_get_version()
	return C.GoString(version)
}

// IsCUDAAvailable checks if CUDA is available
func IsCUDAAvailable() bool {
	return int(C.torch_cuda_is_available()) != 0
}

// GetCUDADeviceCount returns the number of CUDA devices
func GetCUDADeviceCount() int {
	return int(C.torch_cuda_device_count())
}