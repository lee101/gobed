// +build libtorch

package gpu

/*
#cgo CPPFLAGS: -I${SRCDIR} -I${SRCDIR}/../libtorch/include -I${SRCDIR}/../libtorch/include/torch/csrc/api/include
#cgo LDFLAGS: -L${SRCDIR}/../libtorch/lib -L${SRCDIR} -ltorch -ltorch_cuda -ltorch_cpu -ldl
#include "torch_cgo_wrapper.h"
#include <stdlib.h>
#include <dlfcn.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// TorchNativeIndexer provides high-performance GPU indexing using LibTorch C++ API
type TorchNativeIndexer struct {
	handle C.TorchIndexerHandle
	config TorchNativeConfig
}

// TorchNativeConfig holds configuration for the native LibTorch indexer
type TorchNativeConfig struct {
	VectorDim         int `json:"vector_dim"`
	NumSubquantizers  int `json:"num_subquantizers"`
	CodebookSize      int `json:"codebook_size"`
	IVFClusters       int `json:"ivf_clusters"`
	ProbeLists        int `json:"probe_lists"`
	RerankK           int `json:"rerank_k"`
	DeviceID          int `json:"device_id"`
}

// DefaultTorchNativeConfig returns a reasonable default configuration
func DefaultTorchNativeConfig() TorchNativeConfig {
	return TorchNativeConfig{
		VectorDim:        512,
		NumSubquantizers: 64,
		CodebookSize:     256,
		IVFClusters:      4096,
		ProbeLists:       64,
		RerankK:          1000,
		DeviceID:         0,
	}
}

// NewTorchNativeIndexer creates a new LibTorch-based indexer
func NewTorchNativeIndexer(config TorchNativeConfig) (*TorchNativeIndexer, error) {
	// Check CUDA availability
	if C.torch_cuda_is_available() == 0 {
		return nil, errors.New("CUDA is not available")
	}

	deviceCount := int(C.torch_cuda_device_count())
	if config.DeviceID >= deviceCount {
		return nil, fmt.Errorf("device ID %d is not available (found %d devices)", config.DeviceID, deviceCount)
	}

	// Convert config to C struct
	cConfig := C.IndexConfig{
		vector_dim:         C.int(config.VectorDim),
		num_subquantizers:  C.int(config.NumSubquantizers),
		codebook_size:      C.int(config.CodebookSize),
		ivf_clusters:       C.int(config.IVFClusters),
		probe_lists:        C.int(config.ProbeLists),
		rerank_k:          C.int(config.RerankK),
		device_id:         C.int(config.DeviceID),
	}

	// Create indexer
	handle := C.torch_indexer_create(cConfig)
	if handle == nil {
		return nil, errors.New("failed to create LibTorch indexer")
	}

	indexer := &TorchNativeIndexer{
		handle: handle,
		config: config,
	}

	// Set finalizer to ensure cleanup
	runtime.SetFinalizer(indexer, (*TorchNativeIndexer).finalize)

	fmt.Printf("🚀 Created LibTorch native indexer\n")
	fmt.Printf("   Vector dim: %d\n", config.VectorDim)
	fmt.Printf("   PQ: %dx%d\n", config.NumSubquantizers, config.CodebookSize)
	fmt.Printf("   IVF clusters: %d\n", config.IVFClusters)
	fmt.Printf("   Device: CUDA:%d\n", config.DeviceID)

	return indexer, nil
}

// TrainIndex trains the indexer using training vectors
func (t *TorchNativeIndexer) TrainIndex(vectors [][]int8) error {
	if len(vectors) == 0 {
		return errors.New("empty training vectors")
	}

	if len(vectors[0]) != t.config.VectorDim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", 
			t.config.VectorDim, len(vectors[0]))
	}

	// Flatten vectors to C array
	n := len(vectors)
	d := len(vectors[0])
	flatVectors := make([]int8, n*d)

	for i, vec := range vectors {
		copy(flatVectors[i*d:(i+1)*d], vec)
	}

	// Call C function
	result := C.torch_indexer_train(
		t.handle,
		(*C.schar)(unsafe.Pointer(&flatVectors[0])),
		C.int(n),
		C.int(d),
	)

	if result == 0 {
		return errors.New("failed to train index")
	}

	fmt.Printf("✅ Trained index with %d vectors\n", n)
	return nil
}

// AddVectors adds vectors to the trained index
func (t *TorchNativeIndexer) AddVectors(vectors [][]int8) error {
	if len(vectors) == 0 {
		return errors.New("empty vectors")
	}

	if len(vectors[0]) != t.config.VectorDim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", 
			t.config.VectorDim, len(vectors[0]))
	}

	// Flatten vectors to C array
	n := len(vectors)
	d := len(vectors[0])
	flatVectors := make([]int8, n*d)

	for i, vec := range vectors {
		copy(flatVectors[i*d:(i+1)*d], vec)
	}

	// Call C function
	result := C.torch_indexer_add_vectors(
		t.handle,
		(*C.schar)(unsafe.Pointer(&flatVectors[0])),
		C.int(n),
		C.int(d),
	)

	if result == 0 {
		return errors.New("failed to add vectors to index")
	}

	fmt.Printf("📚 Added %d vectors to index\n", n)
	return nil
}

// Search performs k-nearest neighbor search
func (t *TorchNativeIndexer) Search(query []int8, k int) ([]int, []float32, error) {
	if len(query) != t.config.VectorDim {
		return nil, nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", 
			t.config.VectorDim, len(query))
	}

	// Call C function
	result := C.torch_indexer_search(
		t.handle,
		(*C.schar)(unsafe.Pointer(&query[0])),
		C.int(len(query)),
		C.int(k),
	)

	if result.count == 0 {
		return []int{}, []float32{}, nil
	}

	// Convert C arrays to Go slices
	count := int(result.count)
	ids := make([]int, count)
	scores := make([]float32, count)

	// Copy data from C arrays
	cIds := (*[1 << 30]C.int)(unsafe.Pointer(result.ids))[:count:count]
	cScores := (*[1 << 30]C.float)(unsafe.Pointer(result.scores))[:count:count]

	for i := 0; i < count; i++ {
		ids[i] = int(cIds[i])
		scores[i] = float32(cScores[i])
	}

	// Free C memory
	C.torch_search_result_free(&result)

	return ids, scores, nil
}

// GetStats returns indexer statistics
func (t *TorchNativeIndexer) GetStats() (TorchNativeStats, error) {
	cStats := C.torch_indexer_get_stats(t.handle)

	stats := TorchNativeStats{
		NumVectors:      int(cStats.num_vectors),
		VectorDim:       int(cStats.vector_dim),
		IVFClusters:     int(cStats.ivf_clusters),
		PQSubquantizers: int(cStats.pq_subquantizers),
		GPUMemoryMB:     float64(cStats.gpu_memory_mb),
		IsTrained:       cStats.is_trained != 0,
		IndexBuilt:      cStats.index_built != 0,
	}

	return stats, nil
}

// Close releases the indexer resources
func (t *TorchNativeIndexer) Close() error {
	if t.handle != nil {
		C.torch_indexer_destroy(t.handle)
		t.handle = nil
		runtime.SetFinalizer(t, nil)
	}
	return nil
}

// finalize is called by the garbage collector
func (t *TorchNativeIndexer) finalize() {
	t.Close()
}

// TorchNativeStats holds indexer statistics
type TorchNativeStats struct {
	NumVectors      int     `json:"num_vectors"`
	VectorDim       int     `json:"vector_dim"`
	IVFClusters     int     `json:"ivf_clusters"`
	PQSubquantizers int     `json:"pq_subquantizers"`
	GPUMemoryMB     float64 `json:"gpu_memory_mb"`
	IsTrained       bool    `json:"is_trained"`
	IndexBuilt      bool    `json:"index_built"`
}

// TorchNativePipeline provides a complete GPU pipeline using LibTorch native API
type TorchNativePipeline struct {
	indexer  *TorchNativeIndexer
	embedder EmbedderInterface
	texts    []string
	config   TorchNativeConfig
}

// EmbedderInterface defines the interface for embedding models
type EmbedderInterface interface {
	Encode(text string) ([]float32, error)
}

// NewTorchNativePipeline creates a new pipeline with LibTorch native indexer
func NewTorchNativePipeline(config TorchNativeConfig, embedder EmbedderInterface) (*TorchNativePipeline, error) {
	indexer, err := NewTorchNativeIndexer(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create indexer: %w", err)
	}

	return &TorchNativePipeline{
		indexer:  indexer,
		embedder: embedder,
		texts:    make([]string, 0),
		config:   config,
	}, nil
}

// TrainPipeline trains the pipeline with training texts
func (p *TorchNativePipeline) TrainPipeline(trainingTexts []string) error {
	if len(trainingTexts) == 0 {
		return errors.New("empty training texts")
	}

	fmt.Printf("🔧 Training pipeline with %d texts...\n", len(trainingTexts))

	// Generate embeddings for training
	vectors := make([][]int8, len(trainingTexts))
	for i, text := range trainingTexts {
		embedding, err := p.embedder.Encode(text)
		if err != nil {
			return fmt.Errorf("failed to encode training text %d: %w", i, err)
		}
		vectors[i] = Float32ToInt8(embedding)
	}

	// Train the indexer
	return p.indexer.TrainIndex(vectors)
}

// IndexTexts processes and indexes texts
func (p *TorchNativePipeline) IndexTexts(texts []string) error {
	if len(texts) == 0 {
		return nil
	}

	fmt.Printf("📚 Indexing %d texts...\n", len(texts))

	// Generate embeddings
	vectors := make([][]int8, len(texts))
	for i, text := range texts {
		embedding, err := p.embedder.Encode(text)
		if err != nil {
			return fmt.Errorf("failed to encode text %d: %w", i, err)
		}
		vectors[i] = Float32ToInt8(embedding)
	}

	// Store texts for search results
	p.texts = append(p.texts, texts...)

	// Add to indexer
	return p.indexer.AddVectors(vectors)
}

// Search performs similarity search
func (p *TorchNativePipeline) Search(query string, k int) ([]Result, error) {
	// Encode query
	queryEmb, err := p.embedder.Encode(query)
	if err != nil {
		return nil, fmt.Errorf("failed to encode query: %w", err)
	}

	queryInt8 := Float32ToInt8(queryEmb)

	// Search using indexer
	ids, scores, err := p.indexer.Search(queryInt8, k)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Build results
	results := make([]Result, len(ids))
	for i, id := range ids {
		if id < len(p.texts) {
			results[i] = Result{
				Text:  p.texts[id],
				Score: scores[i],
				ID:    id,
			}
		}
	}

	return results, nil
}

// GetPipelineStats returns pipeline statistics
func (p *TorchNativePipeline) GetPipelineStats() (*Stats, error) {
	nativeStats, err := p.indexer.GetStats()
	if err != nil {
		return nil, err
	}

	return &Stats{
		NumTexts:       len(p.texts),
		NumEmbeddings:  nativeStats.NumVectors,
		GPUDevice:      fmt.Sprintf("CUDA:%d", p.config.DeviceID),
		GPUMemoryMB:    nativeStats.GPUMemoryMB,
		CPUMemoryMB:    0, // LibTorch handles memory
		SingleQueryMS:  0, // Would need benchmarking
		SingleQueryQPS: 0,
		BatchQPS:       0,
	}, nil
}

// Close releases all pipeline resources
func (p *TorchNativePipeline) Close() error {
	if p.indexer != nil {
		return p.indexer.Close()
	}
	return nil
}

// GetTorchInfo returns information about LibTorch
func GetTorchInfo() (string, bool, int) {
	version := C.GoString(C.torch_get_version())
	cudaAvailable := C.torch_cuda_is_available() != 0
	deviceCount := int(C.torch_cuda_device_count())
	
	return version, cudaAvailable, deviceCount
}