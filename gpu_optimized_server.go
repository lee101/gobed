//go:build legacy && gpu

package gobed

import (
	"errors"
	"log"
	"sync"

	"github.com/lee101/gobed/pkg/ann/simd"
)

var (
	// ErrNoIndexer is returned when no indexer is available
	ErrNoIndexer = errors.New("no indexer available")
)

// GPUBufferPool manages reusable buffers for GPU operations
type GPUBufferPool struct {
	// Pools for different buffer sizes
	int8Pool512   sync.Pool // For 512-dim int8 vectors
	float32Pool   sync.Pool // For float32 scales
	int32Pool     sync.Pool // For token IDs
	embeddingPool sync.Pool // For embedding results

	// Pre-allocated buffers for batch operations
	batchBuffers []BatchBuffer
	bufferLock   sync.Mutex
}

// BatchBuffer holds pre-allocated buffers for batch processing
type BatchBuffer struct {
	TokenIDs   []int32   // Reusable token buffer
	Embeddings []int8    // Reusable embedding buffer
	Scales     []float32 // Reusable scale buffer
	InUse      bool
}

// Global buffer pool instance
var globalBufferPool *GPUBufferPool
var poolOnce sync.Once

// GetBufferPool returns the global buffer pool instance
func GetBufferPool() *GPUBufferPool {
	poolOnce.Do(func() {
		globalBufferPool = NewGPUBufferPool()
	})
	return globalBufferPool
}

// NewGPUBufferPool creates a new buffer pool
func NewGPUBufferPool() *GPUBufferPool {
	pool := &GPUBufferPool{
		batchBuffers: make([]BatchBuffer, 16), // Pre-allocate 16 batch buffers
	}

	// Initialize int8 pool for 512-dim vectors
	pool.int8Pool512 = sync.Pool{
		New: func() interface{} {
			return make([]int8, 512)
		},
	}

	// Initialize float32 pool for scales
	pool.float32Pool = sync.Pool{
		New: func() interface{} {
			return make([]float32, 1)
		},
	}

	// Initialize int32 pool for token IDs
	pool.int32Pool = sync.Pool{
		New: func() interface{} {
			return make([]int32, 512) // Default max tokens
		},
	}

	// Initialize embedding result pool
	pool.embeddingPool = sync.Pool{
		New: func() interface{} {
			return &EmbedInt8Result{
				Vector: make([]int8, 512),
				Scale:  1.0,
			}
		},
	}

	// Pre-allocate batch buffers
	for i := range pool.batchBuffers {
		pool.batchBuffers[i] = BatchBuffer{
			TokenIDs:   make([]int32, 512),
			Embeddings: make([]int8, 512),
			Scales:     make([]float32, 1),
			InUse:      false,
		}
	}

	return pool
}

// GetInt8Buffer gets a reusable int8 buffer
func (p *GPUBufferPool) GetInt8Buffer() []int8 {
	return p.int8Pool512.Get().([]int8)
}

// PutInt8Buffer returns an int8 buffer to the pool
func (p *GPUBufferPool) PutInt8Buffer(buf []int8) {
	if cap(buf) == 512 {
		buf = buf[:512] // Reset to full capacity
		p.int8Pool512.Put(buf)
	}
}

// GetBatchBuffer gets a reusable batch buffer
func (p *GPUBufferPool) GetBatchBuffer() *BatchBuffer {
	p.bufferLock.Lock()
	defer p.bufferLock.Unlock()

	for i := range p.batchBuffers {
		if !p.batchBuffers[i].InUse {
			p.batchBuffers[i].InUse = true
			return &p.batchBuffers[i]
		}
	}

	// All buffers in use, create a new one
	return &BatchBuffer{
		TokenIDs:   make([]int32, 512),
		Embeddings: make([]int8, 512),
		Scales:     make([]float32, 1),
		InUse:      true,
	}
}

// ReleaseBatchBuffer returns a batch buffer to the pool
func (p *GPUBufferPool) ReleaseBatchBuffer(buf *BatchBuffer) {
	buf.InUse = false
}

// OptimizedGPUSearch performs GPU search with minimal allocations
func OptimizedGPUSearch(indexer *GPUIndexer, embedding *EmbedInt8Result, k int) ([]int32, []float32, error) {
	pool := GetBufferPool()

	// Get a buffer from pool (already sized to 512)
	int8Vec := pool.GetInt8Buffer()
	defer pool.PutInt8Buffer(int8Vec)

	// Direct copy, no allocation - we know it's exactly 512 dimensions
	copy(int8Vec, embedding.Vector[:512])

	// Perform search
	return indexer.Search(int8Vec, embedding.Scale, k)
}

// OptimizedBatchIndex indexes vectors with minimal allocations
func OptimizedBatchIndex(indexer *GPUIndexer, embeddings []*EmbedInt8Result) error {
	if len(embeddings) == 0 {
		return nil
	}

	pool := GetBufferPool()

	// Pre-allocate once for entire batch
	vectors := make([][]int8, 0, len(embeddings))
	scales := make([]float32, 0, len(embeddings))

	for _, emb := range embeddings {
		if emb == nil {
			continue
		}

		// Get buffer from pool
		vec := pool.GetInt8Buffer()

		// Copy only first 512 dimensions
		truncSize := 512
		if len(emb.Vector) < truncSize {
			truncSize = len(emb.Vector)
		}
		copy(vec[:truncSize], emb.Vector[:truncSize])

		vectors = append(vectors, vec)
		scales = append(scales, emb.Scale)
	}

	// Add vectors to GPU
	err := indexer.AddVectors(vectors, scales)

	// Return buffers to pool
	for _, vec := range vectors {
		pool.PutInt8Buffer(vec)
	}

	return err
}

// OptimizedGPUSearchHandler handles search requests with optimized memory usage
type OptimizedGPUSearchHandler struct {
	gpuIndexer *GPUIndexer
	cpuIndexer *SharedMemoryIndex
	bufferPool *GPUBufferPool
	useGPU     bool
}

// NewOptimizedGPUSearchHandler creates an optimized search handler
func NewOptimizedGPUSearchHandler(gpuIndexer *GPUIndexer, cpuIndexer *SharedMemoryIndex) *OptimizedGPUSearchHandler {
	return &OptimizedGPUSearchHandler{
		gpuIndexer: gpuIndexer,
		cpuIndexer: cpuIndexer,
		bufferPool: GetBufferPool(),
		useGPU:     gpuIndexer != nil,
	}
}

// Search performs optimized search
func (h *OptimizedGPUSearchHandler) Search(embedding *EmbedInt8Result, k int) ([]int32, []float32, error) {
	if h.useGPU && h.gpuIndexer != nil {
		// Use optimized GPU search with buffer pool
		return OptimizedGPUSearch(h.gpuIndexer, embedding, k)
	}

	// Fallback to CPU
	if h.cpuIndexer != nil {
		var vec simd.Vec512
		copy(vec[:], embedding.Vector[:512])
		results := h.cpuIndexer.SearchTopK(&vec, k)

		indices := make([]int32, len(results))
		scores := make([]float32, len(results))
		for i, r := range results {
			indices[i] = int32(r.ID)
			scores[i] = r.Similarity
		}
		return indices, scores, nil
	}

	return nil, nil, ErrNoIndexer
}

// BatchAddVectors adds multiple vectors with optimized memory usage
func (h *OptimizedGPUSearchHandler) BatchAddVectors(embeddings []*EmbedInt8Result) error {
	if h.useGPU && h.gpuIndexer != nil {
		return OptimizedBatchIndex(h.gpuIndexer, embeddings)
	}

	// CPU fallback
	if h.cpuIndexer != nil {
		for i, emb := range embeddings {
			if emb == nil {
				continue
			}
			var vec simd.Vec512
			copy(vec[:], emb.Vector[:512])
			if err := h.cpuIndexer.AddVector(&vec, emb.Scale, i); err != nil {
				log.Printf("Failed to add vector %d: %v", i, err)
			}
		}
		h.cpuIndexer.Sync()
		return nil
	}

	return ErrNoIndexer
}

// ZeroCopySearch performs search with zero-copy GPU transfer
func ZeroCopySearch(indexer *GPUIndexer, tokenIDs []int32, k int) ([]int32, []float32, error) {
	// Direct GPU search with tokens - no intermediate embedding copy
	return indexer.SearchWithTokens(tokenIDs, k)
}

// OptimizedEmbedAndSearch embeds and searches in one operation
func OptimizedEmbedAndSearch(model *EmbeddingModel, indexer *GPUIndexer, text string, k int) ([]int32, []float32, error) {
	// Tokenize
	text = normalizeText(text)
	encoding, err := model.tokenizer.EncodeSingle(text, false)
	if err != nil {
		return nil, nil, err
	}

	// Convert tokens directly to int32 array (no intermediate allocation)
	tokenIDs := make([]int32, len(encoding.Ids))
	for i, id := range encoding.Ids {
		tokenIDs[i] = int32(id)
	}

	// Direct GPU search with tokens
	return ZeroCopySearch(indexer, tokenIDs, k)
}
