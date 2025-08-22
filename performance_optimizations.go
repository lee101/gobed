package gobed

import (
	"sync"

	"github.com/lee101/gobed/ann/simd"
)

// ObjectPool provides reusable object pools to reduce allocations
type ObjectPool struct {
	vectorPool    sync.Pool
	slicePool     sync.Pool
	embeddingPool sync.Pool
}

// NewObjectPool creates optimized object pools
func NewObjectPool() *ObjectPool {
	return &ObjectPool{
		vectorPool: sync.Pool{
			New: func() interface{} {
				return &simd.Vec512{}
			},
		},
		slicePool: sync.Pool{
			New: func() interface{} {
				// Pre-allocate slice with reasonable capacity
				slice := make([]float32, 0, 1024)
				return &slice
			},
		},
		embeddingPool: sync.Pool{
			New: func() interface{} {
				return &EmbedInt8Result{
					Vector: make([]int8, 512),
					Scale:  1.0,
				}
			},
		},
	}
}

// GetVector gets a reusable vector from pool
func (p *ObjectPool) GetVector() *simd.Vec512 {
	return p.vectorPool.Get().(*simd.Vec512)
}

// PutVector returns vector to pool
func (p *ObjectPool) PutVector(vec *simd.Vec512) {
	// Clear the vector before reuse
	for i := range vec {
		vec[i] = 0
	}
	p.vectorPool.Put(vec)
}

// GetSlice gets a reusable float32 slice from pool
func (p *ObjectPool) GetSlice() *[]float32 {
	slice := p.slicePool.Get().(*[]float32)
	*slice = (*slice)[:0] // Reset length but keep capacity
	return slice
}

// PutSlice returns slice to pool
func (p *ObjectPool) PutSlice(slice *[]float32) {
	// Don't store slices that are too large to avoid memory waste
	if cap(*slice) < 4096 {
		p.slicePool.Put(slice)
	}
}

// GetEmbedding gets a reusable embedding result from pool
func (p *ObjectPool) GetEmbedding() *EmbedInt8Result {
	result := p.embeddingPool.Get().(*EmbedInt8Result)
	// Clear the vector
	for i := range result.Vector {
		result.Vector[i] = 0
	}
	result.Scale = 1.0
	return result
}

// PutEmbedding returns embedding to pool
func (p *ObjectPool) PutEmbedding(emb *EmbedInt8Result) {
	p.embeddingPool.Put(emb)
}

// BatchProcessor provides optimized batch processing
type BatchProcessor struct {
	pool      *ObjectPool
	batchSize int
	workers   int
}

// NewBatchProcessor creates an optimized batch processor
func NewBatchProcessor(batchSize, workers int) *BatchProcessor {
	return &BatchProcessor{
		pool:      NewObjectPool(),
		batchSize: batchSize,
		workers:   workers,
	}
}

// ProcessBatch processes documents in optimized batches
func (bp *BatchProcessor) ProcessBatch(texts []string, model *EmbeddingModel) ([]simd.Vec512, []float32, error) {
	n := len(texts)
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)

	// Process in chunks to reduce memory allocation peaks
	chunkSize := bp.batchSize
	if chunkSize <= 0 {
		chunkSize = 100 // Default chunk size
	}

	for i := 0; i < n; i += chunkSize {
		end := min(i+chunkSize, n)
		chunk := texts[i:end]

		// Process chunk
		for j, text := range chunk {
			embedding, err := model.EmbedInt8(text)
			if err != nil {
				return nil, nil, err
			}

			copy(vectors[i+j][:], embedding.Vector)
			scales[i+j] = embedding.Scale
		}
	}

	return vectors, scales, nil
}

// VectorBuffer provides a reusable buffer for vector operations
type VectorBuffer struct {
	vectors  []simd.Vec512
	scales   []float32
	capacity int
}

// NewVectorBuffer creates a vector buffer with specified capacity
func NewVectorBuffer(capacity int) *VectorBuffer {
	return &VectorBuffer{
		vectors:  make([]simd.Vec512, 0, capacity),
		scales:   make([]float32, 0, capacity),
		capacity: capacity,
	}
}

// Reset clears the buffer for reuse
func (vb *VectorBuffer) Reset() {
	vb.vectors = vb.vectors[:0]
	vb.scales = vb.scales[:0]
}

// Add adds a vector to the buffer
func (vb *VectorBuffer) Add(vec simd.Vec512, scale float32) {
	vb.vectors = append(vb.vectors, vec)
	vb.scales = append(vb.scales, scale)
}

// IsFull returns true if buffer is at capacity
func (vb *VectorBuffer) IsFull() bool {
	return len(vb.vectors) >= vb.capacity
}

// GetVectors returns the current vectors and scales
func (vb *VectorBuffer) GetVectors() ([]simd.Vec512, []float32) {
	return vb.vectors, vb.scales
}

// Len returns current buffer length
func (vb *VectorBuffer) Len() int {
	return len(vb.vectors)
}

// MemoryOptimizedCache provides a cache with memory management
type MemoryOptimizedCache struct {
	cache       map[string]*EmbedInt8Result
	maxSize     int
	currentSize int
	mu          sync.RWMutex
}

// NewMemoryOptimizedCache creates a memory-optimized embedding cache
func NewMemoryOptimizedCache(maxSize int) *MemoryOptimizedCache {
	return &MemoryOptimizedCache{
		cache:   make(map[string]*EmbedInt8Result),
		maxSize: maxSize,
	}
}

// Get retrieves an embedding from cache
func (c *MemoryOptimizedCache) Get(text string) (*EmbedInt8Result, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result, exists := c.cache[text]
	return result, exists
}

// Put stores an embedding in cache
func (c *MemoryOptimizedCache) Put(text string, embedding *EmbedInt8Result) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if we're at capacity
	if c.currentSize >= c.maxSize {
		// Simple eviction: remove oldest entries (in Go map iteration order)
		count := 0
		for key := range c.cache {
			delete(c.cache, key)
			count++
			if count >= c.maxSize/4 { // Remove 25% of cache
				break
			}
		}
		c.currentSize = len(c.cache)
	}

	// Copy the embedding to avoid external modifications
	cached := &EmbedInt8Result{
		Vector: make([]int8, len(embedding.Vector)),
		Scale:  embedding.Scale,
	}
	copy(cached.Vector, embedding.Vector)

	c.cache[text] = cached
	c.currentSize++
}

// Clear clears the cache
func (c *MemoryOptimizedCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache = make(map[string]*EmbedInt8Result)
	c.currentSize = 0
}

// Size returns current cache size
func (c *MemoryOptimizedCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.currentSize
}

// Utility function for min (removed - using existing one in search_api.go)
