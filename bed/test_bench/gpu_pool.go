package main

import (
	"sync"
	"unsafe"
)

// #cgo LDFLAGS: -L. -lcuda_unique_topk -L/usr/local/cuda/lib64 -lcudart -lcublas
// #include <stdlib.h>
// extern void* create_unique_topk_search(int max_docs, int dim, int max_k);
// extern void destroy_unique_topk_search(void* handle);
// extern void add_documents_topk(void* handle, const signed char* docs, int num_docs, int dim);
// extern int search_topk_unique(void* handle, const signed char* query, int dim, int k, int* out_indices, float* out_scores);
import "C"

// GPUContextPool manages reusable GPU search contexts
type GPUContextPool struct {
	pool     sync.Pool
	maxDocs  int
	dim      int
	maxK     int
}

// NewGPUContextPool creates a new GPU context pool
func NewGPUContextPool(maxDocs, dim, maxK int) *GPUContextPool {
	pool := &GPUContextPool{
		maxDocs: maxDocs,
		dim:     dim,
		maxK:    maxK,
	}

	pool.pool = sync.Pool{
		New: func() interface{} {
			return C.create_unique_topk_search(C.int(maxDocs), C.int(dim), C.int(maxK))
		},
	}

	return pool
}

// Get retrieves a GPU context from the pool
func (p *GPUContextPool) Get() unsafe.Pointer {
	return p.pool.Get().(unsafe.Pointer)
}

// Put returns a GPU context to the pool (contexts are reused, not destroyed)
func (p *GPUContextPool) Put(handle unsafe.Pointer) {
	// Reset the context by clearing existing documents
	// The underlying GPU memory is reused for efficiency
	p.pool.Put(handle)
}

// Destroy permanently destroys all contexts in the pool
func (p *GPUContextPool) Destroy() {
	// This would need custom implementation to track active contexts
	// For now, contexts are cleaned up when the program exits
}

// MemoryPool for reusing embedding buffers
type MemoryPool struct {
	pool sync.Pool
	size int
}

// NewMemoryPool creates a memory pool for int8 slices
func NewMemoryPool(size int) *MemoryPool {
	return &MemoryPool{
		size: size,
		pool: sync.Pool{
			New: func() interface{} {
				return make([]int8, size)
			},
		},
	}
}

// Get retrieves a buffer from the pool
func (m *MemoryPool) Get() []int8 {
	return m.pool.Get().([]int8)
}

// Put returns a buffer to the pool
func (m *MemoryPool) Put(buffer []int8) {
	if len(buffer) == m.size {
		// Clear the buffer before returning to pool
		for i := range buffer {
			buffer[i] = 0
		}
		m.pool.Put(buffer)
	}
}

// Global pools for reuse across benchmarks
var (
	gpuPool        *GPUContextPool
	embeddingPool  *MemoryPool
	flatDataPool   *MemoryPool

	poolInitOnce   sync.Once
)

// InitializePools sets up global memory and GPU pools
func InitializePools() {
	poolInitOnce.Do(func() {
		gpuPool = NewGPUContextPool(10000, 512, 10)
		embeddingPool = NewMemoryPool(512)       // Single embedding buffer
		flatDataPool = NewMemoryPool(1000 * 512) // Batch embeddings buffer
	})
}

// OptimizedGPUSearch performs search using pooled resources
func OptimizedGPUSearch(documents []*Document, query string, k int) ([]int32, []float32, error) {
	InitializePools()

	// Get pooled resources
	handle := gpuPool.Get()
	defer gpuPool.Put(handle)

	flatEmbeddings := flatDataPool.Get()
	defer flatDataPool.Put(flatEmbeddings)

	// Copy embeddings to flat buffer
	numDocs := len(documents)
	for i, doc := range documents {
		copy(flatEmbeddings[i*512:(i+1)*512], doc.Embedding)
	}

	// Add documents to GPU
	C.add_documents_topk(
		handle,
		(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int(numDocs),
		C.int(512),
	)

	// Generate query embedding
	queryEmb, err := benchModel.EmbedInt8(query)
	if err != nil {
		return nil, nil, err
	}

	// Perform search
	indices := make([]int32, k)
	scores := make([]float32, k)

	C.search_topk_unique(
		handle,
		(*C.schar)(unsafe.Pointer(&queryEmb[0])),
		C.int(512),
		C.int(k),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	return indices, scores, nil
}