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

// DynamicGPUPool manages GPU contexts of different sizes
type DynamicGPUPool struct {
	pools map[int]*GPUContextPool
	mutex sync.RWMutex
	dim   int
	maxK  int
}

// NewDynamicGPUPool creates a dynamic GPU pool that can handle various document counts
func NewDynamicGPUPool(dim, maxK int) *DynamicGPUPool {
	return &DynamicGPUPool{
		pools: make(map[int]*GPUContextPool),
		dim:   dim,
		maxK:  maxK,
	}
}

// GetContextForSize returns a GPU context that can handle the specified number of documents
func (d *DynamicGPUPool) GetContextForSize(numDocs int) unsafe.Pointer {
	// Round up to the nearest power of 2 with minimum of 1024
	size := nextPowerOf2(max(numDocs+100, 1024))

	d.mutex.RLock()
	pool, exists := d.pools[size]
	d.mutex.RUnlock()

	if !exists {
		// Create new pool for this size
		d.mutex.Lock()
		// Double-check in case another goroutine created it
		if pool, exists = d.pools[size]; !exists {
			pool = NewGPUContextPool(size, d.dim, d.maxK)
			d.pools[size] = pool
		}
		d.mutex.Unlock()
	}

	return pool.Get()
}

// PutContext returns a context to the appropriate pool
func (d *DynamicGPUPool) PutContext(handle unsafe.Pointer, numDocs int) {
	size := nextPowerOf2(max(numDocs+100, 1024))

	d.mutex.RLock()
	pool, exists := d.pools[size]
	d.mutex.RUnlock()

	if exists {
		pool.Put(handle)
	} else {
		// This shouldn't happen, but if it does, destroy the context
		C.destroy_unique_topk_search(handle)
	}
}

// DynamicMemoryPool manages memory buffers of different sizes
type DynamicMemoryPool struct {
	pools map[int]*MemoryPool
	mutex sync.RWMutex
}

// NewDynamicMemoryPool creates a dynamic memory pool
func NewDynamicMemoryPool() *DynamicMemoryPool {
	return &DynamicMemoryPool{
		pools: make(map[int]*MemoryPool),
	}
}

// GetBufferForSize returns a buffer that can handle the specified size
func (d *DynamicMemoryPool) GetBufferForSize(size int) []int8 {
	// Round up to nearest power of 2 with minimum of 512
	poolSize := nextPowerOf2(max(size, 512))

	d.mutex.RLock()
	pool, exists := d.pools[poolSize]
	d.mutex.RUnlock()

	if !exists {
		d.mutex.Lock()
		if pool, exists = d.pools[poolSize]; !exists {
			pool = NewMemoryPool(poolSize)
			d.pools[poolSize] = pool
		}
		d.mutex.Unlock()
	}

	buffer := pool.Get()
	// Return only the requested size
	return buffer[:size]
}

// PutBuffer returns a buffer to the appropriate pool
func (d *DynamicMemoryPool) PutBuffer(buffer []int8) {
	poolSize := nextPowerOf2(max(cap(buffer), 512))

	d.mutex.RLock()
	pool, exists := d.pools[poolSize]
	d.mutex.RUnlock()

	if exists {
		// Restore buffer to full capacity before returning to pool
		fullBuffer := buffer[:cap(buffer)]
		pool.Put(fullBuffer)
	}
	// If pool doesn't exist, just let GC handle the buffer
}

// Utility functions
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func nextPowerOf2(n int) int {
	if n <= 1 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n++
	return n
}

// Global dynamic pools
var (
	dynamicGPUPool    *DynamicGPUPool
	dynamicMemoryPool *DynamicMemoryPool
	dynamicInitOnce   sync.Once
)

// InitializeDynamicPools sets up dynamic pools that can handle any size
func InitializeDynamicPools() {
	dynamicInitOnce.Do(func() {
		dynamicGPUPool = NewDynamicGPUPool(512, 10)
		dynamicMemoryPool = NewDynamicMemoryPool()
	})
}

// OptimizedGPUSearchDynamic performs search using dynamic pooled resources
func OptimizedGPUSearchDynamic(documents []*Document, query string, k int) ([]int32, []float32, error) {
	InitializeDynamicPools()

	numDocs := len(documents)

	// Get dynamically sized resources
	handle := dynamicGPUPool.GetContextForSize(numDocs)
	defer dynamicGPUPool.PutContext(handle, numDocs)

	flatEmbeddings := dynamicMemoryPool.GetBufferForSize(numDocs * 512)
	defer dynamicMemoryPool.PutBuffer(flatEmbeddings)

	// Copy embeddings to flat buffer
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

// BatchOptimizedSearch processes documents in chunks if they exceed GPU memory
func BatchOptimizedSearch(documents []*Document, query string, k int, maxBatchSize int) ([]int32, []float32, error) {
	if len(documents) <= maxBatchSize {
		return OptimizedGPUSearchDynamic(documents, query, k)
	}

	// Process in batches and merge results
	type Result struct {
		index int
		score float32
	}

	var allResults []Result

	for i := 0; i < len(documents); i += maxBatchSize {
		end := min(i+maxBatchSize, len(documents))
		batch := documents[i:end]

		indices, scores, err := OptimizedGPUSearchDynamic(batch, query, k)
		if err != nil {
			return nil, nil, err
		}

		// Adjust indices to global document space and collect results
		for j := 0; j < len(indices) && indices[j] >= 0; j++ {
			allResults = append(allResults, Result{
				index: i + int(indices[j]),
				score: scores[j],
			})
		}
	}

	// Sort all results by score (descending)
	for i := 0; i < len(allResults)-1; i++ {
		for j := i + 1; j < len(allResults); j++ {
			if allResults[j].score > allResults[i].score {
				allResults[i], allResults[j] = allResults[j], allResults[i]
			}
		}
	}

	// Return top k results
	resultCount := min(k, len(allResults))
	indices := make([]int32, resultCount)
	scores := make([]float32, resultCount)

	for i := 0; i < resultCount; i++ {
		indices[i] = int32(allResults[i].index)
		scores[i] = allResults[i].score
	}

	return indices, scores, nil
}