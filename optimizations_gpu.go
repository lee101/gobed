// +build gpu

package gobed

import (
	"runtime"
	"sync"
)

// BufferPool manages reusable buffers to reduce allocations
type BufferPool struct {
	tokenBuffers sync.Pool
	embedBuffers sync.Pool
	int8Buffers  sync.Pool
}

// Create a wrapper that adapts GPUBufferPool to BufferPool interface
var cpuCompatBufferPool = &BufferPool{
	tokenBuffers: sync.Pool{
		New: func() interface{} {
			// Pre-allocate for max sequence length
			return make([]int, 0, 512)
		},
	},
	embedBuffers: sync.Pool{
		New: func() interface{} {
			// Pre-allocate for embedding dimension
			return make([]float32, 1024)
		},
	},
	int8Buffers: sync.Pool{
		New: func() interface{} {
			// Pre-allocate for quantized embeddings
			return make([]int8, 512)
		},
	},
}

// GetTokenBuffer retrieves a reusable token buffer
func GetTokenBuffer() []int {
	return cpuCompatBufferPool.tokenBuffers.Get().([]int)
}

// PutTokenBuffer returns a token buffer to the pool
func PutTokenBuffer(buf []int) {
	if cap(buf) <= 2048 { // Don't pool huge buffers
		buf = buf[:0]
		cpuCompatBufferPool.tokenBuffers.Put(buf)
	}
}

// GetEmbedBuffer retrieves a reusable embedding buffer
func GetEmbedBuffer() []float32 {
	return cpuCompatBufferPool.embedBuffers.Get().([]float32)
}

// PutEmbedBuffer returns an embedding buffer to the pool
func PutEmbedBuffer(buf []float32) {
	cpuCompatBufferPool.embedBuffers.Put(buf)
}

// GetInt8Buffer retrieves a reusable int8 buffer
func GetInt8Buffer() []int8 {
	return cpuCompatBufferPool.int8Buffers.Get().([]int8)
}

// PutInt8Buffer returns an int8 buffer to the pool
func PutInt8Buffer(buf []int8) {
	cpuCompatBufferPool.int8Buffers.Put(buf)
}

// GPU optimizations use the shared MemoryOptimizedCache from performance_optimizations.go

// GetOptimalGPUBatchSize returns optimal batch size for GPU processing
func GetOptimalGPUBatchSize() int {
	// With GPU acceleration, we can handle larger batches
	// Optimized for RTX 3090 with 24GB VRAM
	return 256 // Optimal for GPU parallelism
}

// GetOptimalBatchSize returns optimal batch size for processing
func GetOptimalBatchSize() int {
	// GPU mode can handle larger batches
	return 256
}

// OptimizedSearchEngine provides optimized search operations
type OptimizedSearchEngine struct {
	model          *EmbeddingModel
	searchEngine   *SearchEngine
	embeddingCache *MemoryOptimizedCache
	bufferPool     *BufferPool
	workers        int
}

// NewOptimizedSearchEngine creates a new optimized search engine
func NewOptimizedSearchEngine(model *EmbeddingModel, config SearchConfig) *OptimizedSearchEngine {
	return &OptimizedSearchEngine{
		model:          model,
		searchEngine:   NewSearchEngineWithConfig(model, config),
		embeddingCache: NewMemoryOptimizedCache(100), // 100MB cache
		bufferPool:     cpuCompatBufferPool,
		workers:        runtime.NumCPU(),
	}
}

// ProcessBatch processes a batch of texts efficiently
func (e *OptimizedSearchEngine) ProcessBatch(texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	
	// Use worker pool for parallel processing
	ch := make(chan int, len(texts))
	for i := range texts {
		ch <- i
	}
	close(ch)

	var wg sync.WaitGroup
	wg.Add(e.workers)
	
	for w := 0; w < e.workers; w++ {
		go func() {
			defer wg.Done()
			for i := range ch {
				// Generate embedding (skip cache for GPU version)
				embedding, err := e.model.Encode(texts[i])
				if err != nil {
					continue
				}
				
				results[i] = embedding
			}
		}()
	}
	
	wg.Wait()
	return results, nil
}

// Search performs an optimized search
func (e *OptimizedSearchEngine) Search(query string, k int) ([]SearchResult, error) {
	// Use buffer pool for temporary allocations
	buf := GetTokenBuffer()
	defer PutTokenBuffer(buf)
	
	return e.searchEngine.Search(query, k)
}

// IndexBatch indexes multiple documents efficiently
func (e *OptimizedSearchEngine) IndexBatch(texts []string) error {
	embeddings, err := e.ProcessBatch(texts)
	if err != nil {
		return err
	}
	
	// Add to search engine using proper API
	for i, embedding := range embeddings {
		if embedding != nil {
			// Use the search engine's proper index method instead of direct access
			e.searchEngine.IndexWithID(i, texts[i])
		}
	}
	
	return nil
}

// quantizeVector converts float32 to int8 with scaling
func quantizeVector(vec []float32) []int8 {
	result := GetInt8Buffer()
	if len(result) < len(vec) {
		result = make([]int8, len(vec))
	} else {
		result = result[:len(vec)]
	}
	
	// Find scale
	var maxVal float32
	for _, v := range vec {
		if v > maxVal {
			maxVal = v
		}
		if -v > maxVal {
			maxVal = -v
		}
	}
	
	scale := maxVal / 127.0
	if scale == 0 {
		scale = 1
	}
	
	// Quantize
	for i, v := range vec {
		quantized := int8(v / scale)
		if quantized > 127 {
			quantized = 127
		} else if quantized < -128 {
			quantized = -128
		}
		result[i] = quantized
	}
	
	return result
}