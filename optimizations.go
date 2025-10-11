package gobed

import (
	"runtime"
	"sync"
	"unsafe"
)

// BufferPool manages reusable buffers to reduce allocations
type BufferPool struct {
	tokenBuffers   sync.Pool
	embedBuffers   sync.Pool
	int8Buffers    sync.Pool
	float32Buffers sync.Pool
}

var globalBufferPool = &BufferPool{
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
	float32Buffers: sync.Pool{
		New: func() interface{} {
			// Pre-allocate for batch processing
			return make([]float32, 0, 2048)
		},
	},
}

// GetTokenBuffer returns a reusable token buffer
func GetTokenBuffer() []int {
	return globalBufferPool.tokenBuffers.Get().([]int)[:0]
}

// PutTokenBuffer returns a token buffer to the pool
func PutTokenBuffer(buf []int) {
	if cap(buf) <= 2048 { // Don't pool huge buffers
		globalBufferPool.tokenBuffers.Put(buf)
	}
}

// GetEmbedBuffer returns a reusable embedding buffer
func GetEmbedBuffer() []float32 {
	return globalBufferPool.embedBuffers.Get().([]float32)
}

// PutEmbedBuffer returns an embedding buffer to the pool
func PutEmbedBuffer(buf []float32) {
	if cap(buf) <= 4096 {
		globalBufferPool.embedBuffers.Put(buf)
	}
}

// GetInt8Buffer returns a reusable int8 buffer
func GetInt8Buffer() []int8 {
	return globalBufferPool.int8Buffers.Get().([]int8)[:0]
}

// PutInt8Buffer returns an int8 buffer to the pool
func PutInt8Buffer(buf []int8) {
	if cap(buf) <= 2048 {
		globalBufferPool.int8Buffers.Put(buf)
	}
}

// BatchConfig holds optimized batch processing configuration
type BatchConfig struct {
	MaxBatchSize   int
	OptimalBatch   int
	MinBatch       int
	GPUMemoryLimit uint64
}

// GetOptimalBatchSize returns the optimal batch size based on available memory
func GetOptimalBatchSize() int {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Available memory in MB
	availableMB := (m.Sys - m.Alloc) / 1024 / 1024

	// Estimate batch size based on memory
	// Each embedding ~4KB, aim to use 25% of available memory
	optimalBatch := int(availableMB * 256 / 4) // 256 = 1024 * 0.25

	// Clamp to reasonable range
	if optimalBatch < 32 {
		return 32
	}
	if optimalBatch > 2048 {
		return 2048
	}

	// Round to power of 2 for better alignment
	return nearestPowerOf2(optimalBatch)
}

// GetOptimalGPUBatchSize returns optimal batch size for GPU processing
func GetOptimalGPUBatchSize() int {
	// Check if we're building with GPU support
	return GetOptimalBatchSize() // Fall back to CPU sizing for now
}

func nearestPowerOf2(n int) int {
	if n <= 0 {
		return 1
	}

	// Find the nearest power of 2
	power := 1
	for power < n {
		power <<= 1
	}

	// Check if the previous power is closer
	if power-n > n-power/2 && power > 1 {
		return power / 2
	}

	return power
}

// TokenizerOptimizations provides fast tokenization helpers
type TokenizerOptimizations struct {
	tokenCache   map[string][]int
	cacheMu      sync.RWMutex
	maxCacheSize int
}

// NewTokenizerOptimizations creates optimized tokenizer wrapper
func NewTokenizerOptimizations(maxCacheSize int) *TokenizerOptimizations {
	return &TokenizerOptimizations{
		tokenCache:   make(map[string][]int),
		maxCacheSize: maxCacheSize,
	}
}

// TokenizeCached performs cached tokenization
func (t *TokenizerOptimizations) TokenizeCached(text string, tokenizeFn func(string) ([]uint32, error)) ([]int, error) {
	// Check cache first
	t.cacheMu.RLock()
	if cached, ok := t.tokenCache[text]; ok {
		t.cacheMu.RUnlock()
		return cached, nil
	}
	t.cacheMu.RUnlock()

	// Tokenize
	ids, err := tokenizeFn(text)
	if err != nil {
		return nil, err
	}

	// Convert with buffer pool
	tokens := GetTokenBuffer()
	for _, id := range ids {
		tokens = append(tokens, int(id))
	}

	// Cache if not too large
	if len(t.tokenCache) < t.maxCacheSize {
		t.cacheMu.Lock()
		t.tokenCache[text] = tokens
		t.cacheMu.Unlock()
	}

	return tokens, nil
}

// ClearCache clears the tokenization cache
func (t *TokenizerOptimizations) ClearCache() {
	t.cacheMu.Lock()
	t.tokenCache = make(map[string][]int)
	t.cacheMu.Unlock()
}

// ZeroCopyConversion performs zero-copy type conversions when possible
func ZeroCopyInt32ToFloat32(src []int32) []float32 {
	// This is unsafe but fast - use only when you control the lifetime
	if len(src) == 0 {
		return nil
	}

	// Warning: This shares the underlying memory
	header := (*[1 << 30]float32)(unsafe.Pointer(&src[0]))
	return header[:len(src):len(src)]
}

// FastQuantize performs optimized quantization with SIMD hints
func FastQuantize(input []float32) ([]int8, float32) {
	if len(input) == 0 {
		return nil, 0
	}

	// Find min/max in single pass
	minVal, maxVal := input[0], input[0]
	for i := 1; i < len(input); i++ {
		if input[i] < minVal {
			minVal = input[i]
		}
		if input[i] > maxVal {
			maxVal = input[i]
		}
	}

	// Calculate scale
	scale := (maxVal - minVal) / 255.0
	if scale == 0 {
		scale = 1.0
	}
	invScale := 1.0 / scale

	// Quantize with buffer pool
	output := GetInt8Buffer()
	if cap(output) < len(input) {
		output = make([]int8, len(input))
	} else {
		output = output[:len(input)]
	}

	// Vectorizable loop
	for i := 0; i < len(input); i++ {
		val := (input[i] - minVal) * invScale
		if val < 0 {
			output[i] = 0
		} else if val > 127 {
			output[i] = 127
		} else {
			output[i] = int8(val)
		}
	}

	return output, scale
}

// ParallelProcessor handles parallel batch processing with controlled concurrency
type ParallelProcessor struct {
	workers    int
	jobQueue   chan func()
	wg         sync.WaitGroup
	bufferPool *BufferPool
}

// NewParallelProcessor creates a processor with optimal worker count
func NewParallelProcessor() *ParallelProcessor {
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8 // Cap at 8 for diminishing returns
	}

	p := &ParallelProcessor{
		workers:    workers,
		jobQueue:   make(chan func(), workers*2),
		bufferPool: globalBufferPool,
	}

	// Start workers
	for i := 0; i < workers; i++ {
		go p.worker()
	}

	return p
}

func (p *ParallelProcessor) worker() {
	for job := range p.jobQueue {
		job()
		p.wg.Done()
	}
}

// ProcessBatch processes items in parallel with controlled concurrency
func (p *ParallelProcessor) ProcessBatch(items []func()) {
	p.wg.Add(len(items))
	for _, item := range items {
		p.jobQueue <- item
	}
	p.wg.Wait()
}

// Close shuts down the processor
func (p *ParallelProcessor) Close() {
	close(p.jobQueue)
}
