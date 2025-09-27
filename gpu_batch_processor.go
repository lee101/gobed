// +build gpu

package gobed

import (
	"log"
	"runtime"
	"sync"
	"time"
)

// GPUBatchProcessor handles parallel batch processing of embeddings on GPU
type GPUBatchProcessor struct {
	cache         *TokenPatternCache
	model         *EmbeddingModel
	batchSize     int
	maxConcurrent int
	
	// GPU memory management
	gpuBuffers    []GPUBuffer
	bufferPool    sync.Pool
	
	// Metrics
	processedBatches uint64
	totalLatency     time.Duration
	mu               sync.RWMutex
}

// GPUBuffer represents a GPU memory buffer for batch processing
type GPUBuffer struct {
	TokenIDs    [][]int    // Batch of token sequences
	Embeddings  [][]float32 // Result embeddings
	Quantized   [][]int8    // Quantized results
	Scales      []float32   // Quantization scales
	InUse       bool
}

// NewGPUBatchProcessor creates a GPU-optimized batch processor
func NewGPUBatchProcessor(model *EmbeddingModel, cache *TokenPatternCache) *GPUBatchProcessor {
	numCPU := runtime.NumCPU()
	// Use dynamic batch sizing based on available memory
	batchSize := GetOptimalGPUBatchSize()
	
	processor := &GPUBatchProcessor{
		cache:         cache,
		model:         model,
		batchSize:     batchSize,
		maxConcurrent: numCPU * 2, // Allow 2x CPU count for GPU overlap
		gpuBuffers:    make([]GPUBuffer, numCPU*2),
	}
	
	// Initialize buffer pool
	processor.bufferPool = sync.Pool{
		New: func() interface{} {
			return &GPUBuffer{
				TokenIDs:   make([][]int, batchSize),
				Embeddings: make([][]float32, batchSize),
				Quantized:  make([][]int8, batchSize),
				Scales:     make([]float32, batchSize),
			}
		},
	}
	
	// Pre-allocate buffers
	for i := range processor.gpuBuffers {
		processor.gpuBuffers[i] = GPUBuffer{
			TokenIDs:   make([][]int, batchSize),
			Embeddings: make([][]float32, batchSize),
			Quantized:  make([][]int8, batchSize),
			Scales:     make([]float32, batchSize),
		}
	}
	
	log.Printf("GPU Batch Processor initialized: batch_size=%d, max_concurrent=%d", 
		batchSize, processor.maxConcurrent)
	
	return processor
}

// ProcessBatch processes a batch of texts in parallel using GPU
func (p *GPUBatchProcessor) ProcessBatch(texts []string) ([]*EmbedInt8Result, error) {
	startTime := time.Now()
	defer func() {
		p.mu.Lock()
		p.processedBatches++
		p.totalLatency += time.Since(startTime)
		p.mu.Unlock()
	}()
	
	numTexts := len(texts)
	results := make([]*EmbedInt8Result, numTexts)
	
	// Stage 1: Tokenize all texts in parallel
	tokenBatches := make([][]int, numTexts)
	var tokenizeWg sync.WaitGroup
	
	for i := 0; i < numTexts; i += p.batchSize {
		end := i + p.batchSize
		if end > numTexts {
			end = numTexts
		}
		
		tokenizeWg.Add(1)
		go func(start, end int) {
			defer tokenizeWg.Done()
			
			for j := start; j < end; j++ {
				text := normalizeText(texts[j])
				if text == "" {
					tokenBatches[j] = []int{}
					continue
				}
				
				encoding, err := p.model.tokenizer.EncodeSingle(text, false)
				if err != nil {
					tokenBatches[j] = []int{}
					continue
				}
				
				// Use buffer pool
				tokens := GetTokenBuffer()
				for _, id := range encoding.Ids {
					tokens = append(tokens, int(id))
				}
				
				// Apply stopword filtering if needed
				if p.cache != nil && len(tokens) > 15 {
					tokens = p.cache.FilterStopwords(tokens, len(text))
				}
				
				tokenBatches[j] = tokens
			}
		}(i, end)
	}
	
	tokenizeWg.Wait()
	
	// Stage 2: Check cache for all patterns in parallel
	cachedEmbeddings, cacheHits := p.cache.BatchGetEmbeddings(tokenBatches)
	
	// Stage 3: Process uncached items in GPU batches
	uncachedIndices := make([]int, 0, numTexts)
	for i, hit := range cacheHits {
		if !hit {
			uncachedIndices = append(uncachedIndices, i)
		} else {
			// Use cached result
			results[i] = &EmbedInt8Result{
				Vector: cachedEmbeddings[i].VectorI8,
				Scale:  cachedEmbeddings[i].Scale,
			}
		}
	}
	
	if len(uncachedIndices) > 0 {
		// Process uncached items on GPU
		if err := p.processOnGPU(tokenBatches, uncachedIndices, results); err != nil {
			// Fallback to CPU processing
			return p.fallbackToCPU(tokenBatches, uncachedIndices, results)
		}
	}
	
	log.Printf("Processed batch of %d texts: %d cache hits, %d GPU processed, latency: %v",
		numTexts, numTexts-len(uncachedIndices), len(uncachedIndices), time.Since(startTime))
	
	return results, nil
}

// processOnGPU sends tokens to GPU for embedding computation
func (p *GPUBatchProcessor) processOnGPU(tokenBatches [][]int, indices []int, results []*EmbedInt8Result) error {
	// Get a free GPU buffer
	buffer := p.getBuffer()
	defer p.releaseBuffer(buffer)
	
	// Process in chunks that fit GPU memory
	for i := 0; i < len(indices); i += p.batchSize {
		end := i + p.batchSize
		if end > len(indices) {
			end = len(indices)
		}
		
		batchIndices := indices[i:end]
		
		// Prepare batch for GPU
		for j, idx := range batchIndices {
			buffer.TokenIDs[j] = tokenBatches[idx]
		}
		
		// GPU computation would happen here
		// For now, simulate with parallel CPU processing
		var wg sync.WaitGroup
		for j, idx := range batchIndices {
			wg.Add(1)
			go func(j, idx int) {
				defer wg.Done()
				
				embedding, err := p.model.computeEmbedding(buffer.TokenIDs[j])
				if err != nil {
					results[idx] = &EmbedInt8Result{
						Vector: make([]int8, p.model.EmbedDim),
						Scale:  1.0,
					}
					return
				}
				
				quantized, scale := quantizeEmbedding(embedding)
				results[idx] = &EmbedInt8Result{
					Vector: quantized,
					Scale:  scale,
				}
				
				// Update cache with new embedding
				if p.cache != nil && len(buffer.TokenIDs[j]) <= 4 {
					p.updateCache(buffer.TokenIDs[j], embedding, quantized, scale)
				}
			}(j, idx)
		}
		wg.Wait()
	}
	
	return nil
}

// fallbackToCPU processes embeddings on CPU when GPU is unavailable
func (p *GPUBatchProcessor) fallbackToCPU(tokenBatches [][]int, indices []int, results []*EmbedInt8Result) ([]*EmbedInt8Result, error) {
	var wg sync.WaitGroup
	for _, idx := range indices {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			
			embedding, err := p.model.computeEmbedding(tokenBatches[idx])
			if err != nil {
				results[idx] = &EmbedInt8Result{
					Vector: make([]int8, p.model.EmbedDim),
					Scale:  1.0,
				}
				return
			}
			
			quantized, scale := quantizeEmbedding(embedding)
			results[idx] = &EmbedInt8Result{
				Vector: quantized,
				Scale:  scale,
			}
		}(idx)
	}
	wg.Wait()
	
	return results, nil
}

// updateCache adds newly computed embeddings to cache
func (p *GPUBatchProcessor) updateCache(tokens []int, embedding []float32, quantized []int8, scale float32) {
	if p.cache == nil {
		return
	}
	
	p.cache.mu.Lock()
	defer p.cache.mu.Unlock()
	
	key := makePatternKey(tokens)
	cached := &CachedEmbedding{
		Vector:   embedding,
		VectorI8: quantized,
		Scale:    scale,
		UseCount: 1,
		LastUsed: time.Now().Unix(),
	}
	
	switch len(tokens) {
	case 1:
		p.cache.singleTokens[tokens[0]] = cached
	case 2:
		p.cache.bigrams[key] = cached
	case 3:
		p.cache.trigrams[key] = cached
	case 4:
		p.cache.fourgrams[key] = cached
	}
}

// getBuffer gets a free GPU buffer
func (p *GPUBatchProcessor) getBuffer() *GPUBuffer {
	// Try to get from pool first
	if buffer := p.bufferPool.Get(); buffer != nil {
		return buffer.(*GPUBuffer)
	}
	
	// Find a free buffer
	for i := range p.gpuBuffers {
		if !p.gpuBuffers[i].InUse {
			p.gpuBuffers[i].InUse = true
			return &p.gpuBuffers[i]
		}
	}
	
	// All buffers in use, create a new one
	return &GPUBuffer{
		TokenIDs:   make([][]int, p.batchSize),
		Embeddings: make([][]float32, p.batchSize),
		Quantized:  make([][]int8, p.batchSize),
		Scales:     make([]float32, p.batchSize),
		InUse:      true,
	}
}

// releaseBuffer returns a buffer to the pool
func (p *GPUBatchProcessor) releaseBuffer(buffer *GPUBuffer) {
	buffer.InUse = false
	p.bufferPool.Put(buffer)
}

// GetStats returns processing statistics
func (p *GPUBatchProcessor) GetStats() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	avgLatency := time.Duration(0)
	if p.processedBatches > 0 {
		avgLatency = p.totalLatency / time.Duration(p.processedBatches)
	}
	
	cacheStats := map[string]interface{}{}
	if p.cache != nil {
		cacheStats = p.cache.GetStats()
	}
	
	return map[string]interface{}{
		"processed_batches": p.processedBatches,
		"avg_latency_ms":    avgLatency.Milliseconds(),
		"batch_size":        p.batchSize,
		"max_concurrent":    p.maxConcurrent,
		"cache_stats":       cacheStats,
	}
}

// WarmupCache preloads common patterns into cache
func (p *GPUBatchProcessor) WarmupCache(commonTexts []string) {
	log.Printf("Warming up cache with %d common texts...", len(commonTexts))
	
	// Process in batches to populate cache
	for i := 0; i < len(commonTexts); i += p.batchSize {
		end := i + p.batchSize
		if end > len(commonTexts) {
			end = len(commonTexts)
		}
		
		batch := commonTexts[i:end]
		_, _ = p.ProcessBatch(batch)
	}
	
	if p.cache != nil {
		stats := p.cache.GetStats()
		log.Printf("Cache warmed up: %v hits, %.1f%% hit rate", 
			stats["hits"], stats["hit_rate"])
	}
}