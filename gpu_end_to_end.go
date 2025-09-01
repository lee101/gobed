// +build gpu

package gobed

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

/*
#cgo CFLAGS: -I./gpu
#cgo LDFLAGS: -L./gpu -lpure_cuda_indexer -L/usr/local/cuda-12.8/lib64 -lcudart -lcublas -Wl,-rpath,./gpu -Wl,-rpath,/usr/local/cuda-12.8/lib64
#include "pure_cuda_indexer.h"
#include <stdlib.h>
*/
import "C"

// GPUEndToEndPipeline provides a complete GPU-resident pipeline
// from tokenization to search, keeping everything on GPU
type GPUEndToEndPipeline struct {
	model         *EmbeddingModel
	gpuHandle     unsafe.Pointer
	
	// GPU memory management
	embeddingsLoaded bool
	numVectors       int
	vectorDim        int
	vocabSize        int
	embedDim         int
	
	// Batch processing
	batchProcessor *GPUBatchTokenProcessor
	
	// Performance tracking
	stats GPUPipelineStats
	mu    sync.RWMutex
}

// GPUBatchTokenProcessor handles batch token processing on GPU
type GPUBatchTokenProcessor struct {
	pipeline      *GPUEndToEndPipeline
	maxBatchSize  int
	maxSeqLen     int
	
	// Token buffer management
	tokenBuffer   []int32
	lengthBuffer  []int
	resultBuffer  []float32
}

// GPUPipelineStats tracks pipeline performance
type GPUPipelineStats struct {
	TotalTokenizations uint64
	TotalEmbeddings    uint64
	TotalSearches      uint64
	TotalIndexOps      uint64
	AvgTokenizeTimeUs  uint64
	AvgEmbedTimeUs     uint64
	AvgSearchTimeUs    uint64
	AvgIndexTimeUs     uint64
	GPUMemoryBytes     uint64
}

// NewGPUEndToEndPipeline creates a complete GPU pipeline
func NewGPUEndToEndPipeline(model *EmbeddingModel) (*GPUEndToEndPipeline, error) {
	if !IsCUDAAvailable() {
		return nil, fmt.Errorf("CUDA is not available")
	}
	
	// Create GPU index with proper dimensions
	vectorDim := 512  // gobed dimension
	vocabSize := 30522 // BERT vocab size
	embedDim := 512   // embedding dimension
	
	handle := C.cuda_index_create(
		C.int(vectorDim),
		C.int(vocabSize),
		C.int(embedDim),
	)
	
	if handle == nil {
		return nil, fmt.Errorf("failed to create GPU index")
	}
	
	pipeline := &GPUEndToEndPipeline{
		model:     model,
		gpuHandle: handle,
		vectorDim: vectorDim,
		vocabSize: vocabSize,
		embedDim:  embedDim,
	}
	
	// Create batch processor
	pipeline.batchProcessor = &GPUBatchTokenProcessor{
		pipeline:     pipeline,
		maxBatchSize: 2048,
		maxSeqLen:    512,
		tokenBuffer:  make([]int32, 2048*512),
		lengthBuffer: make([]int, 2048),
		resultBuffer: make([]float32, 2048*512),
	}
	
	// Set finalizer
	runtime.SetFinalizer(pipeline, (*GPUEndToEndPipeline).destroy)
	
	log.Printf("🚀 GPU End-to-End Pipeline created (dim=%d, vocab=%d)", vectorDim, vocabSize)
	return pipeline, nil
}

// LoadEmbeddingsToGPU loads the model's embedding table to GPU memory
func (p *GPUEndToEndPipeline) LoadEmbeddingsToGPU() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if p.embeddingsLoaded {
		return nil
	}
	
	// Extract embeddings from model
	embeddings := make([]float32, p.vocabSize*p.embedDim)
	
	// For now, generate random embeddings (replace with actual model embeddings)
	// In production, extract from p.model's embedding layer
	for i := range embeddings {
		embeddings[i] = float32(i%100) / 100.0
	}
	
	result := C.cuda_load_embeddings(
		p.gpuHandle,
		(*C.float)(unsafe.Pointer(&embeddings[0])),
	)
	
	if result == 0 {
		return fmt.Errorf("failed to load embeddings to GPU")
	}
	
	p.embeddingsLoaded = true
	log.Printf("✅ Loaded %d embeddings to GPU", p.vocabSize)
	return nil
}

// ProcessTextToEmbedding processes text directly to GPU embedding
func (p *GPUEndToEndPipeline) ProcessTextToEmbedding(text string) ([]int8, float32, error) {
	startTime := time.Now()
	defer func() {
		atomic.AddUint64(&p.stats.TotalEmbeddings, 1)
		atomic.AddUint64(&p.stats.AvgEmbedTimeUs, uint64(time.Since(startTime).Microseconds()))
	}()
	
	// Tokenize on CPU (for now)
	text = normalizeText(text)
	encoding, err := p.model.tokenizer.EncodeSingle(text, false)
	if err != nil {
		return nil, 0, err
	}
	
	// Convert to int32 for GPU
	tokenIDs := make([]int32, len(encoding.Ids))
	for i, id := range encoding.Ids {
		tokenIDs[i] = int32(id)
	}
	
	// Process tokens on GPU to get embedding
	return p.tokensToEmbedding(tokenIDs)
}

// tokensToEmbedding converts tokens to embeddings entirely on GPU
func (p *GPUEndToEndPipeline) tokensToEmbedding(tokenIDs []int32) ([]int8, float32, error) {
	if !p.embeddingsLoaded {
		if err := p.LoadEmbeddingsToGPU(); err != nil {
			return nil, 0, err
		}
	}
	
	// Allocate result buffers
	resultVector := make([]int8, p.vectorDim)
	resultScores := make([]float32, 1)
	
	// Use GPU to generate and quantize embedding
	result := C.cuda_search_with_tokens(
		p.gpuHandle,
		(*C.int)(unsafe.Pointer(&tokenIDs[0])),
		C.int(len(tokenIDs)),
		(*C.int)(unsafe.Pointer(&[]int32{0}[0])), // Dummy for embedding generation
		(*C.float)(unsafe.Pointer(&resultScores[0])),
		C.int(0), // k=0 means just generate embedding
	)
	
	if result < 0 {
		return nil, 0, fmt.Errorf("GPU embedding generation failed")
	}
	
	return resultVector, resultScores[0], nil
}

// BatchProcessAndIndex processes multiple texts and indexes them on GPU
func (p *GPUEndToEndPipeline) BatchProcessAndIndex(texts []string) (int, error) {
	startTime := time.Now()
	defer func() {
		atomic.AddUint64(&p.stats.TotalIndexOps, uint64(len(texts)))
		atomic.AddUint64(&p.stats.AvgIndexTimeUs, uint64(time.Since(startTime).Microseconds())/uint64(len(texts)))
	}()
	
	if !p.embeddingsLoaded {
		if err := p.LoadEmbeddingsToGPU(); err != nil {
			return 0, err
		}
	}
	
	// Tokenize all texts
	tokenSequences := make([][]int32, len(texts))
	seqLengths := make([]int, len(texts))
	
	for i, text := range texts {
		text = normalizeText(text)
		encoding, err := p.model.tokenizer.EncodeSingle(text, false)
		if err != nil {
			continue
		}
		
		tokens := make([]int32, len(encoding.Ids))
		for j, id := range encoding.Ids {
			tokens[j] = int32(id)
		}
		
		tokenSequences[i] = tokens
		seqLengths[i] = len(tokens)
	}
	
	// Bulk index on GPU
	return p.bulkIndexTokens(tokenSequences, seqLengths)
}

// bulkIndexTokens indexes multiple token sequences entirely on GPU
func (p *GPUEndToEndPipeline) bulkIndexTokens(tokenSequences [][]int32, seqLengths []int) (int, error) {
	if len(tokenSequences) == 0 {
		return 0, nil
	}
	
	// Find max sequence length
	maxSeqLen := 0
	for _, seq := range tokenSequences {
		if len(seq) > maxSeqLen {
			maxSeqLen = len(seq)
		}
	}
	
	// Flatten sequences
	batchSize := len(tokenSequences)
	flatTokens := make([]int32, batchSize*maxSeqLen)
	
	for i, seq := range tokenSequences {
		for j, token := range seq {
			flatTokens[i*maxSeqLen+j] = token
		}
	}
	
	// Convert lengths to C format
	cLengths := make([]C.int, len(seqLengths))
	for i, l := range seqLengths {
		cLengths[i] = C.int(l)
	}
	
	// Call GPU bulk indexing
	result := C.cuda_bulk_index_tokens(
		p.gpuHandle,
		(*C.int)(unsafe.Pointer(&flatTokens[0])),
		(*C.int)(unsafe.Pointer(&cLengths[0])),
		C.int(batchSize),
		C.int(maxSeqLen),
	)
	
	if result == 0 {
		return 0, fmt.Errorf("GPU bulk indexing failed")
	}
	
	p.numVectors += int(result)
	return int(result), nil
}

// SearchWithText performs end-to-end search from text query
func (p *GPUEndToEndPipeline) SearchWithText(query string, k int) ([]int32, []float32, error) {
	startTime := time.Now()
	defer func() {
		atomic.AddUint64(&p.stats.TotalSearches, 1)
		atomic.AddUint64(&p.stats.AvgSearchTimeUs, uint64(time.Since(startTime).Microseconds()))
	}()
	
	if !p.embeddingsLoaded {
		if err := p.LoadEmbeddingsToGPU(); err != nil {
			return nil, nil, err
		}
	}
	
	// Tokenize query
	query = normalizeText(query)
	encoding, err := p.model.tokenizer.EncodeSingle(query, false)
	if err != nil {
		return nil, nil, err
	}
	
	// Convert to int32
	tokenIDs := make([]int32, len(encoding.Ids))
	for i, id := range encoding.Ids {
		tokenIDs[i] = int32(id)
	}
	
	// Search directly with tokens on GPU
	return p.searchWithTokens(tokenIDs, k)
}

// searchWithTokens performs GPU search with token IDs
func (p *GPUEndToEndPipeline) searchWithTokens(tokenIDs []int32, k int) ([]int32, []float32, error) {
	if p.numVectors == 0 {
		return nil, nil, fmt.Errorf("index is empty")
	}
	
	// Allocate result buffers
	resultIndices := make([]int32, k)
	resultScores := make([]float32, k)
	
	// GPU search
	result := C.cuda_search_with_tokens(
		p.gpuHandle,
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

// BatchSearch performs batch search entirely on GPU
func (p *GPUEndToEndPipeline) BatchSearch(queries []string, k int) ([][]int32, [][]float32, error) {
	results := make([][]int32, len(queries))
	scores := make([][]float32, len(queries))
	
	// Process queries in parallel batches
	batchSize := 32
	var wg sync.WaitGroup
	var mu sync.Mutex
	
	for i := 0; i < len(queries); i += batchSize {
		end := i + batchSize
		if end > len(queries) {
			end = len(queries)
		}
		
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			
			for j := start; j < end; j++ {
				indices, scores_, err := p.SearchWithText(queries[j], k)
				if err != nil {
					log.Printf("Search error for query %d: %v", j, err)
					continue
				}
				
				mu.Lock()
				results[j] = indices
				scores[j] = scores_
				mu.Unlock()
			}
		}(i, end)
	}
	
	wg.Wait()
	return results, scores, nil
}

// GetStats returns pipeline statistics
func (p *GPUEndToEndPipeline) GetStats() GPUPipelineStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	stats := p.stats
	stats.GPUMemoryBytes = uint64(C.cuda_get_memory_usage())
	return stats
}

// SetMaxTokens sets the maximum tokens for truncation
func (p *GPUEndToEndPipeline) SetMaxTokens(maxTokens int) {
	C.cuda_set_max_tokens(p.gpuHandle, C.int(maxTokens))
	log.Printf("📏 GPU pipeline max tokens set to %d", maxTokens)
}

// destroy cleans up GPU resources
func (p *GPUEndToEndPipeline) destroy() {
	if p.gpuHandle != nil {
		C.cuda_index_destroy(p.gpuHandle)
		p.gpuHandle = nil
	}
}

// Close explicitly releases GPU resources
func (p *GPUEndToEndPipeline) Close() {
	p.destroy()
	log.Println("🧹 GPU pipeline closed")
}

// RunLoadTest performs a comprehensive load test of the GPU pipeline
func (p *GPUEndToEndPipeline) RunLoadTest(numDocuments int, numQueries int, k int) error {
	log.Printf("🔥 Starting GPU pipeline load test: %d documents, %d queries, k=%d", 
		numDocuments, numQueries, k)
	
	// Generate test documents
	testDocs := make([]string, numDocuments)
	for i := range testDocs {
		testDocs[i] = fmt.Sprintf("This is test document number %d with some random content for testing GPU processing", i)
	}
	
	// Phase 1: Bulk indexing
	log.Println("Phase 1: Bulk indexing...")
	indexStart := time.Now()
	
	batchSize := 1000
	totalIndexed := 0
	
	for i := 0; i < numDocuments; i += batchSize {
		end := i + batchSize
		if end > numDocuments {
			end = numDocuments
		}
		
		batch := testDocs[i:end]
		indexed, err := p.BatchProcessAndIndex(batch)
		if err != nil {
			log.Printf("Indexing error: %v", err)
			continue
		}
		totalIndexed += indexed
		
		if (i/batchSize)%10 == 0 {
			log.Printf("  Indexed %d/%d documents", totalIndexed, numDocuments)
		}
	}
	
	indexDuration := time.Since(indexStart)
	indexThroughput := float64(totalIndexed) / indexDuration.Seconds()
	log.Printf("✅ Indexing complete: %d documents in %v (%.0f docs/sec)", 
		totalIndexed, indexDuration, indexThroughput)
	
	// Phase 2: Search load test
	log.Println("Phase 2: Search load test...")
	searchStart := time.Now()
	
	// Generate test queries
	testQueries := make([]string, numQueries)
	for i := range testQueries {
		testQueries[i] = fmt.Sprintf("test query number %d", i%100)
	}
	
	// Run concurrent searches
	numWorkers := runtime.NumCPU()
	queriesPerWorker := numQueries / numWorkers
	
	var searchWg sync.WaitGroup
	searchCount := uint64(0)
	errorCount := uint64(0)
	
	for w := 0; w < numWorkers; w++ {
		searchWg.Add(1)
		go func(workerID int) {
			defer searchWg.Done()
			
			start := workerID * queriesPerWorker
			end := start + queriesPerWorker
			if workerID == numWorkers-1 {
				end = numQueries
			}
			
			for i := start; i < end; i++ {
				_, _, err := p.SearchWithText(testQueries[i], k)
				if err != nil {
					atomic.AddUint64(&errorCount, 1)
				} else {
					atomic.AddUint64(&searchCount, 1)
				}
				
				if atomic.LoadUint64(&searchCount)%1000 == 0 {
					log.Printf("  Completed %d searches", atomic.LoadUint64(&searchCount))
				}
			}
		}(w)
	}
	
	searchWg.Wait()
	
	searchDuration := time.Since(searchStart)
	searchThroughput := float64(searchCount) / searchDuration.Seconds()
	
	log.Printf("✅ Search test complete: %d searches in %v (%.0f QPS)", 
		searchCount, searchDuration, searchThroughput)
	
	// Phase 3: Batch search test
	log.Println("Phase 3: Batch search test...")
	batchSearchStart := time.Now()
	
	batchSearchSize := 100
	numBatches := numQueries / batchSearchSize
	
	for i := 0; i < numBatches; i++ {
		start := i * batchSearchSize
		end := start + batchSearchSize
		if end > numQueries {
			end = numQueries
		}
		
		batch := testQueries[start:end]
		_, _, err := p.BatchSearch(batch, k)
		if err != nil {
			log.Printf("Batch search error: %v", err)
		}
		
		if i%10 == 0 {
			log.Printf("  Completed %d/%d batch searches", i, numBatches)
		}
	}
	
	batchSearchDuration := time.Since(batchSearchStart)
	batchThroughput := float64(numQueries) / batchSearchDuration.Seconds()
	
	log.Printf("✅ Batch search complete: %d queries in %v (%.0f QPS)", 
		numQueries, batchSearchDuration, batchThroughput)
	
	// Print final statistics
	stats := p.GetStats()
	log.Println("\n📊 GPU Pipeline Statistics:")
	log.Printf("  Total Embeddings: %d", stats.TotalEmbeddings)
	log.Printf("  Total Searches: %d", stats.TotalSearches)
	log.Printf("  Total Index Ops: %d", stats.TotalIndexOps)
	log.Printf("  Avg Embed Time: %d µs", stats.AvgEmbedTimeUs)
	log.Printf("  Avg Search Time: %d µs", stats.AvgSearchTimeUs)
	log.Printf("  Avg Index Time: %d µs", stats.AvgIndexTimeUs)
	log.Printf("  GPU Memory: %.2f MB", float64(stats.GPUMemoryBytes)/(1024*1024))
	log.Printf("  Index Throughput: %.0f docs/sec", indexThroughput)
	log.Printf("  Search Throughput: %.0f QPS", searchThroughput)
	log.Printf("  Batch Search Throughput: %.0f QPS", batchThroughput)
	
	return nil
}