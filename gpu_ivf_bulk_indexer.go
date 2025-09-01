// +build gpu

package gobed

/*
#cgo CFLAGS: -I./gpu
#cgo LDFLAGS: -L./gpu -lgpu_ivf_bulk -L/usr/local/cuda-12.8/lib64 -lcudart -lcublas -Wl,-rpath,./gpu -Wl,-rpath,/usr/local/cuda-12.8/lib64
#include <stdlib.h>

// GPU IVF Bulk Indexing C API
typedef struct {
    void* handle;
    int nlist;
    int nprobe;
    int vector_dim;
    int vocab_size;
    int embed_dim;
} gpu_ivf_bulk_t;

// Core functions
gpu_ivf_bulk_t* gpu_ivf_bulk_create(int nlist, int nprobe, int vector_dim, int vocab_size, int embed_dim);
void gpu_ivf_bulk_destroy(gpu_ivf_bulk_t* indexer);

// Training and setup
int gpu_ivf_bulk_load_embeddings(gpu_ivf_bulk_t* indexer, const signed char* embeddings, const float* scales);
int gpu_ivf_bulk_train_kmeans(gpu_ivf_bulk_t* indexer, const signed char* training_vectors, const float* scales, int num_training);

// Bulk indexing operations  
int gpu_ivf_bulk_index_batch(gpu_ivf_bulk_t* indexer, const int* token_sequences, const int* seq_lengths, 
                             int batch_size, int max_seq_len, int* assigned_ids);
                             
// Progressive indexing
int gpu_ivf_bulk_index_stream(gpu_ivf_bulk_t* indexer, const int* tokens, const int* lengths,
                              int num_sequences, float* progress);

// Memory management
unsigned long gpu_ivf_bulk_get_memory_usage(gpu_ivf_bulk_t* indexer);
int gpu_ivf_bulk_optimize_batch_size(int available_vram_mb);

// Search operations
int gpu_ivf_bulk_search_batch(gpu_ivf_bulk_t* indexer, const signed char* queries, const float* query_scales,
                              int num_queries, int k, int* result_ids, float* result_scores);
*/
import "C"
import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// GPUIVFBulkIndexer provides high-performance GPU-accelerated bulk indexing with IVF
type GPUIVFBulkIndexer struct {
	handle       *C.gpu_ivf_bulk_t
	config       BulkIndexConfig
	mutex        sync.RWMutex
	memoryPool   *GPUMemoryPool
	stats        IndexingStats
	
	// Progressive indexing state
	totalVectors     int64
	processedVectors int64
	startTime        time.Time
	
	// Batch optimization
	optimalBatchSize int
	maxBatchSize     int
}

// BulkIndexConfig configures the GPU bulk indexer
type BulkIndexConfig struct {
	NList          int     // Number of IVF clusters
	NProbe         int     // Number of clusters to search
	VectorDim      int     // Vector dimension (e.g., 512)
	VocabSize      int     // Vocabulary size
	EmbedDim       int     // Embedding dimension
	MaxMemoryMB    int     // Maximum GPU memory to use
	ProgressiveMode bool   // Enable progressive indexing
	OnlineUpdates  bool    // Allow online index updates
}

// DefaultBulkIndexConfig returns optimized config for bulk indexing
func DefaultBulkIndexConfig() BulkIndexConfig {
	// Get available GPU memory
	availableVRAM := GetAvailableVRAM()
	
	return BulkIndexConfig{
		NList:          1024,  // Good balance for large datasets
		NProbe:         32,    // Search accuracy vs speed
		VectorDim:      512,   // Optimized dimension
		VocabSize:      30522, // BERT vocab
		EmbedDim:       1024,  // Full embedding dimension
		MaxMemoryMB:    int(float64(availableVRAM) * 0.8), // Use 80% of VRAM
		ProgressiveMode: true,
		OnlineUpdates:  true,
	}
}

// NewGPUIVFBulkIndexer creates a new GPU bulk indexer
func NewGPUIVFBulkIndexer(config BulkIndexConfig) (*GPUIVFBulkIndexer, error) {
	if !IsCUDAAvailable() {
		return nil, fmt.Errorf("CUDA not available")
	}
	
	// Create C handle
	handle := C.gpu_ivf_bulk_create(
		C.int(config.NList),
		C.int(config.NProbe), 
		C.int(config.VectorDim),
		C.int(config.VocabSize),
		C.int(config.EmbedDim),
	)
	if handle == nil {
		return nil, fmt.Errorf("failed to create GPU IVF bulk indexer")
	}
	
	// Calculate optimal batch size based on VRAM
	optimalBatch := int(C.gpu_ivf_bulk_optimize_batch_size(C.int(config.MaxMemoryMB)))
	if optimalBatch < 100 {
		optimalBatch = 100
	}
	
	indexer := &GPUIVFBulkIndexer{
		handle:           handle,
		config:           config,
		memoryPool:       NewGPUMemoryPool(config.MaxMemoryMB),
		stats:            IndexingStats{},
		optimalBatchSize: optimalBatch,
		maxBatchSize:     optimalBatch * 5, // Allow 5x for large batches
	}
	
	runtime.SetFinalizer(indexer, (*GPUIVFBulkIndexer).destroy)
	
	log.Printf("GPU IVF Bulk Indexer created: nlist=%d, nprobe=%d, batch_size=%d", 
		config.NList, config.NProbe, optimalBatch)
	
	return indexer, nil
}

// LoadEmbeddings loads quantized embeddings to GPU for bulk processing
func (idx *GPUIVFBulkIndexer) LoadEmbeddings(embeddings []int8, scales []float32) error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()
	
	if len(embeddings) != idx.config.VocabSize * idx.config.EmbedDim {
		return fmt.Errorf("embeddings size mismatch: expected %d, got %d",
			idx.config.VocabSize * idx.config.EmbedDim, len(embeddings))
	}
	
	if len(scales) != idx.config.VocabSize {
		return fmt.Errorf("scales size mismatch: expected %d, got %d",
			idx.config.VocabSize, len(scales))
	}
	
	result := C.gpu_ivf_bulk_load_embeddings(
		idx.handle,
		(*C.schar)(unsafe.Pointer(&embeddings[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
	)
	
	if result == 0 {
		return fmt.Errorf("failed to load embeddings to GPU")
	}
	
	log.Printf("Loaded %d quantized embeddings to GPU", idx.config.VocabSize)
	return nil
}

// TrainKMeans trains IVF clusters on GPU using sample vectors
func (idx *GPUIVFBulkIndexer) TrainKMeans(trainingVectors []int8, scales []float32, numTraining int) error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()
	
	if len(trainingVectors) != numTraining * idx.config.VectorDim {
		return fmt.Errorf("training vectors size mismatch: expected %d, got %d",
			numTraining * idx.config.VectorDim, len(trainingVectors))
	}
	
	start := time.Now()
	result := C.gpu_ivf_bulk_train_kmeans(
		idx.handle,
		(*C.schar)(unsafe.Pointer(&trainingVectors[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int(numTraining),
	)
	
	if result == 0 {
		return fmt.Errorf("k-means training failed on GPU")
	}
	
	trainTime := time.Since(start)
	log.Printf("GPU k-means training completed: %d clusters, %d training vectors, %v", 
		idx.config.NList, numTraining, trainTime)
	
	return nil
}

// BulkIndexTokenSequences indexes multiple token sequences in optimized batches
func (idx *GPUIVFBulkIndexer) BulkIndexTokenSequences(tokenSequences [][]int, progressCallback func(float64)) (int, error) {
	if len(tokenSequences) == 0 {
		return 0, fmt.Errorf("no token sequences provided")
	}
	
	idx.totalVectors = int64(len(tokenSequences))
	idx.processedVectors = 0
	idx.startTime = time.Now()
	
	totalIndexed := 0
	batchSize := idx.optimalBatchSize
	
	// Process in optimized batches
	for i := 0; i < len(tokenSequences); i += batchSize {
		end := i + batchSize
		if end > len(tokenSequences) {
			end = len(tokenSequences)
		}
		
		batch := tokenSequences[i:end]
		indexed, err := idx.processBatch(batch)
		if err != nil {
			return totalIndexed, fmt.Errorf("batch processing failed: %w", err)
		}
		
		totalIndexed += indexed
		idx.processedVectors = int64(totalIndexed)
		
		// Report progress
		if progressCallback != nil {
			progress := float64(idx.processedVectors) / float64(idx.totalVectors)
			progressCallback(progress)
		}
		
		// Update stats
		idx.stats.DocumentsProcessed += end - i
		idx.stats.ProcessingTime = time.Since(idx.startTime)
	}
	
	totalTime := time.Since(idx.startTime)
	throughput := float64(totalIndexed) / totalTime.Seconds()
	
	log.Printf("Bulk indexing completed: %d vectors, %.0f vec/sec, %v total", 
		totalIndexed, throughput, totalTime)
	
	return totalIndexed, nil
}

// processBatch processes a batch of token sequences on GPU
func (idx *GPUIVFBulkIndexer) processBatch(batch [][]int) (int, error) {
	batchSize := len(batch)
	
	// Find max sequence length for this batch
	maxSeqLen := 0
	for _, seq := range batch {
		if len(seq) > maxSeqLen {
			maxSeqLen = len(seq)
		}
	}
	
	// Flatten sequences into GPU-friendly format
	flatTokens := make([]int, batchSize * maxSeqLen)
	seqLengths := make([]int, batchSize)
	
	for i, seq := range batch {
		seqLengths[i] = len(seq)
		for j, token := range seq {
			flatTokens[i*maxSeqLen + j] = token
		}
	}
	
	// Convert to C types
	cSeqLengths := make([]C.int, batchSize)
	for i, l := range seqLengths {
		cSeqLengths[i] = C.int(l)
	}
	
	// Process batch on GPU
	assignedIDs := make([]int, batchSize)
	result := C.gpu_ivf_bulk_index_batch(
		idx.handle,
		(*C.int)(unsafe.Pointer(&flatTokens[0])),
		(*C.int)(unsafe.Pointer(&cSeqLengths[0])),
		C.int(batchSize),
		C.int(maxSeqLen),
		(*C.int)(unsafe.Pointer(&assignedIDs[0])),
	)
	
	if result == 0 {
		return 0, fmt.Errorf("GPU batch indexing failed")
	}
	
	return int(result), nil
}

// StreamingIndex provides progressive indexing with live updates
func (idx *GPUIVFBulkIndexer) StreamingIndex(tokenChan <-chan []int, progressChan chan<- IndexProgress) error {
	if !idx.config.ProgressiveMode {
		return fmt.Errorf("progressive mode not enabled")
	}
	
	batchBuffer := make([][]int, 0, idx.optimalBatchSize)
	processed := 0
	
	for tokens := range tokenChan {
		batchBuffer = append(batchBuffer, tokens)
		
		// Process when batch is full
		if len(batchBuffer) >= idx.optimalBatchSize {
			indexed, err := idx.processBatch(batchBuffer)
			if err != nil {
				return fmt.Errorf("streaming batch failed: %w", err)
			}
			
			processed += indexed
			
			// Send progress update
			if progressChan != nil {
				progress := IndexProgress{
					Current:    processed,
					Total:      int(idx.totalVectors),
					DocsPerSec: float64(idx.stats.DocumentsProcessed) / idx.stats.ProcessingTime.Seconds(),
					TimeLeft:   time.Duration(float64(int64(processed)-idx.totalVectors) / (float64(idx.stats.DocumentsProcessed) / idx.stats.ProcessingTime.Seconds()) * float64(time.Second)),
				}
				
				select {
				case progressChan <- progress:
				default: // Don't block if channel is full
				}
			}
			
			// Reset batch buffer
			batchBuffer = batchBuffer[:0]
		}
	}
	
	// Process remaining items
	if len(batchBuffer) > 0 {
		indexed, err := idx.processBatch(batchBuffer)
		if err != nil {
			return fmt.Errorf("final batch failed: %w", err)
		}
		processed += indexed
	}
	
	close(progressChan)
	
	log.Printf("Streaming indexing completed: %d vectors processed", processed)
	return nil
}

// Search performs fast GPU search across the IVF index
func (idx *GPUIVFBulkIndexer) Search(queries []int8, queryScales []float32, k int) ([]SearchResult, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	
	numQueries := len(queryScales)
	if len(queries) != numQueries * idx.config.VectorDim {
		return nil, fmt.Errorf("query dimension mismatch")
	}
	
	// Allocate result buffers
	resultIDs := make([]int, numQueries * k)
	resultScores := make([]float32, numQueries * k)
	
	result := C.gpu_ivf_bulk_search_batch(
		idx.handle,
		(*C.schar)(unsafe.Pointer(&queries[0])),
		(*C.float)(unsafe.Pointer(&queryScales[0])),
		C.int(numQueries),
		C.int(k),
		(*C.int)(unsafe.Pointer(&resultIDs[0])),
		(*C.float)(unsafe.Pointer(&resultScores[0])),
	)
	
	if result == 0 {
		return nil, fmt.Errorf("GPU search failed")
	}
	
	// Convert to SearchResult format
	results := make([]SearchResult, numQueries * k)
	for i := 0; i < numQueries * k; i++ {
		results[i] = SearchResult{
			ID:         resultIDs[i],
			Similarity: resultScores[i],
		}
	}
	
	return results, nil
}

// GetMemoryUsage returns current GPU memory usage
func (idx *GPUIVFBulkIndexer) GetMemoryUsage() uint64 {
	return uint64(C.gpu_ivf_bulk_get_memory_usage(idx.handle))
}

// GetStats returns comprehensive indexing statistics
func (idx *GPUIVFBulkIndexer) GetStats() IndexingStats {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	
	return idx.stats
}

// GetProgress returns current indexing progress
func (idx *GPUIVFBulkIndexer) GetProgress() IndexProgress {
	if idx.totalVectors == 0 {
		return IndexProgress{}
	}
	
	elapsed := time.Since(idx.startTime)
	throughput := float64(idx.processedVectors) / elapsed.Seconds()
	remaining := idx.totalVectors - idx.processedVectors
	
	var timeLeft time.Duration
	if throughput > 0 {
		timeLeft = time.Duration(float64(remaining)/throughput) * time.Second
	}
	
	return IndexProgress{
		Current:    int(idx.processedVectors),
		Total:      int(idx.totalVectors),
		Percentage: float64(idx.processedVectors) / float64(idx.totalVectors) * 100,
		DocsPerSec: throughput,
		TimeLeft:   timeLeft,
	}
}

// OptimizeBatchSize dynamically adjusts batch size based on performance
func (idx *GPUIVFBulkIndexer) OptimizeBatchSize(targetLatencyMs int) {
	currentLatency := float64(idx.stats.ProcessingTime.Milliseconds()) / float64(idx.stats.DocumentsProcessed)
	
	if currentLatency > float64(targetLatencyMs) && idx.optimalBatchSize > 100 {
		idx.optimalBatchSize = int(float64(idx.optimalBatchSize) * 0.8)
	} else if currentLatency < float64(targetLatencyMs)/2 && idx.optimalBatchSize < idx.maxBatchSize {
		idx.optimalBatchSize = int(float64(idx.optimalBatchSize) * 1.2)
	}
	
	log.Printf("Batch size optimized: %d (target latency: %dms, current: %.1fms)",
		idx.optimalBatchSize, targetLatencyMs, currentLatency)
}

// destroy cleans up GPU resources
func (idx *GPUIVFBulkIndexer) destroy() {
	if idx.handle != nil {
		C.gpu_ivf_bulk_destroy(idx.handle)
		idx.handle = nil
	}
	if idx.memoryPool != nil {
		idx.memoryPool.Cleanup()
	}
}

// Close explicitly releases resources
func (idx *GPUIVFBulkIndexer) Close() {
	idx.destroy()
}

