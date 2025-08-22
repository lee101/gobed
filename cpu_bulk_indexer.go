package gobed

import (
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lee101/gobed/ann/simd"
)

// CPUBulkIndexer provides fast bulk indexing using CPU with parallelization
type CPUBulkIndexer struct {
	index      *VectorIndex
	model      *EmbeddingModel
	batchSize  int
	numWorkers int

	// Statistics
	totalIndexed int64
	totalTime    time.Duration
	mu           sync.Mutex
}

// NewCPUBulkIndexer creates a new CPU bulk indexer
func NewCPUBulkIndexer(index *VectorIndex, batchSize int) *CPUBulkIndexer {
	if batchSize <= 0 {
		batchSize = 1000 // Default batch size for CPU
	}

	return &CPUBulkIndexer{
		index:      index,
		model:      index.model,
		batchSize:  batchSize,
		numWorkers: 4, // Use 4 workers for parallel processing
	}
}

// IndexBatch processes a batch of documents using CPU with parallelization
func (idx *CPUBulkIndexer) IndexBatch(docs []Document) error {
	if len(docs) == 0 {
		return nil
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	startTime := time.Now()
	log.Printf("🔄 CPU Bulk indexing %d documents with %d workers", len(docs), idx.numWorkers)

	// Create channels for work distribution
	docChan := make(chan Document, len(docs))
	resultChan := make(chan embeddingResult, len(docs))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < idx.numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for doc := range docChan {
				// Generate embedding
				embedding, err := idx.model.EmbedInt8(doc.Text)
				if err != nil {
					log.Printf("Worker %d failed to embed doc %d: %v", workerID, doc.ID, err)
					continue
				}

				// Convert to simd vector
				var vec simd.Vec512
				copy(vec[:], embedding.Vector)

				resultChan <- embeddingResult{
					doc:   doc,
					vec:   vec,
					scale: embedding.Scale,
					err:   nil,
				}
			}
		}(i)
	}

	// Send work to workers
	go func() {
		for _, doc := range docs {
			docChan <- doc
		}
		close(docChan)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Add to index
	successCount := 0
	for result := range resultChan {
		if result.err != nil {
			log.Printf("Skipping doc %d due to error: %v", result.doc.ID, result.err)
			continue
		}

		err := idx.index.engine.Add(result.vec, result.scale, result.doc.ID)
		if err != nil {
			log.Printf("Failed to add doc %d to index: %v", result.doc.ID, err)
			continue
		}

		successCount++
	}

	elapsed := time.Since(startTime)
	idx.totalTime += elapsed
	atomic.AddInt64(&idx.totalIndexed, int64(successCount))

	throughput := float64(successCount) / elapsed.Seconds()
	log.Printf("✅ CPU indexed %d/%d documents in %.2fs (%.0f docs/sec)",
		successCount, len(docs), elapsed.Seconds(), throughput)

	return nil
}

// embeddingResult holds the result of embedding generation
type embeddingResult struct {
	doc   Document
	vec   simd.Vec512
	scale float32
	err   error
}

// Stats returns indexing statistics
func (idx *CPUBulkIndexer) Stats() CPUBulkIndexerStats {
	return CPUBulkIndexerStats{
		TotalIndexed: atomic.LoadInt64(&idx.totalIndexed),
		TotalTime:    idx.totalTime,
		BatchSize:    idx.batchSize,
		NumWorkers:   idx.numWorkers,
		Throughput:   float64(atomic.LoadInt64(&idx.totalIndexed)) / idx.totalTime.Seconds(),
	}
}

// CPUBulkIndexerStats contains CPU bulk indexing statistics
type CPUBulkIndexerStats struct {
	TotalIndexed int64
	TotalTime    time.Duration
	BatchSize    int
	NumWorkers   int
	Throughput   float64
}

// LogStats logs the indexing statistics
func (stats CPUBulkIndexerStats) LogStats() {
	log.Printf("📊 CPU Bulk Indexing Stats:")
	log.Printf("   Total indexed: %d documents", stats.TotalIndexed)
	log.Printf("   Total time: %.2fs", stats.TotalTime.Seconds())
	log.Printf("   Batch size: %d", stats.BatchSize)
	log.Printf("   Workers: %d", stats.NumWorkers)
	log.Printf("   Throughput: %.0f docs/sec", stats.Throughput)
}
