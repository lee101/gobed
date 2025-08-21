package gobed

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lee101/gobed/ann/simd"
)

// ParallelIndexer provides high-performance parallel indexing
type ParallelIndexer struct {
	engine       *SearchEngine
	numWorkers   int
	batchSize    int
	enableCache  bool
	
	// Worker pool
	workerPool   chan struct{}
	taskQueue    chan indexTask
	resultQueue  chan indexResult
	
	// Statistics
	totalIndexed uint64
	totalTime    uint64
	errors       uint32
}

// indexTask represents a single indexing task
type indexTask struct {
	id       int
	text     string
	docID    int
}

// indexResult represents the result of indexing
type indexResult struct {
	docID    int
	vector   simd.Vec512
	scale    float32
	err      error
	duration time.Duration
}

// ParallelIndexConfig configures parallel indexing
type ParallelIndexConfig struct {
	NumWorkers   int  // Number of parallel workers (default: NumCPU)
	BatchSize    int  // Batch size for processing (default: 100)
	EnableCache  bool // Enable embedding cache (default: true)
	QueueSize    int  // Task queue size (default: 1000)
}

// DefaultParallelIndexConfig returns optimized configuration
func DefaultParallelIndexConfig() ParallelIndexConfig {
	return ParallelIndexConfig{
		NumWorkers:  runtime.NumCPU(),
		BatchSize:   100,
		EnableCache: true,
		QueueSize:   1000,
	}
}

// NewParallelIndexer creates a new parallel indexer
func NewParallelIndexer(engine *SearchEngine, config ParallelIndexConfig) *ParallelIndexer {
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 100
	}
	if config.QueueSize <= 0 {
		config.QueueSize = 1000
	}
	
	indexer := &ParallelIndexer{
		engine:      engine,
		numWorkers:  config.NumWorkers,
		batchSize:   config.BatchSize,
		enableCache: config.EnableCache,
		workerPool:  make(chan struct{}, config.NumWorkers),
		taskQueue:   make(chan indexTask, config.QueueSize),
		resultQueue: make(chan indexResult, config.QueueSize),
	}
	
	// Initialize worker pool
	for i := 0; i < config.NumWorkers; i++ {
		indexer.workerPool <- struct{}{}
	}
	
	return indexer
}

// IndexDocumentsParallel indexes documents using parallel processing
func (p *ParallelIndexer) IndexDocumentsParallel(texts []string) ([]int, error) {
	if len(texts) == 0 {
		return []int{}, nil
	}
	
	start := time.Now()
	numDocs := len(texts)
	
	// Start result collector
	var wg sync.WaitGroup
	wg.Add(1)
	
	ids := make([]int, numDocs)
	errors := make([]error, 0)
	var errorMutex sync.Mutex
	
	go func() {
		defer wg.Done()
		
		received := 0
		for result := range p.resultQueue {
			if result.err != nil {
				errorMutex.Lock()
				errors = append(errors, result.err)
				errorMutex.Unlock()
			} else {
				ids[result.docID] = result.docID
				atomic.AddUint64(&p.totalIndexed, 1)
			}
			
			received++
			if received >= numDocs {
				break
			}
		}
	}()
	
	// Process documents in parallel batches
	for i := 0; i < numDocs; i += p.batchSize {
		end := i + p.batchSize
		if end > numDocs {
			end = numDocs
		}
		
		batch := texts[i:end]
		p.processBatch(batch, i)
	}
	
	// Wait for all results
	wg.Wait()
	
	// Update statistics
	elapsed := time.Since(start)
	atomic.AddUint64(&p.totalTime, uint64(elapsed))
	
	if len(errors) > 0 {
		return ids, fmt.Errorf("indexing completed with %d errors", len(errors))
	}
	
	return ids, nil
}

// processBatch processes a batch of documents in parallel
func (p *ParallelIndexer) processBatch(texts []string, startID int) {
	var wg sync.WaitGroup
	
	for i, text := range texts {
		// Acquire worker
		<-p.workerPool
		wg.Add(1)
		
		go func(idx int, doc string, docID int) {
			defer func() {
				p.workerPool <- struct{}{} // Return worker
				wg.Done()
			}()
			
			start := time.Now()
			
			// Generate embedding
			embedding, err := p.engine.model.EmbedInt8(doc)
			if err != nil {
				p.resultQueue <- indexResult{
					docID: docID,
					err:   fmt.Errorf("embedding error for doc %d: %v", docID, err),
				}
				return
			}
			
			// Convert to SIMD vector
			var vec simd.Vec512
			copy(vec[:], embedding.Vector)
			
			// Send result
			p.resultQueue <- indexResult{
				docID:    docID,
				vector:   vec,
				scale:    embedding.Scale,
				duration: time.Since(start),
			}
			
		}(i, text, startID+i)
	}
	
	wg.Wait()
}

// IndexWithProgress indexes documents with progress reporting
func (p *ParallelIndexer) IndexWithProgress(texts []string) (<-chan IndexProgress, error) {
	progressChan := make(chan IndexProgress, 100)
	
	go func() {
		defer close(progressChan)
		
		totalDocs := len(texts)
		completed := 0
		
		startTime := time.Now()
		
		// Process in batches
		for i := 0; i < totalDocs; i += p.batchSize {
			end := i + p.batchSize
			if end > totalDocs {
				end = totalDocs
			}
			
			batch := texts[i:end]
			p.processBatch(batch, i)
			
			completed += len(batch)
			
			// Send progress update
			progress := IndexProgress{
				Current:    completed,
				Total:      totalDocs,
				Percentage: float64(completed) / float64(totalDocs) * 100,
				DocsPerSec: float64(completed) / time.Since(startTime).Seconds(),
				TimeLeft:   estimateTimeLeft(completed, totalDocs, startTime),
			}
			
			progressChan <- progress
		}
	}()
	
	return progressChan, nil
}

// ParallelSearchEngine extends SearchEngine with parallel capabilities
type ParallelSearchEngine struct {
	*SearchEngine
	parallelIndexer *ParallelIndexer
	config          ParallelIndexConfig
}

// NewParallelSearchEngine creates a search engine with parallel indexing
func NewParallelSearchEngine(model *EmbeddingModel, config SearchConfig) *ParallelSearchEngine {
	// Create base engine
	baseEngine := NewSearchEngineWithConfig(model, config)
	
	// Create parallel config
	parallelConfig := DefaultParallelIndexConfig()
	if config.AsyncWorkers > 0 {
		parallelConfig.NumWorkers = config.AsyncWorkers
	}
	
	// Create parallel indexer
	parallelIndexer := NewParallelIndexer(baseEngine, parallelConfig)
	
	return &ParallelSearchEngine{
		SearchEngine:    baseEngine,
		parallelIndexer: parallelIndexer,
		config:         parallelConfig,
	}
}

// IndexBatchParallel indexes documents using CPU parallelization
func (e *ParallelSearchEngine) IndexBatchParallel(texts []string) ([]int, error) {
	return e.parallelIndexer.IndexDocumentsParallel(texts)
}

// IndexBatchWithComparison compares different indexing methods
func (e *ParallelSearchEngine) IndexBatchWithComparison(texts []string) (*IndexComparison, error) {
	comparison := &IndexComparison{
		NumDocuments: len(texts),
	}
	
	// Test 1: Sequential indexing
	start := time.Now()
	_, err := e.IndexBatch(texts[:min(100, len(texts))])
	comparison.SequentialTime = time.Since(start)
	if err != nil {
		comparison.SequentialError = err
	}
	
	// Test 2: Async indexing
	start = time.Now()
	response := e.IndexBatchAsync(texts[:min(100, len(texts))])
	result := <-response
	comparison.AsyncTime = time.Since(start)
	if result.Error != nil {
		comparison.AsyncError = result.Error
	}
	
	// Test 3: Parallel CPU indexing
	start = time.Now()
	_, err = e.IndexBatchParallel(texts)
	comparison.ParallelTime = time.Since(start)
	if err != nil {
		comparison.ParallelError = err
	}
	
	// Calculate speedups
	if comparison.SequentialTime > 0 {
		comparison.AsyncSpeedup = float64(comparison.SequentialTime) / float64(comparison.AsyncTime)
		comparison.ParallelSpeedup = float64(comparison.SequentialTime) / float64(comparison.ParallelTime)
	}
	
	return comparison, nil
}

// IndexComparison contains comparison results
type IndexComparison struct {
	NumDocuments     int
	SequentialTime   time.Duration
	AsyncTime        time.Duration
	ParallelTime     time.Duration
	GPUTime          time.Duration
	SequentialError  error
	AsyncError       error
	ParallelError    error
	GPUError         error
	AsyncSpeedup     float64
	ParallelSpeedup  float64
	GPUSpeedup       float64
}

// IndexProgress represents indexing progress
type IndexProgress struct {
	Current    int
	Total      int
	Percentage float64
	DocsPerSec float64
	TimeLeft   time.Duration
}

// Helper function to estimate time left
func estimateTimeLeft(current, total int, startTime time.Time) time.Duration {
	if current == 0 {
		return 0
	}
	
	elapsed := time.Since(startTime)
	rate := float64(current) / elapsed.Seconds()
	remaining := total - current
	
	if rate > 0 {
		return time.Duration(float64(remaining)/rate) * time.Second
	}
	
	return 0
}

// Stats returns parallel indexer statistics
func (p *ParallelIndexer) Stats() ParallelIndexStats {
	return ParallelIndexStats{
		TotalIndexed: atomic.LoadUint64(&p.totalIndexed),
		TotalTime:    time.Duration(atomic.LoadUint64(&p.totalTime)),
		Errors:       atomic.LoadUint32(&p.errors),
		NumWorkers:   p.numWorkers,
		BatchSize:    p.batchSize,
		DocsPerSec:   float64(atomic.LoadUint64(&p.totalIndexed)) / time.Duration(atomic.LoadUint64(&p.totalTime)).Seconds(),
	}
}

// ParallelIndexStats contains parallel indexing statistics
type ParallelIndexStats struct {
	TotalIndexed uint64
	TotalTime    time.Duration
	Errors       uint32
	NumWorkers   int
	BatchSize    int
	DocsPerSec   float64
}

// OptimizeWorkers finds the optimal number of workers
func (p *ParallelIndexer) OptimizeWorkers(testDocs []string) (int, error) {
	if len(testDocs) < 100 {
		return p.numWorkers, nil
	}
	
	testSize := min(500, len(testDocs))
	testData := testDocs[:testSize]
	
	bestWorkers := 1
	bestTime := time.Duration(1<<63 - 1)
	
	workerCounts := []int{1, 2, 4, 8, 16, 32}
	if runtime.NumCPU() > 32 {
		workerCounts = append(workerCounts, runtime.NumCPU())
	}
	
	for _, workers := range workerCounts {
		if workers > runtime.NumCPU()*2 {
			break
		}
		
		// Test with this worker count
		p.numWorkers = workers
		
		start := time.Now()
		_, err := p.IndexDocumentsParallel(testData)
		elapsed := time.Since(start)
		
		if err == nil && elapsed < bestTime {
			bestTime = elapsed
			bestWorkers = workers
		}
	}
	
	p.numWorkers = bestWorkers
	return bestWorkers, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}