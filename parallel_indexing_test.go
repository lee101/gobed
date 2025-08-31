package gobed

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestParallelIndexing(t *testing.T) {
	model, err := LoadModel()
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	t.Run("BasicParallelIndexing", func(t *testing.T) {
		// config := DefaultSearchConfig()
		engine := NewSearchEngine(model)

		parallelConfig := DefaultParallelIndexConfig()
		parallelIndexer := NewParallelIndexer(engine, parallelConfig)

		// Test documents
		docs := generateTestDocumentsParallel(100)

		ids, err := parallelIndexer.IndexDocumentsParallel(docs)
		if err != nil {
			t.Fatalf("Parallel indexing failed: %v", err)
		}

		if len(ids) != len(docs) {
			t.Errorf("Expected %d IDs, got %d", len(docs), len(ids))
		}

		// Check stats
		stats := parallelIndexer.Stats()
		if stats.TotalIndexed != uint64(len(docs)) {
			t.Errorf("Expected %d indexed, got %d", len(docs), stats.TotalIndexed)
		}

		t.Logf("Parallel indexing: %d docs in %v (%.0f docs/sec)",
			stats.TotalIndexed, stats.TotalTime, stats.DocsPerSec)
	})

	t.Run("WorkerScaling", func(t *testing.T) {
		engine := NewSearchEngine(model)
		docs := generateTestDocumentsParallel(500)

		workerCounts := []int{1, 2, 4, 8, runtime.NumCPU()}
		results := make(map[int]time.Duration)

		for _, workers := range workerCounts {
			config := ParallelIndexConfig{
				NumWorkers:  workers,
				BatchSize:   50,
				EnableCache: true,
			}

			indexer := NewParallelIndexer(engine, config)

			start := time.Now()
			_, err := indexer.IndexDocumentsParallel(docs)
			elapsed := time.Since(start)

			if err != nil {
				t.Errorf("Failed with %d workers: %v", workers, err)
				continue
			}

			results[workers] = elapsed
			t.Logf("Workers=%d: %v (%.0f docs/sec)",
				workers, elapsed, float64(len(docs))/elapsed.Seconds())
		}

		// Verify scaling improvement
		if results[1] > 0 && results[runtime.NumCPU()] > 0 {
			speedup := float64(results[1]) / float64(results[runtime.NumCPU()])
			t.Logf("Speedup from 1 to %d workers: %.2fx", runtime.NumCPU(), speedup)

			if speedup < 1.5 {
				t.Logf("Warning: Low parallel speedup (%.2fx)", speedup)
			}
		}
	})

	t.Run("BatchSizeOptimization", func(t *testing.T) {
		engine := NewSearchEngine(model)
		docs := generateTestDocumentsParallel(1000)

		batchSizes := []int{10, 50, 100, 200, 500}
		bestBatch := 0
		bestTime := time.Duration(1<<63 - 1)

		for _, batchSize := range batchSizes {
			config := ParallelIndexConfig{
				NumWorkers:  runtime.NumCPU(),
				BatchSize:   batchSize,
				EnableCache: true,
			}

			indexer := NewParallelIndexer(engine, config)

			start := time.Now()
			_, err := indexer.IndexDocumentsParallel(docs)
			elapsed := time.Since(start)

			if err == nil && elapsed < bestTime {
				bestTime = elapsed
				bestBatch = batchSize
			}

			t.Logf("BatchSize=%d: %v", batchSize, elapsed)
		}

		t.Logf("Optimal batch size: %d (time: %v)", bestBatch, bestTime)
	})

	t.Run("ProgressReporting", func(t *testing.T) {
		engine := NewSearchEngine(model)
		config := DefaultParallelIndexConfig()
		indexer := NewParallelIndexer(engine, config)

		docs := generateTestDocumentsParallel(200)

		progressChan, err := indexer.IndexWithProgress(docs)
		if err != nil {
			t.Fatalf("Failed to start progress indexing: %v", err)
		}

		var lastProgress IndexProgress
		progressUpdates := 0

		for progress := range progressChan {
			progressUpdates++
			lastProgress = progress

			if progress.Current > progress.Total {
				t.Errorf("Invalid progress: %d/%d", progress.Current, progress.Total)
			}

			t.Logf("Progress: %.1f%% (%d/%d) - %.0f docs/sec - ETA: %v",
				progress.Percentage, progress.Current, progress.Total,
				progress.DocsPerSec, progress.TimeLeft)
		}

		if lastProgress.Current != len(docs) {
			t.Errorf("Final progress mismatch: %d != %d", lastProgress.Current, len(docs))
		}

		if progressUpdates == 0 {
			t.Error("No progress updates received")
		}
	})

	t.Run("ConcurrentBatches", func(t *testing.T) {
		engine := NewSearchEngine(model)
		config := DefaultParallelIndexConfig()
		indexer := NewParallelIndexer(engine, config)

		// Multiple concurrent batches
		numBatches := 5
		docsPerBatch := 100

		var wg sync.WaitGroup
		errors := make([]error, numBatches)

		start := time.Now()

		for i := 0; i < numBatches; i++ {
			wg.Add(1)
			go func(batchID int) {
				defer wg.Done()

				docs := make([]string, docsPerBatch)
				for j := 0; j < docsPerBatch; j++ {
					docs[j] = fmt.Sprintf("Batch %d document %d", batchID, j)
				}

				_, err := indexer.IndexDocumentsParallel(docs)
				errors[batchID] = err
			}(i)
		}

		wg.Wait()
		elapsed := time.Since(start)

		// Check for errors
		for i, err := range errors {
			if err != nil {
				t.Errorf("Batch %d failed: %v", i, err)
			}
		}

		totalDocs := numBatches * docsPerBatch
		t.Logf("Concurrent batches: %d docs in %v (%.0f docs/sec)",
			totalDocs, elapsed, float64(totalDocs)/elapsed.Seconds())
	})

	t.Run("WorkerOptimization", func(t *testing.T) {
		engine := NewSearchEngine(model)
		config := DefaultParallelIndexConfig()
		indexer := NewParallelIndexer(engine, config)

		testDocs := generateTestDocumentsParallel(500)

		optimalWorkers, err := indexer.OptimizeWorkers(testDocs)
		if err != nil {
			t.Fatalf("Worker optimization failed: %v", err)
		}

		t.Logf("Optimal worker count: %d (CPUs: %d)", optimalWorkers, runtime.NumCPU())

		if optimalWorkers < 1 || optimalWorkers > runtime.NumCPU()*2 {
			t.Errorf("Unexpected optimal workers: %d", optimalWorkers)
		}
	})
}

func TestParallelSearchEngine(t *testing.T) {
	model, err := LoadModel()
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	t.Run("ComparisonTest", func(t *testing.T) {
		config := AsyncSearchConfig()
		engine := NewParallelSearchEngine(model, config)

		docs := generateTestDocumentsParallel(1000)

		comparison, err := engine.IndexBatchWithComparison(docs)
		if err != nil {
			t.Fatalf("Comparison failed: %v", err)
		}

		t.Logf("\n=== Indexing Method Comparison ===")
		t.Logf("Documents: %d", comparison.NumDocuments)
		t.Logf("Sequential: %v", comparison.SequentialTime)
		t.Logf("Async: %v (%.2fx speedup)", comparison.AsyncTime, comparison.AsyncSpeedup)
		t.Logf("Parallel: %v (%.2fx speedup)", comparison.ParallelTime, comparison.ParallelSpeedup)

		// Parallel should be faster than sequential
		if comparison.ParallelSpeedup < 1.0 {
			t.Logf("Warning: Parallel slower than sequential")
		}
	})

	t.Run("MixedWorkload", func(t *testing.T) {
		config := AsyncSearchConfig()
		engine := NewParallelSearchEngine(model, config)

		// Index some initial documents
		initialDocs := generateTestDocumentsParallel(500)
		_, err := engine.IndexBatchParallel(initialDocs)
		if err != nil {
			t.Fatalf("Initial indexing failed: %v", err)
		}

		// Concurrent indexing and searching
		var wg sync.WaitGroup

		// Indexing workers
		for i := 0; i < 2; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()

				docs := generateTestDocumentsParallel(100)
				_, err := engine.IndexBatchParallel(docs)
				if err != nil {
					t.Errorf("Worker %d indexing failed: %v", workerID, err)
				}
			}(i)
		}

		// Search workers
		for i := 0; i < 4; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()

				for j := 0; j < 10; j++ {
					query := fmt.Sprintf("test query %d", j)
					results, err := engine.Search(query, 5)
					if err != nil {
						t.Errorf("Worker %d search failed: %v", workerID, err)
					}

					if len(results) == 0 {
						t.Logf("Worker %d: No results for query %d", workerID, j)
					}
				}
			}(i)
		}

		wg.Wait()

		stats := engine.Stats()
		t.Logf("Final index size: %d documents", stats.IndexSize)
	})
}

func TestGPUIndexing(t *testing.T) {
	model, err := LoadModel()
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	t.Run("GPUAvailability", func(t *testing.T) {
		config := DefaultVectorIndexConfig()
		config.EnableBulkGPU = true

		index := NewVectorIndex(model, config)

		// Check if GPU indexer was created
		if index.bulkIndexer != nil {
			t.Log("GPU indexer available")
			stats := index.bulkIndexer.Stats()
			t.Logf("GPU: Available=%v, Memory=%dMB",
				stats.GPUAvailable, stats.GPUMemoryMB)
		} else {
			t.Log("GPU indexer not available")
		}
	})

	t.Run("BulkGPUIndexing", func(t *testing.T) {
		config := DefaultVectorIndexConfig()
		config.EnableBulkGPU = true
		config.BulkBatchSize = 100

		index := NewVectorIndex(model, config)

		if index.bulkIndexer == nil {
			t.Skip("GPU indexer not available")
		}

		// Create test documents
		docs := make([]Document, 500)
		for i := 0; i < 500; i++ {
			docs[i] = Document{
				ID:   i,
				Text: fmt.Sprintf("GPU test document %d with content", i),
			}
		}

		// Test bulk indexing
		start := time.Now()
		err := index.AddDocumentsBulkGPU(docs)
		elapsed := time.Since(start)

		if err != nil {
			t.Fatalf("Bulk GPU indexing failed: %v", err)
		}

		stats := index.bulkIndexer.Stats()
		t.Logf("GPU Indexing: %d docs in %v (%.0f docs/sec)",
			stats.TotalProcessed, elapsed, stats.DocsPerSecond)
		t.Logf("GPU Utilization: %.1f%%", stats.GPUUtilization)
	})

	t.Run("MonitoredGPUIndexing", func(t *testing.T) {
		config := DefaultVectorIndexConfig()
		config.EnableBulkGPU = true

		index := NewVectorIndex(model, config)

		if index.bulkIndexer == nil {
			t.Skip("GPU indexer not available")
		}

		// Create documents
		docs := make([]Document, 200)
		for i := 0; i < 200; i++ {
			docs[i] = Document{
				ID:   i,
				Text: fmt.Sprintf("Monitored document %d", i),
			}
		}

		// Index with monitoring
		progressChan, err := index.AddDocumentsWithMonitoring(docs)
		if err != nil {
			t.Fatalf("Monitored indexing failed: %v", err)
		}

		var lastProgress EnhancedIndexProgress
		for progress := range progressChan {
			lastProgress = progress

			t.Logf("Progress: %.1f%% - Batch %d/%d - GPU: %.1f%% - Memory: %d/%d MB",
				progress.Percentage,
				progress.CurrentBatch, progress.TotalBatches,
				progress.GPUUtilization,
				progress.GPUMemoryUsed, progress.GPUMemoryTotal)

			if progress.Error != nil {
				t.Errorf("Batch error: %v", progress.Error)
			}
		}

		if lastProgress.Current != len(docs) {
			t.Errorf("Incomplete indexing: %d/%d", lastProgress.Current, len(docs))
		}
	})

	t.Run("CPUvsGPUComparison", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping comparison test in short mode")
		}

		config := DefaultVectorIndexConfig()
		config.EnableBulkGPU = true

		index := NewVectorIndex(model, config)

		// Test documents
		docs := make([]Document, 1000)
		for i := 0; i < 1000; i++ {
			docs[i] = Document{
				ID:   i,
				Text: fmt.Sprintf("Comparison document %d with substantial content for testing", i),
			}
		}

		// CPU indexing
		cpuStart := time.Now()
		err := index.AddDocuments(docs[:500])
		cpuTime := time.Since(cpuStart)

		if err != nil {
			t.Fatalf("CPU indexing failed: %v", err)
		}

		// GPU indexing (if available)
		if index.bulkIndexer != nil {
			gpuStart := time.Now()
			err = index.AddDocumentsBulkGPU(docs[500:])
			gpuTime := time.Since(gpuStart)

			if err != nil {
				t.Fatalf("GPU indexing failed: %v", err)
			}

			speedup := float64(cpuTime) / float64(gpuTime)
			t.Logf("\n=== CPU vs GPU Comparison ===")
			t.Logf("CPU: %v (%.0f docs/sec)", cpuTime, 500/cpuTime.Seconds())
			t.Logf("GPU: %v (%.0f docs/sec)", gpuTime, 500/gpuTime.Seconds())
			t.Logf("GPU Speedup: %.2fx", speedup)
		} else {
			t.Log("GPU not available for comparison")
		}
	})
}

func BenchmarkParallelIndexing(b *testing.B) {
	model, _ := LoadModel()
	engine := NewSearchEngine(model)

	docs := generateTestDocumentsParallel(100)

	b.Run("Sequential", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			engine.IndexBatch(docs)
		}
	})

	b.Run("Parallel-4", func(b *testing.B) {
		config := ParallelIndexConfig{NumWorkers: 4}
		indexer := NewParallelIndexer(engine, config)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			indexer.IndexDocumentsParallel(docs)
		}
	})

	b.Run("Parallel-8", func(b *testing.B) {
		config := ParallelIndexConfig{NumWorkers: 8}
		indexer := NewParallelIndexer(engine, config)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			indexer.IndexDocumentsParallel(docs)
		}
	})
}

// Helper function to generate test documents
func generateTestDocumentsParallel(n int) []string {
	docs := make([]string, n)
	templates := []string{
		"Advanced machine learning techniques for %d",
		"Deep neural networks in application %d",
		"Natural language processing system %d",
		"Computer vision algorithm number %d",
		"Data science methodology item %d",
	}

	for i := 0; i < n; i++ {
		template := templates[i%len(templates)]
		docs[i] = fmt.Sprintf(template, i)
	}

	return docs
}
