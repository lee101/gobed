package gobed

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestAsyncIndexing(t *testing.T) {
	model, err := LoadModel()
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	t.Run("BasicAsync", func(t *testing.T) {
		config := AsyncSearchConfig()
		engine := NewSearchEngineWithConfig(model, config)

		// Test async indexing
		docs := []string{
			"async test document 1",
			"async test document 2",
			"async test document 3",
		}

		response := engine.IndexBatchAsync(docs)
		result := <-response

		if result.Error != nil {
			t.Fatalf("Async indexing failed: %v", result.Error)
		}

		if len(result.IDs) != len(docs) {
			t.Errorf("Expected %d IDs, got %d", len(docs), len(result.IDs))
		}

		// Verify documents are searchable
		results, err := engine.Search("async test", 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results) != 3 {
			t.Errorf("Expected 3 results, got %d", len(results))
		}
	})

	t.Run("ConcurrentAsync", func(t *testing.T) {
		config := AsyncSearchConfig()
		config.AsyncWorkers = 4
		config.AsyncQueueSize = 100
		engine := NewSearchEngineWithConfig(model, config)

		numBatches := 10
		docsPerBatch := 20
		var wg sync.WaitGroup
		var totalIndexed int32

		start := time.Now()
		for b := 0; b < numBatches; b++ {
			wg.Add(1)
			go func(batchID int) {
				defer wg.Done()

				docs := make([]string, docsPerBatch)
				for i := 0; i < docsPerBatch; i++ {
					docs[i] = fmt.Sprintf("concurrent batch %d doc %d", batchID, i)
				}

				response := engine.IndexBatchAsync(docs)
				result := <-response

				if result.Error != nil {
					t.Errorf("Batch %d failed: %v", batchID, result.Error)
				} else {
					atomic.AddInt32(&totalIndexed, int32(len(result.IDs)))
				}
			}(b)
		}

		wg.Wait()
		elapsed := time.Since(start)

		expectedTotal := int32(numBatches * docsPerBatch)
		if totalIndexed != expectedTotal {
			t.Errorf("Expected %d indexed, got %d", expectedTotal, totalIndexed)
		}

		docsPerSec := float64(totalIndexed) / elapsed.Seconds()
		t.Logf("Concurrent async: %d docs in %v (%.0f docs/sec)", totalIndexed, elapsed, docsPerSec)
	})

	t.Run("QueueOverflow", func(t *testing.T) {
		config := AsyncSearchConfig()
		config.AsyncWorkers = 1
		config.AsyncQueueSize = 2 // Very small queue
		engine := NewSearchEngineWithConfig(model, config)

		// Try to overflow the queue
		responses := make([]<-chan IndexResponse, 5)
		for i := 0; i < 5; i++ {
			docs := []string{fmt.Sprintf("overflow doc %d", i)}
			responses[i] = engine.IndexBatchAsync(docs)
		}

		// Collect results
		successCount := 0
		for _, resp := range responses {
			result := <-resp
			if result.Error == nil {
				successCount++
			}
		}

		// Should have processed at least some
		if successCount == 0 {
			t.Error("No documents were indexed")
		}

		t.Logf("Queue overflow test: %d/%d successful", successCount, len(responses))
	})

	t.Run("AsyncVsSync", func(t *testing.T) {
		// Create two engines
		asyncConfig := AsyncSearchConfig()
		asyncEngine := NewSearchEngineWithConfig(model, asyncConfig)

		syncConfig := DefaultSearchConfig()
		syncEngine := NewSearchEngineWithConfig(model, syncConfig)

		numDocs := 100
		docs := make([]string, numDocs)
		for i := 0; i < numDocs; i++ {
			docs[i] = fmt.Sprintf("comparison document %d", i)
		}

		// Benchmark async
		asyncStart := time.Now()
		asyncResp := asyncEngine.IndexBatchAsync(docs)
		asyncResult := <-asyncResp
		asyncElapsed := time.Since(asyncStart)

		if asyncResult.Error != nil {
			t.Fatalf("Async failed: %v", asyncResult.Error)
		}

		// Benchmark sync
		syncStart := time.Now()
		syncIDs, syncErr := syncEngine.IndexBatch(docs)
		syncElapsed := time.Since(syncStart)

		if syncErr != nil {
			t.Fatalf("Sync failed: %v", syncErr)
		}

		// Compare
		speedup := float64(syncElapsed) / float64(asyncElapsed)
		t.Logf("Async: %v, Sync: %v, Speedup: %.2fx", asyncElapsed, syncElapsed, speedup)

		if len(asyncResult.IDs) != len(syncIDs) {
			t.Errorf("Result mismatch: async=%d, sync=%d", len(asyncResult.IDs), len(syncIDs))
		}
	})

	t.Run("ErrorHandling", func(t *testing.T) {
		config := AsyncSearchConfig()
		engine := NewSearchEngineWithConfig(model, config)

		// Test with empty documents
		emptyDocs := []string{"", "", ""}
		response := engine.IndexBatchAsync(emptyDocs)
		result := <-response

		// Should handle gracefully (might index as empty or skip)
		if result.Error != nil {
			t.Logf("Empty docs error (expected): %v", result.Error)
		}

		// Test with nil slice
		var nilDocs []string
		response = engine.IndexBatchAsync(nilDocs)
		result = <-response

		if result.Error == nil && len(result.IDs) > 0 {
			t.Error("Expected no results for nil docs")
		}
	})

	t.Run("WorkerScaling", func(t *testing.T) {
		workerCounts := []int{1, 2, 4, 8}
		numDocs := 200
		docs := make([]string, numDocs)
		for i := 0; i < numDocs; i++ {
			docs[i] = fmt.Sprintf("scaling test document %d with content", i)
		}

		for _, workers := range workerCounts {
			config := AsyncSearchConfig()
			config.AsyncWorkers = workers
			engine := NewSearchEngineWithConfig(model, config)

			start := time.Now()
			response := engine.IndexBatchAsync(docs)
			result := <-response
			elapsed := time.Since(start)

			if result.Error != nil {
				t.Errorf("Workers=%d failed: %v", workers, result.Error)
				continue
			}

			docsPerSec := float64(numDocs) / elapsed.Seconds()
			t.Logf("Workers=%d: %v (%.0f docs/sec)", workers, elapsed, docsPerSec)
		}
	})
}

func TestAsyncSearchPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	model, err := LoadModel()
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	config := AsyncSearchConfig()
	config.AsyncWorkers = 8
	config.AsyncQueueSize = 1000
	config.MaxExactSearchSize = 6000 // Allow 5000 docs without IVF training
	config.AutoMode = false // Disable auto mode to use manual config
	t.Logf("Config MaxExactSearchSize: %d, AutoMode: %v", config.MaxExactSearchSize, config.AutoMode)
	engine := NewSearchEngineWithConfig(model, config)

	// Large scale test
	t.Run("LargeScaleAsync", func(t *testing.T) {
		numDocs := 5000
		batchSize := 100

		start := time.Now()
		responses := make([]<-chan IndexResponse, 0)

		for i := 0; i < numDocs; i += batchSize {
			batch := make([]string, batchSize)
			for j := 0; j < batchSize; j++ {
				batch[j] = fmt.Sprintf("large scale document %d", i+j)
			}
			responses = append(responses, engine.IndexBatchAsync(batch))
		}

		// Collect all results
		totalIndexed := 0
		for _, resp := range responses {
			result := <-resp
			if result.Error != nil {
				t.Errorf("Batch failed: %v", result.Error)
			} else {
				totalIndexed += len(result.IDs)
			}
		}

		elapsed := time.Since(start)
		docsPerSec := float64(totalIndexed) / elapsed.Seconds()

		t.Logf("Large scale: %d docs in %v (%.0f docs/sec)", totalIndexed, elapsed, docsPerSec)

		// Test search on large index
		searchStart := time.Now()
		results, err := engine.Search("large scale document", 10)
		searchElapsed := time.Since(searchStart)

		if err != nil {
			t.Errorf("Search failed: %v", err)
		}

		t.Logf("Search on %d docs: %v, found %d results", totalIndexed, searchElapsed, len(results))
	})

	t.Run("MixedWorkload", func(t *testing.T) {
		// Simulate mixed indexing and searching
		var wg sync.WaitGroup
		stopCh := make(chan struct{})

		var indexCount int32
		var searchCount int32

		// Indexing workers
		for w := 0; w < 2; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for {
					select {
					case <-stopCh:
						return
					default:
						docs := []string{
							fmt.Sprintf("mixed worker %d doc %d", workerID, time.Now().UnixNano()),
						}
						response := engine.IndexBatchAsync(docs)
						result := <-response
						if result.Error == nil {
							atomic.AddInt32(&indexCount, 1)
						}
					}
				}
			}(w)
		}

		// Search workers
		for w := 0; w < 4; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for {
					select {
					case <-stopCh:
						return
					default:
						_, err := engine.Search("mixed", 5)
						if err == nil {
							atomic.AddInt32(&searchCount, 1)
						}
					}
				}
			}(w)
		}

		// Run for 2 seconds
		time.Sleep(2 * time.Second)
		close(stopCh)
		wg.Wait()

		t.Logf("Mixed workload: %d indexed, %d searches in 2s", indexCount, searchCount)
	})
}

func BenchmarkAsyncIndexing(b *testing.B) {
	model, _ := LoadModel()
	config := AsyncSearchConfig()
	engine := NewSearchEngineWithConfig(model, config)

	docs := make([]string, 100)
	for i := 0; i < 100; i++ {
		docs[i] = fmt.Sprintf("benchmark document %d", i)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			response := engine.IndexBatchAsync(docs)
			<-response
		}
	})
}

func BenchmarkAsyncVsSyncIndexing(b *testing.B) {
	model, _ := LoadModel()

	docs := make([]string, 50)
	for i := 0; i < 50; i++ {
		docs[i] = fmt.Sprintf("comparison benchmark document %d with some content", i)
	}

	b.Run("Sync", func(b *testing.B) {
		config := DefaultSearchConfig()
		engine := NewSearchEngineWithConfig(model, config)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			engine.IndexBatch(docs)
		}
	})

	b.Run("Async", func(b *testing.B) {
		config := AsyncSearchConfig()
		engine := NewSearchEngineWithConfig(model, config)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			response := engine.IndexBatchAsync(docs)
			<-response
		}
	})
}
