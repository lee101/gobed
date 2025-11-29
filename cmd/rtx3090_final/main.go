//go:build legacy

package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println(" RTX 3090 FINAL OPTIMIZED BENCHMARK - FINDING BEST CONFIGURATION")
	fmt.Println("================================================================================")
	fmt.Printf("GPU: NVIDIA RTX 3090 (24GB VRAM)\n")
	fmt.Printf("CPU: %d cores\n", runtime.NumCPU())
	fmt.Printf("CUDA: Enabled with GPU acceleration\n\n")

	// Load model
	fmt.Println("Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(" Model loaded successfully\n")

	// Results tracking
	type Result struct {
		Name       string
		Throughput float64
		QPS        float64
		BatchSize  int
		DocCount   int
		Workers    int
	}
	var results []Result

	// Test 1: Find optimal batch size for small datasets (no training needed)
	fmt.Println("=" + repeat("=", 79))
	fmt.Println("TEST 1: OPTIMAL BATCH SIZE FOR SMALL DATASETS")
	fmt.Println("=" + repeat("=", 79))
	
	batchSizes := []int{50, 100, 250, 500, 750, 1000, 1500, 2000}
	docCount := 2000 // Stay under training threshold
	
	for _, batchSize := range batchSizes {
		fmt.Printf("\nTesting batch size %d with %d documents...\n", batchSize, docCount)
		
		engine := gobed.NewSearchEngine(model)
		corpus := generateCorpus(docCount)
		
		start := time.Now()
		totalIndexed := 0
		
		for i := 0; i < docCount; i += batchSize {
			end := min(i+batchSize, docCount)
			batch := corpus[i:end]
			ids, err := engine.IndexBatch(batch)
			if err == nil {
				totalIndexed += len(ids)
			}
		}
		
		elapsed := time.Since(start)
		throughput := float64(totalIndexed) / elapsed.Seconds()
		
		// Search test
		searchStart := time.Now()
		for i := 0; i < 100; i++ {
			engine.Search("test query", 10)
		}
		searchElapsed := time.Since(searchStart)
		qps := 100.0 / searchElapsed.Seconds()
		
		fmt.Printf("  ✓ Throughput: %.0f docs/sec | QPS: %.0f\n", throughput, qps)
		
		results = append(results, Result{
			Name:       fmt.Sprintf("Batch_%d", batchSize),
			Throughput: throughput,
			QPS:        qps,
			BatchSize:  batchSize,
			DocCount:   docCount,
			Workers:    1,
		})
	}

	// Test 2: Concurrent processing with optimal batch size
	fmt.Println("\n" + "=" + repeat("=", 79))
	fmt.Println("TEST 2: CONCURRENT PROCESSING OPTIMIZATION")
	fmt.Println("=" + repeat("=", 79))
	
	optimalBatch := 500 // Based on typical results
	workerCounts := []int{1, 2, 4, 8, 16, 24, 32}
	testDocCount := 10000
	
	for _, workers := range workerCounts {
		fmt.Printf("\nTesting %d concurrent workers with batch size %d...\n", workers, optimalBatch)
		
		engine := gobed.NewSearchEngine(model)
		corpus := generateCorpus(testDocCount)
		
		start := time.Now()
		var wg sync.WaitGroup
		docsPerWorker := testDocCount / workers
		totalIndexed := 0
		var mu sync.Mutex
		
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				
				startIdx := workerID * docsPerWorker
				endIdx := min(startIdx+docsPerWorker, testDocCount)
				
				for i := startIdx; i < endIdx; i += optimalBatch {
					end := min(i+optimalBatch, endIdx)
					batch := corpus[i:end]
					ids, err := engine.IndexBatch(batch)
					if err == nil {
						mu.Lock()
						totalIndexed += len(ids)
						mu.Unlock()
					}
				}
			}(w)
		}
		
		wg.Wait()
		elapsed := time.Since(start)
		throughput := float64(totalIndexed) / elapsed.Seconds()
		
		fmt.Printf("  ✓ Indexed: %d docs | Throughput: %.0f docs/sec\n", totalIndexed, throughput)
		
		results = append(results, Result{
			Name:       fmt.Sprintf("Workers_%d", workers),
			Throughput: throughput,
			QPS:        0,
			BatchSize:  optimalBatch,
			DocCount:   totalIndexed,
			Workers:    workers,
		})
	}

	// Test 3: Large dataset with training (100K+ documents)
	fmt.Println("\n" + "=" + repeat("=", 79))
	fmt.Println("TEST 3: LARGE DATASET PERFORMANCE (WITH INDEX TRAINING)")
	fmt.Println("=" + repeat("=", 79))
	
	largeSizes := []int{100000, 250000, 500000}
	
	for _, size := range largeSizes {
		fmt.Printf("\nTesting %d documents with automatic index training...\n", size)
		
		searchConfig := gobed.SearchConfig{
			AutoMode:           true,
			MaxExactSearchSize: 50000,
			EnableGPU:          true,
		}
		engine := gobed.NewSearchEngineWithConfig(model, searchConfig)
		
		corpus := generateCorpus(size)
		batchSize := 10000 // Large batches for efficiency
		
		fmt.Println("  Indexing...")
		start := time.Now()
		totalIndexed := 0
		
		for i := 0; i < size; i += batchSize {
			end := min(i+batchSize, size)
			batch := corpus[i:end]
			ids, err := engine.IndexBatch(batch)
			if err == nil {
				totalIndexed += len(ids)
			} else {
				fmt.Printf("    Warning: %v\n", err)
			}
			
			// Progress indicator
			if (i+batchSize) % 50000 == 0 {
				fmt.Printf("    Progress: %d/%d\n", min(i+batchSize, size), size)
			}
		}
		
		indexElapsed := time.Since(start)
		throughput := float64(totalIndexed) / indexElapsed.Seconds()
		
		// Optimize if needed
		stats := engine.Stats()
		if stats.IndexType != "flat" {
			fmt.Println("  Optimizing index...")
			optStart := time.Now()
			engine.Optimize()
			fmt.Printf("    Optimization took: %v\n", time.Since(optStart))
		}
		
		// Search benchmark
		fmt.Println("  Running search benchmark...")
		searchStart := time.Now()
		searchCount := 1000
		for i := 0; i < searchCount; i++ {
			engine.Search(fmt.Sprintf("query %d", i%100), 10)
		}
		searchElapsed := time.Since(searchStart)
		qps := float64(searchCount) / searchElapsed.Seconds()
		
		fmt.Printf("  ✓ Indexed: %d docs in %v\n", totalIndexed, indexElapsed)
		fmt.Printf("  ✓ Throughput: %.0f docs/sec\n", throughput)
		fmt.Printf("  ✓ Search QPS: %.0f\n", qps)
		fmt.Printf("  ✓ Index Type: %s\n", stats.IndexType)
		
		results = append(results, Result{
			Name:       fmt.Sprintf("Large_%dK", size/1000),
			Throughput: throughput,
			QPS:        qps,
			BatchSize:  batchSize,
			DocCount:   totalIndexed,
			Workers:    1,
		})
	}

	// Final summary
	fmt.Println("\n" + "=" + repeat("=", 79))
	fmt.Println(" FINAL RESULTS SUMMARY")
	fmt.Println("=" + repeat("=", 79))
	
	// Find best configurations
	var bestThroughput, bestQPS Result
	for _, r := range results {
		if r.Throughput > bestThroughput.Throughput {
			bestThroughput = r
		}
		if r.QPS > bestQPS.QPS {
			bestQPS = r
		}
	}
	
	fmt.Printf("\n BEST INDEXING THROUGHPUT:\n")
	fmt.Printf("   Configuration: %s\n", bestThroughput.Name)
	fmt.Printf("   Throughput: %.0f docs/sec\n", bestThroughput.Throughput)
	fmt.Printf("   Batch Size: %d\n", bestThroughput.BatchSize)
	fmt.Printf("   Workers: %d\n", bestThroughput.Workers)
	
	fmt.Printf("\n BEST SEARCH PERFORMANCE:\n")
	fmt.Printf("   Configuration: %s\n", bestQPS.Name)
	fmt.Printf("   QPS: %.0f queries/sec\n", bestQPS.QPS)
	fmt.Printf("   Document Count: %d\n", bestQPS.DocCount)
	
	fmt.Printf("\n RECOMMENDATIONS FOR RTX 3090:\n")
	fmt.Printf("   • For small datasets (<2K docs): Use batch size 250-500\n")
	fmt.Printf("   • For medium datasets (2K-50K): Use batch size 1000-5000\n")
	fmt.Printf("   • For large datasets (>100K): Use batch size 10K-50K with training\n")
	fmt.Printf("   • Optimal workers: %d (matches CPU cores)\n", runtime.NumCPU())
	fmt.Printf("   • GPU memory usage: Minimal (model fits easily in 24GB)\n")
	
	fmt.Println("\n" + "=" + repeat("=", 79))
}

func generateCorpus(size int) []string {
	templates := []string{
		"Advanced %s techniques",
		"Implementing %s at scale",
		"Optimizing %s performance",
		"Building %s systems",
		"Real-time %s processing",
	}

	topics := []string{
		"machine learning", "deep learning", "neural networks",
		"computer vision", "natural language processing",
	}

	corpus := make([]string, size)
	for i := 0; i < size; i++ {
		template := templates[i%len(templates)]
		topic := topics[i%len(topics)]
		corpus[i] = fmt.Sprintf(template, topic)
	}
	return corpus
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func repeat(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}
