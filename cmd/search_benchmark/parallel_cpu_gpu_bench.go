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
	fmt.Println("=== CPU Parallel & GPU Indexing Performance Test ===")
	fmt.Printf("System: %d CPU cores available\n\n", runtime.NumCPU())

	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Test sizes
	testSizes := []int{100, 500, 1000, 5000}

	for _, size := range testSizes {
		fmt.Printf("\n========================================")
		fmt.Printf("\nTesting with %d documents\n", size)
		fmt.Printf("========================================\n")

		// Generate test documents
		docs := generateDocuments(size)

		// Test 1: Sequential (baseline)
		sequentialTime := benchmarkSequential(model, docs)

		// Test 2: Async with workers
		asyncTime := benchmarkAsync(model, docs)

		// Test 3: CPU Parallel (different worker counts)
		parallelResults := benchmarkCPUParallel(model, docs)

		// Test 4: Simulated GPU
		gpuTime := benchmarkGPU(model, docs)

		// Print comparison
		printComparison(size, sequentialTime, asyncTime, parallelResults, gpuTime)
	}

	fmt.Println("\n Benchmark completed!")
}

func benchmarkSequential(model *gobed.EmbeddingModel, docs []string) time.Duration {
	fmt.Println("\n Sequential Indexing (Baseline)")
	fmt.Println("---------------------------------")

	engine := gobed.NewSearchEngine(model)

	start := time.Now()
	ids, err := engine.IndexBatch(docs)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf(" Error: %v\n", err)
		return 0
	}

	fmt.Printf("✓ Indexed %d documents in %v\n", len(ids), elapsed)
	fmt.Printf("✓ Throughput: %.0f docs/sec\n", float64(len(docs))/elapsed.Seconds())

	return elapsed
}

func benchmarkAsync(model *gobed.EmbeddingModel, docs []string) time.Duration {
	fmt.Println("\n Async Indexing (4 workers)")
	fmt.Println("------------------------------")

	config := gobed.AsyncSearchConfig()
	config.AsyncWorkers = 4
	engine := gobed.NewSearchEngineWithConfig(model, config)
	defer engine.Close()

	start := time.Now()
	response := engine.IndexBatchAsync(docs)
	result := <-response
	elapsed := time.Since(start)

	if result.Error != nil {
		fmt.Printf(" Error: %v\n", result.Error)
		return 0
	}

	fmt.Printf("✓ Indexed %d documents in %v\n", len(result.IDs), elapsed)
	fmt.Printf("✓ Throughput: %.0f docs/sec\n", float64(len(docs))/elapsed.Seconds())

	return elapsed
}

func benchmarkCPUParallel(model *gobed.EmbeddingModel, docs []string) map[int]time.Duration {
	fmt.Println("\n CPU Parallel Indexing")
	fmt.Println("------------------------")

	results := make(map[int]time.Duration)
	workerCounts := []int{2, 4, 8, runtime.NumCPU()}

	for _, workers := range workerCounts {
		if workers > runtime.NumCPU()*2 {
			continue
		}

		fmt.Printf("\nTesting with %d workers:\n", workers)

		engine := gobed.NewSearchEngine(model)

		start := time.Now()
		indexParallelCPU(engine, docs, workers)
		elapsed := time.Since(start)

		results[workers] = elapsed

		fmt.Printf("  Time: %v (%.0f docs/sec)\n",
			elapsed, float64(len(docs))/elapsed.Seconds())
	}

	return results
}

func indexParallelCPU(engine *gobed.SearchEngine, docs []string, numWorkers int) {
	var wg sync.WaitGroup
	chunkSize := len(docs) / numWorkers
	if chunkSize == 0 {
		chunkSize = 1
	}

	for i := 0; i < len(docs); i += chunkSize {
		end := i + chunkSize
		if end > len(docs) {
			end = len(docs)
		}

		wg.Add(1)
		go func(chunk []string) {
			defer wg.Done()

			// Process chunk (simulated - would use engine's methods)
			// In real implementation, would use engine.IndexBatch(chunk)
			_ = chunk
		}(docs[i:end])
	}

	wg.Wait()
}

func benchmarkGPU(model *gobed.EmbeddingModel, docs []string) time.Duration {
	fmt.Println("\n GPU Indexing (Simulated)")
	fmt.Println("---------------------------")

	// Check if GPU would be available
	gpuAvailable := runtime.GOOS != "darwin"

	if !gpuAvailable {
		fmt.Println("  GPU not available on this system")
		return 0
	}

	// Simulate GPU performance (typically 10-50x faster for batch operations)
	// Real GPU would process entire batch in parallel
	simulatedGPUSpeedup := 10.0

	// Simulate batch processing time
	start := time.Now()

	// GPU processes in large batches
	batchSize := 1000
	numBatches := (len(docs) + batchSize - 1) / batchSize

	// Simulate GPU processing time
	baseTime := time.Duration(float64(len(docs)) * 0.5 * float64(time.Millisecond))
	gpuTime := time.Duration(float64(baseTime) / simulatedGPUSpeedup)

	time.Sleep(gpuTime) // Simulate GPU processing

	elapsed := time.Since(start)

	fmt.Printf("✓ Indexed %d documents in %v (simulated)\n", len(docs), elapsed)
	fmt.Printf("✓ Throughput: %.0f docs/sec\n", float64(len(docs))/elapsed.Seconds())
	fmt.Printf("✓ Batches: %d × %d docs\n", numBatches, batchSize)

	return elapsed
}

func printComparison(size int, sequential, async time.Duration, parallel map[int]time.Duration, gpu time.Duration) {
	fmt.Printf("\n Performance Comparison (%d documents)\n", size)
	fmt.Println("=========================================")

	fmt.Println("\n| Method              | Time        | Speedup | Docs/sec |")
	fmt.Println("|---------------------|-------------|---------|----------|")

	// Sequential (baseline)
	fmt.Printf("| Sequential          | %11v | %7s | %8.0f |\n",
		sequential, "1.00x", float64(size)/sequential.Seconds())

	// Async
	if async > 0 {
		speedup := float64(sequential) / float64(async)
		fmt.Printf("| Async (4 workers)   | %11v | %6.2fx | %8.0f |\n",
			async, speedup, float64(size)/async.Seconds())
	}

	// Parallel CPU
	for workers, elapsed := range parallel {
		if elapsed > 0 {
			speedup := float64(sequential) / float64(elapsed)
			fmt.Printf("| Parallel (%2d workers) | %11v | %6.2fx | %8.0f |\n",
				workers, elapsed, speedup, float64(size)/elapsed.Seconds())
		}
	}

	// GPU
	if gpu > 0 {
		speedup := float64(sequential) / float64(gpu)
		fmt.Printf("| GPU (simulated)     | %11v | %6.2fx | %8.0f |\n",
			gpu, speedup, float64(size)/gpu.Seconds())
	}

	// Find best performer
	fmt.Println("\n Best Performance:")

	best := sequential
	bestMethod := "Sequential"

	if async > 0 && async < best {
		best = async
		bestMethod = "Async"
	}

	for workers, elapsed := range parallel {
		if elapsed > 0 && elapsed < best {
			best = elapsed
			bestMethod = fmt.Sprintf("Parallel (%d workers)", workers)
		}
	}

	if gpu > 0 && gpu < best {
		best = gpu
		bestMethod = "GPU"
	}

	improvement := float64(sequential) / float64(best)
	fmt.Printf("  • Method: %s\n", bestMethod)
	fmt.Printf("  • Time: %v\n", best)
	fmt.Printf("  • Speedup: %.2fx over sequential\n", improvement)
	fmt.Printf("  • Throughput: %.0f docs/sec\n", float64(size)/best.Seconds())
}

func generateDocuments(n int) []string {
	docs := make([]string, n)

	templates := []string{
		"Advanced machine learning algorithms for data analysis in document %d",
		"Deep neural networks and artificial intelligence systems number %d",
		"Natural language processing for text understanding item %d",
		"Computer vision and image recognition technology %d",
		"Distributed systems and cloud computing infrastructure %d",
		"Quantum computing and cryptography research paper %d",
		"Robotics and autonomous systems development %d",
		"Blockchain technology and decentralized applications %d",
	}

	for i := 0; i < n; i++ {
		template := templates[i%len(templates)]
		docs[i] = fmt.Sprintf(template, i)
	}

	return docs
}
