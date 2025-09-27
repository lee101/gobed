package main

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

type BenchResult struct {
	Name       string
	Operations int
	Duration   time.Duration
	Throughput float64
	Latency    time.Duration
}

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println(" REAL-WORLD PERFORMANCE BENCHMARK")
	fmt.Println(strings.Repeat("=", 80))

	// System info
	fmt.Printf("\n System: %d cores, %s/%s, Go %s\n",
		runtime.NumCPU(), runtime.GOOS, runtime.GOARCH, runtime.Version())

	// Load model
	fmt.Println("\n🔄 Loading model...")
	startLoad := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		panic(err)
	}
	fmt.Printf(" Model loaded in %v (vocab=%d, dim=%d)\n",
		time.Since(startLoad), model.VocabSize, model.EmbedDim)

	// Test texts - realistic use cases
	shortText := "Hello world"
	mediumText := "Machine learning algorithms are transforming how we process information."
	longText := strings.Repeat("Natural language processing with deep learning models. ", 10)

	fmt.Printf("\n Test texts:\n")
	fmt.Printf("  Short:  %d chars\n", len(shortText))
	fmt.Printf("  Medium: %d chars\n", len(mediumText))
	fmt.Printf("  Long:   %d chars\n", len(longText))

	// Warmup
	fmt.Print("\n Warming up...")
	for i := 0; i < 100; i++ {
		model.Encode(shortText)
	}
	fmt.Println(" done")

	var results []BenchResult

	// Test 1: Short text throughput
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("TEST 1: Short Text Encoding")
	fmt.Println(strings.Repeat("-", 50))

	iterations := 10000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, _ = model.Encode(shortText)
	}
	duration := time.Since(start)

	result := BenchResult{
		Name:       "Short text",
		Operations: iterations,
		Duration:   duration,
		Throughput: float64(iterations) / duration.Seconds(),
		Latency:    duration / time.Duration(iterations),
	}
	results = append(results, result)
	printResult(result)

	// Test 2: Medium text
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("TEST 2: Medium Text Encoding")
	fmt.Println(strings.Repeat("-", 50))

	iterations = 5000
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, _ = model.Encode(mediumText)
	}
	duration = time.Since(start)

	result = BenchResult{
		Name:       "Medium text",
		Operations: iterations,
		Duration:   duration,
		Throughput: float64(iterations) / duration.Seconds(),
		Latency:    duration / time.Duration(iterations),
	}
	results = append(results, result)
	printResult(result)

	// Test 3: Long text
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("TEST 3: Long Text Encoding")
	fmt.Println(strings.Repeat("-", 50))

	iterations = 1000
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, _ = model.Encode(longText)
	}
	duration = time.Since(start)

	result = BenchResult{
		Name:       "Long text",
		Operations: iterations,
		Duration:   duration,
		Throughput: float64(iterations) / duration.Seconds(),
		Latency:    duration / time.Duration(iterations),
	}
	results = append(results, result)
	printResult(result)

	// Test 4: Concurrent encoding
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("TEST 4: Concurrent Encoding (12 workers)")
	fmt.Println(strings.Repeat("-", 50))

	workers := runtime.NumCPU()
	totalOps := 10000
	opsPerWorker := totalOps / workers

	var wg sync.WaitGroup
	start = time.Now()

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < opsPerWorker; i++ {
				model.Encode(mediumText)
			}
		}()
	}
	wg.Wait()
	duration = time.Since(start)

	result = BenchResult{
		Name:       "Concurrent",
		Operations: totalOps,
		Duration:   duration,
		Throughput: float64(totalOps) / duration.Seconds(),
		Latency:    duration / time.Duration(totalOps),
	}
	results = append(results, result)
	printResult(result)

	// Test 5: Similarity computation
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("TEST 5: Similarity Computation")
	fmt.Println(strings.Repeat("-", 50))

	// Generate embeddings
	emb1, _ := model.Encode("Machine learning")
	emb2, _ := model.Encode("Deep learning")

	iterations = 1000000
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_ = gobed.CosineSimilarity(emb1, emb2)
	}
	duration = time.Since(start)

	result = BenchResult{
		Name:       "Similarity",
		Operations: iterations,
		Duration:   duration,
		Throughput: float64(iterations) / duration.Seconds(),
		Latency:    duration / time.Duration(iterations),
	}
	results = append(results, result)
	printResult(result)

	// Test 6: Batch processing simulation
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("TEST 6: Batch Processing (100 texts)")
	fmt.Println(strings.Repeat("-", 50))

	batch := make([]string, 100)
	for i := range batch {
		batch[i] = fmt.Sprintf("Test sentence number %d with some content", i)
	}

	batchRuns := 10
	start = time.Now()
	for run := 0; run < batchRuns; run++ {
		for _, text := range batch {
			model.Encode(text)
		}
	}
	duration = time.Since(start)
	totalBatchOps := batchRuns * len(batch)

	result = BenchResult{
		Name:       "Batch (100)",
		Operations: totalBatchOps,
		Duration:   duration,
		Throughput: float64(totalBatchOps) / duration.Seconds(),
		Latency:    duration / time.Duration(totalBatchOps),
	}
	results = append(results, result)
	printResult(result)

	// Test 7: Real-world simulation (mixed workload)
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("TEST 7: Mixed Workload (5 second stress)")
	fmt.Println(strings.Repeat("-", 50))

	mixedTexts := []string{
		"Short",
		"A medium length sentence here",
		"This is a much longer text that simulates a paragraph of content with multiple words",
		"Query",
		"Document retrieval and semantic search",
	}

	count := 0
	deadline := time.Now().Add(5 * time.Second)
	start = time.Now()

	for time.Now().Before(deadline) {
		text := mixedTexts[count%len(mixedTexts)]
		model.Encode(text)
		count++
	}
	duration = time.Since(start)

	result = BenchResult{
		Name:       "Mixed load",
		Operations: count,
		Duration:   duration,
		Throughput: float64(count) / duration.Seconds(),
		Latency:    duration / time.Duration(count),
	}
	results = append(results, result)
	printResult(result)

	// Summary
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" PERFORMANCE SUMMARY")
	fmt.Println(strings.Repeat("=", 80))

	fmt.Println("\n Throughput (operations/second):")
	for _, r := range results {
		fmt.Printf("  %-15s: %10.0f ops/sec\n", r.Name, r.Throughput)
	}

	fmt.Println("\n Latency (per operation):")
	for _, r := range results {
		if r.Latency < time.Microsecond {
			fmt.Printf("  %-15s: %10.0f ns\n", r.Name, float64(r.Latency.Nanoseconds()))
		} else if r.Latency < time.Millisecond {
			fmt.Printf("  %-15s: %10.1f µs\n", r.Name, float64(r.Latency.Nanoseconds())/1000)
		} else {
			fmt.Printf("  %-15s: %10.2f ms\n", r.Name, float64(r.Latency.Nanoseconds())/1000000)
		}
	}

	// Calculate statistics
	singleThreaded := results[1].Throughput // Medium text
	concurrent := results[3].Throughput
	speedup := concurrent / singleThreaded

	fmt.Printf("\n Key Metrics:\n")
	fmt.Printf("  Single-thread throughput: %.0f ops/sec\n", singleThreaded)
	fmt.Printf("  Multi-thread throughput:  %.0f ops/sec\n", concurrent)
	fmt.Printf("  Parallel speedup:         %.1fx\n", speedup)
	fmt.Printf("  Similarity ops:           %.0f/sec\n", results[4].Throughput)

	// Estimate INT8 improvements
	fmt.Println("\n Projected INT8 Performance:")
	fmt.Printf("  Memory usage:      -75%% (119MB → 30MB)\n")
	fmt.Printf("  Cache efficiency:  ~2x better\n")
	fmt.Printf("  SIMD speedup:      2-4x (with AVX-512)\n")
	fmt.Printf("  Expected throughput: %.0f-%.0f ops/sec\n",
		singleThreaded*2, singleThreaded*4)

	fmt.Println("\n Benchmark completed!")
}

func printResult(r BenchResult) {
	fmt.Printf("  Operations: %d\n", r.Operations)
	fmt.Printf("  Duration:   %v\n", r.Duration)
	fmt.Printf("  Throughput: %.0f ops/sec\n", r.Throughput)

	if r.Latency < time.Microsecond {
		fmt.Printf("  Latency:    %.0f ns/op\n", float64(r.Latency.Nanoseconds()))
	} else if r.Latency < time.Millisecond {
		fmt.Printf("  Latency:    %.1f µs/op\n", float64(r.Latency.Nanoseconds())/1000)
	} else {
		fmt.Printf("  Latency:    %.2f ms/op\n", float64(r.Latency.Nanoseconds())/1000000)
	}
}
