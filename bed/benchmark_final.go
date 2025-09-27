// benchmark_final.go - Final comprehensive benchmark for bed tool
package main

import (
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"
)

type BenchmarkResult struct {
	Query      string
	Time       time.Duration
	Found      bool
	TopScore   float64
	ResultCount int
}

func main() {
	fmt.Println("\n🏁 FINAL BED BENCHMARK SUITE")
	fmt.Println(strings.Repeat("=", 60))

	// Test queries
	queries := []string{
		"Studio Ghibli",
		"anime",
		"Dragon Ball",
		"Naruto",
		"One Piece",
		"machine learning",
		"neural network",
		"transformer model",
		"BERT GPT",
		"CUDA GPU",
		"python programming",
		"semantic search",
		"vector embeddings",
		"cosine similarity",
		"information retrieval",
	}

	// Run benchmarks
	fmt.Println("\n📊 1. SEARCH QUALITY TEST")
	fmt.Println(strings.Repeat("-", 40))
	testSearchQuality(queries)

	fmt.Println("\n⚡ 2. PERFORMANCE TEST")
	fmt.Println(strings.Repeat("-", 40))
	testPerformance(queries)

	fmt.Println("\n🔥 3. STRESS TEST")
	fmt.Println(strings.Repeat("-", 40))
	testStress()

	fmt.Println("\n📈 4. SCALABILITY TEST")
	fmt.Println(strings.Repeat("-", 40))
	testScalability()

	fmt.Println("\n✅ BENCHMARK COMPLETE")
}

func testSearchQuality(queries []string) {
	var successCount int
	var totalTime time.Duration

	for _, query := range queries {
		start := time.Now()
		cmd := exec.Command("./bed_ultra_fixed", "-dir", "testdata", "-k", "5", query)
		output, err := cmd.Output()
		elapsed := time.Since(start)
		totalTime += elapsed

		if err == nil && len(output) > 0 {
			outputStr := string(output)
			// Check if we got meaningful results
			if strings.Contains(outputStr, "✓") || strings.Contains(outputStr, "~") {
				successCount++
				fmt.Printf("  ✅ '%s': Found match in %.2fms\n", query, float64(elapsed.Milliseconds()))
			} else if strings.Contains(outputStr, "Results:") {
				fmt.Printf("  ⚡ '%s': Searched in %.2fms\n", query, float64(elapsed.Milliseconds()))
			} else {
				fmt.Printf("  ❌ '%s': No results\n", query)
			}
		} else {
			fmt.Printf("  ❌ '%s': Error\n", query)
		}
	}

	avgTime := totalTime / time.Duration(len(queries))
	successRate := float64(successCount) / float64(len(queries)) * 100

	fmt.Printf("\n  📊 Success Rate: %.1f%%\n", successRate)
	fmt.Printf("  ⏱️  Avg Time: %.2fms\n", float64(avgTime.Milliseconds()))
}

func testPerformance(queries []string) {
	// Sequential test
	start := time.Now()
	for i := 0; i < 10; i++ {
		query := queries[i%len(queries)]
		cmd := exec.Command("./bed_ultra_fixed", "-dir", "testdata", "-k", "3", query)
		cmd.Output()
	}
	seqTime := time.Since(start)

	fmt.Printf("  Sequential (10 queries): %.2fs\n", seqTime.Seconds())
	fmt.Printf("  Throughput: %.1f queries/sec\n", 10.0/seqTime.Seconds())

	// Parallel test
	var wg sync.WaitGroup
	start = time.Now()

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			query := queries[idx%len(queries)]
			cmd := exec.Command("./bed_ultra_fixed", "-dir", "testdata", "-k", "3", query)
			cmd.Output()
		}(i)
	}

	wg.Wait()
	parTime := time.Since(start)

	fmt.Printf("\n  Parallel (10 queries): %.2fs\n", parTime.Seconds())
	fmt.Printf("  Throughput: %.1f queries/sec\n", 10.0/parTime.Seconds())
	fmt.Printf("  Speedup: %.2fx\n", seqTime.Seconds()/parTime.Seconds())
}

func testStress() {
	numQueries := 100
	queries := []string{"test", "search", "find", "query", "match"}

	start := time.Now()
	var wg sync.WaitGroup
	successCount := 0
	var mu sync.Mutex

	for i := 0; i < numQueries; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			query := queries[idx%len(queries)]
			cmd := exec.Command("./bed_ultra_fixed", "-dir", "testdata", "-k", "1", query)
			_, err := cmd.Output()
			if err == nil {
				mu.Lock()
				successCount++
				mu.Unlock()
			}
		}(i)

		// Limit concurrency
		if i%20 == 19 {
			wg.Wait()
		}
	}

	wg.Wait()
	elapsed := time.Since(start)

	fmt.Printf("  Queries: %d\n", numQueries)
	fmt.Printf("  Success: %d (%.1f%%)\n", successCount, float64(successCount)/float64(numQueries)*100)
	fmt.Printf("  Total Time: %.2fs\n", elapsed.Seconds())
	fmt.Printf("  Throughput: %.1f queries/sec\n", float64(numQueries)/elapsed.Seconds())
}

func testScalability() {
	// Test with different result sizes
	kValues := []int{1, 5, 10, 20, 50}

	fmt.Println("  Testing different K values...")
	for _, k := range kValues {
		start := time.Now()
		cmd := exec.Command("./bed_ultra_fixed", "-dir", "testdata", "-k", fmt.Sprintf("%d", k), "test")
		output, _ := cmd.Output()
		elapsed := time.Since(start)

		// Count actual results returned
		resultCount := strings.Count(string(output), "\n   ")
		fmt.Printf("    K=%2d: %.2fms (%d results)\n", k, float64(elapsed.Milliseconds()), resultCount)
	}

	// Test with pattern filtering
	fmt.Println("\n  Testing with file patterns...")
	patterns := []string{"*.txt", "*.md", "*"}

	for _, pattern := range patterns {
		start := time.Now()
		cmd := exec.Command("./bed_ultra_fixed", "-dir", "testdata", "-pattern", pattern, "-k", "3", "test")
		output, _ := cmd.Output()
		elapsed := time.Since(start)

		// Extract document count if available
		outputStr := string(output)
		docCount := "?"
		if idx := strings.Index(outputStr, "Indexed "); idx >= 0 {
			endIdx := strings.Index(outputStr[idx:], " documents")
			if endIdx > 0 {
				docCount = outputStr[idx+8 : idx+endIdx]
			}
		}

		fmt.Printf("    Pattern '%s': %.2fms (%s docs)\n", pattern, float64(elapsed.Milliseconds()), docCount)
	}
}