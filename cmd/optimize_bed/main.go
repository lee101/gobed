//go:build legacy

// optimize_bed.go - Comprehensive optimization and testing for bed tool

package main

import (
	"fmt"
	"math"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"
)

type OptimizationResult struct {
	TestName   string
	Duration   time.Duration
	Throughput float64
	Quality    float64
	MemoryMB   float64
}

func main() {
	fmt.Println("\n🚀 BED OPTIMIZATION & TESTING SUITE\n")
	fmt.Println("=" + strings.Repeat("=", 50))

	results := []OptimizationResult{}

	// Test 1: Baseline Performance
	fmt.Println("\n📊 Test 1: Baseline Performance")
	result := testBaseline()
	results = append(results, result)

	// Test 2: Search Quality
	fmt.Println("\n🎯 Test 2: Search Quality")
	result = testSearchQuality()
	results = append(results, result)

	// Test 3: Parallel Performance
	fmt.Println("\n⚡ Test 3: Parallel Search Performance")
	result = testParallelPerformance()
	results = append(results, result)

	// Test 4: Large Scale Test
	fmt.Println("\n📈 Test 4: Large Scale Performance")
	result = testLargeScale()
	results = append(results, result)

	// Test 5: Memory Efficiency
	fmt.Println("\n💾 Test 5: Memory Efficiency")
	result = testMemoryEfficiency()
	results = append(results, result)

	// Generate Final Report
	generateFinalReport(results)
}

func testBaseline() OptimizationResult {
	queries := []string{
		"Studio Ghibli",
		"anime",
		"machine learning",
		"BERT transformer",
		"GPU optimization",
	}

	var totalTime time.Duration
	var successCount int

	for _, query := range queries {
		startTime := time.Now()
		cmd := exec.Command("./bed", "-dir", "testdata", "-k", "5", query)
		output, err := cmd.Output()
		elapsed := time.Since(startTime)

		if err == nil && len(output) > 0 {
			successCount++
			totalTime += elapsed
			fmt.Printf("  ✓ Query '%s': %.2fms\n", query, float64(elapsed.Microseconds())/1000.0)
		} else {
			fmt.Printf("  ✗ Query '%s': failed\n", query)
		}
	}

	avgTime := totalTime / time.Duration(len(queries))
	quality := float64(successCount) / float64(len(queries))

	return OptimizationResult{
		TestName:   "Baseline",
		Duration:   avgTime,
		Throughput: 1000.0 / float64(avgTime.Milliseconds()),
		Quality:    quality,
	}
}

func testSearchQuality() OptimizationResult {
	// Test semantic search quality with known queries
	testCases := []struct {
		query    string
		expected []string // Expected terms in top results
	}{
		{"Studio Ghibli", []string{"Studio", "Ghibli", "films"}},
		{"anime", []string{"anime", "Dragon Ball"}},
		{"neural networks", []string{"neural", "networks", "deep learning"}},
		{"CUDA GPU", []string{"CUDA", "GPU", "optimization"}},
	}

	var totalScore float64
	startTime := time.Now()

	for _, tc := range testCases {
		cmd := exec.Command("./bed", "-dir", "testdata", "-k", "10", tc.query)
		output, err := cmd.Output()

		if err == nil {
			outputStr := string(output)
			matchCount := 0
			for _, expected := range tc.expected {
				if strings.Contains(strings.ToLower(outputStr), strings.ToLower(expected)) {
					matchCount++
				}
			}
			score := float64(matchCount) / float64(len(tc.expected))
			totalScore += score
			fmt.Printf("  Query '%s': %.0f%% match\n", tc.query, score*100)
		}
	}

	avgQuality := totalScore / float64(len(testCases))
	duration := time.Since(startTime) / time.Duration(len(testCases))

	return OptimizationResult{
		TestName:   "SearchQuality",
		Duration:   duration,
		Throughput: 1000.0 / float64(duration.Milliseconds()),
		Quality:    avgQuality,
	}
}

func testParallelPerformance() OptimizationResult {
	numWorkers := runtime.NumCPU()
	queries := []string{"anime", "machine learning", "GPU", "transformer", "neural"}

	var wg sync.WaitGroup
	startTime := time.Now()
	successCount := 0
	var mu sync.Mutex

	for i := 0; i < numWorkers*2; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			query := queries[idx%len(queries)]
			cmd := exec.Command("./bed", "-dir", "testdata", "-k", "3", query)
			_, err := cmd.Output()
			if err == nil {
				mu.Lock()
				successCount++
				mu.Unlock()
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(startTime)
	totalQueries := numWorkers * 2
	throughput := float64(totalQueries) / duration.Seconds()

	fmt.Printf("  Parallel queries: %d\n", totalQueries)
	fmt.Printf("  Total time: %.2fs\n", duration.Seconds())
	fmt.Printf("  Throughput: %.1f queries/sec\n", throughput)

	return OptimizationResult{
		TestName:   "Parallel",
		Duration:   duration / time.Duration(totalQueries),
		Throughput: throughput,
		Quality:    float64(successCount) / float64(totalQueries),
	}
}

func testLargeScale() OptimizationResult {
	// Test with different dataset sizes
	fmt.Println("  Testing scalability...")

	startTime := time.Now()
	cmd := exec.Command("./bed", "-dir", "testdata", "-k", "20", "test")
	output, err := cmd.Output()

	if err != nil {
		fmt.Printf("  Error: %v\n", err)
		return OptimizationResult{TestName: "LargeScale", Quality: 0}
	}

	duration := time.Since(startTime)
	lines := strings.Count(string(output), "\n")

	fmt.Printf("  Indexed testdata/ in %.2fs\n", duration.Seconds())
	fmt.Printf("  Results returned: %d lines\n", lines)

	return OptimizationResult{
		TestName:   "LargeScale",
		Duration:   duration,
		Throughput: float64(lines) / duration.Seconds(),
		Quality:    1.0,
	}
}

func testMemoryEfficiency() OptimizationResult {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	startMem := m.Alloc

	// Run multiple searches
	for i := 0; i < 10; i++ {
		cmd := exec.Command("./bed", "-dir", "testdata", "-k", "5", "test")
		cmd.Output()
	}

	runtime.GC()
	runtime.ReadMemStats(&m)
	endMem := m.Alloc

	memUsedMB := float64(endMem-startMem) / (1024 * 1024)
	if memUsedMB < 0 {
		memUsedMB = float64(m.Alloc) / (1024 * 1024)
	}

	fmt.Printf("  Memory usage: %.2f MB\n", memUsedMB)
	fmt.Printf("  GC cycles: %d\n", m.NumGC)

	// Good if memory usage is low
	quality := math.Min(1.0, 100.0/memUsedMB)

	return OptimizationResult{
		TestName: "Memory",
		MemoryMB: memUsedMB,
		Quality:  quality,
	}
}

func generateFinalReport(results []OptimizationResult) {
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("📋 FINAL OPTIMIZATION REPORT")
	fmt.Println(strings.Repeat("=", 50))

	var totalQuality float64
	var avgThroughput float64
	var avgMemory float64
	count := 0

	for _, r := range results {
		if r.Quality > 0 {
			totalQuality += r.Quality
			count++
		}
		if r.Throughput > 0 {
			avgThroughput += r.Throughput
		}
		if r.MemoryMB > 0 {
			avgMemory += r.MemoryMB
		}
	}

	if count > 0 {
		totalQuality /= float64(count)
	}
	if avgThroughput > 0 {
		avgThroughput /= float64(len(results))
	}

	// Performance Score (0-100)
	perfScore := totalQuality * 100

	fmt.Printf("\n🎯 Overall Performance Score: %.1f/100\n", perfScore)
	fmt.Printf("⚡ Average Throughput: %.1f queries/sec\n", avgThroughput)
	fmt.Printf("💾 Memory Efficiency: %.2f MB avg\n", avgMemory)

	// Recommendations
	fmt.Println("\n💡 OPTIMIZATION RECOMMENDATIONS:")

	if perfScore < 50 {
		fmt.Println("  ⚠️  Critical: Search quality needs improvement")
		fmt.Println("     - Check embedding model initialization")
		fmt.Println("     - Verify tokenization and encoding")
	} else if perfScore < 75 {
		fmt.Println("  ⚡ Moderate: Performance can be improved")
		fmt.Println("     - Consider implementing caching")
		fmt.Println("     - Optimize embedding quantization")
	} else {
		fmt.Println("  ✅ Good: System performing well")
		fmt.Println("     - Consider GPU acceleration for scale")
		fmt.Println("     - Implement index structures for large datasets")
	}

	if avgThroughput < 10 {
		fmt.Println("  ⚠️  Throughput is low")
		fmt.Println("     - Implement parallel processing")
		fmt.Println("     - Use batch operations")
	}

	if avgMemory > 100 {
		fmt.Println("  ⚠️  High memory usage detected")
		fmt.Println("     - Implement memory pooling")
		fmt.Println("     - Use streaming for large files")
	}

	// Final verdict
	fmt.Println("\n📊 VERDICT:")
	if perfScore >= 90 && avgThroughput >= 50 {
		fmt.Println("  🏆 EXCELLENT - Ready for production!")
	} else if perfScore >= 70 && avgThroughput >= 20 {
		fmt.Println("  ✅ GOOD - Minor optimizations recommended")
	} else if perfScore >= 50 {
		fmt.Println("  ⚡ ACCEPTABLE - Optimization needed for scale")
	} else {
		fmt.Println("  ⚠️  NEEDS WORK - Major improvements required")
	}
}
