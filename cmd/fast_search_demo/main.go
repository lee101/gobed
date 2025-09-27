// +build gpu

package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Printf("Ultra-Fast GPU Search Demo\n")
	fmt.Printf("Target: 10-20s indexing, <1ms search\n")
	fmt.Printf("================================\n\n")

	// Configuration for your 243K AI images use case
	config := gobed.DefaultFastSearchConfig()
	config.IndexPath = "search_index_243k.bin"
	config.MaxLatencyUs = 1000  // 1ms target
	config.PrecomputeStructures = true
	config.EnableCache = true
	
	// Create indexer
	indexer, err := gobed.NewGPUFastSearchIndexer(config)
	if err != nil {
		log.Fatalf("Failed to create fast search indexer: %v", err)
	}
	defer indexer.Close()

	// Check if pre-built index exists
	indexPath := config.IndexPath
	if indexer.CheckIndexExists(indexPath) {
		fmt.Printf("Found existing index, loading...\n")
		runWithExistingIndex(indexer, indexPath)
	} else {
		fmt.Printf("Building new index for 243K vectors...\n")
		runFullIndexBuild(indexer, indexPath)
	}
}

func runWithExistingIndex(indexer *gobed.GPUFastSearchIndexer, indexPath string) {
	// FAST STARTUP - Load pre-built index
	start := time.Now()
	if err := indexer.LoadIndex(indexPath); err != nil {
		log.Fatalf(" Failed to load index: %v", err)
	}
	loadTime := time.Since(start)
	
	fmt.Printf(" Index loaded in %v (ultra-fast startup!)\n\n", loadTime)
	
	// Optimize for inference
	indexer.OptimizeForInference()
	
	// Warmup with common queries
	fmt.Printf(" Warming up cache with common queries...\n")
	warmupQueries := generateCommonQueries(1000, 512)
	indexer.WarmupCache(warmupQueries)
	
	// Run search benchmarks
	runSearchBenchmarks(indexer)
}

func runFullIndexBuild(indexer *gobed.GPUFastSearchIndexer, indexPath string) {
	// Simulate your 243K AI images dataset
	fmt.Printf(" Generating dataset (243K vectors, 512 dim)...\n")
	numVectors := 243000
	vectorDim := 512
	
	vectors, scales, ids := generateRealisticDataset(numVectors, vectorDim)
	
	// BUILD INDEX - Target 10-20 seconds
	fmt.Printf(" Building search index (target: 10-20s)...\n")
	buildStart := time.Now()
	
	if err := indexer.BuildIndexFast(vectors, scales, ids); err != nil {
		log.Fatalf(" Failed to build index: %v", err)
	}
	
	buildTime := time.Since(buildStart)
	fmt.Printf(" Index built in %v", buildTime)
	
	if buildTime.Seconds() <= 20 {
		fmt.Printf(" ( Target met!)\n")
	} else {
		fmt.Printf(" (  Exceeded 20s target)\n")
	}
	
	// SAVE INDEX for fast future startups
	fmt.Printf(" Saving index for future fast loading...\n")
	saveStart := time.Now()
	if err := indexer.SaveIndex(indexPath); err != nil {
		log.Printf("  Failed to save index: %v", err)
	} else {
		saveTime := time.Since(saveStart)
		fmt.Printf(" Index saved in %v\n", saveTime)
	}
	
	// Optimize for search performance
	indexer.OptimizeForInference()
	
	// Warmup cache
	fmt.Printf(" Warming up search cache...\n")
	warmupQueries := generateCommonQueries(1000, 512)
	indexer.WarmupCache(warmupQueries)
	
	fmt.Printf("\n")
	
	// Run search benchmarks
	runSearchBenchmarks(indexer)
}

func runSearchBenchmarks(indexer *gobed.GPUFastSearchIndexer) {
	fmt.Printf(" Search Performance Testing\n")
	fmt.Printf("Target: <1ms per search\n")
	fmt.Printf("----------------------------\n")
	
	// Test single searches
	testSingleSearch(indexer)
	
	// Test batch searches
	testBatchSearch(indexer)
	
	// Show final stats
	showFinalStats(indexer)
}

func testSingleSearch(indexer *gobed.GPUFastSearchIndexer) {
	fmt.Printf(" Single Search Test (1000 queries):\n")
	
	numTests := 1000
	k := 10 // Top-10 results
	totalTime := time.Duration(0)
	successCount := 0
	
	for i := 0; i < numTests; i++ {
		query := generateSingleQuery(512)
		scale := float32(1.0)
		
		start := time.Now()
		results, err := indexer.SearchSingle(query, scale, k)
		latency := time.Since(start)
		
		if err != nil {
			fmt.Printf("    Query %d failed: %v\n", i, err)
			continue
		}
		
		totalTime += latency
		successCount++
		
		// Check if under 1ms target
		if latency.Microseconds() > 1000 {
			fmt.Printf("     Query %d: %dμs (over 1ms target)\n", i, latency.Microseconds())
		}
		
		// Show progress every 100 queries
		if (i+1)%100 == 0 {
			avgUs := totalTime.Microseconds() / int64(successCount)
			fmt.Printf("   Progress: %d/%d, avg: %dμs\n", i+1, numTests, avgUs)
		}
		
		_ = results // Use results to avoid unused variable warning
	}
	
	// Final single search stats
	avgLatencyUs := totalTime.Microseconds() / int64(successCount)
	qps := float64(successCount) / totalTime.Seconds()
	
	fmt.Printf("    Results: %d successful searches\n", successCount)
	fmt.Printf("     Average latency: %dμs", avgLatencyUs)
	
	if avgLatencyUs <= 1000 {
		fmt.Printf(" ( Target met!)\n")
	} else {
		fmt.Printf(" ( Exceeded 1ms target)\n")
	}
	
	fmt.Printf("    Throughput: %.0f QPS\n\n", qps)
}

func testBatchSearch(indexer *gobed.GPUFastSearchIndexer) {
	fmt.Printf(" Batch Search Test:\n")
	
	batchSizes := []int{10, 50, 100}
	k := 10
	
	for _, batchSize := range batchSizes {
		queries := make([][]int8, batchSize)
		scales := make([]float32, batchSize)
		
		for i := 0; i < batchSize; i++ {
			queries[i] = generateSingleQuery(512)
			scales[i] = 1.0
		}
		
		start := time.Now()
		results, err := indexer.SearchBatch(queries, scales, k)
		batchTime := time.Since(start)
		
		if err != nil {
			fmt.Printf("    Batch size %d failed: %v\n", batchSize, err)
			continue
		}
		
		avgLatencyUs := batchTime.Microseconds() / int64(batchSize)
		
		fmt.Printf("   Batch %d: %dμs total, %dμs avg", 
			batchSize, batchTime.Microseconds(), avgLatencyUs)
		
		if avgLatencyUs <= 1000 {
			fmt.Printf(" ( Target met!)\n")
		} else {
			fmt.Printf(" ( Exceeded 1ms target)\n")
		}
		
		_ = results // Use results
	}
	
	fmt.Printf("\n")
}

func showFinalStats(indexer *gobed.GPUFastSearchIndexer) {
	fmt.Printf(" Final Performance Statistics\n")
	fmt.Printf("===============================\n")
	
	stats := indexer.GetSearchStats()
	
	fmt.Printf("Total searches: %d\n", stats.TotalSearches)
	fmt.Printf("Average latency: %.0fμs\n", stats.AvgLatencyUs)
	fmt.Printf("GPU latency: %.0fμs\n", stats.GPULatencyUs)
	fmt.Printf("Cache hit rate: %.1f%%\n", stats.CacheHitRate*100)
	fmt.Printf("Target latency: %dμs\n", stats.TargetLatencyUs)
	
	if stats.TargetMet {
		fmt.Printf(" Performance target MET! Average %.0fμs < %dμs\n", 
			stats.AvgLatencyUs, stats.TargetLatencyUs)
	} else {
		fmt.Printf("  Performance target MISSED. Average %.0fμs > %dμs\n",
			stats.AvgLatencyUs, stats.TargetLatencyUs)
		
		// Suggestions for improvement
		fmt.Printf("\n Optimization Suggestions:\n")
		fmt.Printf("   - Reduce NProbe (current search accuracy for speed)\n")
		fmt.Printf("   - Enable more aggressive caching\n")
		fmt.Printf("   - Pre-warm more common queries\n")
		fmt.Printf("   - Consider fewer clusters (NList) for faster search\n")
	}
	
	// Show cache effectiveness
	if stats.CacheHitRate > 0.3 {
		fmt.Printf(" Cache working well (%.1f%% hit rate)\n", stats.CacheHitRate*100)
	} else {
		fmt.Printf("  Low cache hit rate (%.1f%%) - consider warming more queries\n", 
			stats.CacheHitRate*100)
	}
	
	fmt.Printf("\n Ready for production inference with ultra-fast search!\n")
}

// Helper functions for realistic data generation

func generateRealisticDataset(numVectors, vectorDim int) ([]int8, []float32, []int) {
	fmt.Printf("   Generating %d vectors of dimension %d...\n", numVectors, vectorDim)
	
	vectors := make([]int8, numVectors*vectorDim)
	scales := make([]float32, numVectors)
	ids := make([]int, numVectors)
	
	// Generate realistic AI image embeddings
	for i := 0; i < numVectors; i++ {
		// Realistic scale distribution for AI embeddings
		scales[i] = 0.5 + rand.Float32()*1.5 // 0.5 to 2.0 range
		ids[i] = i
		
		// Generate embedding that resembles AI image features
		for j := 0; j < vectorDim; j++ {
			// Create somewhat realistic distribution
			// (actual embeddings would come from your model)
			val := rand.NormFloat64() * 64
			if val > 127 {
				val = 127
			}
			if val < -128 {
				val = -128
			}
			vectors[i*vectorDim+j] = int8(val)
		}
		
		// Progress indicator for large datasets
		if (i+1)%50000 == 0 {
			fmt.Printf("   Generated %d/%d vectors\n", i+1, numVectors)
		}
	}
	
	return vectors, scales, ids
}

func generateCommonQueries(numQueries, vectorDim int) [][]int8 {
	queries := make([][]int8, numQueries)
	
	for i := 0; i < numQueries; i++ {
		query := make([]int8, vectorDim)
		for j := 0; j < vectorDim; j++ {
			// Generate common query patterns
			val := rand.NormFloat64() * 64
			if val > 127 {
				val = 127
			}
			if val < -128 {
				val = -128
			}
			query[j] = int8(val)
		}
		queries[i] = query
	}
	
	return queries
}

func generateSingleQuery(vectorDim int) []int8 {
	query := make([]int8, vectorDim)
	for j := 0; j < vectorDim; j++ {
		val := rand.NormFloat64() * 64
		if val > 127 {
			val = 127
		}
		if val < -128 {
			val = -128
		}
		query[j] = int8(val)
	}
	return query
}

// Production Usage Example
func showProductionUsage() {
	fmt.Printf(`
🏭 Production Usage Pattern:

// 1. First time - build and save index (10-20s one-time cost)
config := gobed.DefaultFastSearchConfig()
config.IndexPath = "/path/to/your/search_index.bin"
indexer, _ := gobed.NewGPUFastSearchIndexer(config)

vectors, scales, ids := loadYour243KImages()
indexer.BuildIndexFast(vectors, scales, ids)  // 10-20s
indexer.SaveIndex(config.IndexPath)           // Save for future

// 2. All subsequent startups - ultra-fast loading
indexer.LoadIndex(config.IndexPath)           // <1s startup
indexer.OptimizeForInference()
indexer.WarmupCache(commonQueries)

// 3. Production searches - <1ms each
for query := range incomingQueries {
    results, _ := indexer.SearchSingle(query, 1.0, 10)  // <1ms
    sendResults(results)
}
`)
}