//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/search"
)

type TestConfig struct {
	Name              string
	GraphDegree       int
	IntermediateDegree int
	ItopkSize         int
	SearchWidth       int
	MinIterations     int
	NProbe            int
	RerankSize        int
}

type BenchResult struct {
	Config           TestConfig
	BuildTimeMs      float64
	SearchTimeMs     float64
	ThroughputQPS    float64
	Recall           float64
	ExactMatchRate   float64
}

func generateTestData(n, dim int) [][]float32 {
	data := make([][]float32, n)
	for i := 0; i < n; i++ {
		data[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			data[i][j] = rand.Float32()*2.0 - 1.0 // [-1, 1]
		}
	}
	return data
}

func generateQueries(nQueries, dim int) [][]float32 {
	return generateTestData(nQueries, dim)
}

func bruteForceSearch(dataset [][]float32, query []float32, k int) []int {
	type distIdx struct {
		dist float32
		idx  int
	}

	distances := make([]distIdx, len(dataset))
	for i, vec := range dataset {
		var sum float32
		for j := range query {
			diff := query[j] - vec[j]
			sum += diff * diff
		}
		distances[i] = distIdx{dist: sum, idx: i}
	}

	// Sort by distance (simple bubble sort for top-k)
	for i := 0; i < k && i < len(distances); i++ {
		minIdx := i
		for j := i + 1; j < len(distances); j++ {
			if distances[j].dist < distances[minIdx].dist {
				minIdx = j
			}
		}
		distances[i], distances[minIdx] = distances[minIdx], distances[i]
	}

	results := make([]int, k)
	for i := 0; i < k && i < len(distances); i++ {
		results[i] = distances[i].idx
	}
	return results
}

func calculateRecall(results []search.SearchResult, groundTruth []int) float64 {
	truthMap := make(map[int]bool)
	for _, id := range groundTruth {
		truthMap[id] = true
	}

	hits := 0
	for _, res := range results {
		if truthMap[res.ID] {
			hits++
		}
	}

	return float64(hits) / float64(len(groundTruth))
}

func benchmarkConfig(config TestConfig, dataset [][]float32, queries [][]float32, k int) BenchResult {
	result := BenchResult{Config: config}

	// Create embedding model with dataset
	model := &gobed.EmbeddingModel{
		Embeddings: dataset,
	}

	// Create search config
	searchConfig := gobed.SearchConfig{
		AutoMode:           false,
		Preset:             gobed.CustomPreset,
		MaxExactSearchSize: 100, // Force approximate search
		NumClusters:        len(dataset) / 50,
		SearchClusters:     config.NProbe,
		EnableGPU:          true,
		GPUBatchSize:       1000,
	}

	// Override with test parameters
	internalConfig := search.Config{
		MaxFlatSize: 100,
		NList:       searchConfig.NumClusters,
		NProbe:      config.NProbe,
		M:           config.GraphDegree / 2,
		NBits:       8,
		HNSWEnabled: false,
		RerankSize:  config.RerankSize,
		UseParallel: true,
	}

	// Create search engine
	engine := gobed.NewSearchEngineWithConfig(model, searchConfig)
	// Override internal config
	engine.SetInternalConfig(internalConfig)

	// Build index
	fmt.Printf("\nTesting: %s\n", config.Name)
	fmt.Printf("  GraphDegree: %d, Itopk: %d, Width: %d, Iterations: %d, NProbe: %d\n",
		config.GraphDegree, config.ItopkSize, config.SearchWidth, config.MinIterations, config.NProbe)

	startBuild := time.Now()
	err := engine.BuildIndex()
	if err != nil {
		fmt.Printf("  ❌ Build failed: %v\n", err)
		return result
	}
	result.BuildTimeMs = float64(time.Since(startBuild).Microseconds()) / 1000.0
	fmt.Printf("  ✅ Build time: %.2f ms\n", result.BuildTimeMs)

	// Benchmark search
	totalRecall := 0.0
	exactMatches := 0

	// Warm-up
	engine.Search(queries[0], k)

	startSearch := time.Now()
	for i, query := range queries {
		results, err := engine.Search(query, k)
		if err != nil {
			fmt.Printf("  ❌ Search failed: %v\n", err)
			continue
		}

		// Calculate ground truth for this query
		groundTruth := bruteForceSearch(dataset, query, k)

		// Calculate recall
		recall := calculateRecall(results, groundTruth)
		totalRecall += recall

		// Check if top result is exact match
		if len(results) > 0 && results[0].ID == groundTruth[0] {
			exactMatches++
		}
	}

	searchTime := time.Since(startSearch)
	result.SearchTimeMs = float64(searchTime.Microseconds()) / 1000.0 / float64(len(queries))
	result.ThroughputQPS = float64(len(queries)) * 1000.0 / (float64(searchTime.Microseconds()) / 1000.0)
	result.Recall = totalRecall / float64(len(queries))
	result.ExactMatchRate = float64(exactMatches) / float64(len(queries))

	fmt.Printf("  ✅ Search: %.3f ms/query, %.0f QPS\n", result.SearchTimeMs, result.ThroughputQPS)
	fmt.Printf("  ✅ Recall@%d: %.1f%%, Exact matches: %.1f%%\n",
		k, result.Recall*100, result.ExactMatchRate*100)

	return result
}

func main() {
	fmt.Println("🚀 Go Custom CAGRA Optimization Benchmark")
	fmt.Println("=========================================\n")

	// Test parameters
	nVectors := 5000
	nQueries := 100
	dim := 512
	k := 20

	fmt.Printf("Dataset: %d vectors, %d dimensions\n", nVectors, dim)
	fmt.Printf("Queries: %d, k=%d\n\n", nQueries, k)

	// Generate test data
	fmt.Println("Generating test data...")
	rand.Seed(42)
	dataset := generateTestData(nVectors, dim)
	queries := generateQueries(nQueries, dim)

	// Test configurations - from fast to quality
	configs := []TestConfig{
		// Ultra-fast configurations
		{"Ultra-Fast v1", 16, 32, 32, 1, 0, 2, 32},
		{"Ultra-Fast v2", 16, 32, 64, 1, 0, 4, 50},

		// Fast configurations
		{"Fast v1", 32, 64, 64, 1, 2, 8, 64},
		{"Fast v2", 32, 64, 96, 1, 2, 12, 96},

		// Balanced configurations
		{"Balanced v1", 48, 96, 128, 2, 4, 16, 128},
		{"Balanced v2", 64, 128, 160, 2, 4, 20, 160},

		// Current "optimal" from our tests
		{"Current Optimal", 64, 128, 192, 2, 6, 24, 192},

		// Quality-focused
		{"Quality v1", 80, 160, 256, 3, 8, 32, 256},
		{"Quality v2", 96, 192, 320, 4, 10, 40, 320},

		// Ultra-quality
		{"Ultra-Quality", 128, 256, 512, 4, 16, 64, 512},
	}

	// Run benchmarks
	fmt.Println("🔬 Running Benchmarks")
	fmt.Println("=====================")

	results := make([]BenchResult, 0, len(configs))
	for _, config := range configs {
		result := benchmarkConfig(config, dataset, queries, k)
		results = append(results, result)
		time.Sleep(100 * time.Millisecond) // Brief pause between tests
	}

	// Print summary
	fmt.Println("\n\n📊 BENCHMARK SUMMARY")
	fmt.Println("====================\n")

	fmt.Println("Configuration    | Build(ms) | Search(ms) | QPS     | Recall | Exact%")
	fmt.Println("-----------------|-----------|------------|---------|--------|-------")

	for _, r := range results {
		fmt.Printf("%-16s | %9.2f | %10.3f | %7.0f | %5.1f%% | %5.1f%%\n",
			r.Config.Name,
			r.BuildTimeMs,
			r.SearchTimeMs,
			r.ThroughputQPS,
			r.Recall*100,
			r.ExactMatchRate*100)
	}

	// Find best configurations
	fmt.Println("\n\n🏆 BEST CONFIGURATIONS")
	fmt.Println("======================\n")

	// Best for speed with >90% recall
	var bestSpeed *BenchResult
	for i := range results {
		if results[i].Recall >= 0.9 {
			if bestSpeed == nil || results[i].ThroughputQPS > bestSpeed.ThroughputQPS {
				bestSpeed = &results[i]
			}
		}
	}

	if bestSpeed != nil {
		fmt.Printf("⚡ Best Speed (>90%% recall): %s\n", bestSpeed.Config.Name)
		fmt.Printf("   %.0f QPS, %.3f ms latency, %.1f%% recall\n",
			bestSpeed.ThroughputQPS, bestSpeed.SearchTimeMs, bestSpeed.Recall*100)
		fmt.Printf("   Settings: GraphDegree=%d, Itopk=%d, NProbe=%d\n",
			bestSpeed.Config.GraphDegree, bestSpeed.Config.ItopkSize, bestSpeed.Config.NProbe)
	}

	// Best recall
	var bestRecall *BenchResult
	for i := range results {
		if bestRecall == nil || results[i].Recall > bestRecall.Recall {
			bestRecall = &results[i]
		}
	}

	if bestRecall != nil {
		fmt.Printf("\n🎯 Best Recall: %s\n", bestRecall.Config.Name)
		fmt.Printf("   %.1f%% recall, %.1f%% exact matches\n",
			bestRecall.Recall*100, bestRecall.ExactMatchRate*100)
		fmt.Printf("   Settings: GraphDegree=%d, Itopk=%d, NProbe=%d\n",
			bestRecall.Config.GraphDegree, bestRecall.Config.ItopkSize, bestRecall.Config.NProbe)
	}

	// Best balanced (speed * recall score)
	var bestBalanced *BenchResult
	bestScore := 0.0
	for i := range results {
		score := results[i].Recall * (results[i].ThroughputQPS / 1000.0)
		if score > bestScore {
			bestScore = score
			bestBalanced = &results[i]
		}
	}

	if bestBalanced != nil {
		fmt.Printf("\n⚖️  Best Balanced: %s\n", bestBalanced.Config.Name)
		fmt.Printf("   %.0f QPS, %.1f%% recall, Score: %.1f\n",
			bestBalanced.ThroughputQPS, bestBalanced.Recall*100, bestScore)
		fmt.Printf("   Settings: GraphDegree=%d, Itopk=%d, NProbe=%d\n",
			bestBalanced.Config.GraphDegree, bestBalanced.Config.ItopkSize, bestBalanced.Config.NProbe)
	}

	fmt.Println("\n✅ Optimization complete!")
}
