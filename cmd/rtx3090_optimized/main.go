package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 RTX 3090 OPTIMIZED GPU INDEXING BENCHMARK")
	fmt.Println("================================================================================")
	fmt.Printf("GPU: NVIDIA RTX 3090 (24GB VRAM)\n")
	fmt.Printf("CPU: %d cores\n", runtime.NumCPU())
	fmt.Printf("CUDA: Enabled with GPU build tag\n\n")

	// Load model
	fmt.Println("Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✅ Model loaded successfully\n")

	// Test configurations - start conservative and scale up
	configs := []struct {
		name       string
		docCount   int
		batchSize  int
		searchSize int
	}{
		{"Warmup", 100, 100, 100},
		{"Small_1K", 1000, 250, 1000},
		{"Medium_5K", 5000, 500, 5000},
		{"Large_10K", 10000, 1000, 10000},
		{"XLarge_25K", 25000, 2500, 20000},
		{"XXLarge_50K", 50000, 5000, 30000},
		{"Massive_100K", 100000, 10000, 40000},
		{"Ultra_250K", 250000, 25000, 50000},
		{"Extreme_500K", 500000, 50000, 75000},
		{"Max_1M", 1000000, 100000, 100000},
	}

	bestThroughput := 0.0
	bestConfig := ""
	bestQPS := 0.0

	for _, cfg := range configs {
		fmt.Printf("\n============================================================\n")
		fmt.Printf("📊 Testing: %s\n", cfg.name)
		fmt.Printf("   Documents: %d | Batch Size: %d\n", cfg.docCount, cfg.batchSize)
		fmt.Println("------------------------------------------------------------")

		// Create fresh engine for each test
		searchConfig := gobed.SearchConfig{
			AutoMode:           true,
			MaxExactSearchSize: cfg.searchSize,
			EnableGPU:          true,
		}
		engine := gobed.NewSearchEngineWithConfig(model, searchConfig)

		// Generate corpus
		corpus := generateCorpus(cfg.docCount)

		// Measure indexing time
		indexStart := time.Now()
		totalIndexed := 0
		
		// Index in batches
		for i := 0; i < cfg.docCount; i += cfg.batchSize {
			end := min(i+cfg.batchSize, cfg.docCount)
			batch := corpus[i:end]
			
			ids, err := engine.IndexBatch(batch)
			if err != nil {
				fmt.Printf("   ⚠️  Batch error at %d-%d: %v\n", i, end, err)
				continue
			}
			totalIndexed += len(ids)
		}
		
		indexElapsed := time.Since(indexStart)
		indexThroughput := float64(totalIndexed) / indexElapsed.Seconds()

		// Get stats
		stats := engine.Stats()

		// Display indexing results
		fmt.Printf("\n✅ Indexing Results:\n")
		fmt.Printf("   Indexed: %d/%d documents\n", totalIndexed, cfg.docCount)
		fmt.Printf("   Time: %v\n", indexElapsed)
		fmt.Printf("   Throughput: %.0f docs/sec\n", indexThroughput)
		fmt.Printf("   Memory: %.2f MB\n", stats.MemoryUsageMB)
		fmt.Printf("   Index Type: %s\n", stats.IndexType)

		if indexThroughput > bestThroughput {
			bestThroughput = indexThroughput
			bestConfig = cfg.name
		}

		// Only run search benchmark if we indexed successfully
		if totalIndexed > 0 {
			// Train/optimize index if needed
			if stats.IndexType != "flat" {
				fmt.Printf("\n🔧 Optimizing index...\n")
				optStart := time.Now()
				engine.Optimize()
				fmt.Printf("   Optimization took: %v\n", time.Since(optStart))
			}

			// Search benchmark
			queries := []string{
				"machine learning algorithms",
				"deep neural networks",
				"transformer models",
				"gpu acceleration",
				"vector search",
				"distributed systems",
				"cloud computing",
				"data pipelines",
			}

			fmt.Printf("\n🔍 Search Performance:\n")
			searchStart := time.Now()
			searchCount := min(1000, totalIndexed/10) // Adaptive search count
			searchErrors := 0
			
			for i := 0; i < searchCount; i++ {
				query := queries[i%len(queries)]
				results, err := engine.Search(query, 10)
				if err != nil {
					searchErrors++
				} else if len(results) == 0 {
					// Silent - expected for random corpus
				}
			}
			
			searchElapsed := time.Since(searchStart)
			qps := float64(searchCount) / searchElapsed.Seconds()
			avgLatency := searchElapsed / time.Duration(searchCount)

			fmt.Printf("   Queries: %d\n", searchCount)
			fmt.Printf("   QPS: %.0f\n", qps)
			fmt.Printf("   Avg Latency: %v\n", avgLatency)
			if searchErrors > 0 {
				fmt.Printf("   Errors: %d\n", searchErrors)
			}

			if qps > bestQPS {
				bestQPS = qps
			}
		}

		// Memory check
		fmt.Printf("\n💾 System Status:\n")
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		fmt.Printf("   Go Memory: %.2f MB\n", float64(m.Alloc)/1024/1024)
		fmt.Printf("   Go Goroutines: %d\n", runtime.NumGoroutine())
	}

	// Final summary
	fmt.Printf("\n================================================================================\n")
	fmt.Printf("🏆 BEST RESULTS:\n")
	fmt.Printf("   Best Config: %s\n", bestConfig)
	fmt.Printf("   Best Indexing: %.0f docs/sec\n", bestThroughput)
	fmt.Printf("   Best Search QPS: %.0f\n", bestQPS)
	fmt.Printf("================================================================================\n")
}

func generateCorpus(size int) []string {
	templates := []string{
		"Advanced techniques in %s for modern applications",
		"Implementing scalable %s with cloud infrastructure",
		"Optimizing %s performance using GPU acceleration",
		"Building robust %s systems with high availability",
		"Real-time %s processing for enterprise solutions",
	}

	topics := []string{
		"machine learning", "deep learning", "neural networks",
		"computer vision", "natural language processing",
		"data pipelines", "stream processing", "batch processing",
		"microservices", "serverless computing", "container orchestration",
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

func contains(s string) string {
	result := ""
	for i := 0; i < len(s); i++ {
		result += "="
	}
	return result
}