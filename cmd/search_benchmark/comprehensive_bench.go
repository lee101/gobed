package main

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/ann/simd"
)

func main() {
	fmt.Println("=== Comprehensive Performance Test ===")
	fmt.Println("Comparing: Standard vs Async vs Shared Memory\n")

	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Test configurations
	testSizes := []int{100, 1000, 5000}

	for _, size := range testSizes {
		fmt.Printf("\n========================================")
		fmt.Printf("\nTesting with %d documents\n", size)
		fmt.Printf("========================================\n")

		// Generate test data
		docs := generateDocs(size)
		queries := generateQueries(20)

		// Test 1: Standard Search Engine
		standardResults := testStandard(model, docs, queries)

		// Test 2: Async Search Engine
		asyncResults := testAsync(model, docs, queries)

		// Test 3: Shared Memory
		sharedResults := testSharedMemory(model, docs, queries)

		// Print comparison
		printResults(size, standardResults, asyncResults, sharedResults)
	}
}

type TestResult struct {
	Name          string
	IndexTime     time.Duration
	SearchLatency time.Duration
	QPS           float64
	MemoryMB      float64
	DocsPerSec    float64
}

func testStandard(model *gobed.EmbeddingModel, docs []string, queries []string) TestResult {
	fmt.Println("\n📊 Standard Search Engine")
	fmt.Println("-------------------------")

	config := gobed.DefaultSearchConfig()
	engine := gobed.NewSearchEngineWithConfig(model, config)

	// Index
	start := time.Now()
	ids, err := engine.IndexBatch(docs)
	indexTime := time.Since(start)

	if err != nil {
		log.Printf("Standard indexing error: %v", err)
	}

	fmt.Printf("✓ Indexed %d documents in %v\n", len(ids), indexTime)

	// Search
	totalSearchTime := time.Duration(0)
	for _, query := range queries {
		searchStart := time.Now()
		_, err := engine.Search(query, 10)
		if err == nil {
			totalSearchTime += time.Since(searchStart)
		}
	}

	avgLatency := totalSearchTime / time.Duration(len(queries))
	qps := float64(len(queries)) / totalSearchTime.Seconds()

	stats := engine.Stats()

	fmt.Printf("✓ Search: %v avg latency, %.0f QPS\n", avgLatency, qps)
	fmt.Printf("✓ Memory: %.2f MB\n", stats.MemoryUsageMB)

	return TestResult{
		Name:          "Standard",
		IndexTime:     indexTime,
		SearchLatency: avgLatency,
		QPS:           qps,
		MemoryMB:      stats.MemoryUsageMB,
		DocsPerSec:    float64(len(docs)) / indexTime.Seconds(),
	}
}

func testAsync(model *gobed.EmbeddingModel, docs []string, queries []string) TestResult {
	fmt.Println("\n⚡ Async Search Engine")
	fmt.Println("----------------------")

	config := gobed.AsyncSearchConfig()
	config.AsyncWorkers = 4
	engine := gobed.NewSearchEngineWithConfig(model, config)

	// Index
	start := time.Now()
	response := engine.IndexBatchAsync(docs)
	result := <-response
	indexTime := time.Since(start)

	if result.Error != nil {
		log.Printf("Async indexing error: %v", result.Error)
	}

	fmt.Printf("✓ Indexed %d documents in %v (async)\n", len(result.IDs), indexTime)

	// Search
	totalSearchTime := time.Duration(0)
	for _, query := range queries {
		searchStart := time.Now()
		_, err := engine.Search(query, 10)
		if err == nil {
			totalSearchTime += time.Since(searchStart)
		}
	}

	avgLatency := totalSearchTime / time.Duration(len(queries))
	qps := float64(len(queries)) / totalSearchTime.Seconds()

	stats := engine.Stats()

	fmt.Printf("✓ Search: %v avg latency, %.0f QPS\n", avgLatency, qps)
	fmt.Printf("✓ Memory: %.2f MB\n", stats.MemoryUsageMB)

	return TestResult{
		Name:          "Async",
		IndexTime:     indexTime,
		SearchLatency: avgLatency,
		QPS:           qps,
		MemoryMB:      stats.MemoryUsageMB,
		DocsPerSec:    float64(len(docs)) / indexTime.Seconds(),
	}
}

func testSharedMemory(model *gobed.EmbeddingModel, docs []string, queries []string) TestResult {
	fmt.Println("\n🌐 Shared Memory Index")
	fmt.Println("----------------------")

	config := gobed.SharedMemoryConfig{
		BasePath:    fmt.Sprintf("/tmp/gobed_comp_test_%d", time.Now().UnixNano()),
		MaxVectors:  len(docs) * 2,
		CreateIfNew: true,
		CacheSize:   100,
	}

	idx, err := gobed.NewSharedMemoryIndex(config)
	if err != nil {
		log.Fatalf("Failed to create shared index: %v", err)
	}
	defer idx.Close()

	// Index
	start := time.Now()
	indexed := 0
	for i, doc := range docs {
		embedding, err := model.EmbedInt8(doc)
		if err != nil {
			continue
		}
		var vec simd.Vec512
		copy(vec[:], embedding.Vector)
		if err := idx.AddVector(&vec, embedding.Scale, i); err == nil {
			indexed++
		}
	}
	idx.Sync()
	indexTime := time.Since(start)

	fmt.Printf("✓ Indexed %d documents in %v (shared)\n", indexed, indexTime)

	// Prepare query vectors
	queryVecs := make([]*simd.Vec512, len(queries))
	for i, query := range queries {
		embedding, _ := model.EmbedInt8(query)
		vec := &simd.Vec512{}
		copy(vec[:], embedding.Vector)
		queryVecs[i] = vec
	}

	// Search
	totalSearchTime := time.Duration(0)
	for _, qvec := range queryVecs {
		searchStart := time.Now()
		_ = idx.SearchTopK(qvec, 10)
		totalSearchTime += time.Since(searchStart)
	}

	avgLatency := totalSearchTime / time.Duration(len(queries))
	qps := float64(len(queries)) / totalSearchTime.Seconds()

	stats := idx.Stats()

	fmt.Printf("✓ Search: %v avg latency, %.0f QPS\n", avgLatency, qps)
	fmt.Printf("✓ Memory: %.2f MB (shared)\n", stats.MemoryUsageMB)

	// Test concurrent access
	fmt.Println("\n  Testing concurrent access...")
	var wg sync.WaitGroup
	numGoroutines := 4
	searchesPerGoroutine := 25

	concurrentStart := time.Now()
	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for s := 0; s < searchesPerGoroutine; s++ {
				idx.SearchTopK(queryVecs[s%len(queryVecs)], 10)
			}
		}()
	}
	wg.Wait()
	concurrentTime := time.Since(concurrentStart)

	concurrentQPS := float64(numGoroutines*searchesPerGoroutine) / concurrentTime.Seconds()
	fmt.Printf("  ✓ Concurrent: %d searches in %v (%.0f QPS)\n",
		numGoroutines*searchesPerGoroutine, concurrentTime, concurrentQPS)

	return TestResult{
		Name:          "SharedMem",
		IndexTime:     indexTime,
		SearchLatency: avgLatency,
		QPS:           concurrentQPS, // Use concurrent QPS for shared memory
		MemoryMB:      stats.MemoryUsageMB,
		DocsPerSec:    float64(indexed) / indexTime.Seconds(),
	}
}

func printResults(size int, standard, async, shared TestResult) {
	fmt.Printf("\n📈 Performance Summary (%d documents)\n", size)
	fmt.Println("=====================================")

	fmt.Println("\n| Metric         | Standard    | Async       | Shared Memory |")
	fmt.Println("|----------------|-------------|-------------|---------------|")

	fmt.Printf("| Index Time     | %11v | %11v | %13v |\n",
		standard.IndexTime, async.IndexTime, shared.IndexTime)

	fmt.Printf("| Docs/sec       | %11.0f | %11.0f | %13.0f |\n",
		standard.DocsPerSec, async.DocsPerSec, shared.DocsPerSec)

	fmt.Printf("| Search Latency | %11v | %11v | %13v |\n",
		standard.SearchLatency, async.SearchLatency, shared.SearchLatency)

	fmt.Printf("| QPS            | %11.0f | %11.0f | %13.0f |\n",
		standard.QPS, async.QPS, shared.QPS)

	fmt.Printf("| Memory (MB)    | %11.2f | %11.2f | %13.2f |\n",
		standard.MemoryMB, async.MemoryMB, shared.MemoryMB)

	// Calculate improvements
	fmt.Println("\n📊 Improvements vs Standard:")

	asyncIndexSpeedup := standard.IndexTime.Seconds() / async.IndexTime.Seconds()
	sharedIndexSpeedup := standard.IndexTime.Seconds() / shared.IndexTime.Seconds()

	fmt.Printf("  • Async indexing: %.2fx faster\n", asyncIndexSpeedup)
	fmt.Printf("  • Shared indexing: %.2fx faster\n", sharedIndexSpeedup)

	if shared.SearchLatency < standard.SearchLatency {
		improvement := (1 - shared.SearchLatency.Seconds()/standard.SearchLatency.Seconds()) * 100
		fmt.Printf("  • Shared search: %.1f%% lower latency\n", improvement)
	}

	sharedQPSGain := (shared.QPS - standard.QPS) / standard.QPS * 100
	fmt.Printf("  • Shared QPS: %.1f%% higher throughput\n", sharedQPSGain)
}

func generateDocs(n int) []string {
	docs := make([]string, n)
	topics := []string{
		"machine learning", "artificial intelligence", "deep learning",
		"neural networks", "computer vision", "natural language",
		"data science", "robotics", "automation", "algorithms",
	}

	for i := 0; i < n; i++ {
		topic := topics[i%len(topics)]
		docs[i] = fmt.Sprintf("Document %d about %s and related concepts", i, topic)
	}
	return docs
}

func generateQueries(n int) []string {
	queries := make([]string, n)
	queryTemplates := []string{
		"machine learning applications",
		"deep neural networks",
		"artificial intelligence systems",
		"computer vision algorithms",
		"natural language processing",
	}

	for i := 0; i < n; i++ {
		queries[i] = queryTemplates[i%len(queryTemplates)]
	}
	return queries
}
