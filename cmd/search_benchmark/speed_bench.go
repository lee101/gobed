package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("=== Gobed Speed-Optimized Search Benchmark ===\n")

	// Load model
	fmt.Println("Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Model loaded\n")

	// Test different dataset sizes to show index transitions
	sizes := []int{1000, 5000, 10000, 20000, 50000}

	for _, size := range sizes {
		fmt.Printf("=== Testing %d documents ===\n", size)

		// Generate simple test corpus
		corpus := generateCorpus(size)

		// Create search engine with default (speed-optimized) config
		engine := gobed.NewSearchEngine(model)

		// Measure indexing
		indexStart := time.Now()
		batchSize := 500
		for i := 0; i < size; i += batchSize {
			end := min(i+batchSize, size)
			_, err := engine.IndexBatch(corpus[i:end])
			if err != nil {
				log.Printf("Index error: %v", err)
			}
		}
		indexTime := time.Since(indexStart)

		// Get stats to see what index type was chosen
		stats := engine.Stats()

		// Run search benchmarks
		numQueries := 100
		queries := []string{
			"machine learning algorithms",
			"database optimization",
			"cloud computing services",
			"web development frameworks",
			"data science pipelines",
		}

		totalSearchTime := time.Duration(0)
		for i := 0; i < numQueries; i++ {
			query := queries[i%len(queries)]
			searchStart := time.Now()
			_, err := engine.Search(query, 10)
			if err != nil {
				log.Printf("Search error: %v", err)
			}
			totalSearchTime += time.Since(searchStart)
		}

		avgSearchLatency := totalSearchTime / time.Duration(numQueries)

		// Print results
		fmt.Printf("Index Type:      %s\n", stats.IndexType)
		fmt.Printf("Memory:          %.2f MB\n", stats.MemoryUsageMB)
		fmt.Printf("Index Time:      %v (%.0f docs/sec)\n",
			indexTime, float64(size)/indexTime.Seconds())
		fmt.Printf("Avg Search:      %v\n", avgSearchLatency)
		fmt.Printf("Search QPS:      %.0f\n", float64(numQueries)/totalSearchTime.Seconds())

		// Highlight if we're using approximate search
		if stats.IndexType != "flat" {
			fmt.Printf("Mode:            APPROXIMATE (fast)\n")
		} else {
			fmt.Printf("Mode:            EXACT\n")
		}
		fmt.Println()
	}

	// Demonstrate the speed difference between exact and approximate
	fmt.Println("=== Speed Comparison: Exact vs Approximate ===\n")

	testSize := 15000 // Will trigger approximate search with new defaults
	corpus := generateCorpus(testSize)

	// Test 1: Force exact search
	fmt.Printf("Testing %d documents with EXACT search (slow):\n", testSize)
	configExact := gobed.SearchConfig{
		AutoMode:           false,
		MaxExactSearchSize: 100000, // Force exact search
	}
	engineExact := gobed.NewSearchEngineWithConfig(model, configExact)

	// Index
	for i := 0; i < testSize; i += 1000 {
		end := min(i+1000, testSize)
		engineExact.IndexBatch(corpus[i:end])
	}

	// Benchmark exact search
	start := time.Now()
	for i := 0; i < 50; i++ {
		engineExact.Search("test query", 10)
	}
	exactTime := time.Since(start) / 50

	// Test 2: Automatic (approximate) search
	fmt.Printf("Testing %d documents with APPROXIMATE search (fast):\n", testSize)
	engineApprox := gobed.NewSearchEngine(model) // Uses new speed-optimized defaults

	// Index
	for i := 0; i < testSize; i += 1000 {
		end := min(i+1000, testSize)
		engineApprox.IndexBatch(corpus[i:end])
	}

	// Optimize for best performance
	engineApprox.Optimize()

	// Benchmark approximate search
	start = time.Now()
	for i := 0; i < 50; i++ {
		engineApprox.Search("test query", 10)
	}
	approxTime := time.Since(start) / 50

	statsExact := engineExact.Stats()
	statsApprox := engineApprox.Stats()

	fmt.Printf("\nResults:\n")
	fmt.Printf("Exact Search:       %v (Index: %s)\n", exactTime, statsExact.IndexType)
	fmt.Printf("Approximate Search: %v (Index: %s)\n", approxTime, statsApprox.IndexType)
	fmt.Printf("Speedup:            %.1fx faster\n", float64(exactTime)/float64(approxTime))

	if approxTime < exactTime/2 {
		fmt.Println("\n Approximate search is significantly faster!")
	}
}

func generateCorpus(size int) []string {
	templates := []string{
		"Introduction to %s in modern software development",
		"Advanced techniques for %s optimization",
		"Building scalable %s systems",
		"Understanding %s best practices",
		"The future of %s technology",
	}

	topics := []string{
		"machine learning", "database", "cloud computing",
		"web development", "data science", "artificial intelligence",
		"blockchain", "microservices", "DevOps", "cybersecurity",
	}

	corpus := make([]string, size)
	for i := 0; i < size; i++ {
		template := templates[rand.Intn(len(templates))]
		topic := topics[rand.Intn(len(topics))]
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
