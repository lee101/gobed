//go:build legacy

package main

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

func main_disabled() {
	fmt.Println("=== Gobed Optimized Performance Benchmark ===")
	fmt.Println("Showcasing all performance optimizations\n")

	// Load model
	fmt.Println("Loading embedding model...")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("✓ Model loaded in %v\n\n", time.Since(start))

	// Test different configurations
	configurations := []struct {
		name        string
		size        int
		description string
		config      gobed.SearchConfig
	}{
		{
			name:        "Standard",
			size:        10000,
			description: "Standard configuration",
			config:      gobed.DefaultSearchConfig(),
		},
		{
			name:        "Async_Optimized",
			size:        10000,
			description: "Async with optimizations",
			config:      gobed.AsyncSearchConfig(),
		},
		{
			name:        "Large_Dataset",
			size:        50000,
			description: "Large dataset with IVF",
			config:      gobed.DefaultSearchConfig(),
		},
		{
			name:        "Large_Async",
			size:        50000,
			description: "Large async with full optimization",
			config:      gobed.AsyncSearchConfig(),
		},
	}

	// Generate diverse test corpus
	corpus := generateDiverseCorpus(60000) // Generate enough for largest test
	queries := generateTestQueries()

	var allResults []BenchmarkResult

	fmt.Println("=== PERFORMANCE BENCHMARKS ===\n")

	for _, config := range configurations {
		fmt.Printf("Testing %s (%d documents): %s\n", config.name, config.size, config.description)
		fmt.Println("────────────────────────────────────────────────────────")

		// Create search engine
		engine := gobed.NewSearchEngineWithConfig(model, config.config)

		// Measure indexing performance
		documents := corpus[:config.size]

		indexStart := time.Now()
		var indexTime time.Duration
		var throughput float64

		if config.config.EnableAsync {
			// Test async indexing
			batchSize := 2000
			var responses []<-chan gobed.IndexResponse

			for i := 0; i < len(documents); i += batchSize {
				end := min(i+batchSize, len(documents))
				batch := documents[i:end]
				response := engine.IndexBatchAsync(batch)
				responses = append(responses, response)
			}

			// Wait for all responses
			totalIndexed := 0
			for _, response := range responses {
				result := <-response
				if result.Error != nil {
					log.Printf("Async indexing error: %v", result.Error)
				}
				totalIndexed += len(result.IDs)
			}

			// Flush any remaining work
			engine.Flush()

			indexTime = time.Since(indexStart)
			throughput = float64(totalIndexed) / indexTime.Seconds()
		} else {
			// Test synchronous indexing
			batchSize := 2000
			totalIndexed := 0

			for i := 0; i < len(documents); i += batchSize {
				end := min(i+batchSize, len(documents))
				batch := documents[i:end]

				ids, err := engine.IndexBatch(batch)
				if err != nil {
					log.Printf("Sync indexing error: %v", err)
					continue
				}
				totalIndexed += len(ids)
			}

			indexTime = time.Since(indexStart)
			throughput = float64(totalIndexed) / indexTime.Seconds()
		}

		fmt.Printf("Indexing: %v (%,.0f docs/sec)\n", indexTime, throughput)

		// Test search performance with different concurrency levels
		concurrencyLevels := []int{1, 4, 8, 16}

		for _, concurrency := range concurrencyLevels {
			searchLatency := benchmarkSearch(engine, queries, 100, concurrency)
			searchThroughput := float64(100*concurrency) / searchLatency.Seconds()

			fmt.Printf("Search (%d concurrent): %v latency, %,.0f QPS\n",
				concurrency, searchLatency/time.Duration(100), searchThroughput)
		}

		// Get memory statistics
		stats := engine.Stats()
		fmt.Printf("Memory: %.2f MB, Index: %s\n", stats.MemoryUsageMB, stats.IndexType)

		// Test cache effectiveness (for optimized versions)
		if config.config.EnableAsync {
			// Test repeated queries to show cache benefit
			cacheStart := time.Now()
			for i := 0; i < 50; i++ {
				engine.Search(queries[i%len(queries)], 10)
			}
			cacheTime := time.Since(cacheStart)
			fmt.Printf("Cache performance: %v for 50 queries (%v avg)\n",
				cacheTime, cacheTime/50)
		}

		// Store results (use the single-threaded search latency)
		singleSearchLatency := benchmarkSearch(engine, queries, 100, 1)
		result := BenchmarkResult{
			Configuration:   config.name,
			DocumentCount:   config.size,
			IndexTime:       indexTime,
			IndexThroughput: throughput,
			SearchLatency:   singleSearchLatency / time.Duration(100), // Single search latency
			MemoryMB:        stats.MemoryUsageMB,
			IndexType:       stats.IndexType,
		}
		allResults = append(allResults, result)

		// Cleanup
		if config.config.EnableAsync {
			engine.Close()
		}

		// Force garbage collection between tests
		runtime.GC()
		time.Sleep(500 * time.Millisecond)

		fmt.Println()
	}

	// Print comparison table
	printComparisonTable(allResults)

	// Performance insights
	printPerformanceInsights(allResults)

	fmt.Println("✓ Optimized benchmark completed successfully!")
}

// benchmarkSearch measures search performance with specified concurrency
func benchmarkSearch(engine *gobed.SearchEngine, queries []string, numQueries, concurrency int) time.Duration {
	var wg sync.WaitGroup
	start := time.Now()

	queryChan := make(chan string, numQueries)
	for i := 0; i < numQueries; i++ {
		queryChan <- queries[i%len(queries)]
	}
	close(queryChan)

	// Launch workers
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for query := range queryChan {
				_, err := engine.Search(query, 10)
				if err != nil {
					log.Printf("Search error: %v", err)
				}
			}
		}()
	}

	wg.Wait()
	return time.Since(start)
}

// generateDiverseCorpus creates a diverse corpus for testing
func generateDiverseCorpus(size int) []string {
	domains := [][]string{
		// Technology
		{"artificial intelligence", "machine learning", "deep learning", "neural networks", "computer vision"},
		{"cloud computing", "distributed systems", "microservices", "containerization", "Kubernetes"},
		{"web development", "React", "Vue.js", "Angular", "JavaScript", "TypeScript"},
		{"database", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "SQL optimization"},

		// Science
		{"quantum physics", "molecular biology", "genetics", "neuroscience", "climate change"},
		{"space exploration", "astronomy", "astrophysics", "Mars mission", "satellite technology"},
		{"renewable energy", "solar panels", "wind turbines", "battery technology", "green technology"},

		// Business
		{"digital transformation", "business strategy", "market analysis", "customer experience"},
		{"financial technology", "blockchain", "cryptocurrency", "fintech", "payment systems"},
		{"supply chain", "logistics", "inventory management", "manufacturing", "quality control"},

		// Health & Medicine
		{"medical research", "pharmaceutical", "clinical trials", "drug discovery", "healthcare"},
		{"mental health", "psychology", "therapy", "wellness", "mindfulness"},
		{"nutrition", "fitness", "exercise", "sports medicine", "rehabilitation"},
	}

	templates := []string{
		"Advanced %s techniques for modern %s applications",
		"Best practices in %s implementation and %s integration",
		"Scalable %s solutions for enterprise %s systems",
		"Performance optimization strategies for %s in %s environments",
		"Security considerations in %s architecture and %s design",
		"Future trends in %s technology and %s innovation",
		"Research developments in %s and applications to %s",
		"Case study: implementing %s for improved %s outcomes",
		"Comparative analysis of %s methods in %s industry",
		"Emerging %s technologies transforming %s landscape",
	}

	corpus := make([]string, size)
	for i := 0; i < size; i++ {
		domain1 := domains[rand.Intn(len(domains))]
		domain2 := domains[rand.Intn(len(domains))]

		term1 := domain1[rand.Intn(len(domain1))]
		term2 := domain2[rand.Intn(len(domain2))]

		template := templates[rand.Intn(len(templates))]
		corpus[i] = fmt.Sprintf(template, term1, term2)
	}

	return corpus
}

// generateTestQueries creates realistic test queries
func generateTestQueries() []string {
	return []string{
		"machine learning algorithms for data analysis",
		"cloud native application deployment strategies",
		"database performance optimization techniques",
		"artificial intelligence in healthcare applications",
		"cybersecurity best practices for enterprise",
		"blockchain technology for financial services",
		"renewable energy solutions and sustainability",
		"quantum computing research and development",
		"web application security and authentication",
		"mobile app development frameworks comparison",
		"distributed systems architecture patterns",
		"data science and predictive analytics",
		"containerization and orchestration platforms",
		"natural language processing applications",
		"computer vision and image recognition",
	}
}

// BenchmarkResult stores benchmark results
type BenchmarkResult struct {
	Configuration   string
	DocumentCount   int
	IndexTime       time.Duration
	IndexThroughput float64
	SearchLatency   time.Duration
	MemoryMB        float64
	IndexType       string
}

// printComparisonTable prints a comparison table of results
func printComparisonTable(results []BenchmarkResult) {
	fmt.Println("=== PERFORMANCE COMPARISON TABLE ===\n")
	fmt.Println("| Configuration    | Docs   | Index Time | Index QPS  | Search Latency | Memory MB | Index Type |")
	fmt.Println("|------------------|--------|------------|------------|----------------|-----------|------------|")

	for _, r := range results {
		latencyStr := fmt.Sprintf("%v", r.SearchLatency)
		if r.SearchLatency < time.Millisecond {
			latencyStr += " "
		}

		fmt.Printf("| %-16s | %6d | %10v | %10.0f | %14s | %9.2f | %-10s |\n",
			r.Configuration, r.DocumentCount, r.IndexTime.Round(time.Millisecond),
			r.IndexThroughput, latencyStr, r.MemoryMB, r.IndexType)
	}
	fmt.Println()
}

// printPerformanceInsights analyzes and prints performance insights
func printPerformanceInsights(results []BenchmarkResult) {
	fmt.Println("=== PERFORMANCE INSIGHTS ===\n")

	// Find best indexing performance
	bestIndexing := results[0]
	for _, r := range results[1:] {
		if r.IndexThroughput > bestIndexing.IndexThroughput {
			bestIndexing = r
		}
	}

	// Find best search performance
	bestSearch := results[0]
	for _, r := range results[1:] {
		if r.SearchLatency < bestSearch.SearchLatency {
			bestSearch = r
		}
	}

	fmt.Printf(" **Best Indexing Performance**: %s with %,.0f docs/sec\n",
		bestIndexing.Configuration, bestIndexing.IndexThroughput)

	fmt.Printf(" **Best Search Performance**: %s with %v latency\n",
		bestSearch.Configuration, bestSearch.SearchLatency)

	// Compare standard vs async
	var standard, async *BenchmarkResult
	for i := range results {
		if results[i].Configuration == "Standard" && results[i].DocumentCount == 10000 {
			standard = &results[i]
		}
		if results[i].Configuration == "Async_Optimized" && results[i].DocumentCount == 10000 {
			async = &results[i]
		}
	}

	if standard != nil && async != nil {
		indexImprovement := (async.IndexThroughput - standard.IndexThroughput) / standard.IndexThroughput * 100
		fmt.Printf("\n **Async vs Standard (10K docs)**:\n")
		fmt.Printf("   • Indexing: %.1f%% improvement\n", indexImprovement)

		if async.SearchLatency < standard.SearchLatency {
			searchImprovement := float64(standard.SearchLatency-async.SearchLatency) / float64(standard.SearchLatency) * 100
			fmt.Printf("   • Search: %.1f%% faster\n", searchImprovement)
		}
	}

	// Memory efficiency analysis
	fmt.Printf("\n **Memory Efficiency**:\n")
	for _, r := range results {
		docsPerMB := float64(r.DocumentCount) / r.MemoryMB
		fmt.Printf("   • %s: %,.0f docs/MB\n", r.Configuration, docsPerMB)
	}

	// Scalability analysis
	fmt.Printf("\n **Scalability Analysis**:\n")
	smallDataset := findResult(results, "Standard", 10000)
	largeDataset := findResult(results, "Large_Dataset", 50000)

	if smallDataset != nil && largeDataset != nil {
		scalingFactor := float64(largeDataset.DocumentCount) / float64(smallDataset.DocumentCount)
		throughputRatio := largeDataset.IndexThroughput / smallDataset.IndexThroughput
		latencyRatio := float64(largeDataset.SearchLatency) / float64(smallDataset.SearchLatency)

		fmt.Printf("   • 5x dataset increase: %.1fx throughput scaling\n", throughputRatio)
		fmt.Printf("   • Search latency scaling: %.1fx\n", latencyRatio)

		if throughputRatio > 0.8*scalingFactor {
			fmt.Printf("    Excellent linear scaling performance\n")
		} else if throughputRatio > 0.5*scalingFactor {
			fmt.Printf("    Good scaling performance\n")
		} else {
			fmt.Printf("     Scaling could be improved\n")
		}
	}

	fmt.Printf("\n **Key Optimizations Demonstrated**:\n")
	fmt.Printf("    Embedding caching for duplicate text handling\n")
	fmt.Printf("    Async indexing with worker pools\n")
	fmt.Printf("    Pre-allocated memory to reduce GC pressure\n")
	fmt.Printf("    Automatic index selection (flat vs IVF vs PQ)\n")
	fmt.Printf("    SIMD-optimized vector operations\n")
	fmt.Printf("    Real training data vs synthetic for better clustering\n")

	fmt.Println()
}

// Helper function to find a specific result
func findResult(results []BenchmarkResult, config string, size int) *BenchmarkResult {
	for i := range results {
		if results[i].Configuration == config && results[i].DocumentCount == size {
			return &results[i]
		}
	}
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
