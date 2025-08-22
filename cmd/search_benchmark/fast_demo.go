package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/lee101/gobed"
)

func main_disabled() {
	fmt.Println("=== Gobed Fast Approximate Search Demo ===\n")
	fmt.Println("Demonstrating speed-optimized defaults (approximate search from 5K docs)\n")

	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Model loaded\n")

	// Test at different scales to show the automatic optimization
	testSizes := []struct {
		size int
		desc string
	}{
		{1000, "Very small - exact search"},
		{5000, "Threshold - still exact"},
		{10000, "Medium - switches to approximate"},
		{25000, "Large - fully approximate"},
	}

	for _, test := range testSizes {
		fmt.Printf("--- Testing %d documents (%s) ---\n", test.size, test.desc)

		// Generate corpus
		corpus := make([]string, test.size)
		for i := 0; i < test.size; i++ {
			topics := []string{"AI", "database", "cloud", "web", "security", "data"}
			topic := topics[rand.Intn(len(topics))]
			corpus[i] = fmt.Sprintf("Document about %s technology #%d", topic, i)
		}

		// Create engine with default speed-optimized config
		engine := gobed.NewSearchEngine(model)

		// Index
		start := time.Now()
		batchSize := 1000
		for i := 0; i < test.size; i += batchSize {
			end := min(i+batchSize, test.size)
			_, err := engine.IndexBatch(corpus[i:end])
			if err != nil {
				// For small datasets, this is expected - just continue
				continue
			}
		}
		indexTime := time.Since(start)

		// Get index type
		stats := engine.Stats()

		// Run search benchmark
		searchQueries := 50
		start = time.Now()
		for i := 0; i < searchQueries; i++ {
			engine.Search("AI technology", 5)
		}
		searchTime := time.Since(start) / time.Duration(searchQueries)

		// Results
		fmt.Printf("  Index type: %s\n", stats.IndexType)
		fmt.Printf("  Index time: %v\n", indexTime)
		fmt.Printf("  Search latency: %v\n", searchTime)
		fmt.Printf("  Memory: %.1f MB\n", stats.MemoryUsageMB)

		if searchTime < time.Millisecond {
			fmt.Printf("  ✓ SUB-MILLISECOND search!\n")
		}
		fmt.Println()
	}

	// Show a practical example with real queries
	fmt.Println("=== Practical Example: 15K Document Search ===\n")

	// Create a realistic corpus
	corpusSize := 15000
	topics := []string{
		"Python programming", "Machine learning", "Cloud computing",
		"Web development", "Database systems", "Cybersecurity",
		"Data science", "DevOps practices", "Mobile apps",
		"Blockchain technology", "AI research", "Software architecture",
	}

	corpus := make([]string, corpusSize)
	for i := 0; i < corpusSize; i++ {
		topic1 := topics[rand.Intn(len(topics))]
		topic2 := topics[rand.Intn(len(topics))]
		corpus[i] = fmt.Sprintf("%s combined with %s for modern applications", topic1, topic2)
	}

	// Index with speed-optimized settings
	engine := gobed.NewSearchEngine(model)

	fmt.Printf("Indexing %d documents...\n", corpusSize)
	start := time.Now()
	for i := 0; i < corpusSize; i += 1000 {
		end := min(i+1000, corpusSize)
		engine.IndexBatch(corpus[i:end])
	}
	fmt.Printf("✓ Indexed in %v\n\n", time.Since(start))

	// Perform realistic searches
	queries := []string{
		"Python machine learning frameworks",
		"Cloud native DevOps",
		"Mobile app security",
		"Blockchain data science",
		"Web development with AI",
	}

	stats := engine.Stats()
	fmt.Printf("Using %s index (optimized for speed)\n", stats.IndexType)
	fmt.Printf("Memory usage: %.1f MB\n\n", stats.MemoryUsageMB)

	fmt.Println("Search results:")
	for _, query := range queries {
		start := time.Now()
		results, _ := engine.Search(query, 3)
		latency := time.Since(start)

		fmt.Printf("\nQuery: '%s' (%v)\n", query, latency)
		for i, r := range results {
			text := r.Text
			if len(text) > 60 {
				text = text[:57] + "..."
			}
			fmt.Printf("  %d. [%.3f] %s\n", i+1, r.Similarity, text)
		}
	}

	fmt.Println("\n✓ Fast approximate search demonstrated!")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
