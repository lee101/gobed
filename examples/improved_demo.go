package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println(" Gobed Improved Demo - Showing Performance Optimizations")
	fmt.Println("=" + "="*60)

	// Load the model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Example 1: Using preset configurations for search
	fmt.Println("\n Example 1: Simplified Search with Presets")
	fmt.Println("-" * 50)

	// Create search engines with different presets
	fastSearch, err := gobed.NewSearchEngineWithPreset(model, gobed.FastPreset)
	if err != nil {
		log.Fatalf("Failed to create fast search engine: %v", err)
	}

	balancedSearch, err := gobed.NewSearchEngineWithPreset(model, gobed.BalancedPreset)
	if err != nil {
		log.Fatalf("Failed to create balanced search engine: %v", err)
	}

	// Index some sample documents
	documents := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning is transforming the world",
		"Natural language processing enables computers to understand text",
		"Deep learning models can generate human-like text",
		"Search engines help users find relevant information quickly",
	}

	for i, doc := range documents {
		if err := fastSearch.Index(i, doc); err != nil {
			log.Printf("Failed to index document %d: %v", i, err)
		}
		if err := balancedSearch.Index(i, doc); err != nil {
			log.Printf("Failed to index document %d: %v", i, err)
		}
	}

	// Search with different presets
	query := "find information about machine learning"

	start := time.Now()
	fastResults, err := fastSearch.Search(query, 3)
	fastTime := time.Since(start)

	start = time.Now()
	balancedResults, err := balancedSearch.Search(query, 3)
	balancedTime := time.Since(start)

	fmt.Printf("\nQuery: %s\n", query)
	fmt.Printf("\nFast Preset (optimized for speed):\n")
	fmt.Printf("  Time: %v\n", fastTime)
	for _, r := range fastResults {
		fmt.Printf("  - Doc %d (score: %.3f): %s\n", r.ID, r.Score, documents[r.ID][:50]+"...")
	}

	fmt.Printf("\nBalanced Preset (balance speed/accuracy):\n")
	fmt.Printf("  Time: %v\n", balancedTime)
	for _, r := range balancedResults {
		fmt.Printf("  - Doc %d (score: %.3f): %s\n", r.ID, r.Score, documents[r.ID][:50]+"...")
	}

	// Example 2: Memory-optimized batch processing
	fmt.Println("\n Example 2: Memory-Optimized Batch Processing")
	fmt.Println("-" * 50)

	// Generate a larger batch of texts
	batchTexts := make([]string, 100)
	for i := range batchTexts {
		batchTexts[i] = fmt.Sprintf("Document number %d with some content about AI and technology", i)
	}

	// Process batch with memory pooling (now built-in)
	start = time.Now()
	for _, text := range batchTexts {
		_, err := model.Encode(text)
		if err != nil {
			log.Printf("Failed to encode: %v", err)
		}
	}
	batchTime := time.Since(start)

	fmt.Printf("Processed %d documents in %v\n", len(batchTexts), batchTime)
	fmt.Printf("Average time per document: %v\n", batchTime/time.Duration(len(batchTexts)))

	// Example 3: Show simplified configuration
	fmt.Println("\n Example 3: Simplified Configuration")
	fmt.Println("-" * 50)

	fmt.Println("Available presets:")
	fmt.Println("  - FastPreset: Best for <50K vectors, prioritizes speed")
	fmt.Println("  - BalancedPreset: Best for 50K-500K vectors, balances speed/accuracy")
	fmt.Println("  - AccuratePreset: Best for >500K vectors, prioritizes accuracy")

	fmt.Println("\n Demo completed successfully!")
	fmt.Println("\nKey improvements demonstrated:")
	fmt.Println("  1. Simplified search configuration with presets")
	fmt.Println("  2. Built-in memory pooling for better performance")
	fmt.Println("  3. Cleaner API with fewer configuration options")
}
