//go:build legacy


package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("=== Testing Gobed Cache Performance ===")
	
	cachePath := "/tmp/gobed_cache_test.bin"
	indexedPath := cachePath + ".indexed"
	
	// Clean up any existing cache
	os.Remove(cachePath)
	os.Remove(indexedPath)
	defer os.Remove(cachePath)
	defer os.Remove(indexedPath)

	// Test data
	texts := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning is transforming industries",
		"Natural language processing enables AI to understand text",
		"Vector embeddings capture semantic meaning",
		"Search engines use similarity metrics",
		"Deep learning models power modern AI",
		"Transformers revolutionized NLP",
		"BERT and GPT are transformer models",
		"Semantic search improves relevance",
		"Neural networks learn patterns",
	}

	// Load model
	fmt.Println("\n1. Loading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Test 1: Initial indexing
	fmt.Println("\n2. Initial indexing (no cache)...")
	engine1 := gobed.NewSearchEngine(model)
	
	indexStart := time.Now()
	_, err = engine1.IndexBatch(texts)
	if err != nil {
		log.Fatalf("Failed to index: %v", err)
	}
	indexTime := time.Since(indexStart)
	fmt.Printf("   ✓ Indexed %d documents in %v\n", len(texts), indexTime)

	// Save to cache
	fmt.Println("\n3. Saving to cache...")
	saveStart := time.Now()
	err = engine1.QuickSave(cachePath)
	if err != nil {
		log.Fatalf("Failed to save: %v", err)
	}
	saveTime := time.Since(saveStart)
	fmt.Printf("   ✓ Saved cache in %v\n", saveTime)
	
	// Create indexed marker
	err = os.WriteFile(indexedPath, []byte("indexed"), 0644)
	if err != nil {
		log.Fatalf("Failed to create marker: %v", err)
	}
	fmt.Printf("   ✓ Created indexed marker\n")

	// Test 2: Load from cache
	fmt.Println("\n4. Loading from cache...")
	engine2 := gobed.NewSearchEngine(model)
	
	loadStart := time.Now()
	err = engine2.Load(cachePath)
	if err != nil {
		log.Fatalf("Failed to load: %v", err)
	}
	loadTime := time.Since(loadStart)
	fmt.Printf("   ✓ Loaded cache in %v\n", loadTime)

	// Test search works
	fmt.Println("\n5. Testing search on cached index...")
	results, err := engine2.Search("machine learning AI", 3)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	fmt.Printf("   ✓ Search returned %d results\n", len(results))

	// Show performance comparison
	fmt.Println("\n=== Performance Summary ===")
	fmt.Printf("Initial indexing: %v\n", indexTime)
	fmt.Printf("Loading from cache: %v\n", loadTime)
	speedup := float64(indexTime) / float64(loadTime)
	fmt.Printf("Speedup: %.1fx faster\n", speedup)
	
	if loadTime > indexTime/2 {
		fmt.Println("WARNING: Cache loading seems slow, might be re-indexing!")
	} else {
		fmt.Println("Cache is working correctly - no re-indexing detected!")
	}
}
