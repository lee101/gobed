package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/ann/simd"
)

func main() {
	fmt.Println("=== Simple Shared Memory Test ===")

	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Create a small test dataset
	documents := []string{
		"machine learning",
		"deep learning",
		"neural networks",
		"artificial intelligence",
		"computer vision",
	}

	// Create shared memory index
	config := gobed.SharedMemoryConfig{
		BasePath:    "/tmp/gobed_test_shared",
		MaxVectors:  100,
		CreateIfNew: true,
		CacheSize:   10,
	}

	fmt.Println("\n1. Creating shared memory index...")
	sharedIndex, err := gobed.NewSharedMemoryIndex(config)
	if err != nil {
		log.Fatalf("Failed to create shared index: %v", err)
	}
	defer sharedIndex.Close()

	// Index documents
	fmt.Println("2. Indexing documents...")
	for i, doc := range documents {
		embedding, err := model.EmbedInt8(doc)
		if err != nil {
			log.Printf("Failed to embed: %v", err)
			continue
		}

		var vec simd.Vec512
		copy(vec[:], embedding.Vector)

		if err := sharedIndex.AddVector(&vec, embedding.Scale, i); err != nil {
			log.Printf("Failed to add vector: %v", err)
		}
	}

	// Sync to disk
	sharedIndex.Sync()

	stats := sharedIndex.Stats()
	fmt.Printf("Indexed %d vectors\n", stats.NumVectors)

	// Test search
	fmt.Println("\n3. Testing search...")
	query := "deep neural networks"
	embedding, err := model.EmbedInt8(query)
	if err != nil {
		log.Fatalf("Failed to embed query: %v", err)
	}

	var queryVec simd.Vec512
	copy(queryVec[:], embedding.Vector)

	start := time.Now()
	results := sharedIndex.SearchTopK(&queryVec, 3)
	latency := time.Since(start)

	fmt.Printf("Search completed in %v\n", latency)
	fmt.Println("Results:")
	for i, r := range results {
		fmt.Printf("  %d. Doc %d (similarity: %.3f)\n", i+1, r.ID, r.Similarity)
	}

	// Test concurrent search
	fmt.Println("\n4. Testing concurrent search...")
	start = time.Now()
	for i := 0; i < 10; i++ {
		go func() {
			sharedIndex.SearchTopK(&queryVec, 3)
		}()
	}
	time.Sleep(100 * time.Millisecond)
	fmt.Printf("10 concurrent searches completed in %v\n", time.Since(start))

	fmt.Println("\n✓ Test completed successfully!")
}
