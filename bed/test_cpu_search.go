package main

import (
	"fmt"
	"log"
	"sort"
	"time"
)

type SearchResult struct {
	Index int
	Score float32
	Text  string
}

// CPU-only search to verify embeddings work correctly
func main() {
	fmt.Println("CPU Search Test")

	// Load model
	model, err := LoadFastModel("../model/modelint8_512dim.safetensors", "../model/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Test documents
	docs := []string{
		"machine learning algorithms are powerful",
		"neural networks and deep learning",
		"anime and manga culture in Japan",
		"cooking recipes for dinner",
		"programming languages like Go and Python",
		"artificial intelligence research",
		"optimization techniques in software",
	}

	// Generate embeddings
	fmt.Println("Generating embeddings...")
	embeddings := make([][]int8, len(docs))
	for i, doc := range docs {
		emb, err := model.EmbedInt8(doc)
		if err != nil {
			log.Printf("Failed to embed doc %d: %v", i, err)
			continue
		}
		embeddings[i] = emb
	}

	// Test queries
	queries := []string{"machine learning", "anime", "cooking", "programming"}

	for _, query := range queries {
		fmt.Printf("\nQuery: \"%s\"\n", query)

		// Generate query embedding
		queryEmb, err := model.EmbedInt8(query)
		if err != nil {
			log.Printf("Failed to embed query: %v", err)
			continue
		}

		// Compute similarities
		var results []SearchResult
		start := time.Now()

		for i, docEmb := range embeddings {
			if docEmb == nil {
				continue
			}

			// Compute int8 dot product
			score := int32(0)
			for j := 0; j < 512; j++ {
				score += int32(queryEmb[j]) * int32(docEmb[j])
			}

			results = append(results, SearchResult{
				Index: i,
				Score: float32(score),
				Text:  docs[i],
			})
		}

		elapsed := time.Since(start)

		// Sort by score (descending)
		sort.Slice(results, func(i, j int) bool {
			return results[i].Score > results[j].Score
		})

		// Show top 3 results
		fmt.Printf("Time: %.3fμs\n", float64(elapsed.Nanoseconds())/1000.0)
		for i := 0; i < 3 && i < len(results); i++ {
			fmt.Printf("  %d. %.1f - %s\n", i+1, results[i].Score, results[i].Text)
		}
	}
}