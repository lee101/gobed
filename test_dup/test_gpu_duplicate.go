//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"log"
	"os"

	"github.com/lee101/gobed"
)

func main() {
	// Force GPU mode
	os.Setenv("GOBED_USE_GPU", "1")
	os.Setenv("CUDA_VISIBLE_DEVICES", "0")

	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal("Failed to load model:", err)
	}

	// Create GPU search engine
	config := gobed.GPUSearchConfig()
	engine := gobed.NewSearchEngineWithConfig(model, config)

	// Simple test content
	testContent := []string{
		"Finding inner peace through meditation",
		"CUDA kernels optimize GPU performance",
		"Machine learning models require tuning",
		"Peaceful gardens create serenity",
		"Database indexing improves queries",
	}

	// Index content
	fmt.Println("=== Indexing with GPU ===")
	docIDs := make([]int, 0)
	for _, content := range testContent {
		id, err := engine.Index(content)
		if err != nil {
			log.Printf("Failed to index: %v\n", err)
			continue
		}
		docIDs = append(docIDs, id)
	}

	// Test different queries
	queries := []string{"peace", "CUDA", "database", "garden", "machine"}

	fmt.Println("\n=== GPU Search Results ===")
	allResults := make([][]int, 0)

	for _, query := range queries {
		results, err := engine.Search(query, 3)
		if err != nil {
			log.Printf("Search failed for '%s': %v\n", query, err)
			continue
		}

		fmt.Printf("\nQuery: '%s'\n", query)
		resultIDs := make([]int, 0)
		for i, result := range results {
			resultIDs = append(resultIDs, result.ID)

			// Get content
			content := ""
			for idx, id := range docIDs {
				if id == result.ID && idx < len(testContent) {
					content = testContent[idx]
					break
				}
			}

			fmt.Printf("  %d. [ID:%d] (sim: %.4f) %s\n",
				i+1, result.ID, result.Similarity, content)
		}
		allResults = append(allResults, resultIDs)
	}

	// Check for duplicate bug
	fmt.Println("\n=== Duplicate Bug Check (GPU) ===")
	duplicateIssue := true
	if len(allResults) > 1 {
		firstResults := allResults[0]
		for i := 1; i < len(allResults); i++ {
			if len(allResults[i]) != len(firstResults) {
				duplicateIssue = false
				break
			}
			for j := range allResults[i] {
				if allResults[i][j] != firstResults[j] {
					duplicateIssue = false
					break
				}
			}
			if !duplicateIssue {
				break
			}
		}
	}

	if duplicateIssue {
		fmt.Println("❌ BUG DETECTED: All GPU queries return identical results!")
		fmt.Printf("All queries return IDs: %v\n", allResults[0])
	} else {
		fmt.Println("✓ GPU search returns different results for different queries")
	}
}
