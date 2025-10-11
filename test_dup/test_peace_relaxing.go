//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"log"
	"strings"
	"io/ioutil"

	"github.com/lee101/gobed"
)

func main() {
	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal("Failed to load model:", err)
	}

	// Create search engine
	engine := gobed.NewSearchEngine(model)

	// Load and index ai.txt
	fmt.Println("=== Loading ai.txt ===")
	aiContent, err := ioutil.ReadFile("../ai.txt")
	if err != nil {
		log.Fatal("Failed to read ai.txt:", err)
	}

	lines := strings.Split(string(aiContent), "\n")
	fmt.Printf("Found %d lines in ai.txt\n", len(lines))

	// Index first 100 lines
	docIDs := make([]int, 0)
	for i := 0; i < 100 && i < len(lines); i++ {
		if lines[i] != "" {
			id, err := engine.Index(lines[i])
			if err != nil {
				log.Printf("Failed to index line %d: %v\n", i, err)
				continue
			}
			docIDs = append(docIDs, id)
		}
	}
	fmt.Printf("Indexed %d documents\n\n", len(docIDs))

	// Test queries for peace and relaxing
	queries := []string{"peace", "relaxing", "peaceful", "relax", "tranquil", "meditation"}

	fmt.Println("=== Testing Peace and Relaxing Searches ===\n")

	for _, query := range queries {
		results, err := engine.Search(query, 5)
		if err != nil {
			log.Printf("Search failed for '%s': %v\n", query, err)
			continue
		}

		fmt.Printf("Query: '%s'\n", query)
		fmt.Println("Results:")

		foundPeace := false
		foundRelaxing := false

		for i, result := range results {
			// Find the content
			content := ""
			for idx, id := range docIDs {
				if id == result.ID && idx < len(lines) {
					content = lines[idx]
					break
				}
			}

			// Check if it contains peace or relaxing
			lowerContent := strings.ToLower(content)
			if strings.Contains(lowerContent, "peace") || strings.Contains(lowerContent, "peaceful") {
				foundPeace = true
			}
			if strings.Contains(lowerContent, "relax") {
				foundRelaxing = true
			}

			// Truncate for display
			if len(content) > 100 {
				content = content[:100] + "..."
			}

			marker := ""
			if strings.Contains(strings.ToLower(content), "peace") {
				marker = " [PEACE]"
			}
			if strings.Contains(strings.ToLower(content), "relax") {
				marker = " [RELAX]"
			}

			fmt.Printf("  %d. (sim: %.4f)%s %s\n", i+1, result.Similarity, marker, content)
		}

		if query == "peace" && !foundPeace {
			fmt.Println("  ❌ No peace-related content found!")
		}
		if query == "relaxing" && !foundRelaxing {
			fmt.Println("  ❌ No relaxing-related content found!")
		}

		fmt.Println()
	}

	// Test that the first two entries (peace and relaxing) rank high
	fmt.Println("=== Verifying New Content Ranks High ===\n")

	// Search for exact content
	peaceQuery := "Finding inner peace through AI-assisted meditation"
	results, err := engine.Search(peaceQuery, 10)
	if err == nil {
		fmt.Printf("Query: '%s'\n", peaceQuery)
		foundAtRank := -1
		for i, result := range results {
			for idx, id := range docIDs {
				if id == result.ID && idx == 0 { // First line is the peace article
					foundAtRank = i + 1
					break
				}
			}
		}
		if foundAtRank == 1 {
			fmt.Printf("✓ Peace article ranked #%d (top result!)\n", foundAtRank)
		} else if foundAtRank > 0 {
			fmt.Printf("⚠️  Peace article ranked #%d\n", foundAtRank)
		} else {
			fmt.Println("❌ Peace article not in top 10")
		}
	}

	relaxQuery := "Creating relaxing ambient music with generative AI"
	results, err = engine.Search(relaxQuery, 10)
	if err == nil {
		fmt.Printf("\nQuery: '%s'\n", relaxQuery)
		foundAtRank := -1
		for i, result := range results {
			for idx, id := range docIDs {
				if id == result.ID && idx == 1 { // Second line is the relaxing article
					foundAtRank = i + 1
					break
				}
			}
		}
		if foundAtRank == 1 {
			fmt.Printf("✓ Relaxing article ranked #%d (top result!)\n", foundAtRank)
		} else if foundAtRank > 0 {
			fmt.Printf("⚠️  Relaxing article ranked #%d\n", foundAtRank)
		} else {
			fmt.Println("❌ Relaxing article not in top 10")
		}
	}
}
