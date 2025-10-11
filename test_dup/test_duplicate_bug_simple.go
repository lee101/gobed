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

	// Test content with peace/relaxing themes
	testContent := []string{
		// Peace-related
		"Finding inner peace through meditation and mindfulness brings tranquility to life",
		"Peaceful gardens with flowing water create a serene atmosphere for relaxation",
		"The calm peaceful morning by the lake was perfect for quiet reflection",
		"World peace requires understanding, compassion and dialogue between nations",
		"A peaceful mind leads to better decisions and clearer thinking",

		// Relaxing-related
		"Relaxing on the beach with waves gently lapping creates ultimate tranquility",
		"Deep breathing exercises help in relaxing tense muscles and calming the mind",
		"A relaxing massage can relieve stress and promote overall wellbeing",
		"Soft relaxing music helps create a comfortable environment for rest",
		"Taking time for relaxing activities is essential for mental health",

		// Technical content (different domain)
		"CUDA kernels optimize GPU memory access patterns for better performance",
		"Machine learning models require careful hyperparameter tuning for accuracy",
		"Database indexing improves query performance by reducing search time",
		"Neural networks use backpropagation to adjust weights during training",
		"API rate limiting prevents server overload and ensures service stability",

		// Mixed content
		"Finding peaceful solutions to technical problems requires patience",
		"A relaxing coding environment improves developer productivity",
		"Stress-free programming leads to fewer bugs and better code",
		"Calm debugging sessions are more effective than rushed fixes",
		"Peaceful collaboration between teams creates better software",
	}

	// Load ai.txt content if it exists
	aiContent, err := ioutil.ReadFile("../ai.txt")
	if err == nil {
		lines := strings.Split(string(aiContent), "\n")
		// Add first 30 lines from ai.txt
		for i := 0; i < 30 && i < len(lines); i++ {
			if lines[i] != "" {
				testContent = append(testContent, lines[i])
			}
		}
		fmt.Printf("Loaded %d additional lines from ai.txt\n", len(lines))
	}

	// Create search engine with optimal CAGRA config
	engine := gobed.NewSearchEngine(model)

	// Index all content and keep track of IDs
	fmt.Println("\n=== Indexing Content ===")
	docIDs := make([]int, 0, len(testContent))
	for _, content := range testContent {
		id, err := engine.Index(content)
		if err != nil {
			log.Printf("Failed to index: %v\n", err)
			continue
		}
		docIDs = append(docIDs, id)
	}
	fmt.Printf("Indexed %d documents\n", len(docIDs))

	// Test queries
	queries := []string{
		"peace",
		"relaxing",
		"peaceful",
		"relaxation",
		"calm",
		"tranquil",
		"stress relief",
		"meditation",
		"technical",
		"programming",
		"CUDA",
		"neural network",
	}

	fmt.Println("\n=== Testing for Duplicate Bug ===")

	// Store results for duplicate detection
	type queryResults struct {
		query string
		ids   []int
		texts []string
	}
	allResults := make([]queryResults, 0, len(queries))

	for _, query := range queries {
		fmt.Printf("\nQuery: '%s'\n", query)
		results, err := engine.Search(query, 5)
		if err != nil {
			log.Printf("Search failed for '%s': %v\n", query, err)
			continue
		}

		qr := queryResults{
			query: query,
			ids:   make([]int, 0),
			texts: make([]string, 0),
		}

		fmt.Println("Results:")
		for j, result := range results {
			// Get the actual text content
			text := ""
			for idx, id := range docIDs {
				if id == result.ID {
					text = testContent[idx]
					break
				}
			}

			if len(text) > 80 {
				text = text[:80] + "..."
			}

			qr.ids = append(qr.ids, result.ID)
			qr.texts = append(qr.texts, text)

			fmt.Printf("  %d. [ID:%d] (sim: %.4f) %s\n",
				j+1, result.ID, result.Similarity, text)
		}

		allResults = append(allResults, qr)
	}

	// Check for duplicate bug
	fmt.Println("\n=== Duplicate Bug Analysis ===")
	if len(allResults) > 1 {
		duplicateIssue := true
		firstResults := allResults[0].ids

		for i := 1; i < len(allResults); i++ {
			if len(allResults[i].ids) != len(firstResults) {
				duplicateIssue = false
				break
			}

			for j := range allResults[i].ids {
				if allResults[i].ids[j] != firstResults[j] {
					duplicateIssue = false
					break
				}
			}
			if !duplicateIssue {
				break
			}
		}

		if duplicateIssue {
			fmt.Println("❌ CRITICAL BUG: All queries return identical results!")
			fmt.Printf("All queries return: %v\n", firstResults)
		} else {
			fmt.Println("✓ Different queries return different results")
		}
	}

	// Check semantic quality for peace/relaxing
	fmt.Println("\n=== Semantic Search Quality ===")

	// Peace query should find peace-related content
	peaceResults, _ := engine.Search("peace", 10)
	peaceMatches := 0
	for _, result := range peaceResults {
		for idx, id := range docIDs {
			if id == result.ID && idx < 10 { // Check first 10 (peace/relaxing content)
				text := testContent[idx]
				if strings.Contains(strings.ToLower(text), "peace") ||
				   strings.Contains(strings.ToLower(text), "peaceful") {
					peaceMatches++
				}
				break
			}
		}
	}
	fmt.Printf("Peace query: %d/%d results are peace-related\n", peaceMatches, len(peaceResults))

	// Relaxing query should find relaxation content
	relaxResults, _ := engine.Search("relaxing", 10)
	relaxMatches := 0
	for _, result := range relaxResults {
		for idx, id := range docIDs {
			if id == result.ID && idx < 10 {
				text := testContent[idx]
				if strings.Contains(strings.ToLower(text), "relax") {
					relaxMatches++
				}
				break
			}
		}
	}
	fmt.Printf("Relaxing query: %d/%d results are relaxation-related\n", relaxMatches, len(relaxResults))

	// Check for duplicates within single query results
	fmt.Println("\n=== Checking for Duplicates in Individual Results ===")
	for _, qr := range allResults[:3] { // Check first 3 queries
		seen := make(map[int]bool)
		duplicates := 0
		for _, id := range qr.ids {
			if seen[id] {
				duplicates++
			}
			seen[id] = true
		}
		if duplicates > 0 {
			fmt.Printf("Query '%s': Found %d duplicate results!\n", qr.query, duplicates)
		} else {
			fmt.Printf("Query '%s': No duplicates (✓)\n", qr.query)
		}
	}

	// Test exact match quality
	fmt.Println("\n=== Exact Match Quality ===")
	exactQuery := "Finding inner peace through meditation"
	exactResults, _ := engine.Search(exactQuery, 5)
	if len(exactResults) > 0 {
		fmt.Printf("Exact query: '%s'\n", exactQuery)
		fmt.Printf("Top result ID: %d (sim: %.4f)\n", exactResults[0].ID, exactResults[0].Similarity)

		// Check if top result is the exact match
		topText := ""
		for idx, id := range docIDs {
			if id == exactResults[0].ID {
				topText = testContent[idx]
				break
			}
		}

		if strings.HasPrefix(topText, exactQuery) {
			fmt.Println("✓ Exact match ranked #1")
		} else {
			fmt.Printf("❌ Exact match not ranked #1\nTop result: %s\n", topText)
		}
	}
}
