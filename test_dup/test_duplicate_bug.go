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
	model, err := gobed.LoadModel("all-MiniLM-L6-v2", &gobed.ModelOptions{
		EmbeddingSize: 384,
		MaxTokens:     512,
		GPULayers:     99,
		Threads:       8,
	})
	if err != nil {
		log.Fatal("Failed to load model:", err)
	}
	defer model.Close()

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
	aiContent, err := ioutil.ReadFile("ai.txt")
	if err == nil {
		lines := strings.Split(string(aiContent), "\n")
		// Add first 30 lines from ai.txt
		for i := 0; i < 30 && i < len(lines); i++ {
			if lines[i] != "" {
				testContent = append(testContent, lines[i])
			}
		}
		fmt.Printf("Loaded %d lines from ai.txt\n", len(testContent))
	}

	// Create search engine with optimal CAGRA config
	engine := gobed.NewSearchEngine(model)

	// Index all content
	fmt.Println("\n=== Indexing Content ===")
	for i, content := range testContent {
		err := engine.AddDocument(fmt.Sprintf("doc_%d", i), content)
		if err != nil {
			log.Printf("Failed to index doc %d: %v\n", i, err)
		}
	}

	// Build index
	err = engine.BuildIndex()
	if err != nil {
		log.Fatal("Failed to build index:", err)
	}
	fmt.Printf("Indexed %d documents\n", len(testContent))

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
	allResults := make([][]string, len(queries))

	for i, query := range queries {
		fmt.Printf("\nQuery: '%s'\n", query)
		results, scores, err := engine.Search(query, 5)
		if err != nil {
			log.Printf("Search failed for '%s': %v\n", query, err)
			continue
		}

		allResults[i] = results

		fmt.Println("Results:")
		for j, docID := range results {
			// Get content preview
			content := ""
			idx := -1
			for k, c := range testContent {
				if fmt.Sprintf("doc_%d", k) == docID {
					idx = k
					content = c
					break
				}
			}

			if len(content) > 80 {
				content = content[:80] + "..."
			}

			fmt.Printf("  %d. [%s] (score: %.4f) %s\n",
				j+1, docID, scores[j], content)
		}
	}

	// Check for duplicate bug
	fmt.Println("\n=== Duplicate Bug Analysis ===")
	duplicateIssue := true
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

	if duplicateIssue {
		fmt.Println("❌ CRITICAL BUG: All queries return identical results!")
		fmt.Println("First query results:", firstResults)
	} else {
		fmt.Println("✓ Different queries return different results")
	}

	// Check semantic quality for peace/relaxing
	fmt.Println("\n=== Semantic Search Quality ===")

	// Peace query should find peace-related content
	peaceResults, _, _ := engine.Search("peace", 10)
	peaceMatches := 0
	for _, docID := range peaceResults {
		for i, content := range testContent[:10] { // Check first 10 (peace/relaxing content)
			if fmt.Sprintf("doc_%d", i) == docID {
				if strings.Contains(strings.ToLower(content), "peace") ||
				   strings.Contains(strings.ToLower(content), "peaceful") {
					peaceMatches++
				}
				break
			}
		}
	}
	fmt.Printf("Peace query: %d/%d results are peace-related\n", peaceMatches, len(peaceResults))

	// Relaxing query should find relaxation content
	relaxResults, _, _ := engine.Search("relaxing", 10)
	relaxMatches := 0
	for _, docID := range relaxResults {
		for i, content := range testContent[:10] {
			if fmt.Sprintf("doc_%d", i) == docID {
				if strings.Contains(strings.ToLower(content), "relax") {
					relaxMatches++
				}
				break
			}
		}
	}
	fmt.Printf("Relaxing query: %d/%d results are relaxation-related\n", relaxMatches, len(relaxResults))

	// Check for duplicates within single query results
	fmt.Println("\n=== Checking for Duplicates in Individual Results ===")
	for i, query := range queries[:3] { // Check first 3 queries
		seen := make(map[string]bool)
		duplicates := 0
		for _, docID := range allResults[i] {
			if seen[docID] {
				duplicates++
			}
			seen[docID] = true
		}
		if duplicates > 0 {
			fmt.Printf("Query '%s': Found %d duplicate results!\n", query, duplicates)
		} else {
			fmt.Printf("Query '%s': No duplicates (✓)\n", query)
		}
	}

	// Test exact match quality
	fmt.Println("\n=== Exact Match Quality ===")
	exactQuery := "Finding inner peace through meditation"
	exactResults, exactScores, _ := engine.Search(exactQuery, 5)
	if len(exactResults) > 0 {
		fmt.Printf("Exact query: '%s'\n", exactQuery)
		fmt.Printf("Top result: %s (score: %.4f)\n", exactResults[0], exactScores[0])
		if exactResults[0] == "doc_0" {
			fmt.Println("✓ Exact match ranked #1")
		} else {
			fmt.Println("❌ Exact match not ranked #1")
		}
	}
}
