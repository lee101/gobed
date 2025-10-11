package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"
)

// Test real semantic search with actual ai.txt content
func main() {
	fmt.Println("Real Content Search Test - ai.txt")

	// Load model
	model, err := LoadFastModel("../model/modelint8_512dim.safetensors", "../model/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Load real ai.txt content
	file, err := os.Open("testdata/ai.txt")
	if err != nil {
		log.Fatalf("Failed to open ai.txt: %v", err)
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	lineCount := 0
	maxLines := 1000 // Test with first 1000 lines for speed

	for scanner.Scan() && lineCount < maxLines {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			lines = append(lines, line)
			lineCount++
		}
	}

	fmt.Printf("Loaded %d lines from ai.txt\n", len(lines))

	// Test queries that should find exact matches as top1
	testQueries := []struct {
		query    string
		expected string // What word should be found in top result
	}{
		{"art", "art"},           // Should find "artstation-art" files
		{"friend", "friend"},     // Should find "girlfriend" or "friendly" files
		{"fun", "fun"},           // Should find "fun" in filenames
		{"anime", "anime"},       // Should find "anime" files
		{"dragon", "dragon"},     // Should find dragon references
	}

	fmt.Println("\nTesting exact word matches...")

	for _, test := range testQueries {
		fmt.Printf("\nQuery: \"%s\"\n", test.query)

		// Generate query embedding
		queryEmb, err := model.EmbedInt8(test.query)
		if err != nil {
			log.Printf("Failed to embed query: %v", err)
			continue
		}

		// Search through all lines
		type Result struct {
			Index int
			Score int32
			Text  string
		}

		var results []Result
		start := time.Now()

		for i, line := range lines {
			// Generate line embedding
			lineEmb, err := model.EmbedInt8(line)
			if err != nil {
				continue
			}

			// Compute similarity
			score := int32(0)
			for j := 0; j < 512; j++ {
				score += int32(queryEmb[j]) * int32(lineEmb[j])
			}

			results = append(results, Result{
				Index: i,
				Score: score,
				Text:  line,
			})
		}

		elapsed := time.Since(start)

		// Sort by score (descending)
		sort.Slice(results, func(i, j int) bool {
			return results[i].Score > results[j].Score
		})

		// Check results
		fmt.Printf("Time: %.2fms\n", float64(elapsed.Microseconds())/1000.0)

		topResult := results[0]
		containsExpected := strings.Contains(strings.ToLower(topResult.Text), test.expected)

		fmt.Printf("Top result (score: %d):\n", topResult.Score)
		fmt.Printf("  %s\n", topResult.Text)

		if containsExpected {
			fmt.Printf("✓ PASS: Found '%s' in top result\n", test.expected)
		} else {
			fmt.Printf("✗ FAIL: Expected '%s' not found in top result\n", test.expected)
		}

		// Show top 3 for context
		fmt.Printf("Top 3 results:\n")
		for i := 0; i < 3 && i < len(results); i++ {
			containsWord := strings.Contains(strings.ToLower(results[i].Text), test.expected)
			marker := "  "
			if containsWord {
				marker = "→ "
			}
			fmt.Printf("%s%d. %d - %s\n", marker, i+1, results[i].Score,
				truncate(results[i].Text, 80))
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("Real semantic search test completed!")
	fmt.Printf("Tested against %d real lines from ai.txt\n", len(lines))
	fmt.Printf("Each query should find exact word matches as top results\n")
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}