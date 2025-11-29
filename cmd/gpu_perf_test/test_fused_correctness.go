//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	fmt.Println("🧪 Fused CAGRA Correctness Test")
	fmt.Println("===============================")

	// Test configuration - small for debugging
	config := gobed.FusedCAGRAConfig{
		VocabSize:   1000,
		EmbedDim:    512,
		MaxVectors:  20,
		TopK:        5,
		GraphDegree: 8, // Small graph for debugging
	}

	fmt.Printf("Config: vocab=%d, dim=%d, vectors=%d, top_k=%d\n",
		config.VocabSize, config.EmbedDim, config.MaxVectors, config.TopK)

	// Create engine
	fmt.Print("Creating FusedCAGRAEngine: ")
	engine, err := gobed.NewFusedCAGRAEngine(config)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	fmt.Println("OK")

	// Create simple, predictable embeddings for testing
	embedWeights, embedScales := createTestEmbeddings(config.VocabSize, config.EmbedDim)

	// Create test database with known content
	testTexts := []string{
		"hello world",      // DB[0] - tokens [1, 2]
		"hello friend",     // DB[1] - tokens [1, 3]
		"goodbye world",    // DB[2] - tokens [4, 2]
		"goodbye friend",   // DB[3] - tokens [4, 3]
		"hello world",      // DB[4] - exact duplicate of DB[0]
		"the quick brown",  // DB[5] - tokens [5, 6, 7]
		"hello",            // DB[6] - token [1] only
		"world",            // DB[7] - token [2] only
		"friend",           // DB[8] - token [3] only
		"goodbye",          // DB[9] - token [4] only
	}

	database, dbScales := createTestDatabase(testTexts, embedWeights, embedScales, config.EmbedDim)

	fmt.Printf("Created test database with %d entries\n", len(database))
	for i, text := range testTexts {
		fmt.Printf("  DB[%d]: %q\n", i, text)
	}

	// Build index
	fmt.Print("Building index: ")
	err = engine.BuildIndex(embedWeights, embedScales, database, dbScales)
	if err != nil {
		log.Fatalf("Failed to build index: %v", err)
	}
	fmt.Println("OK")

	// Test queries with expected results
	testQueries := []struct {
		text           string
		tokens         []uint16
		expectedExact  []int  // DB indices that should be exact matches
		expectedSimilar []int // DB indices that should be similar
	}{
		{
			text:           "hello world",
			tokens:         []uint16{1, 2},
			expectedExact:  []int{0, 4}, // Exact duplicates
			expectedSimilar: []int{6, 7}, // "hello" and "world" individually
		},
		{
			text:           "hello friend",
			tokens:         []uint16{1, 3},
			expectedExact:  []int{1},     // Exact match
			expectedSimilar: []int{6, 8}, // "hello" and "friend" individually
		},
		{
			text:           "goodbye world",
			tokens:         []uint16{4, 2},
			expectedExact:  []int{2},     // Exact match
			expectedSimilar: []int{7, 9}, // "world" and "goodbye" individually
		},
		{
			text:           "hello",
			tokens:         []uint16{1},
			expectedExact:  []int{6},     // Exact match
			expectedSimilar: []int{0, 1, 4}, // Contains "hello"
		},
		{
			text:           "world",
			tokens:         []uint16{2},
			expectedExact:  []int{7},     // Exact match
			expectedSimilar: []int{0, 2, 4}, // Contains "world"
		},
		{
			text:           "unknown",
			tokens:         []uint16{999}, // Unknown token
			expectedExact:  []int{},       // No exact matches
			expectedSimilar: []int{},      // Should return something but we don't know what
		},
	}

	fmt.Println("\nRunning correctness tests:")
	fmt.Println(strings.Repeat("=", 50))

	passCount := 0
	totalTests := len(testQueries)

	for i, test := range testQueries {
		fmt.Printf("\nTest %d: %q\n", i+1, test.text)
		fmt.Printf("Tokens: %v\n", test.tokens)

		// Search
		start := time.Now()
		results, err := engine.Search(test.tokens)
		searchTime := time.Since(start)

		if err != nil {
			fmt.Printf("❌ ERROR: %v\n", err)
			continue
		}

		fmt.Printf("Results (%v):\n", searchTime)

		// Check results
		exactFound := make(map[int]bool)
		similarFound := make(map[int]bool)
		validResults := 0

		for j, result := range results {
			if result.ID >= 0 && result.ID < len(testTexts) {
				validResults++
				fmt.Printf("  %d: DB[%d] %q (dist=%.4f)",
					j+1, result.ID, testTexts[result.ID], result.Similarity)

				// Check for exact matches (very low distance)
				isExact := false
				for _, expectedID := range test.expectedExact {
					if result.ID == expectedID && result.Similarity < 0.1 {
						exactFound[expectedID] = true
						isExact = true
						fmt.Printf(" ✓ EXACT")
						break
					}
				}

				// Check for similar matches
				if !isExact {
					for _, expectedID := range test.expectedSimilar {
						if result.ID == expectedID {
							similarFound[expectedID] = true
							fmt.Printf(" ✓ SIMILAR")
							break
						}
					}
				}
				fmt.Println()
			} else {
				fmt.Printf("  %d: INVALID ID=%d (dist=%.4f)\n",
					j+1, result.ID, result.Similarity)
			}
		}

		// Evaluate test
		testPassed := true

		// Check if we found all expected exact matches
		for _, expectedID := range test.expectedExact {
			if !exactFound[expectedID] {
				fmt.Printf("❌ Missing exact match: DB[%d] %q\n",
					expectedID, testTexts[expectedID])
				testPassed = false
			}
		}

		// Check for valid results
		if validResults == 0 {
			fmt.Printf("❌ No valid results returned\n")
			testPassed = false
		}

		// Check distances make sense (not crazy large numbers)
		for _, result := range results {
			if math.Abs(float64(result.Similarity)) > 1000000 {
				fmt.Printf("❌ Unreasonable distance: %.4f\n", result.Similarity)
				testPassed = false
			}
		}

		if testPassed {
			fmt.Printf("✅ Test PASSED\n")
			passCount++
		} else {
			fmt.Printf("❌ Test FAILED\n")
		}
	}

	fmt.Println(strings.Repeat("=", 50))
	fmt.Printf("📊 Results: %d/%d tests passed (%.1f%%)\n",
		passCount, totalTests, float64(passCount)/float64(totalTests)*100)

	if passCount == totalTests {
		fmt.Println("🎉 All tests passed! Fused CAGRA is working correctly.")
	} else {
		fmt.Printf("⚠️  %d tests failed. Check implementation.\n", totalTests-passCount)
	}

	// Show engine stats
	stats := engine.GetStats()
	fmt.Printf("\n📈 Engine Stats:\n")
	fmt.Printf("  Vectors: %d\n", stats.NumVectors)
	fmt.Printf("  Searches: %d\n", stats.SearchCount)
	fmt.Printf("  Avg search time: %.3fms\n", stats.AvgSearchTimeMs)
}

// createTestEmbeddings creates simple, predictable embeddings for testing
func createTestEmbeddings(vocabSize, embedDim int) ([]int8, []float32) {
	embedWeights := make([]int8, vocabSize*embedDim)
	embedScales := make([]float32, vocabSize)

	// Simple vocabulary mapping:
	// Token 0: UNK - random
	// Token 1: "hello" - pattern [10, 10, 10, ...]
	// Token 2: "world" - pattern [20, 20, 20, ...]
	// Token 3: "friend" - pattern [30, 30, 30, ...]
	// Token 4: "goodbye" - pattern [40, 40, 40, ...]
	// Token 5: "the" - pattern [50, 50, 50, ...]
	// Token 6: "quick" - pattern [60, 60, 60, ...]
	// Token 7: "brown" - pattern [70, 70, 70, ...]
	// Rest: random

	for i := 0; i < vocabSize; i++ {
		embedScales[i] = 1.0 // Simple scale

		var pattern int8
		switch i {
		case 0:
			pattern = 0   // UNK
		case 1:
			pattern = 10  // hello
		case 2:
			pattern = 20  // world
		case 3:
			pattern = 30  // friend
		case 4:
			pattern = 40  // goodbye
		case 5:
			pattern = 50  // the
		case 6:
			pattern = 60  // quick
		case 7:
			pattern = 70  // brown
		default:
			pattern = int8((i % 100) - 50) // Various patterns
		}

		for j := 0; j < embedDim; j++ {
			embedWeights[i*embedDim+j] = pattern
		}
	}

	return embedWeights, embedScales
}

// createTestDatabase creates database vectors from test texts
func createTestDatabase(texts []string, embedWeights []int8, embedScales []float32, embedDim int) ([]simd.Vec512, []float32) {
	// Simple tokenizer mapping
	tokenMap := map[string]uint16{
		"hello":   1,
		"world":   2,
		"friend":  3,
		"goodbye": 4,
		"the":     5,
		"quick":   6,
		"brown":   7,
	}

	database := make([]simd.Vec512, len(texts))
	dbScales := make([]float32, len(texts))

	for i, text := range texts {
		dbScales[i] = 1.0

		// Tokenize text
		words := strings.Fields(strings.ToLower(text))
		tokens := make([]uint16, 0, len(words))
		for _, word := range words {
			if token, exists := tokenMap[word]; exists {
				tokens = append(tokens, token)
			} else {
				tokens = append(tokens, 0) // UNK
			}
		}

		// Create embedding by averaging token embeddings
		var embedding simd.Vec512
		if len(tokens) > 0 {
			// Accumulate embeddings
			accumulator := make([]float32, embedDim)
			for _, token := range tokens {
				if int(token)*embedDim < len(embedWeights) {
					scale := embedScales[token]
					for j := 0; j < embedDim; j++ {
						accumulator[j] += float32(embedWeights[int(token)*embedDim+j]) * scale
					}
				}
			}

			// Average and quantize
			invCount := 1.0 / float32(len(tokens))
			for j := 0; j < embedDim; j++ {
				val := accumulator[j] * invCount
				if val > 127 {
					embedding[j] = 127
				} else if val < -128 {
					embedding[j] = -128
				} else {
					embedding[j] = int8(val)
				}
			}
		}

		database[i] = embedding
	}

	return database, dbScales
}
