//go:build legacy
// +build legacy

package main

/*
#cgo CFLAGS: -I../../
#cgo LDFLAGS: -L../../ -lsimple_cagra -L/usr/local/cuda/lib64 -lcudart -ldl -lstdc++ -lm
#include <stdlib.h>
#include <stdint.h>

void* create_simple_context(
    int8_t* embed_weights,
    float* embed_scales_raw,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales_raw,
    int num_vectors,
    int top_k);

void simple_search(
    void* context,
    uint16_t* token_batch,
    int* token_lengths,
    int batch_size,
    int max_tokens,
    float* output_distances,
    int* output_indices);

void destroy_simple_context(void* context);
*/
import "C"
import (
	"fmt"
	"math"
	"strings"
	"time"
	"unsafe"
)

// Simple vocabulary for testing
var vocab = map[string]uint16{
	"hello":   1,
	"world":   2,
	"friend":  3,
	"goodbye": 4,
	"good":    5,
	"bad":     6,
	"cat":     7,
	"dog":     8,
	"red":     9,
	"blue":    10,
}

func main() {
	fmt.Println("🧪 Simple Kernel Correctness Test")
	fmt.Println("=================================")

	// Small, manageable test parameters
	vocabSize := 20
	embedDim := 128  // Smaller for easier debugging
	numVectors := 8
	topK := 5

	fmt.Printf("Testing: vocab=%d, vectors=%d, dim=%d\n", vocabSize, numVectors, embedDim)

	// Create predictable embeddings based on simple vocabulary
	embedWeights := make([]int8, vocabSize*embedDim)
	embedScales := make([]float32, vocabSize)

	// Create distinct patterns for each vocabulary token
	for i := 0; i < vocabSize; i++ {
		embedScales[i] = 1.0
		pattern := int8((i % 20) * 6) // Values: 0, 6, 12, 18, ... 114
		for j := 0; j < embedDim; j++ {
			embedWeights[i*embedDim+j] = pattern
		}
	}

	// Create test database with known text content
	testTexts := []string{
		"hello world",    // DB[0] - tokens [1, 2]
		"hello friend",   // DB[1] - tokens [1, 3]
		"goodbye world",  // DB[2] - tokens [4, 2]
		"hello",          // DB[3] - token [1] only
		"world",          // DB[4] - token [2] only
		"hello world",    // DB[5] - exact duplicate of DB[0]
		"good cat",       // DB[6] - tokens [5, 7]
		"bad dog",        // DB[7] - tokens [6, 8]
	}

	database := make([]int8, numVectors*embedDim)
	dbScales := make([]float32, numVectors)

	fmt.Printf("\nDatabase contents:\n")
	for i, text := range testTexts {
		fmt.Printf("  DB[%d]: %q", i, text)

		// Tokenize and create embedding
		tokens := tokenize(text)
		fmt.Printf(" -> tokens %v", tokens)

		dbScales[i] = 1.0

		// Create embedding as average of token embeddings
		if len(tokens) > 0 {
			accumulator := make([]float32, embedDim)
			for _, token := range tokens {
				if int(token) < vocabSize {
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
				database[i*embedDim+j] = clampInt8(val)
			}
		}
		fmt.Printf(" -> pattern [%d, %d, ...]\n", database[i*embedDim], database[i*embedDim+1])
	}

	// Create context
	fmt.Print("\nCreating context: ")
	context := C.create_simple_context(
		(*C.int8_t)(unsafe.Pointer(&embedWeights[0])),
		(*C.float)(unsafe.Pointer(&embedScales[0])),
		C.int(vocabSize),
		C.int(embedDim),
		(*C.int8_t)(unsafe.Pointer(&database[0])),
		(*C.float)(unsafe.Pointer(&dbScales[0])),
		C.int(numVectors),
		C.int(topK),
	)

	if context == nil {
		fmt.Println("ERROR")
		return
	}
	defer C.destroy_simple_context(context)
	fmt.Println("OK")

	// Test queries
	testQueries := []struct {
		text         string
		expectedExact []int // DB indices that should be exact matches
		expectedTop   []int // DB indices that should be in top results
	}{
		{
			text:         "hello world",
			expectedExact: []int{0, 5}, // Exact duplicates
			expectedTop:   []int{0, 5, 1, 3, 4}, // Should include hello, world variants
		},
		{
			text:         "hello friend",
			expectedExact: []int{1},     // Exact match
			expectedTop:   []int{1, 3, 0}, // hello variations
		},
		{
			text:         "hello",
			expectedExact: []int{3},     // Exact match
			expectedTop:   []int{3, 0, 1, 5}, // Contains hello
		},
		{
			text:         "world",
			expectedExact: []int{4},     // Exact match
			expectedTop:   []int{4, 0, 2, 5}, // Contains world
		},
		{
			text:         "goodbye world",
			expectedExact: []int{2},     // Exact match
			expectedTop:   []int{2, 4},  // Should find goodbye world and world
		},
	}

	fmt.Println("\nRunning correctness tests:")
	fmt.Println(strings.Repeat("=", 60))

	passCount := 0
	totalTests := len(testQueries)

	for testIdx, test := range testQueries {
		fmt.Printf("\nTest %d: %q\n", testIdx+1, test.text)

		tokens := tokenize(test.text)
		fmt.Printf("Tokens: %v\n", tokens)

		// Pad tokens
		maxTokens := 10
		paddedTokens := make([]uint16, maxTokens)
		copy(paddedTokens, tokens)
		tokenLengths := []int{len(tokens)}

		// Search
		outputDistances := make([]float32, topK)
		outputIndices := make([]int, topK)

		start := time.Now()
		C.simple_search(
			context,
			(*C.uint16_t)(unsafe.Pointer(&paddedTokens[0])),
			(*C.int)(unsafe.Pointer(&tokenLengths[0])),
			C.int(1), // batch size
			C.int(maxTokens),
			(*C.float)(unsafe.Pointer(&outputDistances[0])),
			(*C.int)(unsafe.Pointer(&outputIndices[0])),
		)
		searchTime := time.Since(start)

		fmt.Printf("Results (%v):\n", searchTime)

		// Analyze results
		validResults := 0
		exactFound := make(map[int]bool)
		topFound := make(map[int]bool)

		for i := 0; i < topK; i++ {
			if outputIndices[i] >= 0 && outputIndices[i] < numVectors {
				validResults++
				dbIdx := outputIndices[i]
				distance := outputDistances[i]

				fmt.Printf("  %d: DB[%d] %q (dist=%.4f)",
					i+1, dbIdx, testTexts[dbIdx], distance)

				// Check for exact matches (distance should be 0 or very small)
				isExact := false
				for _, expectedIdx := range test.expectedExact {
					if dbIdx == expectedIdx {
						if math.Abs(float64(distance)) < 1.0 { // Allow small tolerance
							exactFound[expectedIdx] = true
							isExact = true
							fmt.Printf(" ✓ EXACT")
						} else {
							fmt.Printf(" ⚠ SHOULD BE EXACT (dist=%.4f)", distance)
						}
						break
					}
				}

				// Check if in expected top results
				if !isExact {
					for _, expectedIdx := range test.expectedTop {
						if dbIdx == expectedIdx {
							topFound[expectedIdx] = true
							fmt.Printf(" ✓ EXPECTED")
							break
						}
					}
				}
				fmt.Println()
			} else {
				fmt.Printf("  %d: INVALID ID=%d (dist=%.4f)\n",
					i+1, outputIndices[i], outputDistances[i])
			}
		}

		// Evaluate test
		testPassed := true

		// Check exact matches
		for _, expectedIdx := range test.expectedExact {
			if !exactFound[expectedIdx] {
				fmt.Printf("❌ Missing exact match: DB[%d] %q\n",
					expectedIdx, testTexts[expectedIdx])
				testPassed = false
			}
		}

		// Check for reasonable distances
		for i := 0; i < topK; i++ {
			if outputIndices[i] >= 0 && math.Abs(float64(outputDistances[i])) > 100000 {
				fmt.Printf("❌ Unreasonable distance: %.4f\n", outputDistances[i])
				testPassed = false
			}
		}

		// Check for valid results
		if validResults == 0 {
			fmt.Printf("❌ No valid results\n")
			testPassed = false
		}

		if testPassed {
			fmt.Printf("✅ Test PASSED\n")
			passCount++
		} else {
			fmt.Printf("❌ Test FAILED\n")
		}
	}

	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("📊 Summary: %d/%d tests passed (%.1f%%)\n",
		passCount, totalTests, float64(passCount)/float64(totalTests)*100)

	if passCount == totalTests {
		fmt.Println("🎉 All tests passed! Simple kernel working correctly.")
	} else {
		fmt.Printf("⚠️  %d tests failed. Issues found:\n", totalTests-passCount)
		fmt.Println("  - Check token embeddings are properly distinct")
		fmt.Println("  - Verify distance calculations are reasonable")
		fmt.Println("  - Ensure exact matches return near-zero distance")
	}
}

func tokenize(text string) []uint16 {
	words := strings.Fields(strings.ToLower(text))
	tokens := make([]uint16, 0, len(words))

	for _, word := range words {
		if token, exists := vocab[word]; exists {
			tokens = append(tokens, token)
		} else {
			tokens = append(tokens, 0) // UNK token
		}
	}

	if len(tokens) == 0 {
		tokens = append(tokens, 0) // UNK for empty input
	}

	return tokens
}

func clampInt8(val float32) int8 {
	if val > 127 {
		return 127
	}
	if val < -128 {
		return -128
	}
	return int8(val)
}
