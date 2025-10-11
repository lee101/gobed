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
	"time"
	"unsafe"
)

func main() {
	fmt.Println("🧪 Correctness Test")
	fmt.Println("===================")

	// Small test parameters for debugging
	vocabSize := 10
	embedDim := 64
	numVectors := 5
	topK := 5

	fmt.Printf("Testing: vocab=%d, vectors=%d, dim=%d\n", vocabSize, numVectors, embedDim)

	// Create simple, predictable embeddings
	embedWeights := make([]int8, vocabSize*embedDim)
	embedScales := make([]float32, vocabSize)

	// Simple vocabulary: tokens 0-9
	// Token 0: all zeros
	// Token 1: all ones
	// Token 2: all twos
	// etc.
	for i := 0; i < vocabSize; i++ {
		embedScales[i] = 1.0 // Simple scale
		for j := 0; j < embedDim; j++ {
			embedWeights[i*embedDim+j] = int8(i * 10) // Distinct patterns
		}
	}

	// Create database with known entries
	database := make([]int8, numVectors*embedDim)
	dbScales := make([]float32, numVectors)

	// Database entry 0: corresponds to query [1, 2] (tokens 1,2)
	// Database entry 1: corresponds to query [3, 4] (tokens 3,4)
	// Database entry 2: corresponds to query [1, 2] again (exact duplicate)
	// Database entry 3: corresponds to query [5] (token 5)
	// Database entry 4: corresponds to query [1, 3] (tokens 1,3)

	queries := [][]uint16{
		{1, 2},    // Should match database[0] and database[2]
		{3, 4},    // Should match database[1]
		{5},       // Should match database[3]
		{1, 3},    // Should match database[4]
		{9},       // Should not match anything well
	}

	for i := 0; i < numVectors; i++ {
		dbScales[i] = 1.0

		var queryTokens []uint16
		if i < len(queries) {
			queryTokens = queries[i]
		} else {
			queryTokens = []uint16{0} // Default
		}

		// Simulate average of token embeddings for this database entry
		for j := 0; j < embedDim; j++ {
			sum := int32(0)
			for _, token := range queryTokens {
				if int(token) < vocabSize {
					sum += int32(embedWeights[int(token)*embedDim+j])
				}
			}
			if len(queryTokens) > 0 {
				database[i*embedDim+j] = int8(sum / int32(len(queryTokens)))
			}
		}
	}

	fmt.Print("Creating context: ")
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

	// Test each query
	for testIdx, query := range queries {
		fmt.Printf("\nTest %d: Query %v\n", testIdx+1, query)

		// Pad query to max tokens
		maxTokens := 10
		paddedTokens := make([]uint16, maxTokens)
		copy(paddedTokens, query)
		tokenLengths := []int{len(query)}

		// Output buffers
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
		exactMatch := false
		for i := 0; i < topK && i < len(outputDistances); i++ {
			if outputIndices[i] >= 0 && outputIndices[i] < numVectors {
				fmt.Printf("  %d: ID=%d, distance=%.4f", i+1, outputIndices[i], outputDistances[i])

				// Check if this should be an exact match
				if testIdx < numVectors && (outputIndices[i] == testIdx ||
					(testIdx == 0 && outputIndices[i] == 2)) { // Query 0 matches both DB 0 and 2
					if outputDistances[i] < 0.1 {
						fmt.Printf(" ✓ EXACT MATCH")
						exactMatch = true
					} else {
						fmt.Printf(" ⚠ SHOULD BE EXACT MATCH")
					}
				}
				fmt.Println()
			} else {
				fmt.Printf("  %d: ID=%d (invalid), distance=%.4f\n", i+1, outputIndices[i], outputDistances[i])
			}
		}

		if testIdx < numVectors && !exactMatch {
			fmt.Printf("  ❌ Expected exact match for database entry %d\n", testIdx)
		}
	}

	fmt.Println("\n✅ Correctness test completed")
}
