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
	"math/rand"
	"time"
	"unsafe"
)

func main() {
	fmt.Println("🧪 Simple Kernel Test")
	fmt.Println("====================")

	// Test parameters
	vocabSize := 1000
	embedDim := 512
	numVectors := 100
	topK := 10

	fmt.Printf("Testing: vocab=%d, vectors=%d, dim=%d\n", vocabSize, numVectors, embedDim)

	// Generate test data
	embedWeights := make([]int8, vocabSize*embedDim)
	embedScales := make([]float32, vocabSize)
	database := make([]int8, numVectors*embedDim)
	dbScales := make([]float32, numVectors)

	// Fill with random data
	for i := 0; i < vocabSize; i++ {
		embedScales[i] = 0.05
		for j := 0; j < embedDim; j++ {
			embedWeights[i*embedDim+j] = int8(rand.Intn(255) - 128)
		}
	}

	for i := 0; i < numVectors; i++ {
		dbScales[i] = 0.05
		for j := 0; j < embedDim; j++ {
			database[i*embedDim+j] = int8(rand.Intn(255) - 128)
		}
	}

	// Create context
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

	// Test search
	fmt.Print("Testing search: ")

	// Create test query
	tokenBatch := []uint16{1, 2, 3, 4, 5}
	tokenLengths := []int{5}
	maxTokens := 10

	// Pad token batch
	paddedTokens := make([]uint16, maxTokens)
	copy(paddedTokens, tokenBatch)

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
	fmt.Printf("OK (%v)\n", searchTime)

	// Print results
	fmt.Println("\nTop results:")
	for i := 0; i < topK && i < len(outputDistances); i++ {
		if outputIndices[i] >= 0 {
			fmt.Printf("  %d: ID=%d, distance=%.4f\n", i+1, outputIndices[i], outputDistances[i])
		}
	}

	fmt.Printf("\n✅ Test completed successfully in %v\n", searchTime)
}
