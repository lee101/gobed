//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	fmt.Println("🧪 Simple Fused CAGRA Test")
	fmt.Println("=========================")

	// Small test with just 100 vectors
	n := 100
	fmt.Printf("Testing with %d vectors\n", n)

	// Generate small test data
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)

	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8(rand.Intn(255) - 128)
		}
		scales[i] = 0.05 // Fixed scale for simplicity
	}

	// Create minimal fused CAGRA engine
	config := gobed.DefaultFusedCAGRAConfig()
	config.VocabSize = 1000 // Much smaller vocab for testing
	config.MaxVectors = n

	fmt.Print("Creating fused engine: ")
	engine, err := gobed.NewFusedCAGRAEngine(config)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		return
	}
	defer engine.Close()
	fmt.Println("OK")

	// Generate minimal embedding weights
	embedWeights := make([]int8, config.VocabSize*config.EmbedDim)
	embedScales := make([]float32, config.VocabSize)
	for i := 0; i < config.VocabSize; i++ {
		embedScales[i] = 0.05 // Fixed scale
		for j := 0; j < config.EmbedDim; j++ {
			embedWeights[i*config.EmbedDim+j] = int8(rand.Intn(255) - 128)
		}
	}

	// Test indexing
	fmt.Print("Building index: ")
	start := time.Now()
	err = engine.BuildIndex(embedWeights, embedScales, vectors, scales)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		return
	}
	fmt.Printf("OK (%v)\n", time.Since(start))

	// Test simple search with timeout
	fmt.Print("Testing search: ")

	// Create a simple query
	tokens := []uint16{1, 2, 3, 4, 5} // Simple token sequence

	// Use a channel to implement timeout
	resultChan := make(chan error, 1)

	go func() {
		searchStart := time.Now()
		_, err := engine.Search(tokens)
		searchTime := time.Since(searchStart)

		if err != nil {
			resultChan <- fmt.Errorf("search failed: %v", err)
		} else {
			fmt.Printf("OK (%v)", searchTime)
			resultChan <- nil
		}
	}()

	// Wait for result or timeout
	select {
	case err := <-resultChan:
		if err != nil {
			fmt.Printf("ERROR: %v\n", err)
		} else {
			fmt.Println()
		}
	case <-time.After(10 * time.Second):
		fmt.Println("TIMEOUT - search took too long")
		return
	}

	fmt.Println("✅ Basic test completed successfully")
}
