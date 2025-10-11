//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	fmt.Println("🧪 Debug Fused CAGRA Test")
	fmt.Println("=========================")

	// Very minimal configuration
	config := gobed.FusedCAGRAConfig{
		VocabSize:   5,
		EmbedDim:    512,
		MaxVectors:  2,
		TopK:        2,
		GraphDegree: 2,
	}

	fmt.Printf("Creating engine: vocab=%d, vectors=%d\n", config.VocabSize, config.MaxVectors)

	engine, err := gobed.NewFusedCAGRAEngine(config)
	if err != nil {
		log.Fatalf("Create failed: %v", err)
	}
	defer engine.Close()

	fmt.Println("Engine created successfully")

	// Minimal test data
	embedWeights := make([]int8, config.VocabSize*config.EmbedDim)
	embedScales := make([]float32, config.VocabSize)

	for i := 0; i < config.VocabSize; i++ {
		embedScales[i] = 1.0
		for j := 0; j < config.EmbedDim; j++ {
			embedWeights[i*config.EmbedDim+j] = int8(i + 1)
		}
	}

	database := make([]simd.Vec512, 2)
	dbScales := make([]float32, 2)
	dbScales[0] = 1.0
	dbScales[1] = 1.0

	for j := 0; j < 512; j++ {
		database[0][j] = 10
		database[1][j] = 20
	}

	fmt.Print("Building index... ")
	err = engine.BuildIndex(embedWeights, embedScales, database, dbScales)
	if err != nil {
		log.Fatalf("Build failed: %v", err)
	}
	fmt.Println("OK")

	fmt.Print("Searching... ")
	start := time.Now()

	results, err := engine.Search([]uint16{1})

	elapsed := time.Since(start)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("OK (%v)\n", elapsed)
	fmt.Printf("Results: %d entries\n", len(results))
	for i, r := range results {
		fmt.Printf("  [%d] ID=%d dist=%.4f\n", i, r.ID, r.Similarity)
	}

	fmt.Println("✅ Debug test passed")
}
