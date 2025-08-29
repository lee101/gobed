// +build gpu

package gobed

import (
	"fmt"
	"testing"
)

func TestGPUBatchSearch(t *testing.T) {
	// Check CUDA availability
	if !IsCUDAAvailable() {
		t.Skip("CUDA not available, skipping GPU batch search test")
	}

	// Create GPU indexer with test configuration
	config := IndexConfig{
		VectorDim: 128,
		VocabSize: 1000,
		EmbedDim:  128,
		DeviceID:  0,
	}

	indexer, err := NewGPUIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create GPU indexer: %v", err)
	}
	defer indexer.Close()

	// Add some test vectors
	numVectors := 100
	vectors := make([][]int8, numVectors)
	scales := make([]float32, numVectors)

	for i := 0; i < numVectors; i++ {
		vec := make([]int8, config.VectorDim)
		for j := range vec {
			vec[j] = int8((i + j) % 127)
		}
		vectors[i] = vec
		scales[i] = 1.0
	}

	// Add vectors to index
	err = indexer.AddVectors(vectors, scales)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Prepare batch queries
	numQueries := 10
	queries := make([][]int8, numQueries)
	for i := 0; i < numQueries; i++ {
		query := make([]int8, config.VectorDim)
		for j := range query {
			query[j] = int8((i * 2 + j) % 127)
		}
		queries[i] = query
	}

	// Perform batch search
	k := 5
	results, err := indexer.BatchSearch(queries, k)
	if err != nil {
		t.Fatalf("Batch search failed: %v", err)
	}

	// Verify results
	if len(results) != numQueries {
		t.Errorf("Expected %d result sets, got %d", numQueries, len(results))
	}

	for i, queryResults := range results {
		if len(queryResults) != k {
			t.Errorf("Query %d: expected %d results, got %d", i, k, len(queryResults))
		}

		// Verify results are sorted by similarity (descending)
		for j := 1; j < len(queryResults); j++ {
			if queryResults[j-1].Similarity < queryResults[j].Similarity {
				t.Errorf("Query %d: results not sorted by similarity", i)
				break
			}
		}
	}

	fmt.Printf("✅ GPU Batch Search test passed: %d queries processed successfully\n", numQueries)
}