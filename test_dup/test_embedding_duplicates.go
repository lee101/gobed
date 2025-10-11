//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"log"
	"math"

	"github.com/lee101/gobed"
)

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func areEmbeddingsIdentical(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}

	const epsilon = 1e-6
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > epsilon {
			return false
		}
	}
	return true
}

func main() {
	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal("Failed to load model:", err)
	}

	// Test queries - very different semantically
	queries := []string{
		"peace",
		"war",
		"CUDA",
		"database",
		"Finding inner peace through meditation",
		"Machine learning algorithms for classification",
		"The quick brown fox jumps over the lazy dog",
		"1234567890",
		"",  // Empty query
		"peace",  // Duplicate of first query
	}

	fmt.Println("=== Testing Query Embeddings for Duplicates ===\n")

	// Generate embeddings
	embeddings := make([][]float32, len(queries))
	for i, query := range queries {
		if query == "" {
			fmt.Printf("Query %d: [EMPTY]\n", i)
		} else {
			fmt.Printf("Query %d: '%s'\n", i, query)
		}

		// Generate embedding
		embedding, err := model.Encode(query)
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}

		embeddings[i] = embedding

		// Show embedding stats
		var sum, min, max float32
		min = embedding[0]
		max = embedding[0]
		for _, val := range embedding {
			sum += val
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}
		mean := sum / float32(len(embedding))

		fmt.Printf("  Dim: %d, Mean: %.6f, Min: %.6f, Max: %.6f\n",
			len(embedding), mean, min, max)

		// Show first 5 values
		fmt.Printf("  First 5 values: [")
		for j := 0; j < 5 && j < len(embedding); j++ {
			fmt.Printf("%.4f ", embedding[j])
		}
		fmt.Printf("...]\n\n")
	}

	// Check for duplicate embeddings
	fmt.Println("=== Checking for Duplicate Embeddings ===\n")

	duplicateFound := false
	for i := 0; i < len(embeddings)-1; i++ {
		if embeddings[i] == nil {
			continue
		}
		for j := i + 1; j < len(embeddings); j++ {
			if embeddings[j] == nil {
				continue
			}

			similarity := cosineSimilarity(embeddings[i], embeddings[j])
			identical := areEmbeddingsIdentical(embeddings[i], embeddings[j])

			// Report high similarity or identical embeddings
			if similarity > 0.95 || identical {
				queryI := queries[i]
				queryJ := queries[j]
				if queryI == "" {
					queryI = "[EMPTY]"
				}
				if queryJ == "" {
					queryJ = "[EMPTY]"
				}

				if identical {
					fmt.Printf("❌ IDENTICAL embeddings: Query %d ('%s') == Query %d ('%s')\n",
						i, queryI, j, queryJ)
					duplicateFound = true
				} else {
					fmt.Printf("⚠️  Very similar: Query %d ('%s') ~= Query %d ('%s'), similarity=%.6f\n",
						i, queryI, j, queryJ, similarity)
				}
			}
		}
	}

	if !duplicateFound {
		fmt.Println("✓ No unexpected duplicate embeddings found")
	}

	// Check if the intentional duplicate (queries 0 and 9) produces identical embeddings
	if len(embeddings) > 9 && embeddings[0] != nil && embeddings[9] != nil {
		if areEmbeddingsIdentical(embeddings[0], embeddings[9]) {
			fmt.Println("✓ Identical queries produce identical embeddings (expected)")
		} else {
			similarity := cosineSimilarity(embeddings[0], embeddings[9])
			fmt.Printf("⚠️  Identical queries 'peace' produce slightly different embeddings (similarity=%.6f)\n", similarity)
		}
	}

	// Test pairwise similarities
	fmt.Println("\n=== Pairwise Cosine Similarities (sample) ===\n")
	fmt.Println("         peace    war      CUDA     database")
	for i := 0; i < 4 && i < len(embeddings); i++ {
		if embeddings[i] == nil {
			continue
		}
		fmt.Printf("%-8s ", queries[i][:min(8, len(queries[i]))])
		for j := 0; j < 4 && j < len(embeddings); j++ {
			if embeddings[j] == nil {
				fmt.Printf("  N/A    ")
			} else {
				sim := cosineSimilarity(embeddings[i], embeddings[j])
				fmt.Printf("%.4f   ", sim)
			}
		}
		fmt.Println()
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
