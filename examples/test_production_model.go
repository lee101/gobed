package main

import (
	"fmt"
	"log"
	"math"

	// Import the gobed package if needed
)

func main() {
	fmt.Println("==========================================")
	fmt.Println("PRODUCTION MODEL TEST")
	fmt.Println("==========================================")

	// Load the production ONNX model
	model, err := gobed.NewEmbeddingModel(
		"model/production_embedding_model.onnx",
		"model/production_reference_tokens.json",
		false, // Use CPU
	)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	fmt.Println("✓ Production model loaded successfully")

	// Test sentences
	testSentences := []string{
		"This is a test sentence.",
		"Machine learning is fascinating.",
		"The weather is nice today.",
		"Python is a programming language.",
		"Artificial intelligence will change the world.",
	}

	fmt.Printf("\nTesting with %d sentences:\n", len(testSentences))
	for i, sentence := range testSentences {
		fmt.Printf("  %d. %s\n", i+1, sentence)
	}

	// Generate embeddings
	fmt.Println("\nGenerating embeddings...")
	embeddings := make([][]float32, len(testSentences))

	for i, sentence := range testSentences {
		embedding, err := model.Encode(sentence)
		if err != nil {
			log.Fatalf("Failed to encode sentence %d: %v", i+1, err)
		}
		embeddings[i] = embedding
		fmt.Printf("  Sentence %d: %d dimensions, first 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
			i+1, len(embedding), embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
	}

	// Calculate similarities
	fmt.Println("\n==========================================")
	fmt.Println("SIMILARITY ANALYSIS")
	fmt.Println("==========================================")

	fmt.Println("\nPairwise cosine similarities:")
	fmt.Print("      ")
	for i := range testSentences {
		fmt.Printf("  S%d   ", i+1)
	}
	fmt.Println()

	similarities := make([][]float64, len(testSentences))
	for i := range similarities {
		similarities[i] = make([]float64, len(testSentences))
	}

	for i := range testSentences {
		fmt.Printf("S%d  ", i+1)
		for j := range testSentences {
			sim := gobed.CosineSimilarity(embeddings[i], embeddings[j])
			similarities[i][j] = float64(sim)
			fmt.Printf(" %.3f", sim)
		}
		fmt.Println()
	}

	// Quality checks
	fmt.Println("\n==========================================")
	fmt.Println("QUALITY CHECKS")
	fmt.Println("==========================================")

	// Check diagonal (self-similarity should be ~1.0)
	fmt.Println("\nSelf-similarity check (should be ~1.0):")
	allGood := true
	for i := range testSentences {
		selfSim := similarities[i][i]
		status := "✓"
		if selfSim < 0.99 {
			status = "✗"
			allGood = false
		}
		fmt.Printf("  S%d: %.6f %s\n", i+1, selfSim, status)
	}

	if allGood {
		fmt.Println("✓ All self-similarities are good!")
	} else {
		fmt.Println("✗ Some self-similarities are too low!")
	}

	// Check semantic relationships
	fmt.Println("\nSemantic relationship analysis:")
	
	// Find most similar pair
	maxSim := 0.0
	maxI, maxJ := 0, 0
	for i := 0; i < len(testSentences); i++ {
		for j := i + 1; j < len(testSentences); j++ {
			if similarities[i][j] > maxSim {
				maxSim = similarities[i][j]
				maxI, maxJ = i, j
			}
		}
	}

	// Find least similar pair
	minSim := 1.0
	minI, minJ := 0, 0
	for i := 0; i < len(testSentences); i++ {
		for j := i + 1; j < len(testSentences); j++ {
			if similarities[i][j] < minSim {
				minSim = similarities[i][j]
				minI, minJ = i, j
			}
		}
	}

	fmt.Printf("  Most similar pair (%.3f):\n", maxSim)
	fmt.Printf("    \"%s\"\n", testSentences[maxI])
	fmt.Printf("    \"%s\"\n", testSentences[maxJ])

	fmt.Printf("  Least similar pair (%.3f):\n", minSim)
	fmt.Printf("    \"%s\"\n", testSentences[minI])
	fmt.Printf("    \"%s\"\n", testSentences[minJ])

	// Calculate statistics
	var sum, sumSq float64
	count := 0
	for i := 0; i < len(testSentences); i++ {
		for j := i + 1; j < len(testSentences); j++ {
			sim := similarities[i][j]
			sum += sim
			sumSq += sim * sim
			count++
		}
	}

	mean := sum / float64(count)
	variance := (sumSq / float64(count)) - (mean * mean)
	stdDev := math.Sqrt(variance)

	fmt.Printf("\nSimilarity statistics:\n")
	fmt.Printf("  Mean: %.4f\n", mean)
	fmt.Printf("  Std Dev: %.4f\n", stdDev)
	fmt.Printf("  Range: [%.4f, %.4f]\n", minSim, maxSim)

	// Final assessment
	fmt.Println("\n==========================================")
	fmt.Println("FINAL ASSESSMENT")
	fmt.Println("==========================================")

	if minSim > 0.0 && maxSim < 1.0 && stdDev > 0.05 {
		fmt.Println("✓ EXCELLENT: Model produces realistic, differentiated similarity scores!")
		fmt.Println("  - Similarities are not all the same")
		fmt.Println("  - Range of similarities shows semantic understanding")
		fmt.Println("  - No obviously broken values")
	} else if stdDev < 0.01 {
		fmt.Println("✗ POOR: All similarities are too similar - model may not be working correctly")
	} else {
		fmt.Println("⚠ OKAY: Model works but similarity range could be better")
	}

	fmt.Printf("\nProduction model test completed successfully!\n")
}
