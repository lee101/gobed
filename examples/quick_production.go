package main

import (
	"fmt"
	"log"

	// Import the gobed package if needed
)

func main() {
	fmt.Println("PRODUCTION MODEL TEST - With Reference Tokens")
	fmt.Println("==============================================")

	// Load the production model
	model, err := gobed.NewEmbeddingModel(
		"model/production_embedding_model.onnx",
		"model/production_reference_tokens.json",
		false,
	)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Use sentences that have reference tokens
	testSentences := []string{
		"This is a test sentence.",
		"Machine learning is fascinating.",
		"The weather is nice today.",
		"Python is a programming language.",
		"Hello world",
	}

	fmt.Println("\nTesting with sentences that have reference tokens:")
	embeddings := make([][]float32, len(testSentences))

	for i, sentence := range testSentences {
		embedding, err := model.Encode(sentence)
		if err != nil {
			log.Fatalf("Failed to encode sentence %d: %v", i+1, err)
		}
		embeddings[i] = embedding
		fmt.Printf("  %d. '%s' -> [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
			i+1, sentence, embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
	}

	// Test similarity
	fmt.Println("\nSimilarity Matrix:")
	fmt.Print("    ")
	for i := range testSentences {
		fmt.Printf("  S%d  ", i+1)
	}
	fmt.Println()

	for i := range testSentences {
		fmt.Printf("S%d ", i+1)
		for j := range testSentences {
			sim := gobed.CosineSimilarity(embeddings[i], embeddings[j])
			fmt.Printf(" %.3f", sim)
		}
		fmt.Println()
	}

	// Check if embeddings are different
	fmt.Println("\nQuality Check:")
	allSame := true
	for i := 1; i < len(embeddings); i++ {
		sim := gobed.CosineSimilarity(embeddings[0], embeddings[i])
		if sim < 0.999 {
			allSame = false
			break
		}
	}

	if allSame {
		fmt.Println("  ✗ All embeddings are nearly identical - possible issue!")
	} else {
		fmt.Println("  ✓ Embeddings are different - good!")
	}

	fmt.Println("\nProduction model test completed!")
}
