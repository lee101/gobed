package main

import (
	"fmt"
	"log"
	"math"

	// Import the gobed package if needed
)

func main() {
	diagnosticMain()
}

func diagnosticMain() {
	fmt.Println("================================================================================")
	fmt.Println("GO EMBEDDING DIAGNOSTIC")
	fmt.Println("================================================================================")

	// Load embedding model
	embedding, err := gobed.NewEmbeddingModel("model/production_embedding_model.onnx", "model/debug_tokens.json", false)
	if err != nil {
		log.Fatal("Error loading model:", err)
	}
	defer embedding.Close()

	sentences := []string{
		"This is a test sentence.",
		"Machine learning is fascinating.",
		"The weather is nice today.",
		"Hello world",
		"Python is a programming language.",
	}

	// Generate embeddings using the reference tokens
	var embeddings [][]float32
	for i, sentence := range sentences {
		fmt.Printf("S%d: %s\n", i+1, sentence)

		embedding_vec, err := embedding.Encode(sentence)
		if err != nil {
			log.Fatal("Error generating embedding:", err)
		}

		embeddings = append(embeddings, embedding_vec)

		// Print first 5 dimensions and norm for comparison
		norm := float32(0)
		for _, val := range embedding_vec {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))

		fmt.Printf("S%d Go: [%.3f, %.3f, %.3f, %.3f, %.3f] (norm: %.3f)\n",
			i+1, embedding_vec[0], embedding_vec[1], embedding_vec[2], embedding_vec[3], embedding_vec[4], norm)
	}

	// Calculate similarity matrix
	fmt.Println("\nGo similarity matrix:")
	for i := 0; i < len(embeddings); i++ {
		fmt.Print("[")
		for j := 0; j < len(embeddings); j++ {
			sim := gobed.CosineSimilarity(embeddings[i], embeddings[j])
			fmt.Printf("%9.7f", sim)
			if j < len(embeddings)-1 {
				fmt.Print(" ")
			}
		}
		fmt.Println("]")
	}

	// Diversity analysis
	fmt.Println("\nGo diversity analysis:")
	var similarities []float32
	for i := 0; i < len(embeddings); i++ {
		for j := i + 1; j < len(embeddings); j++ {
			sim := gobed.CosineSimilarity(embeddings[i], embeddings[j])
			similarities = append(similarities, sim)
		}
	}

	if len(similarities) > 0 {
		minSim := similarities[0]
		maxSim := similarities[0]
		sum := float32(0)

		for _, sim := range similarities {
			if sim < minSim {
				minSim = sim
			}
			if sim > maxSim {
				maxSim = sim
			}
			sum += sim
		}

		mean := sum / float32(len(similarities))

		// Calculate standard deviation
		sumSq := float32(0)
		for _, sim := range similarities {
			diff := sim - mean
			sumSq += diff * diff
		}
		std := float32(math.Sqrt(float64(sumSq / float32(len(similarities)))))

		fmt.Printf("  Min similarity: %.6f\n", minSim)
		fmt.Printf("  Max similarity: %.6f\n", maxSim)
		fmt.Printf("  Mean similarity: %.6f\n", mean)
		fmt.Printf("  Std similarity: %.6f\n", std)

		if maxSim-minSim < 0.01 {
			fmt.Println("  ❌ POOR diversity - embeddings are too similar!")
		} else {
			fmt.Println("  ✓ Good diversity - embeddings are different")
		}
	}

	fmt.Println("\n================================================================================")
	fmt.Println("Expected baseline (from Python/ONNX):")
	fmt.Println("  Min similarity: ~-0.067")
	fmt.Println("  Max similarity: ~0.144")
	fmt.Println("  Mean similarity: ~0.029")
	fmt.Println("  Std similarity: ~0.062")
	fmt.Println("================================================================================")
}
