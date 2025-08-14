package main

import (
	"fmt"
	"log"
	"strings"

	// Import the gobed package if needed
)

func main() {
	fmt.Println("🚀 Go Embedding Package Example")
	fmt.Println(strings.Repeat("=", 40))

	// Create a new embedding model
	model, err := gobed.NewEmbeddingModel(
		"model/embedding_model.onnx",  // Path to ONNX model
		"model/reference_tokens.json", // Path to reference tokens
		false,                         // Use CPU (set to true for GPU)
	)
	if err != nil {
		log.Fatalf("Failed to create embedding model: %v", err)
	}
	defer model.Close()

	// Example 1: Single text embedding
	text := "machine learning is fascinating"
	embedding, err := model.Encode(text)
	if err != nil {
		log.Fatalf("Failed to encode text: %v", err)
	}

	fmt.Printf("📝 Encoded: \"%s\"\n", text)
	fmt.Printf("📊 Embedding dimension: %d\n", len(embedding))
	fmt.Printf("📈 First 5 values: %.6f\n", embedding[:5])
	fmt.Printf("📏 L2 norm: %.6f\n", gobed.CalculateNorm(embedding))

	// Example 2: Similarity comparison
	texts := []string{
		"machine learning is fascinating",
		"artificial intelligence and deep learning",
		"hello world",
		"the weather is nice today",
	}

	fmt.Println("\n🔍 Computing embeddings for similarity comparison...")
	embeddings, err := model.BatchEncode(texts)
	if err != nil {
		log.Fatalf("Failed to encode texts: %v", err)
	}

	fmt.Println("\n📊 Similarity Matrix:")
	fmt.Printf("%-30s", "")
	for _, text := range texts {
		if len(text) > 8 {
			fmt.Printf("%-10s", text[:8]+"...")
		} else {
			fmt.Printf("%-10s", text)
		}
	}
	fmt.Println()

	for i, text1 := range texts {
		displayText := text1
		if len(displayText) > 30 {
			displayText = displayText[:27] + "..."
		}
		fmt.Printf("%-30s", displayText)
		for j := range texts {
			if i == j {
				fmt.Printf("%-10s", "1.000")
			} else {
				sim := gobed.CosineSimilarity(embeddings[i], embeddings[j])
				fmt.Printf("%-10.3f", sim)
			}
		}
		fmt.Println()
	}

	// Example 3: Find most similar pair
	fmt.Println("\n🎯 Finding most similar text pair...")
	maxSim := float32(-2.0)
	var bestTexts [2]string

	for i := 0; i < len(texts); i++ {
		for j := i + 1; j < len(texts); j++ {
			sim := gobed.CosineSimilarity(embeddings[i], embeddings[j])
			if sim > maxSim {
				maxSim = sim
				bestTexts = [2]string{texts[i], texts[j]}
			}
		}
	}

	fmt.Printf("🏆 Most similar pair (%.6f):\n", maxSim)
	fmt.Printf("  1. \"%s\"\n", bestTexts[0])
	fmt.Printf("  2. \"%s\"\n", bestTexts[1])

	// Example 4: Quality assessment
	fmt.Println("\n✅ Quality Assessment:")
	if maxSim > 0.3 && maxSim < 0.9 {
		fmt.Println("   ✓ Similarity scores look realistic")
	} else if maxSim > 0.9 {
		fmt.Println("   ⚠️ Very high similarity - check if expected")
	} else {
		fmt.Println("   ⚠️ Very low similarity - check model or texts")
	}

	// Show distance comparison too
	fmt.Println("\n📏 Euclidean Distances:")
	for i := 0; i < len(texts); i++ {
		for j := i + 1; j < len(texts); j++ {
			dist := gobed.SquaredEuclideanDistance(embeddings[i], embeddings[j])
			text1 := texts[i]
			if len(text1) > 20 {
				text1 = text1[:17] + "..."
			}
			text2 := texts[j]
			if len(text2) > 20 {
				text2 = text2[:17] + "..."
			}
			fmt.Printf("  \"%s\" vs \"%s\": %.2f\n", text1, text2, dist)
		}
	}

	fmt.Println("\n🎉 Example completed successfully!")
}
