package main

import (
	"fmt"
	"log"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("Gobed Demo - Testing Basic Functionality")
	fmt.Println("=========================================")

	// Load the model
	fmt.Println("Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("✓ Model loaded successfully")

	// Test encoding
	text := "Machine learning is fascinating."
	fmt.Printf("\nEncoding text: '%s'\n", text)
	embedding, err := model.Encode(text)
	if err != nil {
		log.Fatalf("Failed to encode: %v", err)
	}
	fmt.Printf("✓ Generated embedding with %d dimensions\n", len(embedding))

	// Test similarity
	text1 := "Deep learning models are powerful."
	text2 := "Machine learning is fascinating."
	similarity, err := model.Similarity(text1, text2)
	if err != nil {
		log.Fatalf("Failed to calculate similarity: %v", err)
	}
	fmt.Printf("\nSimilarity between:\n  '%s'\n  '%s'\n  = %.4f\n", text1, text2, similarity)

	fmt.Println("\n✓ Demo completed successfully!")
}
