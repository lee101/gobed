package main

import (
	"fmt"
	"log"
	"os"
)

func main() {
	fmt.Println("Debug Model Loading")

	modelPath := "../model/modelint8_512dim.safetensors"
	tokenizerPath := "../model/tokenizer.json"

	// Check if files exist
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Printf("Model file missing: %s", modelPath)
	} else {
		fmt.Printf("✓ Model file exists: %s\n", modelPath)
	}

	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		log.Printf("Tokenizer file missing: %s", tokenizerPath)
	} else {
		fmt.Printf("✓ Tokenizer file exists: %s\n", tokenizerPath)
	}

	// Try to load the model
	fmt.Println("Attempting to load model...")
	model, err := LoadFastModel(modelPath, tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	fmt.Printf("✓ Model loaded successfully\n")
	fmt.Printf("  Embeddings: %d vectors\n", len(model.embeddings))
	fmt.Printf("  Scales: %d values\n", len(model.scales))

	// Test embedding generation
	fmt.Println("Testing embedding generation...")
	embedding, err := model.EmbedInt8("test query")
	if err != nil {
		log.Fatalf("Failed to generate embedding: %v", err)
	}

	fmt.Printf("✓ Embedding generated: %d dimensions\n", len(embedding))
	fmt.Printf("  First 10 values: %v\n", embedding[:10])

	// Test different queries to see if they produce different embeddings
	queries := []string{"machine learning", "anime", "cooking", "programming"}
	fmt.Println("Testing different queries:")

	for _, query := range queries {
		emb, err := model.EmbedInt8(query)
		if err != nil {
			log.Printf("Failed to embed '%s': %v", query, err)
			continue
		}

		// Calculate simple checksum to see if embeddings differ
		checksum := int32(0)
		for _, val := range emb {
			checksum += int32(val)
		}

		fmt.Printf("  '%s': checksum=%d, first_5=%v\n", query, checksum, emb[:5])
	}
}