package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/lee101/gobed"
)

func main() {
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "similarity":
			RunSimilarityDemo()
		case "bulk":
			runBulkEmbeddingExample()
		case "full":
			runSimilarityExample()
		case "demo":
			RunDemo()
		default:
			fmt.Println("Usage: go run . [similarity|bulk|full|demo]")
			fmt.Println("  similarity - Show basic similarity examples")
			fmt.Println("  bulk       - Run bulk embedding benchmark")
			fmt.Println("  full       - Run full similarity analysis")
			fmt.Println("  demo       - Run original demo")
			fmt.Println("\\nRunning default similarity demo...")
			RunSimilarityDemo()
		}
	} else {
		// Default: run similarity demo
		RunSimilarityDemo()
	}
}

// RunSimilarityDemo shows similarity and distance examples
func RunSimilarityDemo() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("🚀 Text Embedding Similarity & Distance Examples")
	fmt.Println(strings.Repeat("=", 80))

	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		fmt.Printf("❌ Error: %v\\n", err)
		return
	}

	// Test similar texts
	fmt.Println("\\n📊 SIMILARITY BETWEEN RELATED TEXTS")
	fmt.Println(strings.Repeat("-", 70))

	similarPairs := [][]string{
		{"Hello world", "Hi there friend"},
		{"Good morning everyone", "Hello world"},
		{"Python is a programming language.", "JavaScript runs in browsers."},
		{"Machine learning is fascinating.", "Deep learning models are powerful."},
	}

	for _, pair := range similarPairs {
		sim, err := model.Similarity(pair[0], pair[1])
		if err != nil {
			continue
		}
		distance := 1.0 - sim

		fmt.Printf("\\n\\\"%s\\\"\\n↔ \\\"%s\\\"\\n", pair[0], pair[1])
		fmt.Printf("  → Similarity: %.4f | Distance: %.4f\\n", sim, distance)
	}

	// Test unrelated texts
	fmt.Println("\\n📊 SIMILARITY BETWEEN UNRELATED TEXTS")
	fmt.Println(strings.Repeat("-", 70))

	unrelatedPairs := [][]string{
		{"Hello world", "Pizza tastes delicious."},
		{"Python is a programming language.", "The weather is nice today."},
		{"Mathematics requires practice.", "Birds are singing beautifully."},
	}

	for _, pair := range unrelatedPairs {
		sim, err := model.Similarity(pair[0], pair[1])
		if err != nil {
			continue
		}
		distance := 1.0 - sim

		fmt.Printf("\\n\\\"%s\\\"\\n↔ \\\"%s\\\"\\n", pair[0], pair[1])
		fmt.Printf("  → Similarity: %.4f | Distance: %.4f\\n", sim, distance)
	}

	fmt.Println("\\n✅ Demo completed!")
}

// RunDemo runs the original demonstration of the embedding model
func RunDemo() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 Gobed: Real Embedding Model Demo")
	fmt.Println("================================================================================")
	fmt.Println("Model: sentence-transformers/static-retrieval-mrl-en-v1 (REAL WEIGHTS)")
	fmt.Println("")

	// Load the real model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("❌ Failed to load model: %v", err)
	}

	// Get available texts
	availableTexts := model.GetAvailableTexts()
	if len(availableTexts) == 0 {
		log.Fatalf("❌ No texts available for encoding")
	}

	fmt.Printf("📚 Available texts for demo: %v\\n\\n", availableTexts)

	// Demo 1: Basic embedding
	fmt.Println("🔍 DEMO 1: Basic Text Encoding")
	fmt.Println(strings.Repeat("-", 50))

	// Test with arbitrary text first
	testTexts := []string{"This is a test sentence", "Machine learning is fun"}

	for _, text := range testTexts {
		embedding, err := model.Encode(text)
		if err != nil {
			fmt.Printf("❌ Failed to encode '%s': %v\\n", text, err)
			continue
		}

		fmt.Printf("Text: \\\"%s\\\"\\n", text)
		fmt.Printf("  • Dimensions: %d\\n", len(embedding))
		fmt.Printf("  • Sample values: [%.3f, %.3f, %.3f, %.3f, %.3f]\\n",
			embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
		fmt.Println()
	}

	fmt.Println("\\n✅ Demo completed successfully!")
	fmt.Printf("🎯 Model supports arbitrary text tokenization!\\n")
	fmt.Println(strings.Repeat("=", 80))
}

// Placeholder functions for compatibility
func runBulkEmbeddingExample() {
	fmt.Println("Bulk embedding example - not implemented in this simplified version")
}

func runSimilarityExample() {
	fmt.Println("Full similarity example - not implemented in this simplified version")
}
