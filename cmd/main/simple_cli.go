//go:build legacy

package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/lee101/gobed"
)

// Simple CLI that uses the existing working code
func main_simple_cli() { // Renamed to avoid duplicate main
	var text1, text2 string
	var showHelp bool

	flag.StringVar(&text1, "text1", "", "First text to compare")
	flag.StringVar(&text2, "text2", "", "Second text to compare")
	flag.BoolVar(&showHelp, "help", false, "Show help")
	flag.Parse()

	if showHelp || (text1 == "" || text2 == "") {
		fmt.Println(" Real Text Embedding Distance Calculator")
		fmt.Println(strings.Repeat("=", 50))
		fmt.Println("Usage:")
		fmt.Println("  go run simple_cli.go -text1=\"Hello world\" -text2=\"Hi there friend\"")
		fmt.Println("")
		fmt.Println("Note: This uses pre-tokenized texts only for now.")
		fmt.Println("Available texts from model/real_reference_tokens.json")
		return
	}

	// Load the existing model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf(" Error loading model: %v", err)
	}

	// Get available texts
	availableTexts := model.GetAvailableTexts()

	// Check if the input texts are available
	text1Found := false
	text2Found := false

	for _, text := range availableTexts {
		if text == text1 {
			text1Found = true
		}
		if text == text2 {
			text2Found = true
		}
	}

	if !text1Found {
		fmt.Printf(" Text1 not found in pre-tokenized texts: \"%s\"\n", text1)
		fmt.Println("\n📚 Available texts:")
		for i, text := range availableTexts {
			fmt.Printf("  %d. %s\n", i+1, text)
		}
		return
	}

	if !text2Found {
		fmt.Printf(" Text2 not found in pre-tokenized texts: \"%s\"\n", text2)
		fmt.Println("\n📚 Available texts:")
		for i, text := range availableTexts {
			fmt.Printf("  %d. %s\n", i+1, text)
		}
		return
	}

	// Compute embeddings
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println(" COMPUTING REAL EMBEDDINGS")
	fmt.Println(strings.Repeat("=", 70))

	emb1, err := model.Encode(text1)
	if err != nil {
		log.Fatalf(" Error encoding text1: %v", err)
	}

	emb2, err := model.Encode(text2)
	if err != nil {
		log.Fatalf(" Error encoding text2: %v", err)
	}

	// Calculate similarity and distance
	similarity := gobed.CosineSimilarity(emb1, emb2)
	distance := 1.0 - similarity

	// Show results
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("📏 DISTANCE CALCULATION")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("\n Text 1: \"%s\"\n", text1)
	fmt.Printf(" Text 2: \"%s\"\n", text2)
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf(" Cosine Similarity: %.6f\n", similarity)
	fmt.Printf("📐 Distance (1-similarity): %.6f\n", distance)

	// Interpretation
	fmt.Println("\n INTERPRETATION:")
	if similarity > 0.7 {
		fmt.Println(" Very similar texts")
	} else if similarity > 0.4 {
		fmt.Println("🟢 Somewhat similar texts")
	} else if similarity > 0.1 {
		fmt.Println("🟡 Slightly related texts")
	} else if similarity > -0.1 {
		fmt.Println("🔴 Unrelated texts")
	} else {
		fmt.Println("❄  Opposite texts")
	}

	// Show embedding previews
	fmt.Printf("\n Embedding dimensions: %d\n", len(emb1))
	fmt.Printf(" Text 1 embedding sample: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
		emb1[0], emb1[1], emb1[2], emb1[3], emb1[4])
	fmt.Printf(" Text 2 embedding sample: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
		emb2[0], emb2[1], emb2[2], emb2[3], emb2[4])

	fmt.Println("\n Real embedding calculation completed!")
	fmt.Println("🔬 This used REAL safetensors weights and tokenization!")
}
