package main

import (
	"flag"
	"fmt"
	"log"
	
	"github.com/lee101/gobed"
)

func main() {
	var text1, text2 string
	var showHelp, listTexts bool
	
	flag.StringVar(&text1, "text1", "", "First text to compare")
	flag.StringVar(&text2, "text2", "", "Second text to compare")
	flag.BoolVar(&showHelp, "help", false, "Show help")
	flag.BoolVar(&listTexts, "list", false, "List available pre-tokenized texts")
	flag.Parse()

	// Load model
	fmt.Println("🔄 Loading real embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("❌ Error loading model: %v", err)
	}

	// List available texts
	availableTexts := model.GetAvailableTexts()
	
	if listTexts {
		fmt.Printf("\n📚 Available texts (%d total):\n", len(availableTexts))
		for i, text := range availableTexts {
			fmt.Printf("%2d. %s\n", i+1, text)
		}
		return
	}

	if text1 == "" || text2 == "" {
		fmt.Println("❌ Please provide both -text1 and -text2")
		fmt.Println("Use -list to see available texts")
		return
	}

	// Check if texts are available
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
		fmt.Printf("❌ Text1 not found: \"%s\"\n", text1)
		return
	}

	if !text2Found {
		fmt.Printf("❌ Text2 not found: \"%s\"\n", text2)
		return
	}

	// Encode both texts with REAL model
	emb1, err := model.Encode(text1)
	if err != nil {
		log.Fatalf("❌ Error encoding text1: %v", err)
	}

	emb2, err := model.Encode(text2)
	if err != nil {
		log.Fatalf("❌ Error encoding text2: %v", err)
	}

	// Calculate similarity and distance
	similarity := gobed.CosineSimilarity(emb1, emb2)
	distance := 1.0 - similarity

	// Show results
	fmt.Printf("\n📏 DISTANCE CALCULATION\n")
	fmt.Printf("Text 1: \"%s\"\n", text1)
	fmt.Printf("Text 2: \"%s\"\n", text2)
	fmt.Printf("Similarity: %.6f\n", similarity)
	fmt.Printf("Distance: %.6f\n", distance)
	
	fmt.Println("\n✅ Real embedding calculation completed!")
	fmt.Println("🔬 Used REAL static-retrieval-mrl-en-v1 safetensors weights!")
}
