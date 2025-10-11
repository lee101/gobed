package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/daulet/tokenizers"
)

func main() {
	tokenizerPath := filepath.Join("model", "tokenizer.json")

	// Load tokenizer
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	fmt.Println("Tokenizer loaded successfully")

	// Test encoding
	text := "test query"
	encoding := tk.EncodeWithOptions(text, false,
		tokenizers.WithReturnTokens(),
		tokenizers.WithReturnTypeIDs())

	fmt.Printf("Encoding IDs: %v\n", encoding.IDs)
	fmt.Printf("Encoding Tokens: %v\n", encoding.Tokens)
}