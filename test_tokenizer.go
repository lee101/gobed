package main

import (
	"fmt"
	"log"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

func main() {
	fmt.Println("Testing Go BERT tokenizer...")

	// Try to load a BERT tokenizer
	tk, err := pretrained.BertBaseUncased()
	if err != nil {
		log.Printf("Failed to load BERT tokenizer: %v", err)

		// Fallback: try to download and create tokenizer
		fmt.Println("Trying to create tokenizer from model...")

		// For now, let's create a simple test
		fmt.Println("✗ Tokenizer not available, need to implement custom tokenization")
		return
	}

	sentences := []string{
		"This is a test sentence.",
		"Machine learning is fascinating.",
		"Hello world",
	}

	for _, sentence := range sentences {
		// Tokenize
		encoding, err := tk.EncodeSingle(tokenizer.EncodeInput{
			Sequence: sentence,
		})
		if err != nil {
			log.Printf("Failed to tokenize '%s': %v", sentence, err)
			continue
		}

		fmt.Printf("'%s' -> %v\n", sentence, encoding.Ids)
	}
}
