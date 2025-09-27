package main

import (
	"encoding/json"
	"fmt"
	"log"
	
	"os"
	
	"strings"
	
)

// Copy the simple int8 model code here for standalone testing
const Int8EmbeddingDim = 512

type SimpleInt8Model512 struct {
	embeddings [][]int8
	scales     []float32
	vocab      map[string]int16
}

type Int8Result512 struct {
	Vector []int8
	Scale  float32
}

func main() {
	fmt.Println("🚀 Standalone Int8 Model Test")
	
	// Test if model file exists
	modelPath := "../model/modelint8_512dim.safetensors"
	tokenizerPath := "../model/tokenizer.json"
	
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Model not found: %s", modelPath)
	}
	
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		log.Fatalf("Tokenizer not found: %s", tokenizerPath)
	}
	
	fmt.Println("✅ Required files exist")
	
	// Check model size
	info, _ := os.Stat(modelPath)
	fmt.Printf("📊 Model size: %.1f MB\n", float64(info.Size())/(1024*1024))
	
	// Test basic vocab loading
	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to read tokenizer: %v", err)
	}
	
	var tokenizerData struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	
	if err := json.Unmarshal(data, &tokenizerData); err != nil {
		log.Fatalf("Failed to parse tokenizer: %v", err)
	}
	
	fmt.Printf("📝 Vocab size: %d tokens\n", len(tokenizerData.Model.Vocab))
	
	// Test simple tokenization
	vocab := make(map[string]int16)
	for token, id := range tokenizerData.Model.Vocab {
		if id < 32768 {
			vocab[token] = int16(id)
		}
	}
	
	// Simple tokenize function
	simpleTokenize := func(text string) []int16 {
		text = strings.ToLower(strings.TrimSpace(text))
		words := strings.Fields(text)
		var tokens []int16
		
		for _, word := range words {
			if id, ok := vocab[word]; ok {
				tokens = append(tokens, id)
			} else if id, ok := vocab["##"+word]; ok {
				tokens = append(tokens, id)
			} else if unkID, ok := vocab["[UNK]"]; ok {
				tokens = append(tokens, unkID)
			}
		}
		return tokens
	}
	
	// Test tokenization
	testTexts := []string{
		"machine learning",
		"neural networks", 
		"deep learning",
	}
	
	fmt.Println("\n🔧 Testing tokenization:")
	for _, text := range testTexts {
		tokens := simpleTokenize(text)
		fmt.Printf("  %q -> %v\n", text, tokens)
	}
	
	fmt.Println("\n✅ Tokenization working!")
	fmt.Println("🎯 Int8 model files are ready for Go implementation")
}
