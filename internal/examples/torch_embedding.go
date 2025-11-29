//go:build legacy

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
)

// TokenData represents tokenization information
type TokenData struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
}

// EmbeddingLayer represents a simple embedding layer
type EmbeddingLayer struct {
	weights   [][]float32 // [vocab_size, embed_dim]
	vocabSize int
	embedDim  int
}

// NewEmbeddingLayerFromSafetensors creates an embedding layer by loading safetensors weights
// For now, we'll create a placeholder that matches the architecture
func NewEmbeddingLayerFromSafetensors(safetensorsPath string) (*EmbeddingLayer, error) {
	// For this demo, we'll create a simple embedding layer with the right dimensions
	// In a full implementation, you would parse the safetensors file
	return &EmbeddingLayer{
		vocabSize: 30522,
		embedDim:  1024,
		weights:   make([][]float32, 30522), // Initialize empty for now
	}, nil
}

// Forward performs forward pass through embedding layer
func (e *EmbeddingLayer) Forward(tokenIDs []int) []float32 {
	// Simple mean pooling implementation
	embedding := make([]float32, e.embedDim)
	validTokens := 0

	for _, tokenID := range tokenIDs {
		if tokenID > 0 && tokenID < e.vocabSize { // Skip padding tokens (0)
			// For this demo, we'll generate deterministic embeddings based on token ID
			// This creates a simple but consistent embedding
			for i := 0; i < e.embedDim; i++ {
				// Use a simple hash-like function for deterministic embeddings
				val := float32(math.Sin(float64(tokenID*i+i*17)) * 10.0)
				embedding[i] += val
			}
			validTokens++
		}
	}

	// Mean pooling
	if validTokens > 0 {
		for i := range embedding {
			embedding[i] /= float32(validTokens)
		}
	}

	return embedding
}

// CosineSimilarity calculates cosine similarity between two vectors
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := float32(0.0)
	normA := float32(0.0)
	normB := float32(0.0)

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = float32(math.Sqrt(float64(normA)))
	normB = float32(math.Sqrt(float64(normB)))

	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}

	return dotProduct / (normA * normB)
}

func main() {
	fmt.Println("Go PyTorch-Style Embedding Test")
	fmt.Println("===============================")

	// Load the embedding model
	model, err := NewEmbeddingLayerFromSafetensors("cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Load reference tokens
	var referenceTokens map[string]TokenData
	tokensFile, err := os.Open("model/production_reference_tokens.json")
	if err != nil {
		log.Fatalf("Failed to open reference tokens: %v", err)
	}
	defer tokensFile.Close()

	tokensData, err := ioutil.ReadAll(tokensFile)
	if err != nil {
		log.Fatalf("Failed to read reference tokens: %v", err)
	}

	err = json.Unmarshal(tokensData, &referenceTokens)
	if err != nil {
		log.Fatalf("Failed to parse reference tokens: %v", err)
	}

	// Test sentences
	sentences := []string{
		"This is a test sentence.",
		"Machine learning is fascinating.",
		"The weather is nice today.",
		"Python is a programming language.",
		"Hello world",
	}

	embeddings := make([][]float32, len(sentences))

	fmt.Println("\nGenerating embeddings...")
	for i, sentence := range sentences {
		tokenData, exists := referenceTokens[sentence]
		if !exists {
			fmt.Printf("Warning: No tokens found for '%s'\n", sentence)
			continue
		}

		embedding := model.Forward(tokenData.TokenIDs)
		embeddings[i] = embedding

		fmt.Printf("'%s' -> [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
			sentence, embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
	}

	// Calculate similarity matrix
	fmt.Println("\nGo Similarity Matrix:")
	fmt.Println("      S1    S2    S3    S4    S5  ")
	for i, emb1 := range embeddings {
		if emb1 == nil {
			continue
		}
		row := fmt.Sprintf("S%d  ", i+1)
		for _, emb2 := range embeddings {
			if emb2 == nil {
				row += "  --- "
				continue
			}
			sim := CosineSimilarity(emb1, emb2)
			row += fmt.Sprintf("%5.3f ", sim)
		}
		fmt.Println(row)
	}

	fmt.Println("\nGo PyTorch-style embedding test completed!")
	fmt.Println("Note: This is a simplified implementation for demonstration.")
	fmt.Println("A full implementation would load actual safetensors weights.")
}
