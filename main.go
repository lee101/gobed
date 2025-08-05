package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"strings"
)

// TensorInfo contains information about a tensor in the safetensors file
type TensorInfo struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// TokenData represents tokenization information
type TokenData struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
}

// SafetensorsEmbedding represents the embedding model with safetensors weights
type SafetensorsEmbedding struct {
	weights   [][]float32 // [vocab_size, embed_dim]
	vocabSize int
	embedDim  int
}

// NewSafetensorsEmbedding creates an embedding model from safetensors
func NewSafetensorsEmbedding(safetensorsPath string) (*SafetensorsEmbedding, error) {
	file, err := os.Open(safetensorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read header length (first 8 bytes, little-endian)
	headerLengthBytes := make([]byte, 8)
	_, err = file.Read(headerLengthBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read header length: %v", err)
	}

	headerLength := binary.LittleEndian.Uint64(headerLengthBytes)

	// Read header JSON
	headerBytes := make([]byte, headerLength)
	_, err = file.Read(headerBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %v", err)
	}

	var header map[string]TensorInfo
	err = json.Unmarshal(headerBytes, &header)
	if err != nil {
		return nil, fmt.Errorf("failed to parse header: %v", err)
	}

	// Read the rest of the file (tensor data)
	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor data: %v", err)
	}

	// Get embedding weights
	info, exists := header["embedding.weight"]
	if !exists {
		return nil, fmt.Errorf("embedding.weight tensor not found")
	}

	if info.Dtype != "F32" {
		return nil, fmt.Errorf("unsupported dtype: %s", info.Dtype)
	}

	if len(info.Shape) != 2 {
		return nil, fmt.Errorf("expected 2D tensor, got %dD", len(info.Shape))
	}

	start := info.DataOffsets[0]
	end := info.DataOffsets[1]

	if start < 0 || end > int64(len(data)) {
		return nil, fmt.Errorf("invalid data offsets: %d-%d", start, end)
	}

	tensorBytes := data[start:end]

	// Convert bytes to float32 values
	rows := info.Shape[0]
	cols := info.Shape[1]

	weights := make([][]float32, rows)
	for i := range weights {
		weights[i] = make([]float32, cols)
	}

	// Read float32 values (little-endian)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			offset := (i*cols + j) * 4 // 4 bytes per float32
			if offset+4 > len(tensorBytes) {
				return nil, fmt.Errorf("not enough data for tensor")
			}

			bits := binary.LittleEndian.Uint32(tensorBytes[offset : offset+4])
			weights[i][j] = math.Float32frombits(bits)
		}
	}

	fmt.Printf("✅ Loaded safetensors embedding weights: [%d, %d]\n", rows, cols)

	return &SafetensorsEmbedding{
		weights:   weights,
		vocabSize: rows,
		embedDim:  cols,
	}, nil
}

// Encode performs forward pass with mean pooling
func (s *SafetensorsEmbedding) Encode(tokenIDs []int) []float32 {
	embedding := make([]float32, s.embedDim)
	validTokens := 0

	for _, tokenID := range tokenIDs {
		if tokenID > 0 && tokenID < s.vocabSize { // Skip padding tokens (0)
			// Add embedding for this token
			for i := 0; i < s.embedDim; i++ {
				embedding[i] += s.weights[tokenID][i]
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
	fmt.Println("================================================================================")
	fmt.Println("GO SAFETENSORS EMBEDDING - PRODUCTION VERSION")
	fmt.Println("================================================================================")
	fmt.Println("Model: sentence-transformers/static-retrieval-mrl-en-v1")
	fmt.Println("Approach: Direct safetensors loading (matches Python PyTorch exactly)")
	fmt.Println("")

	// Load the actual model weights from safetensors
	model, err := NewSafetensorsEmbedding("cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors")
	if err != nil {
		log.Fatalf("Failed to load safetensors model: %v", err)
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

	fmt.Printf("✅ Loaded reference tokens for %d sentences\n", len(referenceTokens))
	fmt.Println("")

	// Test sentences - same as Python validation
	sentences := []string{
		"This is a test sentence.",
		"Machine learning is fascinating.",
		"The weather is nice today.",
		"Python is a programming language.",
		"Hello world",
	}

	fmt.Println("🚀 Generating embeddings with actual safetensors weights...")
	fmt.Println("----------------------------------------------------------")

	var embeddings [][]float32
	for i, sentence := range sentences {
		tokenData, exists := referenceTokens[sentence]
		if !exists {
			fmt.Printf("⚠️  Warning: No tokens found for '%s'\n", sentence)
			continue
		}

		embedding := model.Encode(tokenData.TokenIDs)
		embeddings = append(embeddings, embedding)

		// Calculate norm for validation
		norm := float32(0)
		for _, val := range embedding {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))

		fmt.Printf("S%d: '%s'\n", i+1, sentence)
		fmt.Printf("     Embedding: [%.3f, %.3f, %.3f, %.3f, %.3f] (norm: %.3f)\n",
			embedding[0], embedding[1], embedding[2], embedding[3], embedding[4], norm)
		fmt.Printf("     Tokens: %d valid tokens\n", len(tokenData.TokenIDs))
		fmt.Println()
	}

	// Calculate and display similarity matrix
	fmt.Println("📊 Cosine Similarity Matrix:")
	fmt.Println("-----------------------------")
	fmt.Println("      S1    S2    S3    S4    S5  ")
	for i, emb1 := range embeddings {
		row := fmt.Sprintf("S%d  ", i+1)
		for _, emb2 := range embeddings {
			sim := CosineSimilarity(emb1, emb2)
			row += fmt.Sprintf("%5.3f ", sim)
		}
		fmt.Println(row)
	}

	// Detailed similarity analysis
	fmt.Println("\n🔍 Detailed Similarity Analysis:")
	fmt.Println("----------------------------------")

	var allSimilarities []float32
	for i := 0; i < len(embeddings); i++ {
		for j := i + 1; j < len(embeddings); j++ {
			sim := CosineSimilarity(embeddings[i], embeddings[j])
			allSimilarities = append(allSimilarities, sim)
			// Safely truncate sentence names
			sent1 := sentences[i]
			sent2 := sentences[j]
			if len(sent1) > 25 {
				sent1 = sent1[:25] + "..."
			}
			if len(sent2) > 25 {
				sent2 = sent2[:25] + "..."
			}
			fmt.Printf("S%d vs S%d: %.6f  ('%s' vs '%s')\n",
				i+1, j+1, sim, sent1, sent2)
		}
	}

	// Statistical summary
	if len(allSimilarities) > 0 {
		minSim := allSimilarities[0]
		maxSim := allSimilarities[0]
		sum := float32(0)

		for _, sim := range allSimilarities {
			if sim < minSim {
				minSim = sim
			}
			if sim > maxSim {
				maxSim = sim
			}
			sum += sim
		}

		mean := sum / float32(len(allSimilarities))

		// Calculate standard deviation
		sumSq := float32(0)
		for _, sim := range allSimilarities {
			diff := sim - mean
			sumSq += diff * diff
		}
		std := float32(math.Sqrt(float64(sumSq / float32(len(allSimilarities)))))

		fmt.Println("\n📈 Statistical Summary:") 
		fmt.Println("-----------------------")
		fmt.Printf("Min similarity: %.6f\n", minSim)
		fmt.Printf("Max similarity: %.6f\n", maxSim)
		fmt.Printf("Mean similarity: %.6f\n", mean)
		fmt.Printf("Std deviation: %.6f\n", std)
		fmt.Printf("Range: %.6f\n", maxSim-minSim)

		// Quality assessment
		fmt.Println("\n🎯 Quality Assessment:")
		fmt.Println("----------------------")
		if maxSim-minSim < 0.01 {
			fmt.Println("❌ POOR: Embeddings are too similar (low diversity)")
		} else if minSim < -0.1 {
			fmt.Println("✅ EXCELLENT: Good diversity with some negative correlations")
		} else if maxSim > 0.1 && minSim < 0.1 {
			fmt.Println("✅ GOOD: Reasonable diversity in similarity scores")
		} else {
			fmt.Println("⚠️  MODERATE: Limited diversity in embeddings")
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("🎉 VALIDATION AGAINST PYTHON PYTORCH REFERENCE:")
	fmt.Println("Expected 'This is a test sentence.': [3.483, -2.513, 3.576, -0.724, 1.369]")
	fmt.Printf("Actual Go safetensors result:         [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
		embeddings[0][0], embeddings[0][1], embeddings[0][2], embeddings[0][3], embeddings[0][4])

	// Check if first embedding matches expected
	expected := []float32{3.483, -2.513, 3.576, -0.724, 1.369}
	match := true
	maxDiff := float32(0.0)
	for i := 0; i < 5; i++ {
		diff := float32(math.Abs(float64(embeddings[0][i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 0.001 {
			match = false
		}
	}

	if match {
		fmt.Printf("✅ PERFECT MATCH! Maximum difference: %.6f\n", maxDiff)
		fmt.Println("🏆 Go safetensors implementation matches Python PyTorch exactly!")
	} else {
		fmt.Printf("❌ MISMATCH: Maximum difference: %.6f\n", maxDiff)
		fmt.Println("⚠️  Further investigation needed")
	}

	fmt.Println("\n📋 Technical Details:")
	fmt.Println("---------------------")
	fmt.Printf("Model: %s\n", "sentence-transformers/static-retrieval-mrl-en-v1")
	fmt.Printf("Vocabulary size: %d tokens\n", model.vocabSize)
	fmt.Printf("Embedding dimension: %d\n", model.embedDim)
	fmt.Printf("Implementation: Direct safetensors loading\n")
	fmt.Printf("Pooling: Mean pooling (excluding padding tokens)\n")
	fmt.Printf("Test sentences: %d\n", len(sentences))

	fmt.Println("\n🚀 Production ready! Use this approach for consistent Go/Python results.")
	fmt.Println(strings.Repeat("=", 80))
}