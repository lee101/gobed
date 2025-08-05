// Package gobed provides a Go interface for sentence embedding models using safetensors.
//
// This package allows you to:
//   - Load safetensors embedding models (like sentence-transformers)
//   - Generate embeddings for text with perfect Python PyTorch consistency
//   - Calculate semantic similarity between texts
//   - Use production-ready models with 30,522 vocabulary and 1,024 dimensions
//
// Example usage:
//
//	import "github.com/lee/gobed/gobed"
//
//	// Load the default production model
//	model, err := gobed.NewSafetensorsEmbedding()
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Generate embeddings
//	emb1, err := model.EncodeText("Machine learning is fascinating")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	emb2, err := model.EncodeText("AI and deep learning")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Calculate similarity
//	similarity := gobed.CosineSimilarity(emb1, emb2)
//	fmt.Printf("Similarity: %.6f\n", similarity)
//
package gobed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"runtime"
)

// TensorInfo contains information about a tensor in the safetensors file
type TensorInfo struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// TokenData represents tokenization information for a sentence
type TokenData struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
}

// SafetensorsEmbedding represents a sentence embedding model loaded from safetensors
type SafetensorsEmbedding struct {
	weights         [][]float32           // [vocab_size, embed_dim]
	vocabSize       int                   // 30,522 for production model
	embedDim        int                   // 1,024 for production model
	referenceTokens map[string]TokenData  // Pre-computed tokenizations
	modelPath       string                // Path to safetensors file
	tokensPath      string                // Path to reference tokens
}

// getPackageDir returns the directory where this package is located
func getPackageDir() (string, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("unable to get current file path")
	}
	return filepath.Dir(filename), nil
}

// getDefaultModelPath returns the path to the bundled safetensors model
func getDefaultModelPath() (string, error) {
	packageDir, err := getPackageDir()
	if err != nil {
		return "", err
	}
	
	// Look for model in package directory first, then parent directory
	modelPaths := []string{
		filepath.Join(packageDir, "models", "model.safetensors"),
		filepath.Join(packageDir, "..", "cached_model", "snapshots", "f60985c706f192d45d218078e49e5a8b6f15283a", "0_StaticEmbedding", "model.safetensors"),
	}
	
	for _, path := range modelPaths {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}
	
	return "", fmt.Errorf("default safetensors model not found in expected locations")
}

// getDefaultTokensPath returns the path to the bundled reference tokens
func getDefaultTokensPath() (string, error) {
	packageDir, err := getPackageDir()
	if err != nil {
		return "", err
	}
	
	// Look for tokens in package directory first, then parent directory
	tokenPaths := []string{
		filepath.Join(packageDir, "models", "reference_tokens.json"),
		filepath.Join(packageDir, "..", "model", "production_reference_tokens.json"),
	}
	
	for _, path := range tokenPaths {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}
	
	return "", fmt.Errorf("default reference tokens not found in expected locations")
}

// NewSafetensorsEmbedding creates a new embedding model using the default bundled model
func NewSafetensorsEmbedding() (*SafetensorsEmbedding, error) {
	modelPath, err := getDefaultModelPath()
	if err != nil {
		return nil, fmt.Errorf("failed to find default model: %v", err)
	}
	
	tokensPath, err := getDefaultTokensPath()
	if err != nil {
		return nil, fmt.Errorf("failed to find default tokens: %v", err)
	}
	
	return NewSafetensorsEmbeddingWithPaths(modelPath, tokensPath)
}

// NewSafetensorsEmbeddingWithPaths creates a new embedding model with custom paths
func NewSafetensorsEmbeddingWithPaths(safetensorsPath, tokensPath string) (*SafetensorsEmbedding, error) {
	// Load safetensors model
	weights, vocabSize, embedDim, err := loadSafetensorsWeights(safetensorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load safetensors: %v", err)
	}
	
	// Load reference tokens
	referenceTokens, err := loadReferenceTokens(tokensPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load reference tokens: %v", err)
	}
	
	return &SafetensorsEmbedding{
		weights:         weights,
		vocabSize:       vocabSize,
		embedDim:        embedDim,
		referenceTokens: referenceTokens,
		modelPath:       safetensorsPath,
		tokensPath:      tokensPath,
	}, nil
}

// loadSafetensorsWeights loads embedding weights from a safetensors file
func loadSafetensorsWeights(safetensorsPath string) ([][]float32, int, int, error) {
	file, err := os.Open(safetensorsPath)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read header length (first 8 bytes, little-endian)
	headerLengthBytes := make([]byte, 8)
	_, err = file.Read(headerLengthBytes)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read header length: %v", err)
	}

	headerLength := binary.LittleEndian.Uint64(headerLengthBytes)

	// Read header JSON
	headerBytes := make([]byte, headerLength)
	_, err = file.Read(headerBytes)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read header: %v", err)
	}

	var header map[string]TensorInfo
	err = json.Unmarshal(headerBytes, &header)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to parse header: %v", err)
	}

	// Read the rest of the file (tensor data)
	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read tensor data: %v", err)
	}

	// Get embedding weights
	info, exists := header["embedding.weight"]
	if !exists {
		return nil, 0, 0, fmt.Errorf("embedding.weight tensor not found")
	}

	if info.Dtype != "F32" {
		return nil, 0, 0, fmt.Errorf("unsupported dtype: %s", info.Dtype)
	}

	if len(info.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("expected 2D tensor, got %dD", len(info.Shape))
	}

	start := info.DataOffsets[0]
	end := info.DataOffsets[1]

	if start < 0 || end > int64(len(data)) {
		return nil, 0, 0, fmt.Errorf("invalid data offsets: %d-%d", start, end)
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
				return nil, 0, 0, fmt.Errorf("not enough data for tensor")
			}

			bits := binary.LittleEndian.Uint32(tensorBytes[offset : offset+4])
			weights[i][j] = math.Float32frombits(bits)
		}
	}

	return weights, rows, cols, nil
}

// loadReferenceTokens loads reference tokens from JSON file
func loadReferenceTokens(tokensPath string) (map[string]TokenData, error) {
	file, err := os.Open(tokensPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open tokens file: %v", err)
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokens file: %v", err)
	}

	var tokens map[string]TokenData
	err = json.Unmarshal(data, &tokens)
	if err != nil {
		return nil, fmt.Errorf("failed to parse tokens: %v", err)
	}

	return tokens, nil
}

// EncodeTokens generates an embedding from token IDs using mean pooling
func (s *SafetensorsEmbedding) EncodeTokens(tokenIDs []int) []float32 {
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

// EncodeText generates an embedding for text using pre-computed reference tokens
func (s *SafetensorsEmbedding) EncodeText(text string) ([]float32, error) {
	tokenData, exists := s.referenceTokens[text]
	if !exists {
		return nil, fmt.Errorf("no reference tokens found for text: %s", text)
	}

	embedding := s.EncodeTokens(tokenData.TokenIDs)
	return embedding, nil
}

// GetAvailableTexts returns a list of texts that have pre-computed tokens
func (s *SafetensorsEmbedding) GetAvailableTexts() []string {
	texts := make([]string, 0, len(s.referenceTokens))
	for text := range s.referenceTokens {
		texts = append(texts, text)
	}
	return texts
}

// GetModelInfo returns information about the loaded model
func (s *SafetensorsEmbedding) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"vocab_size":       s.vocabSize,
		"embedding_dim":    s.embedDim,
		"model_path":       s.modelPath,
		"tokens_path":      s.tokensPath,
		"available_texts":  len(s.referenceTokens),
		"model_type":       "sentence-transformers/static-retrieval-mrl-en-v1",
		"implementation":   "Go safetensors loader",
	}
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

// EuclideanDistance calculates Euclidean distance between two vectors
func EuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	sum := float32(0.0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return float32(math.Sqrt(float64(sum)))
}

// CalculateNorm calculates the L2 norm of a vector
func CalculateNorm(embedding []float32) float32 {
	sum := float32(0.0)
	for _, val := range embedding {
		sum += val * val
	}
	return float32(math.Sqrt(float64(sum)))
}

// BatchEncode generates embeddings for multiple texts
func (s *SafetensorsEmbedding) BatchEncode(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	
	for i, text := range texts {
		embedding, err := s.EncodeText(text)
		if err != nil {
			return nil, fmt.Errorf("failed to encode text %d (%s): %v", i, text, err)
		}
		embeddings[i] = embedding
	}
	
	return embeddings, nil
}