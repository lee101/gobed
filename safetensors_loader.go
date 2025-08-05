package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
)

// SafetensorsHeader represents the header of a safetensors file
type SafetensorsHeader struct {
	Tensors map[string]TensorInfo `json:"__metadata__,omitempty"`
}

// TensorInfo contains information about a tensor in the safetensors file
type TensorInfo struct {
	Dtype   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// SafetensorsLoader loads tensors from a safetensors file
type SafetensorsLoader struct {
	filePath string
	header   map[string]TensorInfo
	data     []byte
}

// NewSafetensorsLoader creates a new safetensors loader
func NewSafetensorsLoader(filePath string) (*SafetensorsLoader, error) {
	file, err := os.Open(filePath)
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
	
	return &SafetensorsLoader{
		filePath: filePath,
		header:   header,
		data:     data,
	}, nil
}

// GetTensor loads a tensor by name and returns it as float32 slice
func (s *SafetensorsLoader) GetTensor(name string) ([][]float32, error) {
	info, exists := s.header[name]
	if !exists {
		return nil, fmt.Errorf("tensor %s not found", name)
	}
	
	if info.Dtype != "F32" {
		return nil, fmt.Errorf("unsupported dtype: %s", info.Dtype)
	}
	
	if len(info.Shape) != 2 {
		return nil, fmt.Errorf("expected 2D tensor, got %dD", len(info.Shape))
	}
	
	start := info.DataOffsets[0]
	end := info.DataOffsets[1]
	
	if start < 0 || end > int64(len(s.data)) {
		return nil, fmt.Errorf("invalid data offsets: %d-%d", start, end)
	}
	
	tensorBytes := s.data[start:end]
	
	// Convert bytes to float32 values
	rows := info.Shape[0]
	cols := info.Shape[1]
	
	tensor := make([][]float32, rows)
	for i := range tensor {
		tensor[i] = make([]float32, cols)
	}
	
	// Read float32 values (little-endian)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			offset := (i*cols + j) * 4 // 4 bytes per float32
			if offset+4 > len(tensorBytes) {
				return nil, fmt.Errorf("not enough data for tensor")
			}
			
			bits := binary.LittleEndian.Uint32(tensorBytes[offset : offset+4])
			tensor[i][j] = math.Float32frombits(bits)
		}
	}
	
	return tensor, nil
}

// EmbeddingModel represents the actual embedding model with safetensors weights
type RealEmbeddingModel struct {
	weights   [][]float32 // [vocab_size, embed_dim]
	vocabSize int
	embedDim  int
}

// NewRealEmbeddingModel creates an embedding model from safetensors
func NewRealEmbeddingModel(safetensorsPath string) (*RealEmbeddingModel, error) {
	loader, err := NewSafetensorsLoader(safetensorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load safetensors: %v", err)
	}
	
	weights, err := loader.GetTensor("embedding.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding weights: %v", err)
	}
	
	fmt.Printf("Loaded embedding weights: [%d, %d]\n", len(weights), len(weights[0]))
	
	return &RealEmbeddingModel{
		weights:   weights,
		vocabSize: len(weights),
		embedDim:  len(weights[0]),
	}, nil
}

// Forward performs forward pass with actual weights
func (r *RealEmbeddingModel) Forward(tokenIDs []int) []float32 {
	embedding := make([]float32, r.embedDim)
	validTokens := 0
	
	for _, tokenID := range tokenIDs {
		if tokenID > 0 && tokenID < r.vocabSize { // Skip padding tokens (0)
			// Add embedding for this token
			for i := 0; i < r.embedDim; i++ {
				embedding[i] += r.weights[tokenID][i]
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

// TokenData represents tokenization information
type TokenData struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
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
	fmt.Println("Go Safetensors Embedding Test")
	fmt.Println("=============================")
	
	// Load the actual model weights from safetensors
	model, err := NewRealEmbeddingModel("cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors")
	if err != nil {
		fmt.Printf("Failed to load model: %v\n", err)
		fmt.Println("Note: This requires the actual safetensors file.")
		return
	}
	
	// Load reference tokens
	var referenceTokens map[string]TokenData
	tokensFile, err := os.Open("model/production_reference_tokens.json")
	if err != nil {
		fmt.Printf("Failed to open reference tokens: %v\n", err)
		return
	}
	defer tokensFile.Close()
	
	tokensData, err := ioutil.ReadAll(tokensFile)
	if err != nil {
		fmt.Printf("Failed to read reference tokens: %v\n", err)
		return
	}
	
	err = json.Unmarshal(tokensData, &referenceTokens)
	if err != nil {
		fmt.Printf("Failed to parse reference tokens: %v\n", err)
		return
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
	
	fmt.Println("\nGenerating embeddings with actual weights...")
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
	fmt.Println("\nGo Safetensors Similarity Matrix:")
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
	
	fmt.Println("\nGo safetensors embedding test completed!")
	fmt.Println("This uses the actual model weights loaded from safetensors.")
}