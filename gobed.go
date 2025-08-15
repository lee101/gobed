package gobed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sort"
	"time"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// EmbeddingModel provides a clean API for text embeddings using the real static-retrieval-mrl-en-v1 model
type EmbeddingModel struct {
	VocabSize       int
	EmbedDim        int
	weights         [][]float32 // Real safetensors weights [vocab_size, embed_dim]
	referenceTokens map[string]TokenData
	embeddingBuffer []float32 // Pre-allocated for performance
	tokenizer       *tokenizer.Tokenizer // BERT tokenizer for arbitrary text
}

// TensorInfo contains safetensors tensor metadata
type TensorInfo struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// TokenData represents tokenization from the real model
type TokenData struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
}

// SimilarityResult represents a similarity comparison
type SimilarityResult struct {
	Text1      string
	Text2      string
	Similarity float32
}

// LoadModel loads the real static-retrieval-mrl-en-v1 embedding model
func LoadModel() (*EmbeddingModel, error) {
	fmt.Println("🔄 Loading real static-retrieval-mrl-en-v1 model...")
	start := time.Now()

	// Load real safetensors weights
	safetensorsPath := "model/real_model.safetensors"
	// Check alternative paths
	if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
		safetensorsPath = "../../model/real_model.safetensors"
		if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
			safetensorsPath = "./model/real_model.safetensors"
		}
	}
	if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("real model file not found: %s", safetensorsPath)
	}

	weights, vocabSize, embedDim, err := loadRealSafetensors(safetensorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load safetensors: %v", err)
	}

	// Load tokenizer
	tokenizerPath := "model/tokenizer.json"
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		tokenizerPath = "../../model/tokenizer.json"
		if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
			tokenizerPath = "./model/tokenizer.json"
		}
	}
	// Load the actual static-retrieval-mrl-en-v1 tokenizer
	var tk *tokenizer.Tokenizer
	
	// Try loading from JSON file
	if _, statErr := os.Stat(tokenizerPath); statErr == nil {
		var tokErr error
		tk, tokErr = pretrained.FromFile(tokenizerPath)
		if tokErr != nil {
			fmt.Printf("Warning: failed to load tokenizer from %s: %v\n", tokenizerPath, tokErr)
			tk = nil
		}
	}
	
	// If loading failed, skip tokenizer (fall back to reference tokens only)
	if tk == nil {
		fmt.Println("Warning: tokenizer not available, using reference tokens only")
	}

	// Load real reference tokens (for backward compatibility)
	tokensPath := "model/real_reference_tokens.json"
	if _, err := os.Stat(tokensPath); os.IsNotExist(err) {
		tokensPath = "../../model/real_reference_tokens.json"
		if _, err := os.Stat(tokensPath); os.IsNotExist(err) {
			tokensPath = "./model/real_reference_tokens.json"
		}
	}
	referenceTokens, err := loadReferenceTokens(tokensPath)
	if err != nil {
		fmt.Printf("Warning: failed to load reference tokens: %v\n", err)
		referenceTokens = make(map[string]TokenData) // Initialize empty map
	}

	model := &EmbeddingModel{
		VocabSize:       vocabSize,
		EmbedDim:        embedDim,
		weights:         weights,
		referenceTokens: referenceTokens,
		embeddingBuffer: make([]float32, embedDim),
		tokenizer:       tk,
	}

	loadTime := time.Since(start)
	fmt.Printf("✅ Model loaded in %v (vocab: %d, dims: %d)\n", loadTime, vocabSize, embedDim)
	return model, nil
}

// Encode converts text to embedding vector using real model weights
func (m *EmbeddingModel) Encode(text string) ([]float32, error) {
	// First try tokenizer for arbitrary text
	if m.tokenizer != nil {
		return m.encodeWithTokenizer(text)
	}
	
	// Fallback: check if we have reference tokens for this text
	tokenData, exists := m.referenceTokens[text]
	if !exists {
		return nil, fmt.Errorf("text not in reference tokens and no tokenizer available: %s", text)
	}

	return m.computeEmbedding(tokenData.TokenIDs)
}

// encodeWithTokenizer tokenizes and encodes arbitrary text
func (m *EmbeddingModel) encodeWithTokenizer(text string) ([]float32, error) {
	// Tokenize the text
    encoding, err := m.tokenizer.EncodeSingle(text, false)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %v", err)
	}
	
	// Convert uint32 token IDs to int
	tokenIDs := make([]int, len(encoding.Ids))
	for i, id := range encoding.Ids {
		tokenIDs[i] = int(id)
	}
	
	return m.computeEmbedding(tokenIDs)
}

// computeEmbedding performs the actual embedding computation with real weights
func (m *EmbeddingModel) computeEmbedding(tokenIDs []int) ([]float32, error) {
	// Reset buffer
	for i := range m.embeddingBuffer {
		m.embeddingBuffer[i] = 0
	}

	validTokens := 0

	// Sum embeddings for all tokens (using real weights)
	for _, tokenID := range tokenIDs {
		if tokenID > 0 && tokenID < m.VocabSize {
			weightRow := m.weights[tokenID]
			for i := 0; i < m.EmbedDim; i++ {
				m.embeddingBuffer[i] += weightRow[i]
			}
			validTokens++
		}
	}

	// Mean pooling (exactly like StaticEmbedding model)
	if validTokens > 0 {
		invValidTokens := 1.0 / float32(validTokens)
		for i := range m.embeddingBuffer {
			m.embeddingBuffer[i] *= invValidTokens
		}
	}

	// StaticEmbedding does NOT normalize - return raw mean pooled values
	result := make([]float32, m.EmbedDim)
	copy(result, m.embeddingBuffer)
	return result, nil
}

// Similarity calculates cosine similarity between two texts
func (m *EmbeddingModel) Similarity(text1, text2 string) (float32, error) {
	emb1, err := m.Encode(text1)
	if err != nil {
		return 0, fmt.Errorf("failed to encode text1: %v", err)
	}

	emb2, err := m.Encode(text2)
	if err != nil {
		return 0, fmt.Errorf("failed to encode text2: %v", err)
	}

	return CosineSimilarity(emb1, emb2), nil
}

// FindMostSimilar finds the most similar texts to a query from a list of candidates
func (m *EmbeddingModel) FindMostSimilar(query string, candidates []string, limit int) ([]SimilarityResult, error) {
	queryEmb, err := m.Encode(query)
	if err != nil {
		return nil, fmt.Errorf("failed to encode query: %v", err)
	}

	var results []SimilarityResult
	for _, candidate := range candidates {
		candEmb, err := m.Encode(candidate)
		if err != nil {
			continue // Skip texts that can't be encoded
		}

		sim := CosineSimilarity(queryEmb, candEmb)
		results = append(results, SimilarityResult{
			Text1:      query,
			Text2:      candidate,
			Similarity: sim,
		})
	}

	// Sort by similarity (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Return top N results
	if limit > 0 && limit < len(results) {
		results = results[:limit]
	}

	return results, nil
}

// GetAvailableTexts returns all texts that can be encoded (from reference tokens)
func (m *EmbeddingModel) GetAvailableTexts() []string {
	texts := make([]string, 0, len(m.referenceTokens))
	for text := range m.referenceTokens {
		texts = append(texts, text)
	}
	return texts
}

// CosineSimilarity calculates cosine similarity between two vectors
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
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

// loadRealSafetensors loads the actual model weights
func loadRealSafetensors(filePath string) ([][]float32, int, int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()

	// Read header
	headerLengthBytes := make([]byte, 8)
	if _, err := file.Read(headerLengthBytes); err != nil {
		return nil, 0, 0, err
	}

	headerLength := binary.LittleEndian.Uint64(headerLengthBytes)
	headerBytes := make([]byte, headerLength)
	if _, err := file.Read(headerBytes); err != nil {
		return nil, 0, 0, err
	}

	var header map[string]TensorInfo
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, 0, 0, err
	}

	// Read tensor data
	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, 0, 0, err
	}

	// Get embedding weights
	info, exists := header["embedding.weight"]
	if !exists {
		return nil, 0, 0, fmt.Errorf("embedding.weight not found")
	}

	if info.Dtype != "F32" || len(info.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("unsupported tensor format")
	}

	start, end := info.DataOffsets[0], info.DataOffsets[1]
	tensorBytes := data[start:end]
	rows, cols := info.Shape[0], info.Shape[1]

	// Load weights
	weights := make([][]float32, rows)
	for i := range weights {
		weights[i] = make([]float32, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			offset := (i*cols + j) * 4
			bits := binary.LittleEndian.Uint32(tensorBytes[offset : offset+4])
			weights[i][j] = math.Float32frombits(bits)
		}
	}

	return weights, rows, cols, nil
}

// loadReferenceTokens loads the tokenization data
func loadReferenceTokens(filePath string) (map[string]TokenData, error) {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	var tokens map[string]TokenData
	err = json.Unmarshal(data, &tokens)
	return tokens, err
}