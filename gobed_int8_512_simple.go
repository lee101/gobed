package gobed

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// SimpleInt8Model512 is a simple version that works without external C deps
type SimpleInt8Model512 struct {
	embeddings [][]int8  // shape: [vocab_size, 512]
	scales     []float32 // shape: [vocab_size] - scale factor for each embedding
	vocab      map[string]int16  // token to ID mapping
}

// LoadSimpleInt8Model512 loads the int8 model with built-in tokenizer
func LoadSimpleInt8Model512() (*SimpleInt8Model512, error) {
	modelPath := filepath.Join("model", "modelint8_512dim.safetensors")
	tokenizerPath := filepath.Join("model", "tokenizer.json")

	log.Printf(" Loading Simple Int8 512-dim model from %s", modelPath)

	// Load the quantized model
	embeddings, scales, err := loadInt8Embeddings(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings: %v", err)
	}

	// Load simple vocab from tokenizer.json
	vocab, err := loadSimpleVocab(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load vocab: %v", err)
	}

	model := &SimpleInt8Model512{
		embeddings: embeddings,
		scales:     scales,
		vocab:      vocab,
	}

	log.Printf(" Simple Int8 model loaded: vocab=%d, dims=%d, memory=%.1f MB",
		len(embeddings), Int8EmbeddingDim,
		float64(len(embeddings)*Int8EmbeddingDim+len(scales)*4)/(1024*1024))

	return model, nil
}

// loadSimpleVocab loads a simple vocabulary mapping from tokenizer.json
func loadSimpleVocab(path string) (map[string]int16, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var tokenizerData struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}

	if err := json.Unmarshal(data, &tokenizerData); err != nil {
		return nil, err
	}

	vocab := make(map[string]int16)
	for token, id := range tokenizerData.Model.Vocab {
		if id < 32768 { // Fits in int16
			vocab[token] = int16(id)
		}
	}

	log.Printf("Loaded %d tokens into vocab", len(vocab))
	return vocab, nil
}

// SimpleTokenize performs basic tokenization (space-separated + subword)
func (m *SimpleInt8Model512) SimpleTokenize(text string) []int16 {
	text = strings.ToLower(strings.TrimSpace(text))
	words := strings.Fields(text)

	var tokens []int16

	// Add [CLS] token (typically ID 101)
	if clsID, ok := m.vocab["[CLS]"]; ok {
		tokens = append(tokens, clsID)
	}

	for _, word := range words {
		// Try exact match first
		if id, ok := m.vocab[word]; ok {
			tokens = append(tokens, id)
			continue
		}

		// Try with ## prefix (BERT subword tokens)
		if id, ok := m.vocab["##"+word]; ok {
			tokens = append(tokens, id)
			continue
		}

		// Try word pieces (simple splitting)
		found := false
		for token, id := range m.vocab {
			if strings.Contains(word, strings.TrimPrefix(token, "##")) && len(token) > 3 {
				tokens = append(tokens, id)
				found = true
				break
			}
		}

		// Fallback to [UNK] token
		if !found {
			if unkID, ok := m.vocab["[UNK]"]; ok {
				tokens = append(tokens, unkID)
			}
		}
	}

	// Add [SEP] token (typically ID 102)
	if sepID, ok := m.vocab["[SEP]"]; ok {
		tokens = append(tokens, sepID)
	}

	return tokens
}

// EmbedTokens embeds int16 token IDs directly
func (m *SimpleInt8Model512) EmbedTokens(tokens []int16) ([]float32, error) {
	if len(tokens) == 0 {
		return make([]float32, Int8EmbeddingDim), nil
	}

	// Initialize result vector
	result := make([]float32, Int8EmbeddingDim)

	// Average the embeddings
	validTokens := 0
	for _, token := range tokens {
		if token < 0 || int(token) >= len(m.embeddings) {
			continue // Skip invalid tokens
		}

		embedding := m.embeddings[token]
		scale := m.scales[token]

		// Dequantize and add to result
		for j := 0; j < Int8EmbeddingDim; j++ {
			result[j] += float32(embedding[j]) * scale
		}
		validTokens++
	}

	// Average
	if validTokens > 0 {
		count := float32(validTokens)
		for i := range result {
			result[i] /= count
		}
	}

	return result, nil
}

// Embed embeds text using simple int8 model
func (m *SimpleInt8Model512) Embed(text string) ([]float32, error) {
	tokens := m.SimpleTokenize(text)
	return m.EmbedTokens(tokens)
}

// EmbedInt8 returns int8 quantized embedding
func (m *SimpleInt8Model512) EmbedInt8(text string) (*Int8Result512, error) {
	tokens := m.SimpleTokenize(text)

	if len(tokens) == 0 {
		return &Int8Result512{
			Vector: make([]int8, Int8EmbeddingDim),
			Scale:  1.0,
		}, nil
	}

	// For single token, return the int8 embedding directly
	if len(tokens) == 1 {
		token := tokens[0]
		if token >= 0 && int(token) < len(m.embeddings) {
			return &Int8Result512{
				Vector: m.embeddings[token],
				Scale:  m.scales[token],
			}, nil
		}
	}

	// For multiple tokens, average and requantize
	embedding, err := m.EmbedTokens(tokens)
	if err != nil {
		return nil, err
	}

	// Quantize the averaged embedding
	return quantizeVector512(embedding), nil
}

// Similarity computes cosine similarity between two texts
func (m *SimpleInt8Model512) Similarity(text1, text2 string) (float32, error) {
	emb1, err := m.EmbedInt8(text1)
	if err != nil {
		return 0, err
	}

	emb2, err := m.EmbedInt8(text2)
	if err != nil {
		return 0, err
	}

	// Compute int8 dot product
	dotProduct := int32(0)
	norm1 := int32(0)
	norm2 := int32(0)

	for i := 0; i < Int8EmbeddingDim; i++ {
		v1 := int32(emb1.Vector[i])
		v2 := int32(emb2.Vector[i])

		dotProduct += v1 * v2
		norm1 += v1 * v1
		norm2 += v2 * v2
	}

	// Apply scales
	scaledDot := float32(dotProduct) * emb1.Scale * emb2.Scale
	scaledNorm1 := float32(norm1) * emb1.Scale * emb1.Scale
	scaledNorm2 := float32(norm2) * emb2.Scale * emb2.Scale

	// Compute cosine similarity
	if scaledNorm1 == 0 || scaledNorm2 == 0 {
		return 0, nil
	}

	return scaledDot / float32(math.Sqrt(float64(scaledNorm1*scaledNorm2))), nil
}