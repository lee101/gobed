package gobed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unicode"
)

const (
	// Model dimensions
	Int8EmbeddingDim = 512
	Int8VocabSize    = 30522
)

// wordPieceTokenizer implements a minimal WordPiece tokenizer optimized for BERT-style vocabularies.
type wordPieceTokenizer struct {
	vocab map[string]int16

	clsID int16
	sepID int16
	unkID int16

	addSpecial bool
}

// tokenize splits text using a lightweight WordPiece algorithm and returns int16 token IDs.
func (t *wordPieceTokenizer) tokenize(text string) ([]int16, error) {
	if len(t.vocab) == 0 {
		return nil, fmt.Errorf("wordpiece vocab not initialized")
	}

	text = strings.ToLower(text)
	words := basicWordSplit(text)

	result := make([]int16, 0, len(words)+2)
	if t.addSpecial && t.clsID >= 0 {
		result = append(result, t.clsID)
	}

	for _, word := range words {
		if id, ok := t.vocab[word]; ok {
			result = append(result, id)
			continue
		}

		subwordIDs := t.tokenizeWordPiece(word)
		if len(subwordIDs) == 0 {
			if t.unkID < 0 {
				return nil, fmt.Errorf("unknown token %q and [UNK] not in vocab", word)
			}
			result = append(result, t.unkID)
			continue
		}
		result = append(result, subwordIDs...)
	}

	if t.addSpecial && t.sepID >= 0 {
		result = append(result, t.sepID)
	}

	return result, nil
}

func (t *wordPieceTokenizer) tokenizeWordPiece(word string) []int16 {
	if word == "" {
		return nil
	}

	var ids []int16
	chars := []rune(word)
	start := 0

	for start < len(chars) {
		end := len(chars)
		var matched int16 = -1
		var matchedLen int

		for end > start {
			sub := string(chars[start:end])
			if start > 0 {
				sub = "##" + sub
			}

			if id, ok := t.vocab[sub]; ok {
				matched = id
				matchedLen = end - start
				break
			}
			end--
		}

		if matched == -1 {
			return nil
		}

		ids = append(ids, matched)
		start += matchedLen
	}

	return ids
}

// basicWordSplit performs a simple word split focusing on ASCII letters/digits and treating punctuation as separators.
func basicWordSplit(text string) []string {
	var tokens []string
	var buf strings.Builder

	flush := func() {
		if buf.Len() > 0 {
			tokens = append(tokens, buf.String())
			buf.Reset()
		}
	}

	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case isWordChar(r):
			buf.WriteRune(r)
		default:
			flush()
			if !unicode.IsSpace(r) {
				tokens = append(tokens, string(r))
			}
		}
	}
	flush()
	return tokens
}

func isWordChar(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' || r == '_'
}

// tokenizeZeroAlloc tokenizes directly into result slice, returns token count
// Uses a pre-allocated scratch buffer for word processing
func (t *wordPieceTokenizer) tokenizeZeroAlloc(text string, result []int16, scratch []byte) int {
	if len(t.vocab) == 0 {
		return 0
	}

	count := 0
	maxLen := len(result)

	// Add CLS token
	if t.addSpecial && t.clsID >= 0 && count < maxLen {
		result[count] = t.clsID
		count++
	}

	// Process text directly without allocating word slices
	textBytes := []byte(strings.ToLower(text))
	wordStart := -1

	processWord := func(word []byte) {
		if len(word) == 0 || count >= maxLen {
			return
		}

		wordStr := string(word)
		if id, ok := t.vocab[wordStr]; ok {
			result[count] = id
			count++
			return
		}

		// WordPiece tokenization inline
		start := 0
		for start < len(word) && count < maxLen {
			end := len(word)
			var matched int16 = -1

			for end > start {
				var sub string
				if start > 0 {
					sub = "##" + string(word[start:end])
				} else {
					sub = string(word[start:end])
				}

				if id, ok := t.vocab[sub]; ok {
					matched = id
					result[count] = id
					count++
					start = end
					break
				}
				end--
			}

			if matched == -1 {
				if t.unkID >= 0 && count < maxLen {
					result[count] = t.unkID
					count++
				}
				break
			}
		}
	}

	for i := 0; i < len(textBytes); i++ {
		b := textBytes[i]
		isWord := (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') || b == '\'' || b == '_'

		if isWord {
			if wordStart < 0 {
				wordStart = i
			}
		} else {
			if wordStart >= 0 {
				processWord(textBytes[wordStart:i])
				wordStart = -1
			}
			// Handle punctuation as single-char token
			if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
				if id, ok := t.vocab[string(b)]; ok && count < maxLen {
					result[count] = id
					count++
				}
			}
		}
	}

	// Process last word
	if wordStart >= 0 {
		processWord(textBytes[wordStart:])
	}

	// Add SEP token
	if t.addSpecial && t.sepID >= 0 && count < maxLen {
		result[count] = t.sepID
		count++
	}

	return count
}

func newWordPieceTokenizer(vocab map[string]int16) *wordPieceTokenizer {
	tk := &wordPieceTokenizer{
		vocab:      vocab,
		clsID:      -1,
		sepID:      -1,
		unkID:      -1,
		addSpecial: true,
	}

	if id, ok := vocab["[CLS]"]; ok {
		tk.clsID = id
	}
	if id, ok := vocab["[SEP]"]; ok {
		tk.sepID = id
	}
	if id, ok := vocab["[UNK]"]; ok {
		tk.unkID = id
	}

	return tk
}

// Int8EmbeddingModel512 represents the int8 quantized model with 512 dimensions
type Int8EmbeddingModel512 struct {
	// Embeddings stored as int8 for memory efficiency
	embeddings [][]int8  // shape: [vocab_size, 512]
	scales     []float32 // shape: [vocab_size] - scale factor for each embedding

	tokenizer *wordPieceTokenizer

	// Cache for frequently used embeddings
	cache     sync.Map
	cacheHits uint64
}

// LoadInt8Model512 loads the int8 quantized model with 512 dimensions
func LoadInt8Model512() (*Int8EmbeddingModel512, error) {
	modelPath := filepath.Join("model", "modelint8_512dim.safetensors")
	tokenizerPath := filepath.Join("model", "tokenizer.json")

	Debugf(" Loading Int8 512-dim model from %s", modelPath)

	// Load the quantized model
	embeddings, scales, err := loadInt8Embeddings(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings: %v", err)
	}

	vocab, err := loadSimpleVocab(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer vocab: %v", err)
	}

	tk := newWordPieceTokenizer(vocab)

	model := &Int8EmbeddingModel512{
		embeddings: embeddings,
		scales:     scales,
		tokenizer:  tk,
	}

	Debugf(" Int8 model loaded: vocab=%d, dims=%d, memory=%.1f MB",
		len(embeddings), Int8EmbeddingDim,
		float64(len(embeddings)*Int8EmbeddingDim+len(scales)*4)/(1024*1024))

	return model, nil
}

// loadInt8Embeddings loads the int8 embeddings from safetensors file
func loadInt8Embeddings(path string) ([][]int8, []float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	// Read header length (8 bytes, little-endian uint64)
	var headerLen uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLen); err != nil {
		return nil, nil, fmt.Errorf("failed to read header length: %v", err)
	}

	// Read header JSON
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return nil, nil, fmt.Errorf("failed to read header: %v", err)
	}

	var header map[string]interface{}
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, nil, fmt.Errorf("failed to parse header: %v", err)
	}

	// Find embeddings and scales tensors
	var embeddingsInfo, scalesInfo map[string]interface{}

	for name, info := range header {
		if name == "__metadata__" {
			continue
		}
		infoMap := info.(map[string]interface{})

		if strings.Contains(name, "embeddings.weight") {
			embeddingsInfo = infoMap
		} else if strings.Contains(name, "embeddings.scales") {
			scalesInfo = infoMap
		}
	}

	if embeddingsInfo == nil || scalesInfo == nil {
		return nil, nil, fmt.Errorf("embeddings or scales not found in model")
	}

	// Load embeddings (int8)
	embeddings, err := loadInt8Tensor(file, embeddingsInfo, 8+int64(headerLen))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load embeddings: %v", err)
	}

	// Load scales (float32)
	scales, err := loadFloat32Tensor(file, scalesInfo, 8+int64(headerLen))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load scales: %v", err)
	}

	// Reshape embeddings from flat array to 2D
	vocabSize := len(scales)
	embeddingDim := len(embeddings) / vocabSize

	if embeddingDim != Int8EmbeddingDim {
		return nil, nil, fmt.Errorf("unexpected embedding dimension: got %d, expected %d",
			embeddingDim, Int8EmbeddingDim)
	}

	// Convert to 2D array
	embeddings2D := make([][]int8, vocabSize)
	for i := 0; i < vocabSize; i++ {
		embeddings2D[i] = embeddings[i*embeddingDim : (i+1)*embeddingDim]
	}

	return embeddings2D, scales, nil
}

// loadInt8Tensor loads an int8 tensor from safetensors
func loadInt8Tensor(file *os.File, info map[string]interface{}, baseOffset int64) ([]int8, error) {
	offsets := info["data_offsets"].([]interface{})
	start := int64(offsets[0].(float64))
	end := int64(offsets[1].(float64))
	size := end - start

	// Seek to tensor data
	if _, err := file.Seek(baseOffset+start, 0); err != nil {
		return nil, err
	}

	// Read int8 data
	data := make([]int8, size)
	buf := make([]byte, size)
	if _, err := io.ReadFull(file, buf); err != nil {
		return nil, err
	}

	// Convert bytes to int8
	for i, b := range buf {
		data[i] = int8(b)
	}

	return data, nil
}

// loadFloat32Tensor loads a float32 tensor from safetensors
func loadFloat32Tensor(file *os.File, info map[string]interface{}, baseOffset int64) ([]float32, error) {
	offsets := info["data_offsets"].([]interface{})
	start := int64(offsets[0].(float64))
	end := int64(offsets[1].(float64))
	size := end - start

	// Seek to tensor data
	if _, err := file.Seek(baseOffset+start, 0); err != nil {
		return nil, err
	}

	// Read float32 data
	numFloats := size / 4
	data := make([]float32, numFloats)
	buf := make([]byte, size)
	if _, err := io.ReadFull(file, buf); err != nil {
		return nil, err
	}

	// Convert bytes to float32
	for i := 0; i < int(numFloats); i++ {
		bits := binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4])
		data[i] = math.Float32frombits(bits)
	}

	return data, nil
}

// Tokenize converts text to int16 token IDs using the in-process wordpiece tokenizer.
func (m *Int8EmbeddingModel512) Tokenize(text string) ([]int16, error) {
	if m.tokenizer == nil {
		return nil, fmt.Errorf("int8 tokenizer not initialized")
	}

	tokens, err := m.tokenizer.tokenize(text)
	if err != nil {
		return nil, err
	}

	for i, id := range tokens {
		if id < 0 || id >= Int8VocabSize {
			return nil, fmt.Errorf("token ID %d at position %d exceeds int16 range", id, i)
		}
	}
	return tokens, nil
}

// EmbedTokens embeds int16 token IDs directly (no tokenization needed)
func (m *Int8EmbeddingModel512) EmbedTokens(tokens []int16) ([]float32, error) {
	if len(tokens) == 0 {
		return make([]float32, Int8EmbeddingDim), nil
	}

	// Initialize result vector
	result := make([]float32, Int8EmbeddingDim)

	// Average the embeddings
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
	}

	// Average
	count := float32(len(tokens))
	for i := range result {
		result[i] /= count
	}

	return result, nil
}

// Embed embeds text using int8 model
func (m *Int8EmbeddingModel512) Embed(text string) ([]float32, error) {
	// Check cache
	if cached, ok := m.cache.Load(text); ok {
		m.cacheHits++
		return cached.([]float32), nil
	}

	tokens, err := m.Tokenize(text)
	if err != nil {
		return nil, err
	}

	embedding, err := m.EmbedTokens(tokens)
	if err != nil {
		return nil, err
	}

	// Cache the result
	m.cache.Store(text, embedding)

	return embedding, nil
}

// EmbedInt8Result represents the result with int8 vector and scale
type Int8Result512 struct {
	Vector []int8  // 512-dimensional int8 vector
	Scale  float32 // Scale factor for dequantization
}

// EmbedInt8 returns int8 quantized embedding
func (m *Int8EmbeddingModel512) EmbedInt8(text string) (*Int8Result512, error) {
	tokens, err := m.Tokenize(text)
	if err != nil {
		return nil, err
	}

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

// quantizeVector512 quantizes a float32 vector to int8
func quantizeVector512(vec []float32) *Int8Result512 {
	// Find max absolute value
	maxAbs := float32(0)
	for _, v := range vec {
		if abs := float32(math.Abs(float64(v))); abs > maxAbs {
			maxAbs = abs
		}
	}

	// Calculate scale
	scale := maxAbs / 127.0
	if scale == 0 {
		scale = 1.0
	}

	// Quantize
	result := make([]int8, len(vec))
	for i, v := range vec {
		quantized := int8(math.Round(float64(v / scale)))
		if quantized > 127 {
			quantized = 127
		} else if quantized < -128 {
			quantized = -128
		}
		result[i] = quantized
	}

	return &Int8Result512{
		Vector: result,
		Scale:  scale,
	}
}

// Similarity computes cosine similarity between two texts using int8 embeddings
func (m *Int8EmbeddingModel512) Similarity(text1, text2 string) (float32, error) {
	emb1, err := m.EmbedInt8(text1)
	if err != nil {
		return 0, err
	}

	emb2, err := m.EmbedInt8(text2)
	if err != nil {
		return 0, err
	}

	// Compute int8 dot product (faster than dequantizing)
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

// GetMemoryUsage returns the memory usage of the model
func (m *Int8EmbeddingModel512) GetMemoryUsage() string {
	// Calculate memory usage
	embeddingBytes := len(m.embeddings) * Int8EmbeddingDim // int8 embeddings
	scaleBytes := len(m.scales) * 4                        // float32 scales
	totalBytes := embeddingBytes + scaleBytes

	// Get cache size estimate
	cacheSize := 0
	m.cache.Range(func(_, _ interface{}) bool {
		cacheSize++
		return true
	})

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	return fmt.Sprintf(
		"Model Memory: %.1f MB (embeddings: %.1f MB, scales: %.1f MB)\n"+
			"Cache: %d entries, %d hits\n"+
			"System Memory: %.1f MB allocated, %.1f MB sys",
		float64(totalBytes)/(1024*1024),
		float64(embeddingBytes)/(1024*1024),
		float64(scaleBytes)/(1024*1024),
		cacheSize, m.cacheHits,
		float64(ms.Alloc)/(1024*1024),
		float64(ms.Sys)/(1024*1024),
	)
}
