package gobed

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// Buffer pools for zero-allocation hot paths
var (
	float32Pool512 = sync.Pool{
		New: func() interface{} {
			buf := make([]float32, Int8EmbeddingDim)
			return &buf
		},
	}
	int8Pool512 = sync.Pool{
		New: func() interface{} {
			buf := make([]int8, Int8EmbeddingDim)
			return &buf
		},
	}
	tokenPool = sync.Pool{
		New: func() interface{} {
			buf := make([]int16, 0, 128)
			return &buf
		},
	}
)

// SimpleInt8Model512 is a simple version that works without external C deps
type SimpleInt8Model512 struct {
	embeddings [][]int8         // shape: [vocab_size, 512]
	scales     []float32        // shape: [vocab_size] - scale factor for each embedding
	vocab      map[string]int16 // token to ID mapping
	tokenizer  *wordPieceTokenizer
}

var simpleInt8ModelOnce sync.Once
var simpleInt8ModelInstance *SimpleInt8Model512
var simpleInt8ModelError error
var simpleInt8Verbose = false

// SetSimpleInt8Verbose controls whether model loading logs are printed
func SetSimpleInt8Verbose(verbose bool) {
	simpleInt8Verbose = verbose
}

// LoadSimpleInt8Model512 loads the int8 model with built-in tokenizer (singleton)
func LoadSimpleInt8Model512() (*SimpleInt8Model512, error) {
	simpleInt8ModelOnce.Do(func() {
		simpleInt8ModelInstance, simpleInt8ModelError = loadSimpleInt8Model512Internal()
	})
	return simpleInt8ModelInstance, simpleInt8ModelError
}

func loadSimpleInt8Model512Internal() (*SimpleInt8Model512, error) {
	modelPath := filepath.Join("model", "modelint8_512dim.safetensors")
	tokenizerPath := filepath.Join("model", "tokenizer.json")

	if simpleInt8Verbose {
		log.Printf("Loading Simple Int8 512-dim model from %s", modelPath)
	}

	embeddings, scales, err := loadInt8Embeddings(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings: %v", err)
	}

	vocab, err := loadSimpleVocab(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load vocab: %v", err)
	}

	model := &SimpleInt8Model512{
		embeddings: embeddings,
		scales:     scales,
		vocab:      vocab,
		tokenizer:  newWordPieceTokenizer(vocab),
	}

	if simpleInt8Verbose {
		log.Printf("Simple Int8 model loaded: vocab=%d, dims=%d, memory=%.1f MB",
			len(embeddings), Int8EmbeddingDim,
			float64(len(embeddings)*Int8EmbeddingDim+len(scales)*4)/(1024*1024))
	}

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

	if simpleInt8Verbose {
		log.Printf("Loaded %d tokens into vocab", len(vocab))
	}
	return vocab, nil
}

// SimpleTokenize performs basic tokenization (space-separated + subword)
func (m *SimpleInt8Model512) SimpleTokenize(text string) []int16 {
	if m.tokenizer == nil {
		return nil
	}
	tokens, err := m.tokenizer.tokenize(text)
	if err != nil {
		return []int16{}
	}
	return tokens
}

// EmbeddingTable returns the vocab embedding matrix (read-only).
func (m *SimpleInt8Model512) EmbeddingTable() [][]int8 {
	return m.embeddings
}

// ScaleTable returns per-token quantization scales (read-only).
func (m *SimpleInt8Model512) ScaleTable() []float32 {
	return m.scales
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

// EmbedTokensInto embeds tokens into a pre-allocated buffer (zero-alloc hot path)
func (m *SimpleInt8Model512) EmbedTokensInto(tokens []int16, result []float32) int {
	// Clear result
	for i := range result {
		result[i] = 0
	}

	validTokens := 0
	vocabLen := len(m.embeddings)

	for _, token := range tokens {
		if token < 0 || int(token) >= vocabLen {
			continue
		}

		embedding := m.embeddings[token]
		scale := m.scales[token]

		// Unrolled loop for 512 dims - process 8 at a time
		for j := 0; j < Int8EmbeddingDim; j += 8 {
			result[j] += float32(embedding[j]) * scale
			result[j+1] += float32(embedding[j+1]) * scale
			result[j+2] += float32(embedding[j+2]) * scale
			result[j+3] += float32(embedding[j+3]) * scale
			result[j+4] += float32(embedding[j+4]) * scale
			result[j+5] += float32(embedding[j+5]) * scale
			result[j+6] += float32(embedding[j+6]) * scale
			result[j+7] += float32(embedding[j+7]) * scale
		}
		validTokens++
	}

	if validTokens > 1 {
		invCount := 1.0 / float32(validTokens)
		for i := 0; i < Int8EmbeddingDim; i += 8 {
			result[i] *= invCount
			result[i+1] *= invCount
			result[i+2] *= invCount
			result[i+3] *= invCount
			result[i+4] *= invCount
			result[i+5] *= invCount
			result[i+6] *= invCount
			result[i+7] *= invCount
		}
	}

	return validTokens
}

// Token buffer pool with fixed size for zero-alloc tokenization
var tokenBufPool = sync.Pool{
	New: func() interface{} {
		buf := make([]int16, 256) // max 256 tokens
		return &buf
	},
}

// EmbedFast is the zero-allocation embedding path using buffer pools
func (m *SimpleInt8Model512) EmbedFast(text string) ([]float32, func()) {
	// Get pooled token buffer
	tokBufPtr := tokenBufPool.Get().(*[]int16)
	tokens := *tokBufPtr

	// Tokenize directly into buffer (zero-alloc)
	var tokenCount int
	if m.tokenizer != nil {
		tokenCount = m.tokenizer.tokenizeZeroAlloc(text, tokens, nil)
	}

	// Get pooled result buffer
	resultPtr := float32Pool512.Get().(*[]float32)
	result := *resultPtr

	m.EmbedTokensInto(tokens[:tokenCount], result)

	return result, func() {
		tokenBufPool.Put(tokBufPtr)
		float32Pool512.Put(resultPtr)
	}
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

// EmbedBatchInt8 efficiently processes multiple texts in a single call
func (m *SimpleInt8Model512) EmbedBatchInt8(texts []string) ([]*Int8Result512, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([]*Int8Result512, len(texts))

	// Process texts in parallel batches for better CPU utilization
	const workerCount = 8 // Use multiple CPU cores
	const batchSize = 256 // Process in smaller chunks for memory efficiency

	type job struct {
		index int
		text  string
	}

	jobs := make(chan job, len(texts))
	resultsChan := make(chan struct {
		index  int
		result *Int8Result512
		err    error
	}, len(texts))

	// Start workers
	for w := 0; w < workerCount; w++ {
		go func() {
			for j := range jobs {
				result, err := m.EmbedInt8(j.text)
				resultsChan <- struct {
					index  int
					result *Int8Result512
					err    error
				}{j.index, result, err}
			}
		}()
	}

	// Send jobs
	go func() {
		defer close(jobs)
		for i, text := range texts {
			jobs <- job{i, text}
		}
	}()

	// Collect results
	var firstErr error
	for i := 0; i < len(texts); i++ {
		result := <-resultsChan
		if result.err != nil && firstErr == nil {
			firstErr = result.err
		}
		if result.result != nil {
			results[result.index] = result.result
		} else {
			// Fallback for failed embeddings
			results[result.index] = &Int8Result512{
				Vector: make([]int8, Int8EmbeddingDim),
				Scale:  1.0,
			}
		}
	}

	return results, firstErr
}

// EmbedBatchInt8Optimized provides the fastest batch processing with memory optimization
func (m *SimpleInt8Model512) EmbedBatchInt8Optimized(texts []string, progressCallback func(processed, total int)) ([]*Int8Result512, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([]*Int8Result512, len(texts))

	// For very large batches, use progressive processing to avoid memory spikes
	const chunkSize = 2000 // Match server batch size

	for start := 0; start < len(texts); start += chunkSize {
		end := start + chunkSize
		if end > len(texts) {
			end = len(texts)
		}

		chunk := texts[start:end]
		chunkResults, err := m.EmbedBatchInt8(chunk)
		if err != nil {
			return nil, fmt.Errorf("batch embedding failed at chunk %d-%d: %v", start, end, err)
		}

		// Copy chunk results to main results
		copy(results[start:end], chunkResults)

		// Report progress
		if progressCallback != nil {
			progressCallback(end, len(texts))
		}
	}

	return results, nil
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

// EmbedTokensInt8 converts pre-tokenized input directly into an int8 embedding result.
func (m *SimpleInt8Model512) EmbedTokensInt8(tokens []int16) (*Int8Result512, error) {
	if len(tokens) == 0 {
		return &Int8Result512{
			Vector: make([]int8, Int8EmbeddingDim),
			Scale:  1.0,
		}, nil
	}

	embedding, err := m.EmbedTokens(tokens)
	if err != nil {
		return nil, err
	}
	return quantizeVector512(embedding), nil
}

// EmbedDim returns the embedding dimensionality for API compatibility.
func (m *SimpleInt8Model512) EmbedDim() int {
	return Int8EmbeddingDim
}

// VocabSize returns the size of the token vocabulary.
func (m *SimpleInt8Model512) VocabSize() int {
	return len(m.embeddings)
}

// Close is provided for API compatibility with heavier model implementations.
func (m *SimpleInt8Model512) Close() error {
	return nil
}
