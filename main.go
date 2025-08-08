package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

// EmbeddingModel provides a clean API for text embeddings using the real static-retrieval-mrl-en-v1 model
type EmbeddingModel struct {
	vocabSize       int
	embedDim        int
	weights         [][]float32 // Real safetensors weights [vocab_size, embed_dim]
	referenceTokens map[string]TokenData
	embeddingBuffer []float32 // Pre-allocated for performance
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
	safetensorsPath := "./model/real_model.safetensors"
	if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("real model file not found: %s", safetensorsPath)
	}

	weights, vocabSize, embedDim, err := loadRealSafetensors(safetensorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load safetensors: %v", err)
	}

	// Load real reference tokens
	tokensPath := "./model/real_reference_tokens.json"
	referenceTokens, err := loadReferenceTokens(tokensPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load reference tokens: %v", err)
	}

	model := &EmbeddingModel{
		vocabSize:       vocabSize,
		embedDim:        embedDim,
		weights:         weights,
		referenceTokens: referenceTokens,
		embeddingBuffer: make([]float32, embedDim),
	}

	loadTime := time.Since(start)
	fmt.Printf("✅ Model loaded in %v (vocab: %d, dims: %d)\n", loadTime, vocabSize, embedDim)
	return model, nil
}

// Encode converts text to embedding vector using real model weights
func (m *EmbeddingModel) Encode(text string) ([]float32, error) {
	// Check if we have reference tokens for this text
	tokenData, exists := m.referenceTokens[text]
	if !exists {
		return nil, fmt.Errorf("text not in reference tokens: %s", text)
	}

	return m.computeEmbedding(tokenData.TokenIDs)
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
		if tokenID > 0 && tokenID < m.vocabSize {
			weightRow := m.weights[tokenID]
			for i := 0; i < m.embedDim; i++ {
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
	result := make([]float32, m.embedDim)
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

// GetAvailableTexts returns all texts that can be encoded
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

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 Gobed: Real Embedding Model Demo")
	fmt.Println("================================================================================")
	fmt.Println("Model: sentence-transformers/static-retrieval-mrl-en-v1 (REAL WEIGHTS)")
	fmt.Println("")

	// Load the real model
	model, err := LoadModel()
	if err != nil {
		log.Fatalf("❌ Failed to load model: %v", err)
	}

	// Get available texts
	availableTexts := model.GetAvailableTexts()
	if len(availableTexts) == 0 {
		log.Fatalf("❌ No texts available for encoding")
	}

	fmt.Printf("📚 Available texts for demo: %v\n\n", availableTexts)

	// Demo 1: Basic embedding
	fmt.Println("🔍 DEMO 1: Basic Text Encoding")
	fmt.Println(strings.Repeat("-", 50))

	for _, text := range availableTexts {
		start := time.Now()
		embedding, err := model.Encode(text)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("❌ Failed to encode '%s': %v\n", text, err)
			continue
		}

		fmt.Printf("Text: \"%s\"\n", text)
		fmt.Printf("  • Encoding time: %v\n", elapsed)
		fmt.Printf("  • Dimensions: %d\n", len(embedding))
		fmt.Printf("  • Sample values: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
			embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
		fmt.Println()
	}

	// Demo 2: Semantic similarity relationships
	fmt.Println("🎯 DEMO 2: Semantic Similarity Analysis")
	fmt.Println(strings.Repeat("-", 50))

	// Calculate all pairwise similarities
	var allSims []SimilarityResult
	for i := 0; i < len(availableTexts); i++ {
		for j := i + 1; j < len(availableTexts); j++ {
			text1, text2 := availableTexts[i], availableTexts[j]
			sim, err := model.Similarity(text1, text2)
			if err != nil {
				continue
			}

			allSims = append(allSims, SimilarityResult{
				Text1:      text1,
				Text2:      text2,
				Similarity: sim,
			})
		}
	}

	// Sort by similarity
	sort.Slice(allSims, func(i, j int) bool {
		return allSims[i].Similarity > allSims[j].Similarity
	})

	// Show most similar pairs
	fmt.Println("🔥 Most Similar Pairs:")
	for i, sim := range allSims {
		if i >= 3 { // Show top 3
			break
		}
		fmt.Printf("  %d. \"%-30s\" ↔ \"%-30s\" → %.4f\n",
			i+1, sim.Text1, sim.Text2, sim.Similarity)
	}

	fmt.Println("\n❄️  Least Similar Pairs:")
	for i := len(allSims) - 1; i >= len(allSims)-3 && i >= 0; i-- {
		sim := allSims[i]
		fmt.Printf("  %d. \"%-30s\" ↔ \"%-30s\" → %.4f\n",
			len(allSims)-i, sim.Text1, sim.Text2, sim.Similarity)
	}

	// Demo 3: Find similar texts
	if len(availableTexts) > 0 {
		fmt.Println("\n🔎 DEMO 3: Find Most Similar Texts")
		fmt.Println(strings.Repeat("-", 50))

		query := availableTexts[0]
		candidates := availableTexts[1:] // Exclude the query itself

		if len(candidates) > 0 {
			similar, err := model.FindMostSimilar(query, candidates, 3)
			if err != nil {
				fmt.Printf("❌ Error finding similar texts: %v\n", err)
			} else {
				fmt.Printf("Query: \"%s\"\n", query)
				fmt.Println("Most similar texts:")
				for i, sim := range similar {
					fmt.Printf("  %d. \"%-30s\" → %.4f\n", i+1, sim.Text2, sim.Similarity)
				}
			}
		}
	}

	// Demo 4: Performance summary
	fmt.Println("\n⚡ DEMO 4: Performance Summary")
	fmt.Println(strings.Repeat("-", 50))

	// Benchmark encoding speed
	if len(availableTexts) > 0 {
		testText := availableTexts[0]
		iterations := 1000

		fmt.Printf("Benchmarking \"%s\" over %d iterations...\n", testText, iterations)

		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, err := model.Encode(testText)
			if err != nil {
				break
			}
		}
		elapsed := time.Since(start)

		avgLatency := elapsed / time.Duration(iterations)
		throughput := float64(iterations) / elapsed.Seconds()

		fmt.Printf("Results:\n")
		fmt.Printf("  • Average latency: %v\n", avgLatency)
		fmt.Printf("  • Throughput: %.0f encodings/sec\n", throughput)
		fmt.Printf("  • Total time: %v\n", elapsed)
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("✅ Demo completed successfully!")
	fmt.Printf("🎯 Model specs: %d vocab × %d dimensions\n", model.vocabSize, model.embedDim)
	fmt.Println("🚀 Real static-retrieval-mrl-en-v1 weights loaded and working!")
	fmt.Println(strings.Repeat("=", 80))
}