package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"strings"
	"time"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// RealEmbeddingModel that can handle arbitrary text input
type RealEmbeddingModel struct {
	vocabSize       int
	embedDim        int
	weights         [][]float32
	tokenizer       *tokenizer.Tokenizer
	embeddingBuffer []float32
}

// TensorInfo contains safetensors tensor metadata
type TensorInfo struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// LoadRealModel loads the model with real tokenizer
func LoadRealModel() (*RealEmbeddingModel, error) {
	fmt.Println("🔄 Loading real embedding model with tokenizer...")
	start := time.Now()

	// Load safetensors weights
	weights, vocabSize, embedDim, err := loadSafetensorsWeights()
	if err != nil {
		return nil, fmt.Errorf("failed to load weights: %v", err)
	}

	// Load real tokenizer
	fmt.Println("📝 Loading tokenizer...")
	tk, err := loadRealTokenizer()
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %v", err)
	}

	model := &RealEmbeddingModel{
		vocabSize:       vocabSize,
		embedDim:        embedDim,
		weights:         weights,
		tokenizer:       tk,
		embeddingBuffer: make([]float32, embedDim),
	}

	loadTime := time.Since(start)
	fmt.Printf("✅ Model loaded in %v (vocab: %d, dims: %d)\n", loadTime, vocabSize, embedDim)
	return model, nil
}

func loadRealTokenizer() (*tokenizer.Tokenizer, error) {
	// Try different tokenizer paths
	paths := []string{
		"model/tokenizer.json",
		"./model/tokenizer.json",
		"model/sentence_transformer/tokenizer.json",
		"./model/sentence_transformer/tokenizer.json",
	}

	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			fmt.Printf("📁 Using tokenizer: %s\n", path)
			return tokenizer.FromFile(path), nil
		}
	}

	// Try pretrained BERT tokenizer as fallback
	fmt.Println("⚠️  Local tokenizer not found, using pretrained BERT tokenizer")
	tk, err := pretrained.BertBaseUncased()
	return tk, err
}

func loadSafetensorsWeights() ([][]float32, int, int, error) {
	// Try different model paths
	paths := []string{
		"model/real_model.safetensors",
		"./model/real_model.safetensors",
	}

	var filePath string
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			filePath = path
			break
		}
	}

	if filePath == "" {
		return nil, 0, 0, fmt.Errorf("safetensors model not found")
	}

	fmt.Printf("📁 Loading weights from: %s\n", filePath)

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
		return nil, 0, 0, fmt.Errorf("embedding.weight not found in safetensors")
	}

	if info.Dtype != "F32" || len(info.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("unsupported tensor format: %s, shape: %v", info.Dtype, info.Shape)
	}

	start, end := info.DataOffsets[0], info.DataOffsets[1]
	tensorBytes := data[start:end]
	rows, cols := info.Shape[0], info.Shape[1]

	fmt.Printf("📊 Loading embedding matrix: [%d, %d]\n", rows, cols)

	// Load weights
	weights := make([][]float32, rows)
	for i := range weights {
		weights[i] = make([]float32, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			offset := (i*cols + j) * 4
			if offset+4 > len(tensorBytes) {
				return nil, 0, 0, fmt.Errorf("tensor data truncated")
			}
			bits := binary.LittleEndian.Uint32(tensorBytes[offset : offset+4])
			weights[i][j] = math.Float32frombits(bits)
		}
		if i%5000 == 0 {
			fmt.Printf("   Loaded %d/%d rows\n", i, rows)
		}
	}

	return weights, rows, cols, nil
}

// EncodeText tokenizes and embeds arbitrary text
func (m *RealEmbeddingModel) EncodeText(text string) ([]float32, error) {
	// Tokenize the text
	encoding, err := m.tokenizer.EncodeSingle(text, true)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %v", err)
	}

	tokens := encoding.GetIds()
	fmt.Printf("🔤 Text: \"%s\"\n", text)
	fmt.Printf("🔢 Tokens (%d): %v\n", len(tokens), tokens)

	return m.computeEmbedding(tokens)
}

// computeEmbedding performs the actual embedding computation
func (m *RealEmbeddingModel) computeEmbedding(tokenIDs []int) ([]float32, error) {
	// Reset buffer
	for i := range m.embeddingBuffer {
		m.embeddingBuffer[i] = 0
	}

	validTokens := 0

	// Sum embeddings for all tokens
	for _, tokenID := range tokenIDs {
		if tokenID > 0 && tokenID < m.vocabSize {
			weightRow := m.weights[tokenID]
			for i := 0; i < m.embedDim; i++ {
				m.embeddingBuffer[i] += weightRow[i]
			}
			validTokens++
		}
	}

	// Mean pooling
	if validTokens > 0 {
		invValidTokens := 1.0 / float32(validTokens)
		for i := range m.embeddingBuffer {
			m.embeddingBuffer[i] *= invValidTokens
		}
	}

	// Return copy
	result := make([]float32, m.embedDim)
	copy(result, m.embeddingBuffer)
	return result, nil
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

func main() {
	var text1, text2 string
	var showHelp bool
	
	flag.StringVar(&text1, "text1", "", "First text to compare")
	flag.StringVar(&text2, "text2", "", "Second text to compare")
	flag.BoolVar(&showHelp, "help", false, "Show help")
	flag.Parse()

	if showHelp || (text1 == "" || text2 == "") {
		fmt.Println("🚀 Real Text Embedding Distance Calculator")
		fmt.Println(strings.Repeat("=", 50))
		fmt.Println("Usage:")
		fmt.Println("  go run cli_distance.go -text1=\"Hello world\" -text2=\"Hi there friend\"")
		fmt.Println("")
		fmt.Println("Examples:")
		fmt.Println("  # Similar texts")
		fmt.Println("  go run cli_distance.go -text1=\"Python programming\" -text2=\"JavaScript coding\"")
		fmt.Println("")
		fmt.Println("  # Different texts")
		fmt.Println("  go run cli_distance.go -text1=\"Hello world\" -text2=\"Pizza is delicious\"")
		fmt.Println("")
		fmt.Println("Output:")
		fmt.Println("  - Similarity: 0.0 to 1.0 (higher = more similar)")
		fmt.Println("  - Distance: 0.0 to 2.0 (lower = more similar)")
		return
	}

	// Load model
	model, err := LoadRealModel()
	if err != nil {
		fmt.Printf("❌ Error loading model: %v\n", err)
		return
	}

	// Encode both texts
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("📊 EMBEDDING TEXT 1")
	fmt.Println(strings.Repeat("-", 70))
	emb1, err := model.EncodeText(text1)
	if err != nil {
		fmt.Printf("❌ Error encoding text1: %v\n", err)
		return
	}

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("📊 EMBEDDING TEXT 2")
	fmt.Println(strings.Repeat("-", 70))
	emb2, err := model.EncodeText(text2)
	if err != nil {
		fmt.Printf("❌ Error encoding text2: %v\n", err)
		return
	}

	// Calculate similarity and distance
	similarity := CosineSimilarity(emb1, emb2)
	distance := 1.0 - similarity

	// Show results
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("📏 DISTANCE CALCULATION")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("\n📝 Text 1: \"%s\"\n", text1)
	fmt.Printf("📝 Text 2: \"%s\"\n", text2)
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("🎯 Cosine Similarity: %.6f\n", similarity)
	fmt.Printf("📐 Distance (1-similarity): %.6f\n", distance)
	
	// Interpretation
	fmt.Println("\n📊 INTERPRETATION:")
	if similarity > 0.7 {
		fmt.Println("🔥 Very similar texts")
	} else if similarity > 0.4 {
		fmt.Println("🟢 Somewhat similar texts")
	} else if similarity > 0.1 {
		fmt.Println("🟡 Slightly related texts")
	} else if similarity > -0.1 {
		fmt.Println("🔴 Unrelated texts")
	} else {
		fmt.Println("❄️  Opposite texts")
	}
	
	// Show embedding previews
	fmt.Printf("\n🔍 Embedding dimensions: %d\n", len(emb1))
	fmt.Printf("📊 Text 1 embedding sample: [%.3f, %.3f, %.3f, %.3f, %.3f]\n", 
		emb1[0], emb1[1], emb1[2], emb1[3], emb1[4])
	fmt.Printf("📊 Text 2 embedding sample: [%.3f, %.3f, %.3f, %.3f, %.3f]\n", 
		emb2[0], emb2[1], emb2[2], emb2[3], emb2[4])

	fmt.Println("\n✅ Real embedding calculation completed!")
}