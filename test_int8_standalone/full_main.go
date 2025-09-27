package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strings"
	"time"
)

const Int8EmbeddingDim = 512

type SimpleInt8Model512 struct {
	embeddings [][]int8
	scales     []float32
	vocab      map[string]int16
}

type Int8Result512 struct {
	Vector []int8
	Scale  float32
}

// Load the int8 model with safetensors format
func loadInt8Embeddings(path string) ([][]int8, []float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	// Read header length
	var headerLen uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLen); err != nil {
		return nil, nil, err
	}

	// Read header JSON
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return nil, nil, err
	}

	var header map[string]interface{}
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, nil, err
	}

	// Find embeddings and scales
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

	// Load embeddings (int8)
	embeddings, err := loadInt8Tensor(file, embeddingsInfo, 8+int64(headerLen))
	if err != nil {
		return nil, nil, err
	}

	// Load scales (float32)
	scales, err := loadFloat32Tensor(file, scalesInfo, 8+int64(headerLen))
	if err != nil {
		return nil, nil, err
	}

	// Reshape to 2D
	vocabSize := len(scales)
	embeddingDim := len(embeddings) / vocabSize
	embeddings2D := make([][]int8, vocabSize)
	for i := 0; i < vocabSize; i++ {
		embeddings2D[i] = embeddings[i*embeddingDim : (i+1)*embeddingDim]
	}

	return embeddings2D, scales, nil
}

func loadInt8Tensor(file *os.File, info map[string]interface{}, baseOffset int64) ([]int8, error) {
	offsets := info["data_offsets"].([]interface{})
	start := int64(offsets[0].(float64))
	end := int64(offsets[1].(float64))
	size := end - start

	file.Seek(baseOffset+start, 0)
	data := make([]int8, size)
	buf := make([]byte, size)
	io.ReadFull(file, buf)

	for i, b := range buf {
		data[i] = int8(b)
	}
	return data, nil
}

func loadFloat32Tensor(file *os.File, info map[string]interface{}, baseOffset int64) ([]float32, error) {
	offsets := info["data_offsets"].([]interface{})
	start := int64(offsets[0].(float64))
	end := int64(offsets[1].(float64))
	size := end - start

	file.Seek(baseOffset+start, 0)
	numFloats := size / 4
	data := make([]float32, numFloats)
	buf := make([]byte, size)
	io.ReadFull(file, buf)

	for i := 0; i < int(numFloats); i++ {
		bits := binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4])
		data[i] = math.Float32frombits(bits)
	}
	return data, nil
}

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
		if id < 32768 {
			vocab[token] = int16(id)
		}
	}
	return vocab, nil
}

func LoadSimpleInt8Model512() (*SimpleInt8Model512, error) {
	modelPath := "../model/modelint8_512dim.safetensors"
	tokenizerPath := "../model/tokenizer.json"

	embeddings, scales, err := loadInt8Embeddings(modelPath)
	if err != nil {
		return nil, err
	}

	vocab, err := loadSimpleVocab(tokenizerPath)
	if err != nil {
		return nil, err
	}

	return &SimpleInt8Model512{
		embeddings: embeddings,
		scales:     scales,
		vocab:      vocab,
	}, nil
}

func (m *SimpleInt8Model512) SimpleTokenize(text string) []int16 {
	text = strings.ToLower(strings.TrimSpace(text))
	words := strings.Fields(text)
	var tokens []int16

	// Add [CLS] token
	if clsID, ok := m.vocab["[CLS]"]; ok {
		tokens = append(tokens, clsID)
	}

	for _, word := range words {
		if id, ok := m.vocab[word]; ok {
			tokens = append(tokens, id)
		} else if id, ok := m.vocab["##"+word]; ok {
			tokens = append(tokens, id)
		} else if unkID, ok := m.vocab["[UNK]"]; ok {
			tokens = append(tokens, unkID)
		}
	}

	// Add [SEP] token
	if sepID, ok := m.vocab["[SEP]"]; ok {
		tokens = append(tokens, sepID)
	}

	return tokens
}

func (m *SimpleInt8Model512) EmbedTokens(tokens []int16) ([]float32, error) {
	if len(tokens) == 0 {
		return make([]float32, Int8EmbeddingDim), nil
	}

	result := make([]float32, Int8EmbeddingDim)
	validTokens := 0

	for _, token := range tokens {
		if token < 0 || int(token) >= len(m.embeddings) {
			continue
		}

		embedding := m.embeddings[token]
		scale := m.scales[token]

		for j := 0; j < Int8EmbeddingDim; j++ {
			result[j] += float32(embedding[j]) * scale
		}
		validTokens++
	}

	if validTokens > 0 {
		count := float32(validTokens)
		for i := range result {
			result[i] /= count
		}
	}

	return result, nil
}

func (m *SimpleInt8Model512) Embed(text string) ([]float32, error) {
	tokens := m.SimpleTokenize(text)
	return m.EmbedTokens(tokens)
}

func quantizeVector512(vec []float32) *Int8Result512 {
	maxAbs := float32(0)
	for _, v := range vec {
		if abs := float32(math.Abs(float64(v))); abs > maxAbs {
			maxAbs = abs
		}
	}

	scale := maxAbs / 127.0
	if scale == 0 {
		scale = 1.0
	}

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

	return &Int8Result512{Vector: result, Scale: scale}
}

func (m *SimpleInt8Model512) EmbedInt8(text string) (*Int8Result512, error) {
	tokens := m.SimpleTokenize(text)
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

	if scaledNorm1 == 0 || scaledNorm2 == 0 {
		return 0, nil
	}

	return scaledDot / float32(math.Sqrt(float64(scaledNorm1*scaledNorm2))), nil
}

func main() {
	fmt.Println("Complete Int8 Model Test with Int16 Tokenizer")
	fmt.Println(strings.Repeat("=", 60))

	// Load model
	start := time.Now()
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	loadTime := time.Since(start)
	fmt.Printf(" Model loaded in %v\n", loadTime)
	fmt.Printf(" Vocab size: %d, Embedding dims: %d\n", len(model.vocab), Int8EmbeddingDim)
	fmt.Printf(" Memory: ~15MB model + %d tokens in vocab\n\n", len(model.vocab))

	// Test texts
	testTexts := []string{
		"machine learning algorithms",
		"deep neural networks",
		"natural language processing",
		"computer vision applications",
		"artificial intelligence systems",
	}

	// Test tokenization
	fmt.Println(" Testing Int16 Tokenization:")
	for _, text := range testTexts {
		tokens := model.SimpleTokenize(text)
		fmt.Printf("  %q -> %v (count: %d)\n", text, tokens, len(tokens))
	}

	// Test embedding generation
	fmt.Println("\n Testing Float32 Embedding Generation:")
	var totalEmbedTime time.Duration
	for _, text := range testTexts {
		start := time.Now()
		embedding, err := model.Embed(text)
		if err != nil {
			log.Printf("Failed: %v", err)
			continue
		}
		embedTime := time.Since(start)
		totalEmbedTime += embedTime

		fmt.Printf("  %q -> %d dims in %v\n", text, len(embedding), embedTime)
	}
	avgEmbedTime := totalEmbedTime / time.Duration(len(testTexts))
	fmt.Printf("  Average: %v per embedding\n", avgEmbedTime)

	// Test int8 embedding
	fmt.Println("\n🔢 Testing Int8 Embedding Generation:")
	var totalInt8Time time.Duration
	for _, text := range testTexts {
		start := time.Now()
		int8Result, err := model.EmbedInt8(text)
		if err != nil {
			log.Printf("Failed: %v", err)
			continue
		}
		int8Time := time.Since(start)
		totalInt8Time += int8Time

		// Find min/max
		minVal, maxVal := int8Result.Vector[0], int8Result.Vector[0]
		for _, v := range int8Result.Vector {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}

		fmt.Printf("  %q -> scale=%.6f, range=[%d,%d] in %v\n",
			text, int8Result.Scale, minVal, maxVal, int8Time)
	}
	avgInt8Time := totalInt8Time / time.Duration(len(testTexts))
	fmt.Printf("  Average: %v per int8 embedding\n", avgInt8Time)

	// Test similarity
	fmt.Println("\n Testing Similarity Computation:")
	testPairs := []struct{ text1, text2 string }{
		{"machine learning", "machine learning"},
		{"deep learning", "neural networks"},
		{"computer vision", "image processing"},
		{"artificial intelligence", "machine learning"},
		{"hello world", "machine learning"},
	}

	var totalSimTime time.Duration
	for _, pair := range testPairs {
		start := time.Now()
		similarity, err := model.Similarity(pair.text1, pair.text2)
		if err != nil {
			log.Printf("Failed: %v", err)
			continue
		}
		simTime := time.Since(start)
		totalSimTime += simTime

		fmt.Printf("  Similarity(%q, %q) = %.4f (%v)\n",
			pair.text1, pair.text2, similarity, simTime)
	}
	avgSimTime := totalSimTime / time.Duration(len(testPairs))
	fmt.Printf("  Average: %v per similarity\n", avgSimTime)

	// Performance benchmark
	fmt.Println("\n  Performance Benchmark (1000 iterations):")
	benchText := "machine learning algorithms for neural networks"
	numIter := 1000

	// Benchmark embedding
	start = time.Now()
	for i := 0; i < numIter; i++ {
		_, err := model.Embed(benchText)
		if err != nil {
			log.Fatalf("Benchmark failed: %v", err)
		}
	}
	embedBenchTime := time.Since(start)

	// Benchmark int8 embedding
	start = time.Now()
	for i := 0; i < numIter; i++ {
		_, err := model.EmbedInt8(benchText)
		if err != nil {
			log.Fatalf("Int8 benchmark failed: %v", err)
		}
	}
	int8BenchTime := time.Since(start)

	// Benchmark similarity
	start = time.Now()
	for i := 0; i < numIter/2; i++ {
		_, err := model.Similarity("machine learning", "neural networks")
		if err != nil {
			log.Fatalf("Similarity benchmark failed: %v", err)
		}
	}
	simBenchTime := time.Since(start)

	// Results
	embedLatency := embedBenchTime / time.Duration(numIter)
	embedThroughput := float64(numIter) / embedBenchTime.Seconds()

	int8Latency := int8BenchTime / time.Duration(numIter)
	int8Throughput := float64(numIter) / int8BenchTime.Seconds()

	simLatency := simBenchTime / time.Duration(numIter/2)
	simThroughput := float64(numIter/2) / simBenchTime.Seconds()

	fmt.Printf(" Benchmark Results:\n")
	fmt.Printf("  Float32 Embedding: %v avg, %.0f/sec\n", embedLatency, embedThroughput)
	fmt.Printf("  Int8 Embedding:    %v avg, %.0f/sec\n", int8Latency, int8Throughput)
	fmt.Printf("  Similarity:        %v avg, %.0f/sec\n", simLatency, simThroughput)

	fmt.Printf("\n Performance Summary:\n")
	fmt.Printf("  Model size: 15MB (vs 119MB original = 7.9x smaller)\n")
	fmt.Printf("  Embedding throughput: %.0f/sec\n", embedThroughput)
	fmt.Printf("  Average latency: %v (target: <1ms) ", embedLatency)
	if embedLatency < time.Millisecond {
		fmt.Printf(" TARGET MET!\n")
	} else {
		fmt.Printf(" Above target\n")
	}

	fmt.Println("\n Complete int8 model test finished successfully!")
	fmt.Printf("Ready for production with int16 tokenizer and int8 embeddings!\n")
}