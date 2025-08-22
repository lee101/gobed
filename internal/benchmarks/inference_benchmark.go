package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"strings"
	"time"
)

// TokenData represents tokenization information
type TokenData struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
}

// SimpleEmbeddingModel demonstrates the LoadModel() approach with separated inference timing
type SimpleEmbeddingModel struct {
	referenceTokens map[string]TokenData
	// In a real implementation, you'd have your actual model weights here
	dummyWeights [][]float32
	vocabSize    int
	embedDim     int
}

// LoadModel demonstrates the separated loading approach (one-time cost)
func LoadModel(referenceTokensPath string) (*SimpleEmbeddingModel, error) {
	log.Printf("🔄 Loading model (one-time initialization)...")
	loadStart := time.Now()

	// Load reference tokens
	tokensFile, err := os.Open(referenceTokensPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open reference tokens: %v", err)
	}
	defer tokensFile.Close()

	tokensData, err := ioutil.ReadAll(tokensFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read reference tokens: %v", err)
	}

	var referenceTokens map[string]TokenData
	err = json.Unmarshal(tokensData, &referenceTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to parse reference tokens: %v", err)
	}

	// Simulate loading model weights (in practice this would be from safetensors/ONNX)
	vocabSize := 30000
	embedDim := 384
	dummyWeights := make([][]float32, vocabSize)
	for i := range dummyWeights {
		dummyWeights[i] = make([]float32, embedDim)
		// Initialize with small random values for demonstration
		for j := range dummyWeights[i] {
			dummyWeights[i][j] = float32(math.Sin(float64(i*embedDim+j))) * 0.1
		}
	}

	model := &SimpleEmbeddingModel{
		referenceTokens: referenceTokens,
		dummyWeights:    dummyWeights,
		vocabSize:       vocabSize,
		embedDim:        embedDim,
	}

	loadTime := time.Since(loadStart)
	log.Printf("✅ Model loaded successfully in %v", loadTime)
	log.Printf("📊 Loaded %d reference tokens, vocab_size=%d, embed_dim=%d", len(referenceTokens), vocabSize, embedDim)

	return model, nil
}

// EncodeText performs pure inference (this is what we benchmark)
func (m *SimpleEmbeddingModel) EncodeText(text string) ([]float32, error) {
	// Get tokens from reference
	tokenData, exists := m.referenceTokens[text]
	if !exists {
		return nil, fmt.Errorf("no reference tokens found for text: %s", text)
	}

	return m.encodeTokenIDs(tokenData.TokenIDs)
}

// encodeTokenIDs performs the actual inference
func (m *SimpleEmbeddingModel) encodeTokenIDs(tokenIDs []int) ([]float32, error) {
	embedding := make([]float32, m.embedDim)
	validTokens := 0

	// Simple mean pooling over token embeddings
	for _, tokenID := range tokenIDs {
		if tokenID > 0 && tokenID < m.vocabSize { // Skip padding tokens
			for i := 0; i < m.embedDim; i++ {
				embedding[i] += m.dummyWeights[tokenID][i]
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

	// L2 normalization
	norm := float32(0)
	for _, val := range embedding {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm > 0 {
		for i := range embedding {
			embedding[i] /= norm
		}
	}

	return embedding, nil
}

// benchmarkPureInference benchmarks only the inference part
func benchmarkPureInference(model *SimpleEmbeddingModel) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("🚀 PURE INFERENCE BENCHMARK\n")
	fmt.Printf("%s\n", strings.Repeat("=", 60))

	// Get available sentences from reference tokens
	sentences := make([]string, 0)
	for sentence := range model.referenceTokens {
		sentences = append(sentences, sentence)
		if len(sentences) >= 10 { // Limit to first 10 for demo
			break
		}
	}

	if len(sentences) == 0 {
		log.Printf("❌ No reference sentences available for benchmarking")
		return
	}

	fmt.Printf("Benchmarking %d sentences with pure inference timing...\n", len(sentences))

	// Warmup runs
	fmt.Println("\n🔥 Warmup runs...")
	for i := 0; i < 3; i++ {
		_, err := model.EncodeText(sentences[0])
		if err != nil {
			log.Printf("Warmup failed: %v", err)
			return
		}
	}
	fmt.Println("✅ Warmup completed")

	// Benchmark individual inference times
	fmt.Println("\n⏱️  Pure inference benchmarks:")
	times := make([]time.Duration, len(sentences))
	embeddings := make([][]float32, len(sentences))

	for i, sentence := range sentences {
		// Time ONLY the inference call
		start := time.Now()
		embedding, err := model.EncodeText(sentence)
		elapsed := time.Since(start)

		if err != nil {
			log.Printf("Failed to encode sentence %d: %v", i+1, err)
			continue
		}

		times[i] = elapsed
		embeddings[i] = embedding

		// Truncate sentence for display
		displaySentence := sentence
		if len(displaySentence) > 30 {
			displaySentence = displaySentence[:27] + "..."
		}

		fmt.Printf("   S%2d: %8.2fμs - \"%s\"\n",
			i+1, float64(elapsed.Nanoseconds())/1000, displaySentence)
	}

	// Calculate statistics
	var totalTime time.Duration
	for _, t := range times {
		totalTime += t
	}

	if len(times) == 0 {
		log.Printf("❌ No successful inferences to analyze")
		return
	}

	avgTime := totalTime / time.Duration(len(times))
	throughput := float64(len(sentences)) / totalTime.Seconds()

	fmt.Printf("\n📊 Performance Summary:\n")
	fmt.Printf("   Total inference time: %v\n", totalTime)
	fmt.Printf("   Average per inference: %v\n", avgTime)
	fmt.Printf("   Throughput: %.0f inferences/sec\n", throughput)
	fmt.Printf("   Latency: %.2fμs per inference\n", float64(avgTime.Nanoseconds())/1000)

	// Find min/max for performance range
	if len(times) > 1 {
		minTime := times[0]
		maxTime := times[0]
		for _, t := range times[1:] {
			if t < minTime {
				minTime = t
			}
			if t > maxTime {
				maxTime = t
			}
		}

		fmt.Printf("   Range: %.2fμs - %.2fμs\n",
			float64(minTime.Nanoseconds())/1000,
			float64(maxTime.Nanoseconds())/1000)
	}

	// Quick similarity test
	if len(embeddings) >= 2 {
		sim := cosineSimilarity(embeddings[0], embeddings[1])
		fmt.Printf("   Sample similarity (S1 vs S2): %.4f\n", sim)
	}
}

func cosineSimilarity(a, b []float32) float32 {
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
	fmt.Println("================================================================================")
	fmt.Println("🚀 GO INFERENCE BENCHMARKING - LOADMODEL() APPROACH")
	fmt.Println("================================================================================")
	fmt.Println("Demonstrates: Separated loading vs pure inference timing")
	fmt.Println("")

	// Check if reference tokens exist
	referenceTokensPath := "model/production_reference_tokens.json"
	if _, err := os.Stat(referenceTokensPath); os.IsNotExist(err) {
		log.Fatalf("❌ Reference tokens file not found: %s", referenceTokensPath)
	}

	// Load model (one-time cost)
	model, err := LoadModel(referenceTokensPath)
	if err != nil {
		log.Fatalf("❌ Failed to load model: %v", err)
	}

	// Demonstrate the separated approach
	fmt.Println("✅ Model loaded successfully!")
	fmt.Println("📝 Now we can benchmark ONLY the inference time...")

	// Benchmark pure inference (what we actually care about)
	benchmarkPureInference(model)

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("✅ Benchmark completed!")
	fmt.Println("🎯 Key insight: Model loading is separated from inference timing.")
	fmt.Println("⚡ This approach allows for accurate performance measurement.")
	fmt.Println(strings.Repeat("=", 80))
}
