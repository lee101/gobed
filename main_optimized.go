package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	"github.com/yalue/onnxruntime_go"
)

// EmbeddingModel represents a loaded E5 model ready for inference
type EmbeddingModel struct {
	tokenizer     *tokenizer.Tokenizer
	orthSession   *onnxruntime_go.AdvancedSession
	inputTensors  []*onnxruntime_go.Tensor[int64]
	outputTensors []*onnxruntime_go.Tensor[float32]
	
	// Configuration
	maxLength   int
	hiddenSize  int
}

// Helper function to convert tensors to ArbitraryTensor interface
func convertToValueSlice[T onnxruntime_go.TensorData](tensors []*onnxruntime_go.Tensor[T]) []onnxruntime_go.ArbitraryTensor {
	values := make([]onnxruntime_go.ArbitraryTensor, len(tensors))
	for i, tensor := range tensors {
		values[i] = tensor
	}
	return values
}

// LoadModel loads the E5 model and tokenizer, preparing everything for inference
func LoadModel(onnxPath, tokenizerPath string) (*EmbeddingModel, error) {
	log.Printf("Loading E5 model from %s", onnxPath)
	loadStart := time.Now()
	
	// Initialize ONNX Runtime environment (one-time setup)
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %v", err)
	}

	// Load tokenizer
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %v", err)
	}

	// Configuration
	maxLength := 512
	hiddenSize := 768
	batchSize := int64(1)
	
	// Create pre-allocated tensors for reuse
	shape := onnxruntime_go.NewShape(batchSize, int64(maxLength))
	
	inputIdsTensor, err := onnxruntime_go.NewEmptyTensor[int64](shape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %v", err)
	}
	
	attentionMaskTensor, err := onnxruntime_go.NewEmptyTensor[int64](shape)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention_mask tensor: %v", err)
	}

	// Create output tensor
	outputShape := onnxruntime_go.NewShape(batchSize, int64(maxLength), int64(hiddenSize))
	outputTensor, err := onnxruntime_go.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %v", err)
	}

	// Create session with pre-allocated tensors
	inputNames := []string{"input_ids", "attention_mask"}
	outputNames := []string{"last_hidden_state"}
	inputTensors := []*onnxruntime_go.Tensor[int64]{inputIdsTensor, attentionMaskTensor}
	outputTensors := []*onnxruntime_go.Tensor[float32]{outputTensor}

	session, err := onnxruntime_go.NewAdvancedSession(onnxPath, inputNames, outputNames, 
		convertToValueSlice(inputTensors), convertToValueSlice(outputTensors), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to load ONNX model: %v", err)
	}

	model := &EmbeddingModel{
		tokenizer:     tk,
		orthSession:   session,
		inputTensors:  inputTensors,
		outputTensors: outputTensors,
		maxLength:     maxLength,
		hiddenSize:    hiddenSize,
	}

	loadTime := time.Since(loadStart)
	log.Printf("Model loaded successfully in %v", loadTime)
	log.Printf("Ready for inference with max_length=%d, hidden_size=%d", maxLength, hiddenSize)

	return model, nil
}

// Close cleans up model resources
func (em *EmbeddingModel) Close() error {
	if em.orthSession != nil {
		em.orthSession.Destroy()
	}
	// Destroy tensors
	for _, tensor := range em.inputTensors {
		if tensor != nil {
			tensor.Destroy()
		}
	}
	for _, tensor := range em.outputTensors {
		if tensor != nil {
			tensor.Destroy()
		}
	}
	onnxruntime_go.DestroyEnvironment()
	log.Println("Model resources cleaned up")
	return nil
}

// tokenizeText tokenizes input text with E5 prefix
func (em *EmbeddingModel) tokenizeText(text string) ([]int64, []int64, error) {
	// Add E5 required prefix
	prefixedText := "query: " + text

	// Encode the text
	encoding, err := em.tokenizer.EncodeSingle(prefixedText, true)
	if err != nil {
		return nil, nil, fmt.Errorf("tokenization failed: %v", err)
	}

	// Convert to int64 and pad/truncate
	inputIds := make([]int64, em.maxLength)
	attentionMask := make([]int64, em.maxLength)

	// Copy tokens up to maxLength
	copyLen := len(encoding.Ids)
	if copyLen > em.maxLength {
		copyLen = em.maxLength
	}

	for i := 0; i < copyLen; i++ {
		inputIds[i] = int64(encoding.Ids[i])
		attentionMask[i] = 1
	}
	// Remaining positions are already 0 (PAD tokens)

	return inputIds, attentionMask, nil
}

// Encode performs pure inference - this is what we benchmark
func (em *EmbeddingModel) Encode(text string) ([]float32, error) {
	// Step 1: Tokenize (very fast, reuses tokenizer)
	inputIds, attentionMask, err := em.tokenizeText(text)
	if err != nil {
		return nil, err
	}

	// Step 2: Copy data to pre-allocated tensors
	inputData := em.inputTensors[0].GetData()
	attentionData := em.inputTensors[1].GetData()
	copy(inputData, inputIds)
	copy(attentionData, attentionMask)

	// Step 3: Run inference (this is the main bottleneck)
	err = em.orthSession.Run()
	if err != nil {
		return nil, fmt.Errorf("ONNX inference failed: %v", err)
	}

	// Step 4: Extract embeddings from output tensor
	lastHiddenState := em.outputTensors[0].GetData()
	
	// Apply average pooling with attention mask
	embedding := make([]float32, em.hiddenSize)
	validTokenCount := float32(0)

	for i := 0; i < em.maxLength; i++ {
		if attentionMask[i] == 1 {
			validTokenCount++
			for j := 0; j < em.hiddenSize; j++ {
				embedding[j] += lastHiddenState[i*em.hiddenSize+j]
			}
		}
	}

	// Average pooling
	if validTokenCount > 0 {
		for i := range embedding {
			embedding[i] /= validTokenCount
		}

		// L2 normalization
		var norm float32
		for _, val := range embedding {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))

		if norm > 0 {
			for i := range embedding {
				embedding[i] /= norm
			}
		}
	}

	return embedding, nil
}

// BatchEncode encodes multiple texts efficiently
func (em *EmbeddingModel) BatchEncode(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	
	for i, text := range texts {
		embedding, err := em.Encode(text)
		if err != nil {
			return nil, fmt.Errorf("failed to encode text %d: %v", i, err)
		}
		embeddings[i] = embedding
	}
	
	return embeddings, nil
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
	
	if normA == 0 || normB == 0 {
		return 0.0
	}
	
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func benchmarkInference(model *EmbeddingModel) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("OPTIMIZED INFERENCE BENCHMARK")
	fmt.Println(strings.Repeat("=", 60))

	benchmarkTexts := []string{
		"hello world",
		"machine learning is fascinating",
		"artificial intelligence and deep learning",
		"natural language processing",
		"computer vision and image recognition",
		"data science and analytics",
		"software engineering best practices",
		"distributed systems architecture",
		"cloud computing and microservices",
		"performance optimization techniques",
	}

	fmt.Printf("Benchmarking %d texts with optimized inference...\n", len(benchmarkTexts))

	// Warmup runs (important for JIT/optimization)
	fmt.Println("\nWarmup runs...")
	for i := 0; i < 3; i++ {
		_, err := model.Encode(benchmarkTexts[0])
		if err != nil {
			log.Printf("Warmup failed: %v", err)
			return
		}
	}
	fmt.Println("Warmup completed")

	// Benchmark individual inference times
	fmt.Println("\nPure inference benchmarks:")
	times := make([]time.Duration, len(benchmarkTexts))
	
	for i, text := range benchmarkTexts {
		// Time only the inference call
		start := time.Now()
		_, err := model.Encode(text)
		elapsed := time.Since(start)
		
		if err != nil {
			log.Printf("Failed to encode text %d: %v", i+1, err)
			continue
		}
		
		times[i] = elapsed
		fmt.Printf("   Text %2d: %8.2fms - \"%s\"\n", i+1, float64(elapsed.Nanoseconds())/1000000, text[:min(40, len(text))])
	}

	// Calculate statistics
	var totalTime time.Duration
	for _, t := range times {
		totalTime += t
	}
	
	avgTime := totalTime / time.Duration(len(times))
	throughput := float64(len(benchmarkTexts)) / totalTime.Seconds()

	fmt.Printf("\nPerformance Summary:\n")
	fmt.Printf("   Total inference time: %v\n", totalTime)
	fmt.Printf("   Average per inference: %v\n", avgTime)
	fmt.Printf("   Throughput: %.2f inferences/sec\n", throughput)
	fmt.Printf("   Latency: %.2fms per inference\n", float64(avgTime.Nanoseconds())/1000000)

	// Find min/max for performance range
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
	
	fmt.Printf("   Range: %.2fms - %.2fms\n", 
		float64(minTime.Nanoseconds())/1000000,
		float64(maxTime.Nanoseconds())/1000000)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	fmt.Println("Optimized Go E5 Embedding Model")
	fmt.Println(strings.Repeat("=", 40))

	// Check required files
	if _, err := os.Stat("model/embedding_model.onnx"); os.IsNotExist(err) {
		log.Fatal("ONNX model file not found at model/embedding_model.onnx")
	}
	if _, err := os.Stat("model/tokenizer.json"); os.IsNotExist(err) {
		log.Fatal("Tokenizer file not found at model/tokenizer.json")
	}

	// Load model (one-time cost)
	model, err := LoadModel("model/embedding_model.onnx", "model/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Test semantic similarity with optimized inference
	fmt.Println("\nSemantic Similarity Test:")
	fmt.Println(strings.Repeat("-", 30))
	
	testTexts := []string{
		"hi",
		"bonjour",
		"actionable business insights",
	}

	// Time only the inference
	fmt.Println("Generating embeddings...")
	embeddings := make([][]float32, len(testTexts))
	
	for i, text := range testTexts {
		start := time.Now()
		embedding, err := model.Encode(text)
		inferenceTime := time.Since(start)
		
		if err != nil {
			log.Fatalf("Failed to encode '%s': %v", text, err)
		}
		
		embeddings[i] = embedding
		fmt.Printf("'%s': %v (dim: %d)\n", text, inferenceTime, len(embedding))
	}

	// Calculate similarities
	fmt.Println("\nSimilarity Results:")
	sim1 := cosineSimilarity(embeddings[0], embeddings[1])
	sim2 := cosineSimilarity(embeddings[0], embeddings[2])
	sim3 := cosineSimilarity(embeddings[1], embeddings[2])

	fmt.Printf("'%s' vs '%s': %.4f\n", testTexts[0], testTexts[1], sim1)
	fmt.Printf("'%s' vs '%s': %.4f\n", testTexts[0], testTexts[2], sim2)
	fmt.Printf("'%s' vs '%s': %.4f\n", testTexts[1], testTexts[2], sim3)

	if sim1 > sim2 && sim1 > sim3 {
		fmt.Println("✓ SUCCESS: Greetings are more similar to each other")
	} else {
		fmt.Println("⚠ Greetings similarity pattern not as expected")
	}

	// Run comprehensive inference benchmark
	benchmarkInference(model)
}