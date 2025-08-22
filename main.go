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

type EmbeddingModel struct {
	tokenizer   *tokenizer.Tokenizer
	orthSession *onnxruntime_go.AdvancedSession
	inputTensors []*onnxruntime_go.Tensor[int64]
	outputTensors []*onnxruntime_go.Tensor[float32]
}

func NewEmbeddingModel(onnxPath, tokenizerPath string) (*EmbeddingModel, error) {
	log.Printf("Loading model from %s and tokenizer from %s", onnxPath, tokenizerPath)

	// Initialize ONNX Runtime
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %v", err)
	}

	// Load tokenizer from JSON file
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %v", err)
	}

	// Create input tensors (we'll update these during inference)
	maxLength := int64(512)
	batchSize := int64(1)
	shape := onnxruntime_go.NewShape(batchSize, maxLength)
	
	inputIdsTensor, err := onnxruntime_go.NewEmptyTensor[int64](shape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %v", err)
	}
	
	attentionMaskTensor, err := onnxruntime_go.NewEmptyTensor[int64](shape)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention_mask tensor: %v", err)
	}

	// Create output tensor
	hiddenSize := int64(768) // E5-base hidden size
	outputShape := onnxruntime_go.NewShape(batchSize, maxLength, hiddenSize)
	outputTensor, err := onnxruntime_go.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %v", err)
	}

	// Create session
	inputNames := []string{"input_ids", "attention_mask"}
	outputNames := []string{"last_hidden_state"}
	inputTensors := []*onnxruntime_go.Tensor[int64]{inputIdsTensor, attentionMaskTensor}
	outputTensors := []*onnxruntime_go.Tensor[float32]{outputTensor}

	session, err := onnxruntime_go.NewAdvancedSession(onnxPath, inputNames, outputNames, 
		convertToValueSlice(inputTensors), convertToValueSlice(outputTensors), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to load ONNX model: %v", err)
	}

	log.Println("Model and tokenizer loaded successfully")

	return &EmbeddingModel{
		tokenizer:     tk,
		orthSession:   session,
		inputTensors:  inputTensors,
		outputTensors: outputTensors,
	}, nil
}

// Helper function to convert tensors to ArbitraryTensor interface
func convertToValueSlice[T onnxruntime_go.TensorData](tensors []*onnxruntime_go.Tensor[T]) []onnxruntime_go.ArbitraryTensor {
	values := make([]onnxruntime_go.ArbitraryTensor, len(tensors))
	for i, tensor := range tensors {
		values[i] = tensor
	}
	return values
}

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
	log.Println("Model session and tensors destroyed")
	return nil
}

func (em *EmbeddingModel) tokenize(text string) ([]int64, []int64, error) {
	// Add "query: " prefix as required by E5 model
	prefixedText := "query: " + text

	// Encode the text
	encoding, err := em.tokenizer.EncodeSingle(prefixedText, true)
	if err != nil {
		return nil, nil, fmt.Errorf("tokenization failed: %v", err)
	}

	// Convert to int64 slices as required by ONNX
	inputIds := make([]int64, len(encoding.Ids))
	attentionMask := make([]int64, len(encoding.Ids))
	
	for i, id := range encoding.Ids {
		inputIds[i] = int64(id)
		attentionMask[i] = 1 // All tokens are attended to
	}

	return inputIds, attentionMask, nil
}

func (em *EmbeddingModel) Encode(text string) ([]float32, error) {
	start := time.Now()

	// Tokenize the input
	inputIds, attentionMask, err := em.tokenize(text)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %v", err)
	}

	// Pad/truncate to max length (512 for E5 model)
	maxLength := 512
	if len(inputIds) > maxLength {
		inputIds = inputIds[:maxLength]
		attentionMask = attentionMask[:maxLength]
	} else {
		// Pad sequences
		for len(inputIds) < maxLength {
			inputIds = append(inputIds, 0) // PAD token
			attentionMask = append(attentionMask, 0)
		}
	}

	// Update the existing tensors with new data
	inputIdsTensor := em.inputTensors[0]
	attentionMaskTensor := em.inputTensors[1]
	
	// Copy data into the tensors
	inputData := inputIdsTensor.GetData()
	maskData := attentionMaskTensor.GetData()
	
	copy(inputData, inputIds)
	copy(maskData, attentionMask)

	// Run inference
	err = em.orthSession.Run()
	if err != nil {
		return nil, fmt.Errorf("ONNX inference failed: %v", err)
	}

	// Extract embeddings from last hidden state and apply average pooling
	outputTensor := em.outputTensors[0]
	lastHiddenState := outputTensor.GetData()
	hiddenSize := 768 // E5-base hidden size
	
	// Apply average pooling with attention mask
	embedding := make([]float32, hiddenSize)
	validTokenCount := float32(0)
	
	for i := 0; i < len(attentionMask); i++ {
		if attentionMask[i] == 1 {
			validTokenCount++
			for j := 0; j < hiddenSize; j++ {
				embedding[j] += lastHiddenState[i*hiddenSize+j]
			}
		}
	}

	// Average and normalize
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

	log.Printf("Inference completed in %v", time.Since(start))
	return embedding, nil
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

func main() {
	fmt.Println("Go Embedding Model Test")
	fmt.Println("=======================")
	
	// Ensure environment cleanup
	defer onnxruntime_go.DestroyEnvironment()
	
	// Check if model files exist
	if _, err := os.Stat("model/embedding_model.onnx"); os.IsNotExist(err) {
		log.Fatal("ONNX model file not found at model/embedding_model.onnx")
	}
	if _, err := os.Stat("model/tokenizer.json"); os.IsNotExist(err) {
		log.Fatal("Tokenizer file not found at model/tokenizer.json")
	}
	
	// Create the ONNX-based model
	model, err := NewEmbeddingModel("model/embedding_model.onnx", "model/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to create embedding model: %v", err)
	}
	defer model.Close()
	
	// Test texts
	testTexts := []string{
		"hi",
		"bonjour", 
		"actionable business insights",
	}
	
	// Generate embeddings
	embeddings := make([][]float32, len(testTexts))
	for i, text := range testTexts {
		embedding, err := model.Encode(text)
		if err != nil {
			log.Fatalf("Failed to encode text '%s': %v", text, err)
		}
		embeddings[i] = embedding
		fmt.Printf("Generated embedding for '%s' (dim: %d)\n", text, len(embedding))
	}
	
	// Calculate similarities
	fmt.Println("\nSimilarity Results:")
	fmt.Println("==================")
	
	sim1 := cosineSimilarity(embeddings[0], embeddings[1])
	sim2 := cosineSimilarity(embeddings[0], embeddings[2])
	sim3 := cosineSimilarity(embeddings[1], embeddings[2])
	
	fmt.Printf("'hi' vs 'bonjour': %.4f\n", sim1)
	fmt.Printf("'hi' vs 'actionable business insights': %.4f\n", sim2)
	fmt.Printf("'bonjour' vs 'actionable business insights': %.4f\n", sim3)
	
	// Test if hi and bonjour are closer than either is to "actionable business insights"
	if sim1 > sim2 && sim1 > sim3 {
		fmt.Println("\n✓ SUCCESS: 'hi' and 'bonjour' are closer to each other than to 'actionable business insights'")
	} else {
		fmt.Println("\n✗ The similarity relationships don't match expected pattern")
		fmt.Println("  (This is expected with the simple hash-based embedding)")
	}
	
	fmt.Println("\nNote: This is using real multilingual E5 model embeddings.")
	
	// Run performance benchmark
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("PERFORMANCE BENCHMARK")
	fmt.Println(strings.Repeat("=", 50))
	
	runBenchmark(model)
}

func runBenchmark(model *EmbeddingModel) {
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
	
	fmt.Printf("Benchmarking with %d different texts...\n", len(benchmarkTexts))
	
	// Warmup run
	fmt.Println("\n1. Warmup run...")
	start := time.Now()
	_, err := model.Encode(benchmarkTexts[0])
	if err != nil {
		log.Printf("Warmup failed: %v", err)
		return
	}
	warmupTime := time.Since(start)
	fmt.Printf("   Warmup completed in: %v\n", warmupTime)
	
	// Benchmark run - 10 embeddings
	fmt.Println("\n2. Benchmark run (10 embeddings)...")
	start = time.Now()
	
	for i, text := range benchmarkTexts {
		embedStart := time.Now()
		embedding, err := model.Encode(text)
		embedTime := time.Since(embedStart)
		
		if err != nil {
			log.Printf("Failed to encode text %d: %v", i+1, err)
			continue
		}
		
		fmt.Printf("   Embedding %2d: %6.2fms (dim: %d, text: \"%.30s...\")\n", 
			i+1, float64(embedTime.Nanoseconds())/1000000, len(embedding), text)
	}
	
	totalTime := time.Since(start)
	avgTime := totalTime / time.Duration(len(benchmarkTexts))
	
	fmt.Printf("\n3. Results Summary:\n")
	fmt.Printf("   Total time: %v\n", totalTime)
	fmt.Printf("   Average time per embedding: %v\n", avgTime)
	fmt.Printf("   Embeddings per second: %.2f\n", float64(len(benchmarkTexts))/totalTime.Seconds())
	fmt.Printf("   Throughput: %.2f embeddings/sec\n", 1.0/avgTime.Seconds())
}
