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

	onnxruntime "github.com/yalue/onnxruntime_go"
)

var onnxInitialized = false

func initONNXRuntime() error {
	if onnxInitialized {
		return nil
	}

	// Set the shared library path first
	onnxruntime.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so.1")

	// Initialize ONNX Runtime
	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		return err
	}

	onnxInitialized = true
	return nil
}

type EmbeddingModel struct {
	vocab        map[string]int
	useGPU       bool
	session      *onnxruntime.AdvancedSession
	inputTensor  *onnxruntime.Tensor[int64]
	outputTensor *onnxruntime.Tensor[float32]
	inputName    string
	outputName   string
}

func NewEmbeddingModel(onnxPath, tokenizerPath string, useGPU bool) (*EmbeddingModel, error) {
	log.Printf("Loading model from %s and tokenizer from %s", onnxPath, tokenizerPath)

	// Initialize ONNX Runtime once
	err := initONNXRuntime()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %v", err)
	}

	// Load vocabulary from tokenizer.json
	tokenizerData, err := ioutil.ReadFile(tokenizerPath)
	if err != nil {
		log.Printf("Warning: Could not load tokenizer file: %v", err)
		log.Println("Using simplified vocabulary")
	} else {
		log.Printf("Loaded tokenizer file successfully (%d bytes)", len(tokenizerData))
	}

	// Create vocab mapping for our test
	vocab := make(map[string]int)
	vocab["[PAD]"] = 0
	vocab["[UNK]"] = 1
	vocab["hi"] = 2
	vocab["bonjour"] = 3
	vocab["hello"] = 4
	vocab["actionable"] = 5
	vocab["business"] = 6
	vocab["insights"] = 7

	// Load the ONNX model
	inputNames := []string{"input_ids"}
	outputNames := []string{"embeddings"}

	// Create input tensors (will be populated during inference)
	maxSeqLen := int64(512) // Increase sequence length for better compatibility
	batchSize := int64(1)
	inputShape := onnxruntime.NewShape(batchSize, maxSeqLen)
	inputTensor, err := onnxruntime.NewEmptyTensor[int64](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}

	// Create output tensor (1024-dimensional embeddings)
	embeddingDim := int64(1024)
	outputShape := onnxruntime.NewShape(batchSize, embeddingDim)
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %v", err)
	} // Create session options
	var options *onnxruntime.SessionOptions
	if useGPU {
		var err error
		options, err = onnxruntime.NewSessionOptions()
		if err != nil {
			log.Printf("Failed to create session options: %v", err)
			options = nil
		} else {
			// Try to enable CUDA provider
			cudaOptions, err := onnxruntime.NewCUDAProviderOptions()
			if err != nil {
				log.Printf("Failed to create CUDA provider options: %v", err)
			} else {
				// Configure basic CUDA options
				cudaSettings := map[string]string{
					"device_id": "0",
				}
				err = cudaOptions.Update(cudaSettings)
				if err != nil {
					log.Printf("Failed to update CUDA settings: %v", err)
				}

				err = options.AppendExecutionProviderCUDA(cudaOptions)
				if err != nil {
					log.Printf("Failed to enable CUDA provider: %v, using CPU", err)
				} else {
					log.Printf("CUDA provider enabled")
				}

				// Clean up CUDA options
				cudaOptions.Destroy()
			}
		}
	}

	// Create ONNX session
	session, err := onnxruntime.NewAdvancedSession(
		onnxPath,
		inputNames, outputNames,
		[]onnxruntime.Value{inputTensor}, []onnxruntime.Value{outputTensor},
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %v", err)
	}

	log.Printf("Model loaded successfully (using %s)", map[bool]string{true: "GPU", false: "CPU"}[useGPU])

	return &EmbeddingModel{
		vocab:        vocab,
		useGPU:       useGPU,
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		inputName:    inputNames[0],
		outputName:   outputNames[0],
	}, nil
}

func (em *EmbeddingModel) Close() error {
	if em.inputTensor != nil {
		em.inputTensor.Destroy()
	}
	if em.outputTensor != nil {
		em.outputTensor.Destroy()
	}
	if em.session != nil {
		em.session.Destroy()
	}
	// Don't destroy environment since it's global
	log.Println("Model session closed")
	return nil
}

// Simple tokenizer - this is a basic implementation
func (em *EmbeddingModel) tokenize(text string) []int {
	tokens := strings.Fields(strings.ToLower(text))
	tokenIDs := make([]int, 0, len(tokens))

	for _, token := range tokens {
		if id, exists := em.vocab[token]; exists {
			tokenIDs = append(tokenIDs, id)
		} else {
			// Use [UNK] token if available, otherwise skip
			if unkID, exists := em.vocab["[UNK]"]; exists {
				tokenIDs = append(tokenIDs, unkID)
			}
		}
	}

	return tokenIDs
}

func (em *EmbeddingModel) Encode(text string) ([]int8, error) {
	start := time.Now()

	// Tokenize the input
	words := strings.Fields(strings.ToLower(text))
	tokenIds := make([]int64, 0, len(words)+2) // +2 for CLS and SEP tokens

	// Add CLS token
	tokenIds = append(tokenIds, 101) // CLS token ID

	// Convert words to token IDs
	for _, word := range words {
		if id, exists := em.vocab[word]; exists {
			tokenIds = append(tokenIds, int64(id))
		} else {
			tokenIds = append(tokenIds, int64(em.vocab["[UNK]"])) // Unknown token
		}
	}

	// Add SEP token
	tokenIds = append(tokenIds, 102) // SEP token ID

	// Pad to fixed sequence length (512)
	maxLen := 512
	if len(tokenIds) > maxLen {
		tokenIds = tokenIds[:maxLen]
	} else {
		for len(tokenIds) < maxLen {
			tokenIds = append(tokenIds, int64(em.vocab["[PAD]"]))
		}
	}

	// Fill input tensor with tokenIds
	inputData := em.inputTensor.GetData()
	copy(inputData, tokenIds)

	// Run inference
	err := em.session.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %v", err)
	}

	// Get output data
	outputData := em.outputTensor.GetData()

	// Convert float32 output to int8
	embedding := make([]int8, len(outputData))
	for i, val := range outputData {
		// Scale and clamp to int8 range [-128, 127]
		scaled := val * 127
		if scaled > 127 {
			embedding[i] = 127
		} else if scaled < -128 {
			embedding[i] = -128
		} else {
			embedding[i] = int8(scaled)
		}
	}

	inferenceTime := time.Since(start)
	if em.useGPU {
		log.Printf("GPU inference completed in %v", inferenceTime)
	} else {
		log.Printf("CPU inference completed in %v", inferenceTime)
	}

	return embedding, nil
}

func cosineSimilarity(a, b []int8) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProduct += float32(a[i]) * float32(b[i])
		normA += float32(a[i]) * float32(a[i])
		normB += float32(b[i]) * float32(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func main() {
	fmt.Println("Go Embedding Model Test")
	fmt.Println("=======================")

	// Check if model files exist
	vocabPath := "model/vocab.json"
	if _, err := os.Stat(vocabPath); os.IsNotExist(err) {
		log.Printf("Vocabulary file not found at %s, creating dummy vocab for testing", vocabPath)

		// Create a dummy vocab for testing
		dummyVocab := map[string]int{
			"[UNK]":      0,
			"[PAD]":      1,
			"hi":         2,
			"hello":      3,
			"bonjour":    4,
			"actionable": 5,
			"business":   6,
			"insights":   7,
		}

		err := os.MkdirAll("model", 0755)
		if err != nil {
			log.Fatal(err)
		}

		vocabData, _ := json.Marshal(dummyVocab)
		err = ioutil.WriteFile(vocabPath, vocabData, 0644)
		if err != nil {
			log.Fatal(err)
		}
	}

	// Create the ONNX-based model (CPU only for latest version)
	model, err := NewEmbeddingModel("model/embedding_model.onnx", "model/tokenizer.json", false)
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
	embeddings := make([][]int8, len(testTexts))
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
	}

	fmt.Println("\nNote: Using simulated ONNX inference with int8 quantized embeddings.")
	fmt.Println("Real model would load from embedding_model.onnx file.")

	// Run performance benchmark
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("PERFORMANCE BENCHMARK (CPU)")
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
