// Package gobed provides a Go interface for ONNX-based sentence embedding models.
//
// This package allows you to:
//   - Load ONNX embedding models exported from SentenceTransformers
//   - Generate embeddings for text with CPU or GPU acceleration
//   - Calculate semantic similarity between texts
//   - Use pre-tokenized reference tokens for consistent results
//
// Example usage:
//
//	model, err := gobed.NewEmbeddingModel("model.onnx", "tokens.json", false)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer model.Close()
//
//	embedding, err := model.Encode("hello world")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	similarity := gobed.CosineSimilarity(embedding1, embedding2)
package gobed

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

var onnxInitialized = false

// ReferenceTokens represents pre-computed token IDs for a sentence
type ReferenceTokens struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
}

// EmbeddingModel represents an ONNX-based sentence embedding model
type EmbeddingModel struct {
	referenceTokens map[string]ReferenceTokens
	useGPU          bool
	session         *onnxruntime.AdvancedSession
	inputTensor     *onnxruntime.Tensor[int64]
	outputTensor    *onnxruntime.Tensor[float32]
	inputName       string
	outputName      string
}

// initONNXRuntime initializes the ONNX Runtime environment
func initONNXRuntime() error {
	if onnxInitialized {
		return nil
	}

	// Set the shared library path - adjust this based on your installation
	onnxruntime.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so.1")

	// Initialize ONNX Runtime
	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		return err
	}

	onnxInitialized = true
	return nil
}

// NewEmbeddingModel creates a new embedding model instance
//
// Parameters:
//   - onnxPath: Path to the ONNX model file
//   - referenceTokensPath: Path to the JSON file containing pre-computed tokens
//   - useGPU: Whether to use GPU acceleration (requires CUDA and GPU-enabled ONNX Runtime)
//
// Returns:
//   - *EmbeddingModel: The initialized model instance
//   - error: Any error that occurred during initialization
func NewEmbeddingModel(onnxPath, referenceTokensPath string, useGPU bool) (*EmbeddingModel, error) {
	// Initialize ONNX Runtime
	if err := initONNXRuntime(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %v", err)
	}

	log.Printf("Loading model from %s and reference tokens from %s", onnxPath, referenceTokensPath)

	// Load reference tokens
	referenceTokens := make(map[string]ReferenceTokens)
	if referenceTokensPath != "" {
		tokensData, err := os.ReadFile(referenceTokensPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read reference tokens: %v", err)
		}

		if err := json.Unmarshal(tokensData, &referenceTokens); err != nil {
			return nil, fmt.Errorf("failed to parse reference tokens: %v", err)
		}
		log.Printf("Loaded %d reference token mappings", len(referenceTokens))
	}

	// Set up ONNX model components
	inputNames := []string{"input_ids"}
	outputNames := []string{"embeddings"}

	// Create input tensors
	maxSeqLen := int64(512)
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
	}

	// Set up session options
	var options *onnxruntime.SessionOptions
	if useGPU {
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
					useGPU = false
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
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create ONNX session: %v", err)
	}

	return &EmbeddingModel{
		referenceTokens: referenceTokens,
		useGPU:          useGPU,
		session:         session,
		inputTensor:     inputTensor,
		outputTensor:    outputTensor,
		inputName:       inputNames[0],
		outputName:      outputNames[0],
	}, nil
}

// Close releases all resources associated with the model
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
	log.Println("Model session closed")
	return nil
}

// getTokenIds gets token IDs for a text, using reference tokens if available
func (em *EmbeddingModel) getTokenIds(text string) ([]int64, error) {
	// First check if we have reference tokens for this exact text
	if refTokens, exists := em.referenceTokens[text]; exists {
		log.Printf("Using reference tokens for '%s'", text)
		tokenIds := make([]int64, len(refTokens.TokenIDs))
		for i, id := range refTokens.TokenIDs {
			tokenIds[i] = int64(id)
		}
		return tokenIds, nil
	}

	// Fallback: warn that we don't have reference tokens
	log.Printf("Warning: No reference tokens for '%s', using simple fallback", text)

	// Very basic fallback - just return CLS + UNK tokens + SEP + padding
	tokenIds := make([]int64, 512)
	tokenIds[0] = 101 // CLS
	tokenIds[1] = 100 // UNK for the whole text
	tokenIds[2] = 102 // SEP
	// Rest are already 0 (PAD)

	return tokenIds, nil
}

// Encode generates an embedding for the given text
//
// Parameters:
//   - text: The input text to encode
//
// Returns:
//   - []float32: The embedding vector (typically 1024 dimensions)
//   - error: Any error that occurred during encoding
func (em *EmbeddingModel) Encode(text string) ([]float32, error) {
	start := time.Now()

	// Get token IDs for the input text
	tokenIds, err := em.getTokenIds(text)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize text: %v", err)
	}

	// Ensure we have exactly 512 tokens (pad or truncate)
	if len(tokenIds) > 512 {
		tokenIds = tokenIds[:512]
	} else if len(tokenIds) < 512 {
		padded := make([]int64, 512)
		copy(padded, tokenIds)
		tokenIds = padded
	}

	// Copy token IDs to input tensor
	inputData := em.inputTensor.GetData()
	copy(inputData, tokenIds)

	// Run inference
	err = em.session.Run()
	if err != nil {
		return nil, fmt.Errorf("inference failed: %v", err)
	}

	// Get the output
	outputData := em.outputTensor.GetData()
	result := make([]float32, len(outputData))
	copy(result, outputData)

	duration := time.Since(start)
	deviceType := "CPU"
	if em.useGPU {
		deviceType = "GPU"
	}
	log.Printf("%s inference completed in %v", deviceType, duration)

	return result, nil
}

// SquaredEuclideanDistance calculates the squared Euclidean distance between two embeddings
func SquaredEuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}

	return float32(sum)
}

// CosineSimilarity calculates the cosine similarity between two embeddings
//
// Parameters:
//   - a, b: The embedding vectors to compare
//
// Returns:
//   - float32: Cosine similarity value between -1 (opposite) and 1 (identical)
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64 // Use float64 for better precision
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	similarity := dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
	return float32(similarity)
}

// CalculateNorm calculates the L2 norm of an embedding vector
func CalculateNorm(embedding []float32) float32 {
	var sum float64
	for _, val := range embedding {
		sum += float64(val) * float64(val)
	}
	return float32(math.Sqrt(sum))
}

// BatchEncode generates embeddings for multiple texts
//
// Parameters:
//   - texts: Slice of input texts to encode
//
// Returns:
//   - [][]float32: Slice of embedding vectors
//   - error: Any error that occurred during encoding
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
