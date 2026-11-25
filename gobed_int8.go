package gobed

import (
	"fmt"
	"math"
	"os"
	"time"
)

// The int8 kernels live in gobed_int8_kernels.go (pure Go + Go assembly,
// default on non-amd64 and when cgo is off) and gobed_int8_cgo.go (cgo
// AVX-512, default on amd64 with cgo).

// EmbeddingModelInt8 provides INT8 quantized embeddings with SIMD acceleration
type EmbeddingModelInt8 struct {
	VocabSize       int
	EmbedDim        int
	weightsInt8     [][]int8    // Quantized weights; rows alias weightsInt8Flat
	weightsInt8Flat []int8      // [VocabSize*EmbedDim] contiguous backing store
	weightsFloat32  [][]float32 // Original weights for comparison
	referenceTokens map[string]TokenData
	scale           float32
	zeroPoint       int8
	useInt8         bool // Flag to enable/disable INT8 mode
}

// LoadModelInt8 loads the model with INT8 quantization support
func LoadModelInt8(useInt8 bool) (*EmbeddingModelInt8, error) {
	fmt.Printf("🔄 Loading model with INT8=%v and SIMD support...\n", useInt8)
	start := time.Now()

	// Load safetensors weights
	safetensorsPath := "model/real_model.safetensors"
	if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
		safetensorsPath = "../../model/real_model.safetensors"
		if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
			safetensorsPath = "./model/real_model.safetensors"
		}
	}

	weightsFloat32, vocabSize, embedDim, err := loadRealSafetensors(safetensorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load safetensors: %v", err)
	}

	// Load reference tokens - reuse the function from gobed.go
	tokensPath := "model/real_reference_tokens.json"
	if _, err := os.Stat(tokensPath); os.IsNotExist(err) {
		tokensPath = "../../model/real_reference_tokens.json"
		if _, err := os.Stat(tokensPath); os.IsNotExist(err) {
			tokensPath = "./model/real_reference_tokens.json"
		}
	}
	// For now, use a simple map - in production would share with main package
	referenceTokens := make(map[string]TokenData)
	// We'll only use pre-tokenized texts from the benchmark anyway

	model := &EmbeddingModelInt8{
		VocabSize:       vocabSize,
		EmbedDim:        embedDim,
		weightsFloat32:  weightsFloat32,
		referenceTokens: referenceTokens,
		useInt8:         useInt8,
	}

	if useInt8 {
		// Quantize weights to INT8
		fmt.Println(" Quantizing weights to INT8...")
		model.quantizeWeights()
	}

	_ = time.Since(start) // loadTime
	// Model loaded successfully
	return model, nil
}

// quantizeWeights converts float32 weights to INT8 with proper scaling
func (m *EmbeddingModelInt8) quantizeWeights() {
	// Find min/max across all weights for optimal quantization
	var minVal, maxVal float32
	minVal = math.MaxFloat32
	maxVal = -math.MaxFloat32

	for i := 0; i < m.VocabSize; i++ {
		for j := 0; j < m.EmbedDim; j++ {
			val := m.weightsFloat32[i][j]
			if val < minVal {
				minVal = val
			}
			if val > maxVal {
				maxVal = val
			}
		}
	}

	// Calculate scale and zero point for symmetric quantization
	m.scale = (maxVal - minVal) / 255.0
	m.zeroPoint = int8(-128 - int(minVal/m.scale))

	fmt.Printf(" Quantization params: scale=%.6f, zero_point=%d, range=[%.3f, %.3f]\n",
		m.scale, m.zeroPoint, minVal, maxVal)

	// Allocate INT8 weights as one contiguous slab so embedding kernels can
	// index rows directly instead of re-flattening the table per call.
	m.weightsInt8Flat = make([]int8, m.VocabSize*m.EmbedDim)
	m.weightsInt8 = make([][]int8, m.VocabSize)
	for i := range m.weightsInt8 {
		m.weightsInt8[i] = m.weightsInt8Flat[i*m.EmbedDim : (i+1)*m.EmbedDim : (i+1)*m.EmbedDim]
	}

	for i := 0; i < m.VocabSize; i++ {
		quantizeRowInt8(m.weightsFloat32[i][:m.EmbedDim], m.weightsInt8[i], m.scale, m.zeroPoint)

		if i%5000 == 0 {
			fmt.Printf("  Quantized %d/%d rows\n", i, m.VocabSize)
		}
	}
}

// Encode converts text to INT8 embedding (0-255 range)
func (m *EmbeddingModelInt8) Encode(text string) ([]uint8, error) {
	// Get token IDs (reuse existing tokenization)
	tokenData, exists := m.referenceTokens[text]
	if !exists {
		return nil, fmt.Errorf("text not in reference tokens: %s", text)
	}

	if m.useInt8 {
		return m.computeEmbeddingInt8(tokenData.TokenIDs)
	} else {
		return m.computeEmbeddingFloat32AsInt8(tokenData.TokenIDs)
	}
}

// ComputeEmbeddingFromTokens computes INT8 embedding from token IDs
func (m *EmbeddingModelInt8) ComputeEmbeddingFromTokens(tokenIDs []int) ([]uint8, error) {
	if m.useInt8 {
		return m.computeEmbeddingInt8(tokenIDs)
	} else {
		return m.computeEmbeddingFloat32AsInt8(tokenIDs)
	}
}

// computeEmbeddingInt8 uses SIMD-accelerated INT8 computation
func (m *EmbeddingModelInt8) computeEmbeddingInt8(tokenIDs []int) ([]uint8, error) {
	result := make([]uint8, m.EmbedDim)

	if len(tokenIDs) == 0 {
		// Return middle value for empty input
		for i := range result {
			result[i] = 128
		}
		return result, nil
	}

	embedInt8(m.flatInt8Weights(), tokenIDs, m.EmbedDim, m.VocabSize, result, m.scale)

	return result, nil
}

// flatInt8Weights returns the contiguous [VocabSize*EmbedDim] weight table.
// quantizeWeights always builds it; a model whose weightsInt8 rows were set
// some other way pays for a temporary copy (not cached, to stay race-free).
func (m *EmbeddingModelInt8) flatInt8Weights() []int8 {
	if len(m.weightsInt8Flat) == m.VocabSize*m.EmbedDim {
		return m.weightsInt8Flat
	}
	flat := make([]int8, m.VocabSize*m.EmbedDim)
	for i := 0; i < m.VocabSize; i++ {
		copy(flat[i*m.EmbedDim:(i+1)*m.EmbedDim], m.weightsInt8[i])
	}
	return flat
}

// computeEmbeddingFloat32AsInt8 computes with float32 then converts to INT8
func (m *EmbeddingModelInt8) computeEmbeddingFloat32AsInt8(tokenIDs []int) ([]uint8, error) {
	// Use float32 computation
	embedding := make([]float32, m.EmbedDim)
	validTokens := 0

	for _, tokenID := range tokenIDs {
		if tokenID >= 0 && tokenID < m.VocabSize {
			weightRow := m.weightsFloat32[tokenID]
			for i := 0; i < m.EmbedDim; i++ {
				embedding[i] += weightRow[i]
			}
			validTokens++
		}
	}

	// Mean pooling
	if validTokens > 0 {
		invTokens := 1.0 / float32(validTokens)
		for i := range embedding {
			embedding[i] *= invTokens
		}
	}

	// Convert to 0-255 range
	result := make([]uint8, m.EmbedDim)
	for i, val := range embedding {
		// Assuming embeddings are roughly in [-1, 1] range
		// Map to [0, 255]
		normalized := (val + 1.0) * 127.5
		if normalized < 0 {
			normalized = 0
		} else if normalized > 255 {
			normalized = 255
		}
		result[i] = uint8(normalized)
	}

	return result, nil
}

// CosineSimilarityInt8 computes similarity between INT8 vectors using SIMD.
// Elements are uint8 values centered at 128 (x-128 in [-128,127]).
func CosineSimilarityInt8(a, b []uint8) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	return cosineInt8(a, b)
}

// CosineSimilarityInt8Fallback is a pure Go fallback for systems without AVX-512
func CosineSimilarityInt8Fallback(a, b []uint8) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	var dotProduct, normA, normB int64

	for i := 0; i < len(a); i++ {
		// Center around 0 by subtracting 128
		aVal := int16(a[i]) - 128
		bVal := int16(b[i]) - 128

		dotProduct += int64(aVal) * int64(bVal)
		normA += int64(aVal) * int64(aVal)
		normB += int64(bVal) * int64(bVal)
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return float32(dotProduct) / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
