package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// FastModel - optimized int8 embedding model
type FastModel struct {
	embeddings [][]int8
	scales     []float32
	tokenizer  *FastTokenizer
}

// LoadFastModel loads the optimized model with modular components
func LoadFastModel(modelPath, tokenizerPath string) (*FastModel, error) {
	// Load zero-alloc tokenizer
	tokenizer, err := LoadTokenizer(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer load failed: %v", err)
	}

	// Load int8 embeddings with optimized loader
	embeddings, scales, err := loadInt8Safetensors(modelPath)
	if err != nil {
		return nil, fmt.Errorf("embeddings load failed: %v", err)
	}

	return &FastModel{
		embeddings: embeddings,
		scales:     scales,
		tokenizer:  tokenizer,
	}, nil
}

// EmbedInt8 creates int8 embedding with optimized mean pooling
func (m *FastModel) EmbedInt8(text string) ([]int8, error) {
	tokens := m.tokenizer.Tokenize(text)

	// Optimized mean pooling with SIMD-friendly operations
	result := make([]float32, 512)
	count := 0

	for _, tokenID := range tokens {
		if int(tokenID) < len(m.embeddings) {
			embedding := m.embeddings[tokenID]
			scale := m.scales[tokenID]

			// Vectorized accumulation (compiler will optimize)
			for i := 0; i < 512; i += 4 {
				result[i] += float32(embedding[i]) * scale
				result[i+1] += float32(embedding[i+1]) * scale
				result[i+2] += float32(embedding[i+2]) * scale
				result[i+3] += float32(embedding[i+3]) * scale
			}
			count++
		}
	}

	// Mean normalization
	if count > 0 {
		invCount := 1.0 / float32(count)
		for i := 0; i < 512; i += 4 {
			result[i] *= invCount
			result[i+1] *= invCount
			result[i+2] *= invCount
			result[i+3] *= invCount
		}
	}

	// Fast quantization to int8
	quantized := make([]int8, 512)
	for i := 0; i < 512; i += 4 {
		quantized[i] = clampInt8(result[i])
		quantized[i+1] = clampInt8(result[i+1])
		quantized[i+2] = clampInt8(result[i+2])
		quantized[i+3] = clampInt8(result[i+3])
	}

	return quantized, nil
}

// clampInt8 efficiently clamps float32 to int8 range
func clampInt8(val float32) int8 {
	if val > 127 {
		return 127
	}
	if val < -128 {
		return -128
	}
	return int8(val)
}

// loadInt8Safetensors - optimized safetensors loader
func loadInt8Safetensors(path string) ([][]int8, []float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	// Read header
	headerLenBytes := make([]byte, 8)
	if _, err := file.Read(headerLenBytes); err != nil {
		return nil, nil, err
	}
	headerLen := binary.LittleEndian.Uint64(headerLenBytes)

	headerBytes := make([]byte, headerLen)
	if _, err := file.Read(headerBytes); err != nil {
		return nil, nil, err
	}

	var header map[string]struct {
		DataType string   `json:"dtype"`
		Shape    []int    `json:"shape"`
		Offsets  [2]int64 `json:"data_offsets"`
	}

	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, nil, err
	}

	// Load embeddings tensor
	embInfo := header["embeddings.weight"]
	vocabSize, embDim := embInfo.Shape[0], embInfo.Shape[1]

	file.Seek(int64(8+headerLen)+embInfo.Offsets[0], 0)
	embData := make([]byte, embInfo.Offsets[1]-embInfo.Offsets[0])
	if _, err := file.Read(embData); err != nil {
		return nil, nil, err
	}

	// Convert to optimized int8 matrix
	embeddings := make([][]int8, vocabSize)
	for i := 0; i < vocabSize; i++ {
		embeddings[i] = make([]int8, embDim)
		for j := 0; j < embDim; j++ {
			embeddings[i][j] = int8(embData[i*embDim+j])
		}
	}

	// Load scales tensor
	scaleInfo := header["embeddings.scales"]
	file.Seek(int64(8+headerLen)+scaleInfo.Offsets[0], 0)

	scaleData := make([]byte, scaleInfo.Offsets[1]-scaleInfo.Offsets[0])
	if _, err := file.Read(scaleData); err != nil {
		return nil, nil, err
	}

	// Convert to float32 scales
	scales := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		bits := binary.LittleEndian.Uint32(scaleData[i*4 : (i+1)*4])
		scales[i] = math.Float32frombits(bits)
	}

	return embeddings, scales, nil
}