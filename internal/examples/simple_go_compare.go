//go:build legacy

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
)

type TensorInfo struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

type TokenData struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
}

func loadWeights() ([][]float32, error) {
	file, err := os.Open("cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read header length
	headerLengthBytes := make([]byte, 8)
	file.Read(headerLengthBytes)
	headerLength := binary.LittleEndian.Uint64(headerLengthBytes)

	// Read header
	headerBytes := make([]byte, headerLength)
	file.Read(headerBytes)

	var header map[string]TensorInfo
	json.Unmarshal(headerBytes, &header)

	// Read tensor data
	data, _ := ioutil.ReadAll(file)

	info := header["embedding.weight"]
	start := info.DataOffsets[0]
	end := info.DataOffsets[1]
	tensorBytes := data[start:end]

	rows := info.Shape[0]
	cols := info.Shape[1]
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

	return weights, nil
}

func forward(weights [][]float32, tokenIDs []int) []float32 {
	embedDim := len(weights[0])
	embedding := make([]float32, embedDim)
	validTokens := 0

	for _, tokenID := range tokenIDs {
		if tokenID > 0 && tokenID < len(weights) {
			for i := 0; i < embedDim; i++ {
				embedding[i] += weights[tokenID][i]
			}
			validTokens++
		}
	}

	if validTokens > 0 {
		for i := range embedding {
			embedding[i] /= float32(validTokens)
		}
	}

	return embedding
}

func main() {
	weights, err := loadWeights()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	var tokens map[string]TokenData
	tokensFile, _ := os.Open("model/production_reference_tokens.json")
	defer tokensFile.Close()
	tokensData, _ := ioutil.ReadAll(tokensFile)
	json.Unmarshal(tokensData, &tokens)

	sentences := []string{
		"This is a test sentence.",
		"Machine learning is fascinating.",
		"Hello world",
	}

	fmt.Println("Go Safetensors Embeddings:")
	for _, sentence := range sentences {
		tokenData := tokens[sentence]
		embedding := forward(weights, tokenData.TokenIDs)
		fmt.Printf("'%s': [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
			sentence, embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
	}
}
