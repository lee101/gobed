package main

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("INT8 Quantization Test (Software Implementation)")
	fmt.Println(strings.Repeat("=", 80))

	// Load standard model
	fmt.Println("\nLoading standard Float32 model...")
	model, err := gobed.LoadModel()
	if err != nil {
		panic(fmt.Sprintf("Failed to load model: %v", err))
	}

	// Test sentences
	testTexts := []string{
		"Hello world",
		"Machine learning is fascinating.",
		"Python is a programming language.",
	}

	fmt.Println("\n1. Testing Float32 embeddings...")
	float32Embeddings := make([][]float32, len(testTexts))

	for i, text := range testTexts {
		emb, err := model.Encode(text)
		if err != nil {
			fmt.Printf("Error encoding '%s': %v\n", text, err)
			continue
		}
		float32Embeddings[i] = emb
		fmt.Printf("  Encoded: %s (dim=%d)\n", text, len(emb))
	}

	fmt.Println("\n2. Converting to INT8 range [0-255]...")
	int8Embeddings := make([][]uint8, len(float32Embeddings))

	for i, emb := range float32Embeddings {
		int8Emb := convertToInt8(emb)
		int8Embeddings[i] = int8Emb

		// Show first few values
		fmt.Printf("  Text %d: F32[%.3f, %.3f, %.3f...] -> INT8[%d, %d, %d...]\n",
			i+1, emb[0], emb[1], emb[2], int8Emb[0], int8Emb[1], int8Emb[2])
	}

	fmt.Println("\n3. Computing similarities...")
	fmt.Println("Float32 similarities:")
	for i := 0; i < len(testTexts); i++ {
		for j := i + 1; j < len(testTexts); j++ {
			sim := gobed.CosineSimilarity(float32Embeddings[i], float32Embeddings[j])
			fmt.Printf("  '%s' vs '%s': %.4f\n",
				truncate(testTexts[i], 25), truncate(testTexts[j], 25), sim)
		}
	}

	fmt.Println("\nINT8 similarities:")
	for i := 0; i < len(testTexts); i++ {
		for j := i + 1; j < len(testTexts); j++ {
			sim := cosineSimilarityInt8(int8Embeddings[i], int8Embeddings[j])
			fmt.Printf("  '%s' vs '%s': %.4f\n",
				truncate(testTexts[i], 25), truncate(testTexts[j], 25), sim)
		}
	}

	fmt.Println("\n4. Performance test...")
	iterations := 1000

	// Benchmark Float32
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for _, text := range testTexts {
			model.Encode(text)
		}
	}
	float32Time := time.Since(start)

	// Benchmark INT8 conversion (simulated)
	start = time.Now()
	for i := 0; i < iterations; i++ {
		for _, text := range testTexts {
			emb, _ := model.Encode(text)
			_ = convertToInt8(emb)
		}
	}
	int8Time := time.Since(start)

	fmt.Printf("\nFloat32 encoding: %v for %d iterations\n", float32Time, iterations*len(testTexts))
	fmt.Printf("INT8 (w/ conversion): %v for %d iterations\n", int8Time, iterations*len(testTexts))
	fmt.Printf("Overhead: %.1f%%\n", (float64(int8Time-float32Time)/float64(float32Time))*100)

	// Memory comparison
	float32Mem := model.VocabSize * model.EmbedDim * 4 / (1024 * 1024)
	int8Mem := model.VocabSize * model.EmbedDim * 1 / (1024 * 1024)

	fmt.Printf("\n5. Memory usage:\n")
	fmt.Printf("Float32 weights: %d MB\n", float32Mem)
	fmt.Printf("INT8 weights:    %d MB\n", int8Mem)
	fmt.Printf("Memory saving:   %.1f%%\n", (1.0-float64(int8Mem)/float64(float32Mem))*100)

	fmt.Println("\n INT8 test completed!")
}

// convertToInt8 converts float32 embedding to uint8 [0-255] range
func convertToInt8(embedding []float32) []uint8 {
	result := make([]uint8, len(embedding))

	// Find min and max for normalization
	minVal, maxVal := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	for _, val := range embedding {
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
	}

	// Handle edge case of all same values
	if minVal == maxVal {
		for i := range result {
			result[i] = 128 // Middle value
		}
		return result
	}

	// Convert to 0-255 range
	scale := 255.0 / (maxVal - minVal)
	for i, val := range embedding {
		normalized := (val - minVal) * scale
		if normalized < 0 {
			normalized = 0
		} else if normalized > 255 {
			normalized = 255
		}
		result[i] = uint8(normalized)
	}

	return result
}

// cosineSimilarityInt8 computes cosine similarity for INT8 vectors
func cosineSimilarityInt8(a, b []uint8) float32 {
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

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
