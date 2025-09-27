package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"
)

// Test quality metrics with GPU-enabled int8 model
func main() {
	fmt.Println("GPU Quality Test - Int8 512-dim Model")

	// Copy the int8 model implementation inline
	model := &SimpleInt8Model512{}
	var err error
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Test pairs for quality validation
	testPairs := []struct {
		text1, text2 string
		expectedSim  float32
		description  string
	}{
		{"machine learning", "machine learning", 1.0, "identical texts"},
		{"neural networks", "deep learning", 0.7, "related concepts"},
		{"algorithm optimization", "performance tuning", 0.6, "similar meaning"},
		{"cat on mat", "feline on carpet", 0.5, "synonyms"},
		{"machine learning", "cooking recipes", 0.1, "unrelated"},
	}

	fmt.Printf("Testing %d similarity pairs...\n", len(testPairs))

	var totalTime time.Duration
	successCount := 0

	for i, pair := range testPairs {
		start := time.Now()

		// Get int8 embeddings
		emb1, err := model.EmbedInt8(pair.text1)
		if err != nil {
			log.Printf("Failed to embed text1: %v", err)
			continue
		}

		emb2, err := model.EmbedInt8(pair.text2)
		if err != nil {
			log.Printf("Failed to embed text2: %v", err)
			continue
		}

		// Compute similarity
		similarity := computeInt8Similarity(emb1, emb2)
		elapsed := time.Since(start)
		totalTime += elapsed

		// Check quality
		diff := math.Abs(float64(similarity - pair.expectedSim))
		quality := "PASS"
		if diff > 0.3 { // Allow 30% tolerance
			quality = "FAIL"
		} else {
			successCount++
		}

		fmt.Printf("%d. %s vs %s\n", i+1, pair.text1, pair.text2)
		fmt.Printf("   Expected: %.3f, Got: %.3f, Diff: %.3f [%s]\n",
			pair.expectedSim, similarity, diff, quality)
		fmt.Printf("   Time: %.3fμs\n", float64(elapsed.Nanoseconds())/1000.0)
		fmt.Println()
	}

	// Summary
	avgTime := totalTime / time.Duration(len(testPairs))
	qps := 1000000.0 / float64(avgTime.Nanoseconds()) * 1000.0

	fmt.Printf("Quality Results:\n")
	fmt.Printf("  Success: %d/%d (%.1f%%)\n", successCount, len(testPairs),
		float64(successCount)/float64(len(testPairs))*100)
	fmt.Printf("  Avg time: %.3fμs per pair\n", float64(avgTime.Nanoseconds())/1000.0)
	fmt.Printf("  Throughput: %.0f pairs/sec\n", qps)

	if successCount == len(testPairs) {
		fmt.Println("✓ All quality tests passed!")
	} else {
		fmt.Printf("⚠ %d quality tests failed\n", len(testPairs)-successCount)
	}
}

// loadInt8Embeddings stub for testing
func loadInt8Embeddings(path string) ([][]int8, []float32, error) {
	// Stub implementation - would load from safetensors
	return nil, nil, fmt.Errorf("safetensors loading not implemented in standalone test")
}

// loadSimpleVocab loads vocab from tokenizer.json
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

func computeInt8Similarity(emb1, emb2 *Int8Result512) float32 {
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

	// Compute cosine similarity
	if scaledNorm1 == 0 || scaledNorm2 == 0 {
		return 0
	}

	return scaledDot / float32(math.Sqrt(float64(scaledNorm1*scaledNorm2)))
}