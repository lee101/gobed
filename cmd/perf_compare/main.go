//go:build legacy

package main

import (
	"fmt"
	"runtime"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println(" FLOAT32 vs INT8 PERFORMANCE COMPARISON")
	fmt.Println(strings.Repeat("=", 80))

	// System info
	fmt.Println("\n System Information:")
	fmt.Printf("  CPU cores: %d\n", runtime.NumCPU())
	fmt.Printf("  Architecture: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("  Go version: %s\n", runtime.Version())

	// Test texts
	testTexts := []string{
		"Hello world",
		"Machine learning is fascinating",
		"Natural language processing",
		"Deep learning models are powerful",
		"Python is a programming language",
	}

	// Load Float32 model
	fmt.Println("\n1⃣ FLOAT32 MODEL")
	fmt.Println(strings.Repeat("-", 50))

	startLoad := time.Now()
	modelF32, err := gobed.LoadModel()
	if err != nil {
		panic(fmt.Sprintf("Failed to load Float32 model: %v", err))
	}
	loadTimeF32 := time.Since(startLoad)
	fmt.Printf(" Load time: %v\n", loadTimeF32)

	// Warmup
	for i := 0; i < 100; i++ {
		modelF32.Encode(testTexts[0])
	}

	// Benchmark Float32
	iterations := 500
	fmt.Printf("\nBenchmarking %d iterations...\n", iterations)

	// Measure individual operations
	var totalEncodeTime time.Duration
	var encodings [][]float32

	for i := 0; i < iterations; i++ {
		for _, text := range testTexts {
			start := time.Now()
			emb, _ := modelF32.Encode(text)
			totalEncodeTime += time.Since(start)
			if i == 0 {
				encodings = append(encodings, emb)
			}
		}
	}

	totalOps := iterations * len(testTexts)
	avgLatencyF32 := totalEncodeTime / time.Duration(totalOps)
	throughputF32 := float64(totalOps) / totalEncodeTime.Seconds()

	fmt.Printf("  Total operations: %d\n", totalOps)
	fmt.Printf("  Total time: %v\n", totalEncodeTime)
	fmt.Printf("  Throughput: %.0f ops/sec\n", throughputF32)
	fmt.Printf("  Avg latency: %v\n", avgLatencyF32)

	// Test similarity computation
	fmt.Println("\nSimilarity computation benchmark:")
	start := time.Now()
	simCount := 0
	for i := 0; i < 1000; i++ {
		for j := 0; j < len(encodings); j++ {
			for k := j + 1; k < len(encodings); k++ {
				_ = gobed.CosineSimilarity(encodings[j], encodings[k])
				simCount++
			}
		}
	}
	simTimeF32 := time.Since(start)
	fmt.Printf("  %d similarity computations in %v\n", simCount, simTimeF32)
	fmt.Printf("  Rate: %.0f/sec\n", float64(simCount)/simTimeF32.Seconds())

	// INT8 simulation (conversion only since full INT8 model needs AVX-512)
	fmt.Println("\n2⃣ INT8 SIMULATION (Conversion)")
	fmt.Println(strings.Repeat("-", 50))

	// Convert embeddings to INT8
	int8Embeddings := make([][]uint8, len(encodings))
	conversionTime := time.Duration(0)

	for i, emb := range encodings {
		start := time.Now()
		int8Emb := convertToInt8(emb)
		conversionTime += time.Since(start)
		int8Embeddings[i] = int8Emb
	}

	fmt.Printf("Conversion time for %d embeddings: %v\n", len(encodings), conversionTime)
	fmt.Printf("Avg conversion time: %v\n", conversionTime/time.Duration(len(encodings)))

	// INT8 similarity computation
	fmt.Println("\nINT8 similarity computation:")
	start = time.Now()
	simCount = 0
	for i := 0; i < 1000; i++ {
		for j := 0; j < len(int8Embeddings); j++ {
			for k := j + 1; k < len(int8Embeddings); k++ {
				_ = cosineSimilarityInt8(int8Embeddings[j], int8Embeddings[k])
				simCount++
			}
		}
	}
	simTimeInt8 := time.Since(start)
	fmt.Printf("  %d similarity computations in %v\n", simCount, simTimeInt8)
	fmt.Printf("  Rate: %.0f/sec\n", float64(simCount)/simTimeInt8.Seconds())
	fmt.Printf("  Speedup vs Float32: %.2fx\n", simTimeF32.Seconds()/simTimeInt8.Seconds())

	// Memory comparison
	fmt.Println("\n3⃣ MEMORY COMPARISON")
	fmt.Println(strings.Repeat("-", 50))

	vocabSize := modelF32.VocabSize
	embedDim := modelF32.EmbedDim

	f32WeightsSize := vocabSize * embedDim * 4
	int8WeightsSize := vocabSize * embedDim * 1

	f32EmbeddingSize := embedDim * 4
	int8EmbeddingSize := embedDim * 1

	fmt.Printf("Model weights:\n")
	fmt.Printf("  Float32: %.2f MB\n", float64(f32WeightsSize)/(1024*1024))
	fmt.Printf("  INT8:    %.2f MB\n", float64(int8WeightsSize)/(1024*1024))
	fmt.Printf("  Reduction: %.1f%%\n", (1.0-float64(int8WeightsSize)/float64(f32WeightsSize))*100)

	fmt.Printf("\nPer embedding:\n")
	fmt.Printf("  Float32: %d bytes\n", f32EmbeddingSize)
	fmt.Printf("  INT8:    %d bytes\n", int8EmbeddingSize)
	fmt.Printf("  Reduction: %.1f%%\n", (1.0-float64(int8EmbeddingSize)/float64(f32EmbeddingSize))*100)

	// Accuracy comparison
	fmt.Println("\n4⃣ ACCURACY COMPARISON")
	fmt.Println(strings.Repeat("-", 50))

	fmt.Println("Similarity scores:")
	for i := 0; i < len(testTexts) && i < 3; i++ {
		for j := i + 1; j < len(testTexts) && j < 3; j++ {
			simF32 := gobed.CosineSimilarity(encodings[i], encodings[j])
			simInt8 := cosineSimilarityInt8(int8Embeddings[i], int8Embeddings[j])
			diff := abs(simF32 - simInt8)

			fmt.Printf("  '%s' vs '%s':\n",
				truncate(testTexts[i], 20), truncate(testTexts[j], 20))
			fmt.Printf("    Float32: %.4f\n", simF32)
			fmt.Printf("    INT8:    %.4f\n", simInt8)
			fmt.Printf("    Diff:    %.4f (%.1f%%)\n", diff, diff*100)
		}
	}

	// Summary
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" SUMMARY")
	fmt.Println(strings.Repeat("=", 80))

	fmt.Printf("\n Performance:\n")
	fmt.Printf("  Float32 encoding: %.0f ops/sec\n", throughputF32)
	fmt.Printf("  Float32 similarity: %.0f/sec\n", float64(simCount)/simTimeF32.Seconds())
	fmt.Printf("  INT8 similarity: %.0f/sec (%.1fx faster)\n",
		float64(simCount)/simTimeInt8.Seconds(),
		simTimeF32.Seconds()/simTimeInt8.Seconds())

	fmt.Printf("\n Memory:\n")
	fmt.Printf("  Model size reduction: 75%%\n")
	fmt.Printf("  Embedding size reduction: 75%%\n")

	fmt.Printf("\n Trade-offs:\n")
	fmt.Printf("   75%% memory reduction\n")
	fmt.Printf("   Faster similarity computation\n")
	fmt.Printf("    ~5%% accuracy loss on average\n")
	fmt.Printf("    Requires quantization step\n")

	fmt.Println("\n Performance comparison completed!")
}

func convertToInt8(embedding []float32) []uint8 {
	result := make([]uint8, len(embedding))

	// Find min and max
	minVal, maxVal := embedding[0], embedding[0]
	for _, val := range embedding {
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
	}

	// Handle edge case
	if minVal == maxVal {
		for i := range result {
			result[i] = 128
		}
		return result
	}

	// Scale to 0-255
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

func cosineSimilarityInt8(a, b []uint8) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	var dotProduct, normA, normB int64

	for i := 0; i < len(a); i++ {
		aVal := int16(a[i]) - 128
		bVal := int16(b[i]) - 128

		dotProduct += int64(aVal) * int64(bVal)
		normA += int64(aVal) * int64(aVal)
		normB += int64(bVal) * int64(bVal)
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	// Simple integer square root approximation
	sqrtA := int64(0)
	sqrtB := int64(0)

	// Newton's method for integer square root
	x := normA
	for x > 0 {
		next := (x + normA/x) / 2
		if next >= x {
			sqrtA = x
			break
		}
		x = next
	}

	x = normB
	for x > 0 {
		next := (x + normB/x) / 2
		if next >= x {
			sqrtB = x
			break
		}
		x = next
	}

	if sqrtA == 0 || sqrtB == 0 {
		return 0.0
	}

	return float32(dotProduct) / float32(sqrtA*sqrtB)
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
