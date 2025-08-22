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
	fmt.Println("🚀 INT8 vs FLOAT32 PERFORMANCE & ACCURACY COMPARISON")
	fmt.Println(strings.Repeat("=", 80))

	// Test sentences
	testSentences := []string{
		"Hello world",
		"Machine learning is fascinating.",
		"Python is a programming language.",
		"The weather is nice today.",
		"Deep learning models are powerful.",
		"Natural language processing",
		"JavaScript runs in browsers.",
		"Mathematics requires practice.",
		"Birds are singing beautifully.",
		"Pizza tastes delicious.",
	}

	// Load both models
	fmt.Println("\n📦 Loading models...")

	// Load standard Float32 model
	fmt.Println("Loading Float32 model...")
	modelFloat32, err := gobed.LoadModel()
	if err != nil {
		panic(fmt.Sprintf("Failed to load Float32 model: %v", err))
	}

	// Load INT8 model (with INT8 enabled)
	fmt.Println("Loading INT8 model with SIMD...")
	modelInt8, err := gobed.LoadModelInt8(true)
	if err != nil {
		panic(fmt.Sprintf("Failed to load INT8 model: %v", err))
	}

	// Load INT8 model with Float32 computation for comparison
	fmt.Println("Loading INT8 model (Float32 mode for comparison)...")
	modelInt8Float, err := gobed.LoadModelInt8(false)
	if err != nil {
		panic(fmt.Sprintf("Failed to load comparison model: %v", err))
	}

	// Accuracy comparison
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("📊 ACCURACY COMPARISON")
	fmt.Println(strings.Repeat("=", 80))

	var totalDiff float64
	var maxDiff float32
	comparisons := 0

	for i := 0; i < len(testSentences); i++ {
		for j := i + 1; j < len(testSentences); j++ {
			text1 := testSentences[i]
			text2 := testSentences[j]

			// Float32 similarity
			emb1_f32, _ := modelFloat32.Encode(text1)
			emb2_f32, _ := modelFloat32.Encode(text2)
			simFloat32 := gobed.CosineSimilarity(emb1_f32, emb2_f32)

			// INT8 similarity
			emb1_int8, _ := modelInt8.Encode(text1)
			emb2_int8, _ := modelInt8.Encode(text2)
			simInt8 := gobed.CosineSimilarityInt8(emb1_int8, emb2_int8)

			// Calculate difference
			diff := float32(math.Abs(float64(simFloat32 - simInt8)))
			totalDiff += float64(diff)
			if diff > maxDiff {
				maxDiff = diff
			}
			comparisons++

			if i == 0 && j < 3 { // Show first few comparisons
				fmt.Printf("\n\"%s\" vs \"%s\"\n",
					truncate(text1, 30), truncate(text2, 30))
				fmt.Printf("  Float32: %.6f\n", simFloat32)
				fmt.Printf("  INT8:    %.6f\n", simInt8)
				fmt.Printf("  Diff:    %.6f (%.2f%%)\n", diff, diff*100)
			}
		}
	}

	avgDiff := totalDiff / float64(comparisons)
	fmt.Printf("\n📈 Accuracy Summary:\n")
	fmt.Printf("  Average difference: %.6f\n", avgDiff)
	fmt.Printf("  Maximum difference: %.6f\n", maxDiff)
	fmt.Printf("  Comparisons made: %d\n", comparisons)

	if avgDiff < 0.01 {
		fmt.Println("  ✅ Excellent accuracy! INT8 is nearly identical to Float32")
	} else if avgDiff < 0.05 {
		fmt.Println("  ✅ Good accuracy! INT8 provides sufficient precision")
	} else {
		fmt.Println("  ⚠️  Moderate accuracy loss with INT8")
	}

	// Performance comparison
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("⚡ PERFORMANCE COMPARISON")
	fmt.Println(strings.Repeat("=", 80))

	iterations := 1000

	// Warmup
	for i := 0; i < 100; i++ {
		modelFloat32.Encode(testSentences[0])
		modelInt8.Encode(testSentences[0])
	}

	// Benchmark Float32
	fmt.Printf("\nBenchmarking Float32 (%d iterations)...\n", iterations)
	startFloat32 := time.Now()
	for i := 0; i < iterations; i++ {
		for _, text := range testSentences {
			_, _ = modelFloat32.Encode(text)
		}
	}
	timeFloat32 := time.Since(startFloat32)

	// Benchmark INT8 with SIMD
	fmt.Printf("Benchmarking INT8+SIMD (%d iterations)...\n", iterations)
	startInt8 := time.Now()
	for i := 0; i < iterations; i++ {
		for _, text := range testSentences {
			_, _ = modelInt8.Encode(text)
		}
	}
	timeInt8 := time.Since(startInt8)

	// Benchmark INT8 without SIMD (Float32 computation)
	fmt.Printf("Benchmarking INT8 (Float32 mode) (%d iterations)...\n", iterations)
	startInt8Float := time.Now()
	for i := 0; i < iterations; i++ {
		for _, text := range testSentences {
			_, _ = modelInt8Float.Encode(text)
		}
	}
	timeInt8Float := time.Since(startInt8Float)

	// Calculate metrics
	totalEmbeddings := iterations * len(testSentences)

	throughputFloat32 := float64(totalEmbeddings) / timeFloat32.Seconds()
	throughputInt8 := float64(totalEmbeddings) / timeInt8.Seconds()
	throughputInt8Float := float64(totalEmbeddings) / timeInt8Float.Seconds()

	speedup := timeFloat32.Seconds() / timeInt8.Seconds()

	fmt.Println("\n📊 Performance Results:")
	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("Float32:\n")
	fmt.Printf("  Total time:        %v\n", timeFloat32)
	fmt.Printf("  Embeddings/sec:    %.0f\n", throughputFloat32)
	fmt.Printf("  Avg latency:       %.2f µs\n", float64(timeFloat32.Microseconds())/float64(totalEmbeddings))

	fmt.Printf("\nINT8+SIMD:\n")
	fmt.Printf("  Total time:        %v\n", timeInt8)
	fmt.Printf("  Embeddings/sec:    %.0f\n", throughputInt8)
	fmt.Printf("  Avg latency:       %.2f µs\n", float64(timeInt8.Microseconds())/float64(totalEmbeddings))
	fmt.Printf("  Speedup:           %.2fx faster\n", speedup)

	fmt.Printf("\nINT8 (Float32 mode):\n")
	fmt.Printf("  Total time:        %v\n", timeInt8Float)
	fmt.Printf("  Embeddings/sec:    %.0f\n", throughputInt8Float)
	fmt.Printf("  Avg latency:       %.2f µs\n", float64(timeInt8Float.Microseconds())/float64(totalEmbeddings))

	// Memory usage comparison
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("💾 MEMORY USAGE COMPARISON")
	fmt.Println(strings.Repeat("=", 80))

	vocabSize := modelFloat32.VocabSize
	embedDim := modelFloat32.EmbedDim

	float32Memory := float64(vocabSize*embedDim*4) / (1024 * 1024)
	int8Memory := float64(vocabSize*embedDim*1) / (1024 * 1024)
	memorySaving := (1.0 - int8Memory/float32Memory) * 100

	fmt.Printf("Float32 weights:  %.2f MB\n", float32Memory)
	fmt.Printf("INT8 weights:     %.2f MB\n", int8Memory)
	fmt.Printf("Memory saving:    %.1f%%\n", memorySaving)

	// Test extreme values
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("🔬 EDGE CASE TESTING")
	fmt.Println(strings.Repeat("=", 80))

	edgeCases := []string{
		"",                           // Empty string
		"a",                          // Single character
		"🔥🚀💯",                        // Emojis
		strings.Repeat("test ", 100), // Long text
	}

	fmt.Println("\nTesting edge cases for stability...")
	for _, text := range edgeCases {
		description := text
		if len(description) > 20 {
			description = description[:20] + "..."
		}
		if description == "" {
			description = "(empty)"
		}

		fmt.Printf("Testing: %-25s ", description)

		// Try encoding with both models
		_, errF32 := modelFloat32.Encode(text)
		_, errInt8 := modelInt8.Encode(text)

		if errF32 != nil || errInt8 != nil {
			fmt.Printf("❌ Error\n")
		} else {
			fmt.Printf("✅ OK\n")
		}
	}

	// Summary
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("📈 SUMMARY")
	fmt.Println(strings.Repeat("=", 80))

	fmt.Printf("\n🎯 Key Findings:\n")
	fmt.Printf("  • INT8+SIMD is %.1fx faster than Float32\n", speedup)
	fmt.Printf("  • Memory usage reduced by %.0f%%\n", memorySaving)
	fmt.Printf("  • Average accuracy difference: %.4f%%\n", avgDiff*100)
	fmt.Printf("  • Throughput improvement: %.0f → %.0f embeddings/sec\n",
		throughputFloat32, throughputInt8)

	if speedup > 2.0 && avgDiff < 0.01 {
		fmt.Println("\n✅ EXCELLENT! INT8+SIMD provides major speedup with minimal accuracy loss")
	} else if speedup > 1.5 && avgDiff < 0.05 {
		fmt.Println("\n✅ GOOD! INT8+SIMD provides significant benefits")
	} else {
		fmt.Println("\n⚠️  Results show moderate improvements")
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
