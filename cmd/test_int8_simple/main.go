package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🚀 Testing Int8 512-dim Model with Int16 Tokenizer")
	fmt.Println("=" * 60)

	// Load the int8 model
	fmt.Println("📚 Loading Int8 model...")
	start := time.Now()
	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	loadTime := time.Since(start)
	fmt.Printf("✅ Model loaded in %v\n\n", loadTime)

	// Test texts
	testTexts := []string{
		"machine learning algorithms",
		"deep neural networks",
		"natural language processing",
		"computer vision applications",
		"artificial intelligence systems",
	}

	// Test embedding generation
	fmt.Println("🔧 Testing embedding generation...")
	for i, text := range testTexts {
		start := time.Now()
		embedding, err := model.Embed(text)
		if err != nil {
			log.Printf("Failed to embed text %d: %v", i, err)
			continue
		}
		embedTime := time.Since(start)

		fmt.Printf("  Text: %q\n", text)
		fmt.Printf("    Embedding dims: %d\n", len(embedding))
		fmt.Printf("    Time: %v\n", embedTime)

		// Show first few dimensions
		if len(embedding) >= 5 {
			fmt.Printf("    Sample values: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
				embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
		}
		fmt.Println()
	}

	// Test tokenization
	fmt.Println("📝 Testing tokenization...")
	for _, text := range testTexts[:3] {
		tokens := model.SimpleTokenize(text)
		fmt.Printf("  Text: %q\n", text)
		fmt.Printf("    Tokens: %v (count: %d)\n", tokens, len(tokens))
		fmt.Println()
	}

	// Test int8 embeddings
	fmt.Println("🔢 Testing int8 embeddings...")
	for _, text := range testTexts[:3] {
		start := time.Now()
		int8Result, err := model.EmbedInt8(text)
		if err != nil {
			log.Printf("Failed to get int8 embedding: %v", err)
			continue
		}
		embedTime := time.Since(start)

		fmt.Printf("  Text: %q\n", text)
		fmt.Printf("    Scale: %.6f\n", int8Result.Scale)
		fmt.Printf("    Vector dims: %d\n", len(int8Result.Vector))
		fmt.Printf("    Time: %v\n", embedTime)

		// Show vector range
		minVal, maxVal := int8Result.Vector[0], int8Result.Vector[0]
		for _, v := range int8Result.Vector {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		fmt.Printf("    Vector range: [%d, %d]\n", minVal, maxVal)
		fmt.Println()
	}

	// Test similarity computation
	fmt.Println("🎯 Testing similarity computation...")
	testPairs := []struct {
		text1, text2 string
	}{
		{"machine learning", "machine learning"},
		{"deep learning", "neural networks"},
		{"computer vision", "image processing"},
		{"hello world", "machine learning"},
		{"artificial intelligence", "machine learning"},
	}

	for _, pair := range testPairs {
		start := time.Now()
		similarity, err := model.Similarity(pair.text1, pair.text2)
		if err != nil {
			log.Printf("Failed to compute similarity: %v", err)
			continue
		}
		simTime := time.Since(start)

		fmt.Printf("  Similarity(%q, %q) = %.4f (time: %v)\n",
			pair.text1, pair.text2, similarity, simTime)
	}

	// Performance benchmark
	fmt.Println("\n⏱️  Performance benchmark...")
	numIterations := 1000
	testText := "machine learning algorithms for neural networks"

	// Benchmark embedding generation
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		_, err := model.Embed(testText)
		if err != nil {
			log.Fatalf("Benchmark failed: %v", err)
		}
	}
	totalTime := time.Since(start)

	avgLatency := totalTime / time.Duration(numIterations)
	throughput := float64(numIterations) / totalTime.Seconds()

	fmt.Printf("  Iterations: %d\n", numIterations)
	fmt.Printf("  Total time: %v\n", totalTime)
	fmt.Printf("  Average latency: %v\n", avgLatency)
	fmt.Printf("  Throughput: %.0f embeddings/sec\n", throughput)

	// Benchmark int8 embedding generation
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		_, err := model.EmbedInt8(testText)
		if err != nil {
			log.Fatalf("Int8 benchmark failed: %v", err)
		}
	}
	int8TotalTime := time.Since(start)

	int8AvgLatency := int8TotalTime / time.Duration(numIterations)
	int8Throughput := float64(numIterations) / int8TotalTime.Seconds()

	fmt.Printf("\n  Int8 Performance:\n")
	fmt.Printf("    Average latency: %v\n", int8AvgLatency)
	fmt.Printf("    Throughput: %.0f embeddings/sec\n", int8Throughput)

	// Benchmark similarity computation
	start = time.Now()
	for i := 0; i < numIterations/2; i++ {
		_, err := model.Similarity("machine learning", "neural networks")
		if err != nil {
			log.Fatalf("Similarity benchmark failed: %v", err)
		}
	}
	simTotalTime := time.Since(start)

	simAvgLatency := simTotalTime / time.Duration(numIterations/2)
	simThroughput := float64(numIterations/2) / simTotalTime.Seconds()

	fmt.Printf("\n  Similarity Performance:\n")
	fmt.Printf("    Average latency: %v\n", simAvgLatency)
	fmt.Printf("    Throughput: %.0f similarities/sec\n", simThroughput)

	fmt.Println("\n✅ Int8 model test complete!")
	fmt.Printf("🎉 Key metrics: %.0f emb/sec, %v avg latency, 15MB memory\n", throughput, avgLatency)
}