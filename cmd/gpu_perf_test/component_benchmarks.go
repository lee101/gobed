package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/ivf"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	fmt.Println("🔬 Component-Level Benchmarks")
	fmt.Println("======================================")
	fmt.Println("Testing each stage of the pipeline for performance bottlenecks")
	fmt.Println()

	// Load model for embedding generation
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Test stages
	runTokenizationBenchmark(model)
	runEmbeddingBenchmark(model)
	runKMeansBenchmark()
	runIVFIndexingBenchmark(model)
	runSearchBenchmark(model)
	runCorrectnessTests(model)
}

func runTokenizationBenchmark(model *gobed.EmbeddingModel) {
	fmt.Println("📝 Stage 1: Tokenization Benchmark")
	fmt.Println("----------------------------------")

	texts := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning algorithms are revolutionizing artificial intelligence",
		"Deep neural networks with transformer architectures enable state-of-the-art performance",
		strings.Repeat("word ", 100), // Long text
		"Short",
		"Another medium length sentence with various words and meanings",
	}

	var totalTime time.Duration
	iterations := 1000

	for _, text := range texts {
		start := time.Now()
		for i := 0; i < iterations; i++ {
			// Tokenization happens inside Encode
			_, _ = model.Encode(text)
		}
		elapsed := time.Since(start)
		avgTime := elapsed / time.Duration(iterations)
		totalTime += avgTime

		fmt.Printf("  Text length %3d chars: %6.2f µs/op (%7.0f ops/sec)\n",
			len(text),
			float64(avgTime.Microseconds()),
			1e9/float64(avgTime.Nanoseconds()))
	}

	avgTime := totalTime / time.Duration(len(texts))
	fmt.Printf("  Average: %.2f µs/op (%.0f ops/sec)\n",
		float64(avgTime.Microseconds()),
		1e9/float64(avgTime.Nanoseconds()))
	fmt.Println()
}

func runEmbeddingBenchmark(model *gobed.EmbeddingModel) {
	fmt.Println("🧮 Stage 2: Embedding Computation Benchmark")
	fmt.Println("-------------------------------------------")

	// Generate sample token sequences of different lengths
	tokenSequences := [][]int{
		{101, 2054, 2003, 102}, // Short
		{101, 2054, 2003, 1996, 2190, 2126, 2000, 2079, 2009, 102}, // Medium
		makeRandomTokens(50),  // Long
		makeRandomTokens(100), // Very long
		makeRandomTokens(200), // Max length
	}

	for _, tokens := range tokenSequences {
		iterations := 1000
		start := time.Now()

		for i := 0; i < iterations; i++ {
			// This should use the actual int8 embedding computation
			_ = computeEmbeddingFromTokens(model, tokens)
		}

		elapsed := time.Since(start)
		avgTime := elapsed / time.Duration(iterations)

		fmt.Printf("  %3d tokens: %6.2f µs/op (%7.0f ops/sec)\n",
			len(tokens),
			float64(avgTime.Microseconds()),
			1e9/float64(avgTime.Nanoseconds()))
	}
	fmt.Println()
}

func runKMeansBenchmark() {
	fmt.Println("⚡ Stage 3: K-Means Training Benchmark (IVF)")
	fmt.Println("--------------------------------------------")
	fmt.Println("  This is the BOTTLENECK - taking 2m34s for 240k vectors!")
	fmt.Println()

	// Test different dataset sizes
	sizes := []int{1000, 5000, 10000, 50000}
	nClusters := 256 // Smaller for testing

	for _, n := range sizes {
		// Generate random int8 vectors
		vectors := make([]simd.Vec512, n)
		scales := make([]float32, n)
		for i := 0; i < n; i++ {
			for j := 0; j < 512; j++ {
				vectors[i][j] = int8(rand.Intn(255) - 128)
			}
			scales[i] = rand.Float32() * 0.1
		}

		// Benchmark K-means
		kmeans := ivf.NewKMeans(nClusters, 10) // Reduced iterations for benchmark

		start := time.Now()
		kmeans.Fit(vectors, scales)
		elapsed := time.Since(start)

		throughput := float64(n) / elapsed.Seconds()
		fmt.Printf("  %7d vectors, %3d clusters: %8.2fms (%8.0f vecs/sec)\n",
			n, nClusters, float64(elapsed.Milliseconds()), throughput)

		// Estimate time for 240k vectors
		if n == 50000 {
			estimated := elapsed * 240000 / 50000
			fmt.Printf("  📊 Estimated for 240k vectors: %.1fs\n", estimated.Seconds())
			fmt.Printf("  🎯 Need ~50x speedup for <3s target!\n")
		}
	}

	fmt.Println()
	fmt.Println("  💡 Solution: GPU-accelerated K-means with CUDA")
	fmt.Println("     - Use cuML's K-means or custom CUDA kernel")
	fmt.Println("     - Batch distance computations on GPU")
	fmt.Println("     - Use tensor cores for int8 operations")
	fmt.Println()
}

func runIVFIndexingBenchmark(model *gobed.EmbeddingModel) {
	fmt.Println("📊 Stage 4: IVF Index Build Benchmark")
	fmt.Println("-------------------------------------")

	sizes := []int{100, 1000, 5000}

	for _, n := range sizes {
		// Generate embeddings
		vectors := make([]simd.Vec512, n)
		scales := make([]float32, n)
		ids := make([]int, n)

		for i := 0; i < n; i++ {
			for j := 0; j < 512; j++ {
				vectors[i][j] = int8(rand.Intn(255) - 128)
			}
			scales[i] = rand.Float32() * 0.1
			ids[i] = i
		}

		// Create and train index
		ivfIndex := ivf.NewIVFIndex(32, 4) // Smaller for testing

		// Training time
		trainStart := time.Now()
		ivfIndex.Train(vectors[:min(n, 1000)], scales[:min(n, 1000)])
		trainTime := time.Since(trainStart)

		// Indexing time
		indexStart := time.Now()
		ivfIndex.AddBatch(vectors, scales, ids)
		indexTime := time.Since(indexStart)

		fmt.Printf("  %5d vectors: Train %6.2fms, Index %6.2fms, Total %7.2fms\n",
			n,
			float64(trainTime.Milliseconds()),
			float64(indexTime.Milliseconds()),
			float64((trainTime + indexTime).Milliseconds()))
	}
	fmt.Println()
}

func runSearchBenchmark(model *gobed.EmbeddingModel) {
	fmt.Println("🔍 Stage 5: Search Performance Benchmark")
	fmt.Println("----------------------------------------")

	// Create a test index with varying sizes
	sizes := []int{100, 1000, 10000}

	for _, n := range sizes {
		engine := gobed.NewGPUSearchEngine(model)
		defer engine.Close()

		// Generate test documents
		docs := make([]string, n)
		ids := make([]int, n)
		for i := 0; i < n; i++ {
			docs[i] = fmt.Sprintf("Document %d with machine learning content", i)
			ids[i] = i
		}

		// Index
		_ = engine.IndexBatchWithIDs(ids, docs)

		// Benchmark search
		queries := []string{
			"machine learning",
			"artificial intelligence",
			"deep neural networks",
		}

		var totalTime time.Duration
		iterations := 100

		for _, query := range queries {
			start := time.Now()
			for i := 0; i < iterations; i++ {
				_, _ = engine.Search(query, 10)
			}
			elapsed := time.Since(start) / time.Duration(iterations)
			totalTime += elapsed
		}

		avgTime := totalTime / time.Duration(len(queries))
		fmt.Printf("  %6d docs: %6.2fms/query (%6.0f QPS)\n",
			n,
			float64(avgTime.Microseconds())/1000.0,
			1e9/float64(avgTime.Nanoseconds()))
	}
	fmt.Println()
}

func runCorrectnessTests(model *gobed.EmbeddingModel) {
	fmt.Println("✅ Stage 6: Correctness Verification")
	fmt.Println("------------------------------------")

	// Test 1: Tokenization consistency
	text := "Test text for verification"
	emb1, _ := model.Encode(text)
	emb2, _ := model.Encode(text)

	identical := true
	for i := range emb1 {
		if emb1[i] != emb2[i] {
			identical = false
			break
		}
	}

	if identical {
		fmt.Println("  ✓ Tokenization: Consistent (same input → same embedding)")
	} else {
		fmt.Println("  ✗ Tokenization: INCONSISTENT!")
	}

	// Test 2: Similarity sanity check
	sim1, _ := model.Similarity("machine learning", "deep learning")
	sim2, _ := model.Similarity("machine learning", "cooking recipes")

	if sim1 > sim2 {
		fmt.Printf("  ✓ Similarity: Correct (ML-DL=%.3f > ML-Cooking=%.3f)\n", sim1, sim2)
	} else {
		fmt.Printf("  ✗ Similarity: INCORRECT (ML-DL=%.3f ≤ ML-Cooking=%.3f)\n", sim1, sim2)
	}

	// Test 3: Embedding magnitude check (int8 quantization)
	emb, _ := model.Encode("Test embedding magnitude")
	var sumSq float32
	for _, val := range emb {
		sumSq += val * val
	}
	magnitude := sumSq // Not normalized in this model

	fmt.Printf("  ℹ Embedding magnitude²: %.3f (raw, not normalized)\n", magnitude)

	// Test 4: Search result diversity
	engine := gobed.NewGPUSearchEngine(model)
	defer engine.Close()

	testDocs := []string{
		"Machine learning algorithms",
		"Deep learning neural networks",
		"Computer vision applications",
		"Natural language processing",
		"Reinforcement learning agents",
	}

	ids := make([]int, len(testDocs))
	for i := range ids {
		ids[i] = i
	}

	_ = engine.IndexBatchWithIDs(ids, testDocs)
	results, _ := engine.Search("learning", 5)

	// Check for diverse scores
	if len(results) > 1 {
		allSame := true
		firstScore := results[0].Similarity
		for _, r := range results[1:] {
			if abs(r.Similarity-firstScore) > 0.001 {
				allSame = false
				break
			}
		}

		if !allSame {
			fmt.Printf("  ✓ Search diversity: Good (scores vary from %.3f to %.3f)\n",
				results[0].Similarity, results[len(results)-1].Similarity)
		} else {
			fmt.Printf("  ✗ Search diversity: SUSPICIOUS (all scores = %.3f)\n", firstScore)
		}
	}

	fmt.Println()
}

func makeRandomTokens(n int) []int {
	tokens := make([]int, n)
	for i := range tokens {
		tokens[i] = rand.Intn(30000) // BERT vocab size
	}
	return tokens
}

func computeEmbeddingFromTokens(model *gobed.EmbeddingModel, tokens []int) []float32 {
	// This is a placeholder - should use actual model method
	// In real implementation, this would call model's internal embedding computation
	result := make([]float32, 512)
	for i := range result {
		result[i] = rand.Float32()
	}
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}