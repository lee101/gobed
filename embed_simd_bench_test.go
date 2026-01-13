package gobed

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/lee101/gobed/metrics"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// Test data for quality validation
var qualityTestQueries = []string{
	"machine learning algorithms",
	"deep neural networks",
	"natural language processing",
	"computer vision applications",
	"reinforcement learning",
	"transformer architecture",
	"attention mechanism",
	"gradient descent optimization",
	"convolutional neural network",
	"recurrent neural network",
}

var qualityTestDocs = []string{
	"Machine learning is a subset of artificial intelligence",
	"Deep learning uses multiple layers of neural networks",
	"NLP involves processing and understanding human language",
	"Computer vision enables machines to interpret visual data",
	"Reinforcement learning learns through trial and error",
	"Transformers revolutionized NLP with self-attention",
	"Attention allows models to focus on relevant parts",
	"Gradient descent minimizes loss functions iteratively",
	"CNNs are effective for image recognition tasks",
	"RNNs process sequential data with memory",
	"Python is popular for data science",
	"GPU acceleration speeds up training",
	"Backpropagation computes gradients efficiently",
	"Dropout prevents overfitting in neural networks",
	"Batch normalization stabilizes training",
}

// BenchmarkEmbedOriginal benchmarks the original embedding method
func BenchmarkEmbedOriginal(b *testing.B) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skip("model not available")
	}

	tokens := model.SimpleTokenize("machine learning deep neural network optimization")
	result := make([]float32, Int8EmbeddingDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.EmbedTokensInto(tokens, result)
	}
}

// BenchmarkEmbedSIMD benchmarks the SIMD-accelerated embedding method
func BenchmarkEmbedSIMD(b *testing.B) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skip("model not available")
	}

	tokens := model.SimpleTokenize("machine learning deep neural network optimization")
	result := make([]float32, Int8EmbeddingDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.EmbedTokensIntoSIMD(tokens, result)
	}
}

// BenchmarkEmbedComparison runs both methods and reports speedup
func BenchmarkEmbedComparison(b *testing.B) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skip("model not available")
	}

	testTexts := []string{
		"short text",
		"medium length text with more tokens to process",
		"a much longer text that contains many more tokens and should stress test the embedding function with significant workload across all dimensions of the embedding space",
	}

	for _, text := range testTexts {
		tokens := model.SimpleTokenize(text)
		tokenCount := len(tokens)

		b.Run(fmt.Sprintf("Original_%dtokens", tokenCount), func(b *testing.B) {
			result := make([]float32, Int8EmbeddingDim)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				model.EmbedTokensInto(tokens, result)
			}
		})

		b.Run(fmt.Sprintf("SIMD_%dtokens", tokenCount), func(b *testing.B) {
			result := make([]float32, Int8EmbeddingDim)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				model.EmbedTokensIntoSIMD(tokens, result)
			}
		})
	}
}

// TestEmbedQualityEquivalence verifies SIMD produces identical results
func TestEmbedQualityEquivalence(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skip("model not available")
	}

	testTexts := append(qualityTestQueries, qualityTestDocs...)

	for _, text := range testTexts {
		tokens := model.SimpleTokenize(text)

		resultOrig := make([]float32, Int8EmbeddingDim)
		resultSIMD := make([]float32, Int8EmbeddingDim)

		model.EmbedTokensInto(tokens, resultOrig)
		model.EmbedTokensIntoSIMD(tokens, resultSIMD)

		// Check all dimensions match
		maxDiff := float32(0)
		for i := 0; i < Int8EmbeddingDim; i++ {
			diff := float32(math.Abs(float64(resultOrig[i] - resultSIMD[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		// Allow tiny floating point differences
		if maxDiff > 1e-5 {
			t.Errorf("text %q: max diff %.8f exceeds tolerance", text, maxDiff)
		}
	}
}

// TestNDCG10Quality tests that search quality is maintained
func TestNDCG10Quality(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skip("model not available")
	}

	// Embed all docs
	docEmbsOrig := make([][]float32, len(qualityTestDocs))
	docEmbsSIMD := make([][]float32, len(qualityTestDocs))

	for i, doc := range qualityTestDocs {
		tokens := model.SimpleTokenize(doc)

		docEmbsOrig[i] = make([]float32, Int8EmbeddingDim)
		docEmbsSIMD[i] = make([]float32, Int8EmbeddingDim)

		model.EmbedTokensInto(tokens, docEmbsOrig[i])
		model.EmbedTokensIntoSIMD(tokens, docEmbsSIMD[i])
	}

	// For each query, compute rankings and NDCG
	for qi, query := range qualityTestQueries {
		tokens := model.SimpleTokenize(query)

		queryOrig := make([]float32, Int8EmbeddingDim)
		querySIMD := make([]float32, Int8EmbeddingDim)

		model.EmbedTokensInto(tokens, queryOrig)
		model.EmbedTokensIntoSIMD(tokens, querySIMD)

		// Compute similarities and rankings
		scoresOrig := make([]float64, len(qualityTestDocs))
		scoresSIMD := make([]float64, len(qualityTestDocs))

		for di := range qualityTestDocs {
			scoresOrig[di] = float64(cosineSim(queryOrig, docEmbsOrig[di]))
			scoresSIMD[di] = float64(cosineSim(querySIMD, docEmbsSIMD[di]))
		}

		rankOrig := rankByScore(scoresOrig)
		rankSIMD := rankByScore(scoresSIMD)

		// Create relevance based on original ranking (top-k are relevant)
		relevance := make(map[int]float64)
		for rank, docIdx := range rankOrig[:minInt(10, len(rankOrig))] {
			relevance[docIdx] = float64(10 - rank) // Higher score for better rank
		}

		ndcgOrig := metrics.ComputeNDCG(rankOrig, relevance, 10)
		ndcgSIMD := metrics.ComputeNDCG(rankSIMD, relevance, 10)

		t.Logf("Query %d: NDCG@10 orig=%.4f simd=%.4f", qi, ndcgOrig, ndcgSIMD)

		// SIMD should produce very similar NDCG (allow small diff due to float precision)
		if math.Abs(ndcgOrig-ndcgSIMD) > 0.01 {
			t.Errorf("Query %d: NDCG@10 diff too large: orig=%.4f simd=%.4f", qi, ndcgOrig, ndcgSIMD)
		}
	}
}

// BenchmarkEndToEndSearch benchmarks full embed+search pipeline
func BenchmarkEndToEndSearch(b *testing.B) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skip("model not available")
	}

	// Create synthetic corpus
	corpusSize := 10000
	corpus := make([][]float32, corpusSize)

	for i := 0; i < corpusSize; i++ {
		tokens := model.SimpleTokenize(fmt.Sprintf("document number %d with random content %d", i, rand.Int()))
		corpus[i] = make([]float32, Int8EmbeddingDim)
		model.EmbedTokensInto(tokens, corpus[i])
	}

	query := "document search query"
	queryTokens := model.SimpleTokenize(query)
	queryEmb := make([]float32, Int8EmbeddingDim)

	b.Run("Original_Embed+BruteSearch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			model.EmbedTokensInto(queryTokens, queryEmb)
			_ = bruteForceTopK(queryEmb, corpus, 10)
		}
	})

	b.Run("SIMD_Embed+BruteSearch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			model.EmbedTokensIntoSIMD(queryTokens, queryEmb)
			_ = bruteForceTopK(queryEmb, corpus, 10)
		}
	})
}

// BenchmarkSIMDFunctions benchmarks individual SIMD functions
func BenchmarkSIMDFunctions(b *testing.B) {
	embedding := make([]int8, 512)
	for i := range embedding {
		embedding[i] = int8(rand.Intn(256) - 128)
	}
	result := make([]float32, 512)
	scale := float32(0.1)

	b.Run("ClearF32", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			simd.ClearF32(result)
		}
	})

	b.Run("EmbedAccum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			simd.EmbedAccum(embedding, scale, result)
		}
	})

	b.Run("ScaleF32", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			simd.ScaleF32(result, 0.5)
		}
	})
}

// TestEmbedSpeedup measures actual speedup
func TestEmbedSpeedup(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skip("model not available")
	}

	text := "machine learning deep neural network optimization transformer attention mechanism gradient descent"
	tokens := model.SimpleTokenize(text)
	result := make([]float32, Int8EmbeddingDim)

	iterations := 100000

	// Warmup
	for i := 0; i < 1000; i++ {
		model.EmbedTokensInto(tokens, result)
		model.EmbedTokensIntoSIMD(tokens, result)
	}

	// Time original
	start := time.Now()
	for i := 0; i < iterations; i++ {
		model.EmbedTokensInto(tokens, result)
	}
	origDuration := time.Since(start)

	// Time SIMD
	start = time.Now()
	for i := 0; i < iterations; i++ {
		model.EmbedTokensIntoSIMD(tokens, result)
	}
	simdDuration := time.Since(start)

	speedup := float64(origDuration) / float64(simdDuration)

	t.Logf("Original: %v (%d ns/op)", origDuration, origDuration.Nanoseconds()/int64(iterations))
	t.Logf("SIMD:     %v (%d ns/op)", simdDuration, simdDuration.Nanoseconds()/int64(iterations))
	t.Logf("Speedup:  %.2fx", speedup)

	if speedup < 1.0 {
		t.Logf("Warning: SIMD is slower than original")
	}
}

// Helper functions

func cosineSim(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(normA*normB)))
}

func rankByScore(scores []float64) []int {
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}
	// Sort descending by score
	for i := 0; i < len(indices)-1; i++ {
		for j := i + 1; j < len(indices); j++ {
			if scores[indices[j]] > scores[indices[i]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}
	return indices
}

func bruteForceTopK(query []float32, corpus [][]float32, k int) []int {
	scores := make([]float64, len(corpus))
	for i, doc := range corpus {
		scores[i] = float64(cosineSim(query, doc))
	}
	ranking := rankByScore(scores)
	if k > len(ranking) {
		k = len(ranking)
	}
	return ranking[:k]
}

