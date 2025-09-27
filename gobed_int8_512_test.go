package gobed

import (
	"math"
	"testing"
	"time"
)

// TestLoadInt8Model512 tests loading the int8 512-dim model
func TestLoadInt8Model512(t *testing.T) {
	start := time.Now()
	model, err := LoadInt8Model512()
	if err != nil {
		t.Skipf("Int8 model not available: %v", err)
	}
	loadTime := time.Since(start)

	t.Logf("Int8 model loaded in %v", loadTime)
	t.Logf("Memory usage:\n%s", model.GetMemoryUsage())

	// Test basic functionality
	text := "machine learning is transforming data science"
	embedding, err := model.Embed(text)
	if err != nil {
		t.Fatalf("Failed to embed text: %v", err)
	}

	if len(embedding) != Int8EmbeddingDim {
		t.Fatalf("Expected embedding dim %d, got %d", Int8EmbeddingDim, len(embedding))
	}

	t.Logf("Embedding computed successfully, dimensions: %d", len(embedding))
}

// TestInt8Tokenizer tests the int16 tokenizer output
func TestInt8Tokenizer(t *testing.T) {
	model, err := LoadInt8Model512()
	if err != nil {
		t.Skipf("Int8 model not available: %v", err)
	}

	texts := []string{
		"hello world",
		"machine learning algorithms",
		"neural networks and deep learning",
		"artificial intelligence applications",
	}

	for _, text := range texts {
		tokens, err := model.Tokenize(text)
		if err != nil {
			t.Errorf("Failed to tokenize %q: %v", text, err)
			continue
		}

		t.Logf("Text: %q", text)
		t.Logf("  Tokens (%d): %v", len(tokens), tokens)

		// Verify all tokens are valid int16
		for i, token := range tokens {
			if token < 0 || token >= Int8VocabSize {
				t.Errorf("Invalid token at position %d: %d (out of range [0, %d))", i, token, Int8VocabSize)
			}
		}

		// Test embedding the tokens
		embedding, err := model.EmbedTokens(tokens)
		if err != nil {
			t.Errorf("Failed to embed tokens: %v", err)
			continue
		}

		if len(embedding) != Int8EmbeddingDim {
			t.Errorf("Expected embedding dim %d, got %d", Int8EmbeddingDim, len(embedding))
		}
	}
}

// TestInt8Embeddings tests the int8 embedding functionality
func TestInt8Embeddings(t *testing.T) {
	model, err := LoadInt8Model512()
	if err != nil {
		t.Skipf("Int8 model not available: %v", err)
	}

	texts := []string{
		"machine learning",
		"deep learning",
		"neural networks",
		"computer vision",
		"natural language processing",
	}

	// Test int8 embeddings
	for _, text := range texts {
		result, err := model.EmbedInt8(text)
		if err != nil {
			t.Errorf("Failed to get int8 embedding for %q: %v", text, err)
			continue
		}

		if len(result.Vector) != Int8EmbeddingDim {
			t.Errorf("Expected vector dim %d, got %d", Int8EmbeddingDim, len(result.Vector))
		}

		if result.Scale <= 0 {
			t.Errorf("Invalid scale factor: %f", result.Scale)
		}

		t.Logf("Text: %q", text)
		t.Logf("  Scale: %.6f", result.Scale)
		t.Logf("  Vector range: [%d, %d]", minInt8(result.Vector), maxInt8(result.Vector))

		// Test reconstruction quality
		float32Emb, err := model.Embed(text)
		if err != nil {
			t.Errorf("Failed to get float32 embedding: %v", err)
			continue
		}

		// Reconstruct from int8
		reconstructed := make([]float32, Int8EmbeddingDim)
		for i, val := range result.Vector {
			reconstructed[i] = float32(val) * result.Scale
		}

		// Calculate cosine similarity between original and reconstructed
		similarity := cosineSimilarity512(float32Emb, reconstructed)
		t.Logf("  Reconstruction similarity: %.4f", similarity)

		if similarity < 0.95 {
			t.Logf("  Warning: Low reconstruction similarity: %.4f", similarity)
		}
	}
}

// TestInt8Similarity tests the int8 similarity computation
func TestInt8Similarity(t *testing.T) {
	model, err := LoadInt8Model512()
	if err != nil {
		t.Skipf("Int8 model not available: %v", err)
	}

	testCases := []struct {
		text1    string
		text2    string
		expected float32 // approximate expected similarity
	}{
		{"machine learning", "machine learning", 1.0},
		{"deep learning", "neural networks", 0.7},
		{"computer vision", "image processing", 0.6},
		{"hello world", "machine learning", 0.1},
	}

	for _, tc := range testCases {
		similarity, err := model.Similarity(tc.text1, tc.text2)
		if err != nil {
			t.Errorf("Failed to compute similarity between %q and %q: %v", tc.text1, tc.text2, err)
			continue
		}

		t.Logf("Similarity(%q, %q) = %.4f", tc.text1, tc.text2, similarity)

		// Check if similarity is in reasonable range
		if similarity < 0 || similarity > 1 {
			t.Errorf("Similarity out of range [0, 1]: %.4f", similarity)
		}

		// For identical texts, similarity should be very high
		if tc.text1 == tc.text2 && similarity < 0.99 {
			t.Errorf("Identical texts should have similarity close to 1, got %.4f", similarity)
		}
	}
}

// TestInt8Performance benchmarks the int8 model performance
func TestInt8Performance(t *testing.T) {
	model, err := LoadInt8Model512()
	if err != nil {
		t.Skipf("Int8 model not available: %v", err)
	}

	texts := []string{
		"machine learning algorithms and applications",
		"deep neural networks for computer vision",
		"natural language processing with transformers",
		"reinforcement learning in robotics",
		"generative adversarial networks for image synthesis",
	}

	// Warmup
	for _, text := range texts {
		_, _ = model.Embed(text)
	}

	// Benchmark embedding speed
	numIterations := 1000
	start := time.Now()

	for i := 0; i < numIterations; i++ {
		text := texts[i%len(texts)]
		_, err := model.Embed(text)
		if err != nil {
			t.Fatalf("Embedding failed: %v", err)
		}
	}

	elapsed := time.Since(start)
	avgLatency := elapsed / time.Duration(numIterations)
	throughput := float64(numIterations) / elapsed.Seconds()

	t.Logf("Performance Results:")
	t.Logf("  Total time: %v", elapsed)
	t.Logf("  Average latency: %v", avgLatency)
	t.Logf("  Throughput: %.0f embeddings/sec", throughput)

	// Test int8 embedding speed
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		text := texts[i%len(texts)]
		_, err := model.EmbedInt8(text)
		if err != nil {
			t.Fatalf("Int8 embedding failed: %v", err)
		}
	}

	int8Elapsed := time.Since(start)
	int8AvgLatency := int8Elapsed / time.Duration(numIterations)
	int8Throughput := float64(numIterations) / int8Elapsed.Seconds()

	t.Logf("Int8 Performance Results:")
	t.Logf("  Total time: %v", int8Elapsed)
	t.Logf("  Average latency: %v", int8AvgLatency)
	t.Logf("  Throughput: %.0f embeddings/sec", int8Throughput)

	// Compare with original model if available
	originalModel, err := LoadModel()
	if err == nil {
		start = time.Now()
		for i := 0; i < 100; i++ { // Fewer iterations for original model
			text := texts[i%len(texts)]
			_, err := originalModel.Encode(text)
			if err != nil {
				t.Fatalf("Original embedding failed: %v", err)
			}
		}
		originalElapsed := time.Since(start)
		originalAvgLatency := originalElapsed / 100

		speedup := float64(originalAvgLatency) / float64(avgLatency)
		t.Logf("Comparison with original model:")
		t.Logf("  Original latency: %v", originalAvgLatency)
		t.Logf("  Int8 speedup: %.1fx", speedup)
	}
}

// BenchmarkInt8Embedding benchmarks the int8 embedding performance
func BenchmarkInt8Embedding(b *testing.B) {
	model, err := LoadInt8Model512()
	if err != nil {
		b.Skipf("Int8 model not available: %v", err)
	}

	text := "machine learning algorithms for neural networks"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := model.Embed(text)
		if err != nil {
			b.Fatalf("Embedding failed: %v", err)
		}
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "embeddings/sec")
}

// BenchmarkInt8EmbeddingInt8 benchmarks the int8 embedding with int8 output
func BenchmarkInt8EmbeddingInt8(b *testing.B) {
	model, err := LoadInt8Model512()
	if err != nil {
		b.Skipf("Int8 model not available: %v", err)
	}

	text := "machine learning algorithms for neural networks"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := model.EmbedInt8(text)
		if err != nil {
			b.Fatalf("Int8 embedding failed: %v", err)
		}
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "embeddings/sec")
}

// BenchmarkInt8Similarity benchmarks similarity computation
func BenchmarkInt8Similarity(b *testing.B) {
	model, err := LoadInt8Model512()
	if err != nil {
		b.Skipf("Int8 model not available: %v", err)
	}

	text1 := "machine learning algorithms"
	text2 := "deep neural networks"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := model.Similarity(text1, text2)
		if err != nil {
			b.Fatalf("Similarity computation failed: %v", err)
		}
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "similarities/sec")
}

// Helper functions
func minInt8(slice []int8) int8 {
	if len(slice) == 0 {
		return 0
	}
	min := slice[0]
	for _, v := range slice[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxInt8(slice []int8) int8 {
	if len(slice) == 0 {
		return 0
	}
	max := slice[0]
	for _, v := range slice[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func cosineSimilarity512(a, b []float32) float32 {
	if len(a) != len(b) || len(a) != Int8EmbeddingDim {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := 0; i < Int8EmbeddingDim; i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}