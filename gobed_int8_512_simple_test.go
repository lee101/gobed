package gobed

import (
	"testing"
	"time"
)

// TestLoadSimpleInt8Model512 tests loading the simple int8 model
func TestLoadSimpleInt8Model512(t *testing.T) {
	start := time.Now()
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("Simple Int8 model not available: %v", err)
	}
	loadTime := time.Since(start)

	t.Logf("Simple Int8 model loaded in %v", loadTime)

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

// TestSimpleTokenizer tests the simple tokenizer
func TestSimpleTokenizer(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("Simple Int8 model not available: %v", err)
	}

	texts := []string{
		"hello world",
		"machine learning",
		"neural networks",
		"artificial intelligence",
	}

	for _, text := range texts {
		tokens := model.SimpleTokenize(text)
		t.Logf("Text: %q", text)
		t.Logf("  Tokens (%d): %v", len(tokens), tokens)

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

// TestSimpleInt8Similarity tests similarity computation
func TestSimpleInt8Similarity(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("Simple Int8 model not available: %v", err)
	}

	testCases := []struct {
		text1    string
		text2    string
		minSim   float32 // minimum expected similarity
	}{
		{"machine learning", "machine learning", 0.99},
		{"deep learning", "neural networks", 0.3},
		{"computer", "machine", 0.1},
		{"hello", "world", 0.0},
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

		// Check minimum expected similarity
		if tc.text1 == tc.text2 && similarity < tc.minSim {
			t.Errorf("Expected similarity >= %.2f for identical texts, got %.4f", tc.minSim, similarity)
		}
	}
}

// TestSimpleInt8Performance tests performance
func TestSimpleInt8Performance(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("Simple Int8 model not available: %v", err)
	}

	texts := []string{
		"machine learning algorithms",
		"deep neural networks",
		"natural language processing",
		"computer vision applications",
		"artificial intelligence systems",
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

	t.Logf("Simple Int8 Performance:")
	t.Logf("  Total time: %v", elapsed)
	t.Logf("  Average latency: %v", avgLatency)
	t.Logf("  Throughput: %.0f embeddings/sec", throughput)

	// Compare with original model if available
	if originalModel, err := LoadModel(); err == nil {
		start = time.Now()
		for i := 0; i < 100; i++ { // Fewer iterations
			text := texts[i%len(texts)]
			_, err := originalModel.Encode(text)
			if err != nil {
				t.Fatalf("Original embedding failed: %v", err)
			}
		}
		originalElapsed := time.Since(start)
		originalAvgLatency := originalElapsed / 100

		speedup := float64(originalAvgLatency) / float64(avgLatency)
		memoryReduction := 1024.0 / 512.0 * 4.0 // float32 to int8 + dimension reduction

		t.Logf("Comparison with original model:")
		t.Logf("  Original latency: %v", originalAvgLatency)
		t.Logf("  Int8 speedup: %.1fx", speedup)
		t.Logf("  Memory reduction: %.1fx", memoryReduction)
	}
}

// BenchmarkSimpleInt8Embedding benchmarks the simple int8 embedding
func BenchmarkSimpleInt8Embedding(b *testing.B) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skipf("Simple Int8 model not available: %v", err)
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

// BenchmarkSimpleInt8EmbeddingInt8 benchmarks int8 output
func BenchmarkSimpleInt8EmbeddingInt8(b *testing.B) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skipf("Simple Int8 model not available: %v", err)
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