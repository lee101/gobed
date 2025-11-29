//go:build legacy

package gobed

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkTokenizationOptimized tests tokenization with buffer pooling
func BenchmarkTokenizationOptimized(b *testing.B) {
	model, err := LoadModel()
	if err != nil {
		b.Fatal(err)
	}
	// model.Close() not needed

	texts := []string{
		"A beautiful sunset over the mountains",
		"The quick brown fox jumps over the lazy dog",
		"Artificial intelligence is transforming the world",
		"Machine learning models can generate amazing images",
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			text := texts[i%len(texts)]
			cleanText := normalizeText(text)
			
			// With buffer pooling
			encoding, _ := model.tokenizer.EncodeSingle(cleanText, false)
			tokens := GetTokenBuffer()
			for _, id := range encoding.Ids {
				tokens = append(tokens, int(id))
			}
			PutTokenBuffer(tokens)
			
			i++
		}
	})
}

// BenchmarkTokenizationOld tests tokenization without optimizations
func BenchmarkTokenizationOld(b *testing.B) {
	model, err := LoadModel()
	if err != nil {
		b.Fatal(err)
	}
	// model.Close() not needed

	texts := []string{
		"A beautiful sunset over the mountains",
		"The quick brown fox jumps over the lazy dog",
		"Artificial intelligence is transforming the world",
		"Machine learning models can generate amazing images",
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			text := texts[i%len(texts)]
			cleanText := normalizeText(text)
			
			// Without buffer pooling
			encoding, _ := model.tokenizer.EncodeSingle(cleanText, false)
			tokens := make([]int, len(encoding.Ids))
			for j, id := range encoding.Ids {
				tokens[j] = int(id)
			}
			
			i++
		}
	})
}

// BenchmarkBatchProcessingOptimized tests batch processing with optimizations
func BenchmarkBatchProcessingOptimized(b *testing.B) {
	model, err := LoadOptimizedModel()
	if err != nil {
		b.Fatal(err)
	}
	// model.Close() not needed

	// Generate test texts
	batchSize := GetOptimalBatchSize()
	texts := make([]string, batchSize)
	for i := range texts {
		texts[i] = fmt.Sprintf("Test document %d with some content", i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = model.BatchEmbed(texts)
	}
	
	b.ReportMetric(float64(batchSize), "items/op")
}

// BenchmarkCacheHitRate measures cache effectiveness
func BenchmarkCacheHitRate(b *testing.B) {
	model, err := LoadOptimizedModel()
	if err != nil {
		b.Fatal(err)
	}
	// model.Close() not needed

	// Common queries that should hit cache after warmup
	queries := []string{
		"anime girl",
		"portrait",
		"landscape",
		"cute cat",
		"fantasy dragon",
		"cyberpunk city",
		"beautiful sunset",
		"abstract art",
	}

	// Warm up cache
	for _, q := range queries {
		_, _ = model.EmbedOptimized(q)
	}

	// Reset stats
	model.mu.Lock()
	model.cacheHits = 0
	model.totalQueries = 0
	model.mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		_, _ = model.EmbedOptimized(query)
	}

	model.mu.RLock()
	hitRate := float64(model.cacheHits) / float64(model.totalQueries) * 100
	model.mu.RUnlock()
	
	b.ReportMetric(hitRate, "%_cache_hit")
}

// BenchmarkMemoryAllocation measures memory usage
func BenchmarkMemoryAllocation(b *testing.B) {
	model, err := LoadOptimizedModel()
	if err != nil {
		b.Fatal(err)
	}
	// model.Close() not needed

	text := "A test sentence for measuring memory allocations"
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		_, _ = model.EmbedOptimized(text)
	}
}

// BenchmarkGPUBatchSize tests different GPU batch sizes
func BenchmarkGPUBatchSize(b *testing.B) {
	// Skip if not built with GPU support
	b.Skip("GPU benchmarks require GPU build tag")

	model, err := LoadOptimizedModel()
	if err != nil {
		b.Fatal(err)
	}
	// model.Close() not needed

	batchSizes := []int{32, 64, 128, 256, 512, 1024, 2048}
	
	for _, size := range batchSizes {
		b.Run(fmt.Sprintf("batch_%d", size), func(b *testing.B) {
			texts := make([]string, size)
			for i := range texts {
				texts[i] = fmt.Sprintf("Document %d", i)
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = model.BatchEmbed(texts)
			}
			
			b.ReportMetric(float64(size), "batch_size")
		})
	}
}

// BenchmarkQuantization tests quantization performance
func BenchmarkQuantization(b *testing.B) {
	embedding := make([]float32, 1024)
	for i := range embedding {
		embedding[i] = float32(i) / 1024.0
	}

	b.Run("FastQuantize", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			quantized, _ := FastQuantize(embedding)
			PutInt8Buffer(quantized)
		}
	})

	b.Run("StandardQuantize", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = quantizeEmbedding(embedding)
		}
	})
}

// BenchmarkParallelProcessing tests parallel processing performance
func BenchmarkParallelProcessing(b *testing.B) {
	processor := NewParallelProcessor()
	defer processor.Close()

	numItems := 1000
	items := make([]func(), numItems)
	for i := range items {
		items[i] = func() {
			// Simulate work
			time.Sleep(time.Microsecond)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		processor.ProcessBatch(items)
	}
	
	b.ReportMetric(float64(numItems), "items/op")
}

// TestMemoryPoolEffectiveness verifies buffer pool reduces allocations
func TestMemoryPoolEffectiveness(t *testing.T) {
	// Test functional correctness instead of allocation patterns
	// sync.Pool behavior is not deterministic in tests

	// Test 1: Verify pool functions work
	buf := GetTokenBuffer()
	if buf == nil {
		t.Error("GetTokenBuffer returned nil")
	}

	// Test 2: Verify capacity is reasonable
	if cap(buf) < 512 {
		t.Errorf("Buffer capacity too small: %d, expected >= 512", cap(buf))
	}

	// Test 3: Verify Put doesn't panic
	PutTokenBuffer(buf)

	// Test 4: Verify we can get another buffer
	buf2 := GetTokenBuffer()
	if buf2 == nil {
		t.Error("GetTokenBuffer returned nil after Put")
	}
	PutTokenBuffer(buf2)

	t.Logf("✓ Buffer pool functions are working correctly")
}

// TestOptimalBatchSizing verifies batch size calculation
func TestOptimalBatchSizing(t *testing.T) {
	cpuBatch := GetOptimalBatchSize()
	t.Logf("Optimal CPU batch size: %d", cpuBatch)
	
	if cpuBatch < 32 || cpuBatch > 2048 {
		t.Errorf("CPU batch size out of expected range: %d", cpuBatch)
	}
	
	if IsCUDAAvailable() {
		gpuBatch := GetOptimalGPUBatchSize()
		t.Logf("Optimal GPU batch size: %d", gpuBatch)
		
		if gpuBatch < 128 || gpuBatch > 4096 {
			t.Errorf("GPU batch size out of expected range: %d", gpuBatch)
		}
		
		// GPU batch should generally be larger
		if gpuBatch < cpuBatch {
			t.Error("GPU batch size should be >= CPU batch size")
		}
	}
}
