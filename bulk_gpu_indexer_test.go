package gobed

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestGPUCPUEmbeddingParity(t *testing.T) {
	modelPath := "model/real_model.safetensors"
	tokenizerPath := "model/tokenizer.json"

	config := BulkIndexerConfig{
		BatchSize:     1,
		MaxSeqLen:     512,
		ModelPath:     modelPath,
		TokenizerPath: tokenizerPath,
		UseGPU:        true,
	}

	gpuIndexer, err := NewBulkGPUIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create GPU indexer: %v", err)
	}
	defer gpuIndexer.Close()

	cpuConfig := config
	cpuConfig.UseGPU = false
	cpuIndexer, err := NewBulkGPUIndexer(cpuConfig)
	if err != nil {
		t.Fatalf("Failed to create CPU indexer: %v", err)
	}
	defer cpuIndexer.Close()

	testTexts := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning is transforming the world",
		"Go is a statically typed, compiled programming language",
	}

	for _, text := range testTexts {
		t.Run(fmt.Sprintf("Text_%d_chars", len(text)), func(t *testing.T) {
			docs := []Document{{
				ID:      1,
				Content: text,
				Scale:   1.0,
			}}

			gpuResult := make([][]float32, 0)
			gpuIndexer.progressCallback = func(p IndexProgress) {
				if p.ProcessedEmbeddings != nil && len(p.ProcessedEmbeddings) > 0 {
					gpuResult = p.ProcessedEmbeddings
				}
			}

			cpuResult := make([][]float32, 0)
			cpuIndexer.progressCallback = func(p IndexProgress) {
				if p.ProcessedEmbeddings != nil && len(p.ProcessedEmbeddings) > 0 {
					cpuResult = p.ProcessedEmbeddings
				}
			}

			if err := gpuIndexer.IndexBatch(docs); err != nil {
				t.Fatalf("GPU indexing failed: %v", err)
			}

			if err := cpuIndexer.IndexBatch(docs); err != nil {
				t.Fatalf("CPU indexing failed: %v", err)
			}

			if len(gpuResult) == 0 || len(cpuResult) == 0 {
				t.Fatal("No embeddings generated")
			}

			if len(gpuResult[0]) != len(cpuResult[0]) {
				t.Fatalf("Embedding dimensions mismatch: GPU=%d, CPU=%d",
					len(gpuResult[0]), len(cpuResult[0]))
			}

			maxDiff := float32(0.0)
			avgDiff := float32(0.0)
			for i := range gpuResult[0] {
				diff := float32(math.Abs(float64(gpuResult[0][i] - cpuResult[0][i])))
				avgDiff += diff
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			avgDiff /= float32(len(gpuResult[0]))

			tolerance := float32(1e-4)
			if maxDiff > tolerance {
				t.Errorf("Embeddings differ too much: maxDiff=%f, avgDiff=%f (tolerance=%f)",
					maxDiff, avgDiff, tolerance)
			}

			t.Logf("✓ GPU/CPU parity check passed: maxDiff=%f, avgDiff=%f", maxDiff, avgDiff)
		})
	}
}

func BenchmarkBulkGPUIndexing(b *testing.B) {
	modelPath := "model/real_model.safetensors"
	tokenizerPath := "model/tokenizer.json"

	batchSizes := []int{1, 8, 16, 32, 64, 128}
	docCounts := []int{100, 500, 1000, 5000}

	for _, batchSize := range batchSizes {
		for _, docCount := range docCounts {
			b.Run(fmt.Sprintf("Batch%d_Docs%d", batchSize, docCount), func(b *testing.B) {
				config := BulkIndexerConfig{
					BatchSize:     batchSize,
					MaxSeqLen:     512,
					ModelPath:     modelPath,
					TokenizerPath: tokenizerPath,
					UseGPU:        true,
				}

				indexer, err := NewBulkGPUIndexer(config)
				if err != nil {
					b.Fatalf("Failed to create indexer: %v", err)
				}
				defer indexer.Close()

				docs := generateTestDocuments(docCount)

				b.ResetTimer()
				b.ReportAllocs()

				for i := 0; i < b.N; i++ {
					if err := indexer.IndexBatch(docs); err != nil {
						b.Fatalf("Indexing failed: %v", err)
					}
				}

				docsPerSec := float64(docCount*b.N) / b.Elapsed().Seconds()
				b.ReportMetric(docsPerSec, "docs/sec")
				b.ReportMetric(float64(docCount)/b.Elapsed().Seconds(), "throughput")
			})
		}
	}
}

func BenchmarkGPUvsCPU(b *testing.B) {
	modelPath := "model/real_model.safetensors"
	tokenizerPath := "model/tokenizer.json"

	docCounts := []int{10, 50, 100, 500}

	for _, docCount := range docCounts {
		docs := generateTestDocuments(docCount)

		b.Run(fmt.Sprintf("GPU_%d_docs", docCount), func(b *testing.B) {
			config := BulkIndexerConfig{
				BatchSize:     32,
				MaxSeqLen:     512,
				ModelPath:     modelPath,
				TokenizerPath: tokenizerPath,
				UseGPU:        true,
			}

			indexer, err := NewBulkGPUIndexer(config)
			if err != nil {
				b.Skipf("GPU not available: %v", err)
			}
			defer indexer.Close()

			b.ResetTimer()
			start := time.Now()

			for i := 0; i < b.N; i++ {
				if err := indexer.IndexBatch(docs); err != nil {
					b.Fatalf("GPU indexing failed: %v", err)
				}
			}

			elapsed := time.Since(start)
			docsPerSec := float64(docCount*b.N) / elapsed.Seconds()
			b.ReportMetric(docsPerSec, "docs/sec")
			b.Logf("GPU: %d docs in %v = %.0f docs/sec", docCount*b.N, elapsed, docsPerSec)
		})

		b.Run(fmt.Sprintf("CPU_%d_docs", docCount), func(b *testing.B) {
			config := BulkIndexerConfig{
				BatchSize:     32,
				MaxSeqLen:     512,
				ModelPath:     modelPath,
				TokenizerPath: tokenizerPath,
				UseGPU:        false,
			}

			indexer, err := NewBulkGPUIndexer(config)
			if err != nil {
				b.Fatalf("Failed to create CPU indexer: %v", err)
			}
			defer indexer.Close()

			b.ResetTimer()
			start := time.Now()

			for i := 0; i < b.N; i++ {
				if err := indexer.IndexBatch(docs); err != nil {
					b.Fatalf("CPU indexing failed: %v", err)
				}
			}

			elapsed := time.Since(start)
			docsPerSec := float64(docCount*b.N) / elapsed.Seconds()
			b.ReportMetric(docsPerSec, "docs/sec")
			b.Logf("CPU: %d docs in %v = %.0f docs/sec", docCount*b.N, elapsed, docsPerSec)
		})
	}
}

func TestGPUMemoryUsage(t *testing.T) {
	if !isGPUAvailable() {
		t.Skip("GPU not available")
	}

	monitor, err := NewGPUMonitor()
	if err != nil {
		t.Skipf("GPU monitoring not available: %v", err)
	}
	defer monitor.Close()

	modelPath := "model/real_model.safetensors"
	tokenizerPath := "model/tokenizer.json"

	config := BulkIndexerConfig{
		BatchSize:     64,
		MaxSeqLen:     512,
		ModelPath:     modelPath,
		TokenizerPath: tokenizerPath,
		UseGPU:        true,
	}

	indexer, err := NewBulkGPUIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()

	initialStats := monitor.GetStats()
	t.Logf("Initial GPU memory: %.2f MB used", float64(initialStats.MemoryUsed)/(1024*1024))

	docCounts := []int{100, 500, 1000}

	for _, docCount := range docCounts {
		docs := generateTestDocuments(docCount)

		beforeStats := monitor.GetStats()

		if err := indexer.IndexBatch(docs); err != nil {
			t.Fatalf("Indexing failed: %v", err)
		}

		afterStats := monitor.GetStats()

		memIncrease := afterStats.MemoryUsed - beforeStats.MemoryUsed
		t.Logf("Batch size %d: Memory increase = %.2f MB, GPU utilization = %.1f%%",
			docCount,
			float64(memIncrease)/(1024*1024),
			afterStats.Utilization)
	}
}

func generateTestDocuments(count int) []Document {
	samples := []string{
		"The quick brown fox jumps over the lazy dog in the sunny meadow",
		"Machine learning algorithms are revolutionizing data analysis and prediction",
		"Golang provides excellent concurrency primitives for building scalable systems",
		"Vector databases enable semantic search across large document collections",
		"GPU acceleration dramatically improves the performance of deep learning models",
		"Natural language processing helps computers understand human language",
		"Distributed systems require careful consideration of consistency and availability",
		"Cloud computing provides on-demand access to computing resources",
	}

	docs := make([]Document, count)
	for i := 0; i < count; i++ {
		docs[i] = Document{
			ID:      i,
			Content: samples[i%len(samples)],
			Scale:   1.0,
		}
	}
	return docs
}

func isGPUAvailable() bool {
	config := BulkIndexerConfig{
		BatchSize:     1,
		MaxSeqLen:     512,
		ModelPath:     "model/real_model.safetensors",
		TokenizerPath: "model/tokenizer.json",
		UseGPU:        true,
	}

	indexer, err := NewBulkGPUIndexer(config)
	if err != nil {
		return false
	}
	indexer.Close()
	return true
}
