package gobed

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"
)

// BenchmarkRTX3090SimpleIndexing performs a simple indexing benchmark optimized for RTX 3090
func BenchmarkRTX3090SimpleIndexing(b *testing.B) {
	// Load the model
	model, err := LoadModel()
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	fmt.Printf("\n🚀 RTX 3090 Simple GPU Indexing Benchmark\n")
	fmt.Printf("   GPU: NVIDIA RTX 3090 (24GB VRAM)\n")
	fmt.Printf("   CPU Cores: %d\n\n", runtime.NumCPU())

	// Test documents with varying lengths
	testTexts := [][]string{
		generateSimpleTestDocs(100),
		generateSimpleTestDocs(500),
		generateSimpleTestDocs(1000),
		generateSimpleTestDocs(5000),
		generateSimpleTestDocs(10000),
	}

	bestThroughput := 0.0
	bestDocCount := 0

	for _, texts := range testTexts {
		docCount := len(texts)
		b.Run(fmt.Sprintf("Docs_%d", docCount), func(b *testing.B) {
			b.ResetTimer()
			
			totalEmbeddings := 0
			start := time.Now()

			for i := 0; i < b.N; i++ {
				// Process documents in batches
				batchSize := 100
				for j := 0; j < len(texts); j += batchSize {
					end := j + batchSize
					if end > len(texts) {
						end = len(texts)
					}
					
					batch := texts[j:end]
					
					// Generate embeddings for batch
					for _, text := range batch {
						_, err := model.Embed(text)
						if err != nil {
							b.Fatalf("Failed to generate embedding: %v", err)
						}
						totalEmbeddings++
					}
				}
			}

			elapsed := time.Since(start)
			docsPerSec := float64(totalEmbeddings) / elapsed.Seconds()
			
			if docsPerSec > bestThroughput {
				bestThroughput = docsPerSec
				bestDocCount = docCount
			}

			b.ReportMetric(docsPerSec, "docs/sec")
			b.Logf("Processed %d documents: %.0f docs/sec", docCount, docsPerSec)
		})
	}

	fmt.Printf("\n🏆 Best throughput: %.0f docs/sec with %d documents\n", bestThroughput, bestDocCount)
}

// BenchmarkRTX3090ConcurrentEmbedding tests concurrent embedding generation
func BenchmarkRTX3090ConcurrentEmbedding(b *testing.B) {
	model, err := LoadModel()
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	fmt.Printf("\n🔥 RTX 3090 Concurrent Embedding Benchmark\n")

	workerCounts := []int{1, 2, 4, 8, 16, 32}
	docCount := 1000
	texts := generateSimpleTestDocs(docCount)

	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("Workers_%d", workers), func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				start := time.Now()
				
				// Process documents concurrently
				var wg sync.WaitGroup
				docsPerWorker := docCount / workers
				
				for w := 0; w < workers; w++ {
					wg.Add(1)
					go func(workerID int) {
						defer wg.Done()
						
						startIdx := workerID * docsPerWorker
						endIdx := startIdx + docsPerWorker
						if workerID == workers-1 {
							endIdx = docCount
						}
						
						for j := startIdx; j < endIdx; j++ {
							_, err := model.Embed(texts[j])
							if err != nil {
								b.Errorf("Worker %d failed: %v", workerID, err)
							}
						}
					}(w)
				}
				
				wg.Wait()
				elapsed := time.Since(start)
				
				docsPerSec := float64(docCount) / elapsed.Seconds()
				b.ReportMetric(docsPerSec, "docs/sec")
				b.ReportMetric(float64(workers), "workers")
			}
		})
	}
}

// BenchmarkRTX3090BatchProcessing tests different batch sizes
func BenchmarkRTX3090BatchProcessing(b *testing.B) {
	model, err := LoadModel()
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	fmt.Printf("\n📊 RTX 3090 Batch Processing Benchmark\n")

	// Test different batch sizes optimized for 24GB VRAM
	batchSizes := []int{1, 10, 50, 100, 250, 500, 1000, 2000}
	totalDocs := 10000
	texts := generateSimpleTestDocs(totalDocs)

	bestBatchSize := 0
	bestThroughput := 0.0

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("Batch_%d", batchSize), func(b *testing.B) {
			b.ResetTimer()
			
			totalProcessed := 0
			start := time.Now()
			
			for i := 0; i < b.N; i++ {
				// Process in batches
				for j := 0; j < len(texts); j += batchSize {
					end := j + batchSize
					if end > len(texts) {
						end = len(texts)
					}
					
					batch := texts[j:end]
					
					// Simulate batch processing
					embeddings := make([][]float32, len(batch))
					for idx, text := range batch {
						emb, err := model.Embed(text)
						if err != nil {
							b.Fatalf("Failed to embed: %v", err)
						}
						embeddings[idx] = emb
						totalProcessed++
					}
				}
			}
			
			elapsed := time.Since(start)
			docsPerSec := float64(totalProcessed) / elapsed.Seconds()
			
			if docsPerSec > bestThroughput {
				bestThroughput = docsPerSec
				bestBatchSize = batchSize
			}
			
			b.ReportMetric(docsPerSec, "docs/sec")
			b.ReportMetric(float64(batchSize), "batch_size")
			b.Logf("Batch size %d: %.0f docs/sec", batchSize, docsPerSec)
		})
	}

	fmt.Printf("\n🏆 Optimal batch size: %d with %.0f docs/sec throughput\n", bestBatchSize, bestThroughput)
}

// Helper function to generate simple test documents
func generateSimpleTestDocs(count int) []string {
	samples := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning is transforming the world of technology",
		"Go provides excellent support for concurrent programming",
		"Vector databases enable semantic search at scale",
		"GPU acceleration dramatically improves deep learning performance",
		"Natural language processing helps computers understand human language",
		"Distributed systems require careful design and implementation",
		"Cloud computing provides scalable infrastructure on demand",
		"Microservices architecture enables independent service deployment",
		"Container orchestration simplifies application management",
		"Real-time data processing is essential for modern applications",
		"Artificial intelligence is revolutionizing various industries",
		"Blockchain technology provides decentralized trust mechanisms",
		"Quantum computing promises exponential speedups for certain problems",
		"Edge computing brings computation closer to data sources",
	}

	docs := make([]string, count)
	for i := 0; i < count; i++ {
		// Add variation to prevent caching
		docs[i] = fmt.Sprintf("%s [ID: %d, Time: %d]", 
			samples[i%len(samples)], i, time.Now().UnixNano())
	}
	return docs
}