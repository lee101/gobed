package gobed

import (
	"fmt"
	"runtime"
	"testing"
	"time"
)

// BenchmarkRTX3090GPUIndexing benchmarks GPU indexing with optimized batch sizes for RTX 3090
func BenchmarkRTX3090GPUIndexing(b *testing.B) {
	// Load the model
	model, err := LoadModel()
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	// Create search engine
	engine := NewSearchEngine(model)

	// Test various batch sizes optimized for RTX 3090 (24GB VRAM)
	batchSizes := []int{
		100,   // Baseline
		250,   // 2.5x
		500,   // 5x 
		1000,  // 10x
		2000,  // 20x
		5000,  // 50x - aggressive for 24GB
		10000, // 100x - very aggressive
	}

	// Document counts for testing
	docCounts := []int{1000, 5000, 10000, 25000, 50000}

	fmt.Printf("\n🚀 RTX 3090 GPU Indexing Benchmark\n")
	fmt.Printf("   GPU: NVIDIA RTX 3090 (24GB VRAM)\n")
	fmt.Printf("   CPU Cores: %d\n", runtime.NumCPU())
	fmt.Printf("   Testing batch sizes: %v\n", batchSizes)
	fmt.Printf("   Document counts: %v\n\n", docCounts)

	bestConfig := struct {
		batchSize  int
		docCount   int
		docsPerSec float64
		workers    int
	}{}

	for _, batchSize := range batchSizes {
		for _, docCount := range docCounts {
			// Test with different worker counts
			workerCounts := []int{runtime.NumCPU() / 2, runtime.NumCPU(), runtime.NumCPU() * 2}
			
			for _, workers := range workerCounts {
				testName := fmt.Sprintf("Batch%d_Docs%d_Workers%d", batchSize, docCount, workers)
				
				b.Run(testName, func(b *testing.B) {
					// Configure parallel indexer for GPU optimization
					config := ParallelIndexConfig{
						NumWorkers:    workers,
						BatchSize:     batchSize,
						EnableCache:   true,
						QueueSize:     batchSize * 2, // Double buffer
						MaxConcurrent: workers * 2,    // Allow 2x concurrency
					}

					indexer := NewParallelIndexer(engine, config)
					
					// Generate test documents
					docs := generateRTX3090TestDocuments(docCount)

					// Warmup run
					if docCount > 100 {
						warmupDocs := docs[:100]
						indexer.IndexDocumentsParallel(warmupDocs)
					}

					b.ResetTimer()
					b.ReportAllocs()

					totalDocs := 0
					start := time.Now()

					for i := 0; i < b.N; i++ {
						_, err := indexer.IndexDocumentsParallel(docs)
						if err != nil {
							b.Fatalf("Indexing failed: %v", err)
						}
						totalDocs += docCount
					}

					elapsed := time.Since(start)
					docsPerSec := float64(totalDocs) / elapsed.Seconds()

					b.ReportMetric(docsPerSec, "docs/sec")
					b.ReportMetric(float64(batchSize), "batch_size")
					b.ReportMetric(float64(workers), "workers")

					// Track best configuration
					if docsPerSec > bestConfig.docsPerSec {
						bestConfig.batchSize = batchSize
						bestConfig.docCount = docCount
						bestConfig.docsPerSec = docsPerSec
						bestConfig.workers = workers
					}

					// Get indexer stats
					stats := indexer.Stats()
					b.Logf("✓ Batch=%d, Workers=%d: %.0f docs/sec, Avg latency: %.2fms",
						batchSize, workers, docsPerSec, float64(stats.AvgLatency)/1000000)
				})
			}
		}
	}

	if bestConfig.docsPerSec > 0 {
		fmt.Printf("\n🏆 Best Configuration for RTX 3090:\n")
		fmt.Printf("   Batch Size: %d\n", bestConfig.batchSize)
		fmt.Printf("   Worker Threads: %d\n", bestConfig.workers)
		fmt.Printf("   Document Count: %d\n", bestConfig.docCount)
		fmt.Printf("   Throughput: %.0f docs/sec\n\n", bestConfig.docsPerSec)
	}
}

// BenchmarkRTX3090ParallelScaling tests parallel scaling efficiency
func BenchmarkRTX3090ParallelScaling(b *testing.B) {
	model, err := LoadModel()
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	engine := NewSearchEngine(model)
	docCount := 10000

	// Test scaling from 1 to 2x CPU cores
	maxWorkers := runtime.NumCPU() * 2
	workerCounts := []int{1, 2, 4, 8, 16, 24, 32}
	
	// Filter to reasonable values
	filteredWorkers := []int{}
	for _, w := range workerCounts {
		if w <= maxWorkers {
			filteredWorkers = append(filteredWorkers, w)
		}
	}
	// Add max workers if not already included
	if filteredWorkers[len(filteredWorkers)-1] != maxWorkers {
		filteredWorkers = append(filteredWorkers, maxWorkers)
	}

	fmt.Printf("\n📊 RTX 3090 Parallel Scaling Test\n")
	fmt.Printf("   Testing worker counts: %v\n", filteredWorkers)
	fmt.Printf("   Documents per test: %d\n\n", docCount)

	baselineTime := time.Duration(0)
	results := make(map[int]float64)

	for _, workers := range filteredWorkers {
		b.Run(fmt.Sprintf("Workers%d", workers), func(b *testing.B) {
			config := ParallelIndexConfig{
				NumWorkers:    workers,
				BatchSize:     1000, // Optimized for GPU
				EnableCache:   true,
				QueueSize:     2000,
				MaxConcurrent: workers * 2,
			}

			indexer := NewParallelIndexer(engine, config)
			docs := generateRTX3090TestDocuments(docCount)

			b.ResetTimer()
			start := time.Now()

			for i := 0; i < b.N; i++ {
				_, err := indexer.IndexDocumentsParallel(docs)
				if err != nil {
					b.Fatalf("Indexing failed: %v", err)
				}
			}

			elapsed := time.Since(start)
			docsPerSec := float64(docCount*b.N) / elapsed.Seconds()
			results[workers] = docsPerSec

			if workers == 1 {
				baselineTime = elapsed
			}

			var speedup float64
			if baselineTime > 0 {
				speedup = float64(baselineTime) / float64(elapsed)
			}

			b.ReportMetric(docsPerSec, "docs/sec")
			b.ReportMetric(speedup, "speedup")
			b.ReportMetric(speedup/float64(workers)*100, "efficiency_%")

			b.Logf("Workers=%d: %.0f docs/sec, Speedup=%.2fx, Efficiency=%.1f%%",
				workers, docsPerSec, speedup, speedup/float64(workers)*100)
		})
	}

	// Print scaling summary
	fmt.Printf("\n📈 Scaling Summary:\n")
	if baseline, ok := results[1]; ok {
		for workers, docsPerSec := range results {
			if workers > 1 {
				speedup := docsPerSec / baseline
				efficiency := (speedup / float64(workers)) * 100
				fmt.Printf("   %d workers: %.2fx speedup, %.1f%% efficiency\n",
					workers, speedup, efficiency)
			}
		}
	}
}

// BenchmarkRTX3090ProgressMonitoring tests indexing with progress monitoring
func BenchmarkRTX3090ProgressMonitoring(b *testing.B) {
	model, err := LoadModel()
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	engine := NewSearchEngine(model)
	
	// Optimized configuration for RTX 3090
	config := ParallelIndexConfig{
		NumWorkers:    runtime.NumCPU(),
		BatchSize:     2000, // Large batch for GPU
		EnableCache:   true,
		QueueSize:     4000,
		MaxConcurrent: runtime.NumCPU() * 2,
	}

	docCounts := []int{10000, 25000, 50000}

	fmt.Printf("\n📊 RTX 3090 Progress Monitoring Benchmark\n")

	for _, docCount := range docCounts {
		b.Run(fmt.Sprintf("Progress_Docs%d", docCount), func(b *testing.B) {
			indexer := NewParallelIndexer(engine, config)
			docs := generateRTX3090TestDocuments(docCount)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				progressChan, err := indexer.IndexWithProgress(docs)
				if err != nil {
					b.Fatalf("Failed to start indexing: %v", err)
				}

				var lastProgress IndexProgress
				progressCount := 0

				for progress := range progressChan {
					progressCount++
					lastProgress = progress
					
					// Log progress at key milestones
					if progress.Percentage >= 25 && progress.Percentage < 26 ||
					   progress.Percentage >= 50 && progress.Percentage < 51 ||
					   progress.Percentage >= 75 && progress.Percentage < 76 {
						b.Logf("Progress: %.1f%% - %.0f docs/sec - ETA: %v",
							progress.Percentage, progress.DocsPerSec, progress.TimeLeft)
					}
				}

				if lastProgress.Current != docCount {
					b.Errorf("Incomplete indexing: %d/%d", lastProgress.Current, docCount)
				}

				b.ReportMetric(lastProgress.DocsPerSec, "final_docs/sec")
				b.ReportMetric(float64(progressCount), "progress_updates")
			}
		})
	}
}

// Helper function to generate varied test documents
func generateRTX3090TestDocuments(count int) []string {
	samples := []string{
		// Short texts (1-2 sentences)
		"The quick brown fox jumps over the lazy dog in the sunny meadow.",
		"Machine learning algorithms transform how we process and understand data.",
		"Go provides excellent concurrency primitives for building scalable systems.",
		"Vector databases enable semantic search across massive document collections.",
		
		// Medium texts (3-4 sentences)
		"GPU acceleration dramatically improves deep learning performance. Modern GPUs like the RTX 3090 offer massive parallel processing capabilities. This enables training and inference at unprecedented speeds. The 24GB of VRAM allows processing large batches efficiently.",
		"Natural language processing has evolved rapidly in recent years. Transformer models now achieve human-level performance on many tasks. These models can understand context and generate coherent text. Applications range from translation to content generation.",
		
		// Longer texts (5+ sentences)
		"Distributed systems require careful consideration of consistency, availability, and partition tolerance. The CAP theorem states that it is impossible for a distributed data store to simultaneously provide more than two out of these three guarantees. In practice, system designers must make trade-offs based on their specific requirements. Modern distributed databases often provide tunable consistency levels. This allows applications to choose the right balance for their use case. Some operations may require strong consistency while others can tolerate eventual consistency.",
		"Cloud computing has revolutionized how businesses deploy and scale applications. Infrastructure as a Service provides virtual machines and storage on demand. Platform as a Service abstracts away infrastructure management entirely. Software as a Service delivers complete applications over the internet. Serverless computing takes this further by eliminating server management. Functions execute in response to events and scale automatically. This model is ideal for variable workloads and reduces operational overhead.",
	}

	docs := make([]string, count)
	for i := 0; i < count; i++ {
		// Mix different text lengths for realistic workload
		sampleIdx := i % len(samples)
		// Add some variation to prevent caching effects
		docs[i] = fmt.Sprintf("%s [Document #%d]", samples[sampleIdx], i)
	}
	return docs
}