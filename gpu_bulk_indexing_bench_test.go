// +build gpu

package gobed

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/lee101/gobed/ann/simd"
)

// BenchmarkGPUBulkIndexing benchmarks the new GPU bulk indexing system
func BenchmarkGPUBulkIndexing(b *testing.B) {
	if !IsCUDAAvailable() {
		b.Skip("CUDA not available")
	}

	testCases := []struct {
		name        string
		numVectors  int
		batchSizes  []int
		vectorDim   int
		nlist       int
		description string
	}{
		{
			name:        "Small_Dataset",
			numVectors:  10000,
			batchSizes:  []int{100, 500, 1000},
			vectorDim:   512,
			nlist:       64,
			description: "Small dataset for initial testing",
		},
		{
			name:        "Medium_Dataset", 
			numVectors:  100000,
			batchSizes:  []int{1000, 2000, 5000},
			vectorDim:   512,
			nlist:       256,
			description: "Medium dataset for real-world simulation",
		},
		{
			name:        "Large_Dataset",
			numVectors:  500000,
			batchSizes:  []int{2000, 5000, 10000},
			vectorDim:   512,
			nlist:       1024,
			description: "Large dataset for scalability testing",
		},
		{
			name:        "XLarge_Dataset",
			numVectors:  1000000,
			batchSizes:  []int{5000, 10000, 20000},
			vectorDim:   512,
			nlist:       2048,
			description: "Extra large dataset for maximum performance",
		},
	}

	fmt.Printf("\n GPU Bulk Indexing Benchmark Suite\n")
	fmt.Printf("GPU: %s, VRAM: %d MB\n", getGPUName(), GetTotalVRAM())
	fmt.Printf("CPU: %d cores\n\n", runtime.NumCPU())

	bestResults := make(map[string]BenchmarkResult)

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			fmt.Printf(" %s: %s\n", tc.name, tc.description)
			fmt.Printf("   Vectors: %d, Dimension: %d, Clusters: %d\n", 
				tc.numVectors, tc.vectorDim, tc.nlist)

			// Generate test data once
			tokenSequences := generateTokenSequences(tc.numVectors, tc.vectorDim)
			embeddings, scales := generateQuantizedEmbeddings(30522, 1024)
			trainingData := generateTrainingVectors(5000, tc.vectorDim)

			for _, batchSize := range tc.batchSizes {
				subTestName := fmt.Sprintf("Batch_%d", batchSize)
				b.Run(subTestName, func(b *testing.B) {
					// Create indexer with test configuration
					config := BulkIndexConfig{
						NList:           tc.nlist,
						NProbe:          tc.nlist / 8, // 1/8 of clusters for search
						VectorDim:       tc.vectorDim,
						VocabSize:       30522,
						EmbedDim:        1024,
						MaxMemoryMB:     int(GetAvailableVRAM() * 0.8),
						ProgressiveMode: true,
						OnlineUpdates:   true,
					}

					indexer, err := NewGPUIVFBulkIndexer(config)
					if err != nil {
						b.Fatalf("Failed to create indexer: %v", err)
					}
					defer indexer.Close()

					// Load embeddings
					if err := indexer.LoadEmbeddings(embeddings, scales); err != nil {
						b.Fatalf("Failed to load embeddings: %v", err)
					}

					// Train k-means
					trainingVecs, trainingScales := trainingData.vectors, trainingData.scales
					if err := indexer.TrainKMeans(trainingVecs, trainingScales, len(trainingScales)); err != nil {
						b.Fatalf("Failed to train k-means: %v", err)
					}

					b.ResetTimer()
					b.ReportAllocs()

					var totalTime time.Duration
					var totalVectors int

					for i := 0; i < b.N; i++ {
						start := time.Now()
						
						indexed, err := indexer.BulkIndexTokenSequences(tokenSequences, nil)
						if err != nil {
							b.Fatalf("Bulk indexing failed: %v", err)
						}
						
						elapsed := time.Since(start)
						totalTime += elapsed
						totalVectors += indexed
					}

					// Calculate metrics
					avgTime := totalTime / time.Duration(b.N)
					vectorsPerSec := float64(totalVectors) / totalTime.Seconds()
					
					// Get memory usage
					memUsage := indexer.GetMemoryUsage()
					stats := indexer.GetStats()

					// Report metrics
					b.ReportMetric(vectorsPerSec, "vectors/sec")
					b.ReportMetric(avgTime.Seconds()*1000, "avg_latency_ms")
					b.ReportMetric(float64(memUsage)/1024/1024, "memory_mb")
					b.ReportMetric(float64(batchSize), "batch_size")
					b.ReportMetric(stats.PeakThroughput, "peak_vectors/sec")

					// Track best result
					result := BenchmarkResult{
						Dataset:       tc.name,
						BatchSize:     batchSize,
						VectorsPerSec: vectorsPerSec,
						LatencyMs:     avgTime.Milliseconds(),
						MemoryMB:      memUsage / 1024 / 1024,
						Efficiency:    calculateEfficiency(vectorsPerSec, tc.numVectors),
					}

					key := fmt.Sprintf("%s_Batch_%d", tc.name, batchSize)
					bestResults[key] = result

					fmt.Printf("   ✓ Batch %d: %.0f vec/sec, %.1fms latency, %d MB\n",
						batchSize, vectorsPerSec, avgTime.Seconds()*1000, memUsage/1024/1024)
				})
			}
			fmt.Printf("\n")
		})
	}

	// Print summary of best results
	printBenchmarkSummary(bestResults)
}

// BenchmarkGPUVsCPUIVF compares GPU bulk indexing vs CPU IVF
func BenchmarkGPUVsCPUIVF(b *testing.B) {
	if !IsCUDAAvailable() {
		b.Skip("CUDA not available")
	}

	numVectors := 50000
	vectorDim := 512
	nlist := 256

	fmt.Printf("\n GPU vs CPU IVF Comparison\n")
	fmt.Printf("Dataset: %d vectors, %d dimensions, %d clusters\n\n", numVectors, vectorDim, nlist)

	tokenSequences := generateTokenSequences(numVectors, vectorDim)
	
	// Generate test vectors for CPU IVF
	cpuVectors, cpuScales, cpuIDs := generateCPUTestVectors(numVectors, vectorDim)

	b.Run("GPU_Bulk", func(b *testing.B) {
		config := DefaultBulkIndexConfig()
		config.NList = nlist
		config.VectorDim = vectorDim

		indexer, err := NewGPUIVFBulkIndexer(config)
		if err != nil {
			b.Fatalf("Failed to create GPU indexer: %v", err)
		}
		defer indexer.Close()

		// Setup
		embeddings, scales := generateQuantizedEmbeddings(30522, 1024)
		trainingData := generateTrainingVectors(2000, vectorDim)
		
		indexer.LoadEmbeddings(embeddings, scales)
		indexer.TrainKMeans(trainingData.vectors, trainingData.scales, len(trainingData.scales))

		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			start := time.Now()
			indexed, err := indexer.BulkIndexTokenSequences(tokenSequences, nil)
			elapsed := time.Since(start)
			
			if err != nil {
				b.Fatalf("GPU indexing failed: %v", err)
			}
			
			vectorsPerSec := float64(indexed) / elapsed.Seconds()
			b.ReportMetric(vectorsPerSec, "gpu_vectors/sec")
		}
	})

	b.Run("CPU_IVF_Original", func(b *testing.B) {
		// Test original CPU IVF implementation
		for i := 0; i < b.N; i++ {
			index := NewIVFIndex(nlist, nlist/8)
			
			start := time.Now()
			// Train
			index.Train(cpuVectors[:2000], cpuScales[:2000])
			
			// Add vectors
			index.AddBatch(cpuVectors, cpuScales, cpuIDs)
			elapsed := time.Since(start)
			
			vectorsPerSec := float64(numVectors) / elapsed.Seconds()
			b.ReportMetric(vectorsPerSec, "cpu_orig_vectors/sec")
		}
	})

	b.Run("CPU_IVF_Optimized", func(b *testing.B) {
		// Test optimized CPU IVF implementation
		for i := 0; i < b.N; i++ {
			index := NewIVFIndexOptimized(nlist, nlist/8)
			
			start := time.Now()
			// Train
			index.Train(cpuVectors[:2000], cpuScales[:2000])
			
			// Add vectors  
			index.AddBatchOptimized(cpuVectors, cpuScales, cpuIDs)
			elapsed := time.Since(start)
			
			vectorsPerSec := float64(numVectors) / elapsed.Seconds()
			b.ReportMetric(vectorsPerSec, "cpu_opt_vectors/sec")
		}
	})
}

// BenchmarkProgressiveIndexing tests progressive/streaming indexing
func BenchmarkProgressiveIndexing(b *testing.B) {
	if !IsCUDAAvailable() {
		b.Skip("CUDA not available")
	}

	fmt.Printf("\n Progressive Indexing Benchmark\n")

	numVectors := 100000
	streamChunkSize := 1000

	b.Run("Streaming", func(b *testing.B) {
		config := DefaultBulkIndexConfig()
		config.ProgressiveMode = true

		indexer, err := NewGPUIVFBulkIndexer(config)
		if err != nil {
			b.Fatalf("Failed to create indexer: %v", err)
		}
		defer indexer.Close()

		// Setup
		embeddings, scales := generateQuantizedEmbeddings(30522, 1024)
		trainingData := generateTrainingVectors(2000, 512)
		
		indexer.LoadEmbeddings(embeddings, scales)
		indexer.TrainKMeans(trainingData.vectors, trainingData.scales, len(trainingData.scales))

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			// Create streaming channels
			tokenChan := make(chan []int, streamChunkSize)
			progressChan := make(chan IndexProgress, 100)

			// Start streaming indexer
			go func() {
				err := indexer.StreamingIndex(tokenChan, progressChan)
				if err != nil {
					b.Errorf("Streaming indexing failed: %v", err)
				}
			}()

			start := time.Now()
			progressCount := 0
			
			// Generate and send token sequences
			go func() {
				defer close(tokenChan)
				for j := 0; j < numVectors; j++ {
					tokens := generateSingleTokenSequence(512)
					tokenChan <- tokens
				}
			}()

			// Monitor progress
			var lastProgress IndexProgress
			for progress := range progressChan {
				progressCount++
				lastProgress = progress
			}

			elapsed := time.Since(start)
			vectorsPerSec := float64(numVectors) / elapsed.Seconds()
			
			b.ReportMetric(vectorsPerSec, "stream_vectors/sec")
			b.ReportMetric(float64(progressCount), "progress_updates")
			b.ReportMetric(lastProgress.DocsPerSec, "final_throughput")
		}
	})
}

// BenchmarkMemoryScaling tests performance across different memory configurations
func BenchmarkMemoryScaling(b *testing.B) {
	if !IsCUDAAvailable() {
		b.Skip("CUDA not available")
	}

	totalVRAM := GetTotalVRAM()
	memoryConfigs := []struct {
		name    string
		sizeMB  int
		percent int
	}{
		{"25_Percent", int(totalVRAM * 0.25), 25},
		{"50_Percent", int(totalVRAM * 0.50), 50},
		{"75_Percent", int(totalVRAM * 0.75), 75},
		{"90_Percent", int(totalVRAM * 0.90), 90},
	}

	numVectors := 75000
	tokenSequences := generateTokenSequences(numVectors, 512)

	fmt.Printf("\n🧠 Memory Scaling Benchmark\n")
	fmt.Printf("Total VRAM: %d MB\n\n", totalVRAM)

	for _, memConfig := range memoryConfigs {
		b.Run(memConfig.name, func(b *testing.B) {
			config := DefaultBulkIndexConfig()
			config.MaxMemoryMB = memConfig.sizeMB

			indexer, err := NewGPUIVFBulkIndexer(config)
			if err != nil {
				b.Fatalf("Failed to create indexer: %v", err)
			}
			defer indexer.Close()

			// Setup
			embeddings, scales := generateQuantizedEmbeddings(30522, 1024)
			trainingData := generateTrainingVectors(2000, 512)
			
			indexer.LoadEmbeddings(embeddings, scales)
			indexer.TrainKMeans(trainingData.vectors, trainingData.scales, len(trainingData.scales))

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				start := time.Now()
				indexed, err := indexer.BulkIndexTokenSequences(tokenSequences, nil)
				elapsed := time.Since(start)

				if err != nil {
					b.Fatalf("Indexing failed: %v", err)
				}

				vectorsPerSec := float64(indexed) / elapsed.Seconds()
				memUsage := indexer.GetMemoryUsage()
				
				b.ReportMetric(vectorsPerSec, "vectors/sec")
				b.ReportMetric(float64(memConfig.sizeMB), "allocated_mb")
				b.ReportMetric(float64(memUsage)/1024/1024, "used_mb")
				b.ReportMetric(float64(memUsage)/float64(memConfig.sizeMB*1024*1024)*100, "utilization_%")

				fmt.Printf("   %s (%d MB): %.0f vec/sec, %.1f%% utilization\n",
					memConfig.name, memConfig.sizeMB, vectorsPerSec, 
					float64(memUsage)/float64(memConfig.sizeMB*1024*1024)*100)
			}
		})
	}
}

// Helper functions for test data generation

func generateTokenSequences(numSequences, maxSeqLen int) [][]int {
	sequences := make([][]int, numSequences)
	for i := 0; i < numSequences; i++ {
		seqLen := rand.Intn(maxSeqLen-10) + 10 // 10 to maxSeqLen tokens
		sequence := make([]int, seqLen)
		for j := 0; j < seqLen; j++ {
			sequence[j] = rand.Intn(30522) // BERT vocab size
		}
		sequences[i] = sequence
	}
	return sequences
}

func generateSingleTokenSequence(maxLen int) []int {
	seqLen := rand.Intn(maxLen-10) + 10
	sequence := make([]int, seqLen)
	for j := 0; j < seqLen; j++ {
		sequence[j] = rand.Intn(30522)
	}
	return sequence
}

func generateQuantizedEmbeddings(vocabSize, embedDim int) ([]int8, []float32) {
	embeddings := make([]int8, vocabSize*embedDim)
	scales := make([]float32, vocabSize)
	
	for i := 0; i < vocabSize; i++ {
		scales[i] = 0.5 + rand.Float32()*1.0 // Scale between 0.5 and 1.5
		for j := 0; j < embedDim; j++ {
			embeddings[i*embedDim+j] = int8(rand.Intn(256) - 128)
		}
	}
	
	return embeddings, scales
}

func generateTrainingVectors(numVectors, vectorDim int) TrainingData {
	vectors := make([]int8, numVectors*vectorDim)
	scales := make([]float32, numVectors)
	
	for i := 0; i < numVectors; i++ {
		scales[i] = 0.5 + rand.Float32()*1.0
		for j := 0; j < vectorDim; j++ {
			vectors[i*vectorDim+j] = int8(rand.Intn(256) - 128)
		}
	}
	
	return TrainingData{vectors: vectors, scales: scales}
}

func generateCPUTestVectors(numVectors, vectorDim int) ([]simd.Vec512, []float32, []int) {
	vectors := make([]simd.Vec512, numVectors)
	scales := make([]float32, numVectors)
	ids := make([]int, numVectors)
	
	for i := 0; i < numVectors; i++ {
		for j := 0; j < vectorDim && j < 512; j++ {
			vectors[i][j] = int8(rand.Intn(256) - 128)
		}
		scales[i] = 0.5 + rand.Float32()*1.0
		ids[i] = i
	}
	
	return vectors, scales, ids
}

func calculateEfficiency(vectorsPerSec float64, numVectors int) float64 {
	// Simple efficiency calculation - could be more sophisticated
	return vectorsPerSec / float64(numVectors) * 100
}

func getGPUName() string {
	// Placeholder - would get actual GPU name from CUDA
	return "RTX 3090"
}

func printBenchmarkSummary(results map[string]BenchmarkResult) {
	fmt.Printf("\n Benchmark Summary\n")
	fmt.Printf("=" + fmt.Sprintf("%80s", "") + "\n")
	fmt.Printf("%-20s %-10s %-15s %-12s %-12s %-10s\n", 
		"Dataset", "Batch", "Vectors/Sec", "Latency(ms)", "Memory(MB)", "Efficiency")
	fmt.Printf("-" + fmt.Sprintf("%80s", "") + "\n")
	
	for key, result := range results {
		fmt.Printf("%-20s %-10d %-15.0f %-12d %-12d %-10.1f\n",
			result.Dataset, result.BatchSize, result.VectorsPerSec,
			result.LatencyMs, result.MemoryMB, result.Efficiency)
	}
	fmt.Printf("\n")
}

// Supporting types

type BenchmarkResult struct {
	Dataset       string
	BatchSize     int
	VectorsPerSec float64
	LatencyMs     int64
	MemoryMB      uint64
	Efficiency    float64
}

type TrainingData struct {
	vectors []int8
	scales  []float32
}

// Placeholder function declarations that would need to be implemented
func NewIVFIndex(nlist, nprobe int) interface{} {
	// Would return actual IVF index
	return nil
}

func NewIVFIndexOptimized(nlist, nprobe int) interface{} {
	// Would return optimized IVF index
	return nil
}