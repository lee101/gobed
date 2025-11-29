package gobed

import (
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/ivf"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// generateRandomVectors generates random int8 vectors for testing
func generateRandomVectors(n int) ([]simd.Vec512, []float32, []int) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	ids := make([]int, n)

	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8(rand.Intn(256) - 128)
		}
		scales[i] = 1.0 + rand.Float32()*0.1 // Small scale variation
		ids[i] = i
	}

	return vectors, scales, ids
}

// BenchmarkIVFIndexBuild benchmarks IVF index construction
func BenchmarkIVFIndexBuild(b *testing.B) {
	testCases := []struct {
		name       string
		dataSize   int
		trainSize  int
		nlist      int
		nprobe     int
		batchSize  int
	}{
		{"Small_1K", 1000, 200, 16, 4, 100},
		{"Medium_10K", 10000, 1000, 64, 8, 500},
		{"Large_100K", 100000, 5000, 256, 16, 2000},
		{"XLarge_500K", 500000, 10000, 1024, 32, 5000},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			// Generate data outside benchmark
			vectors, scales, ids := generateRandomVectors(tc.dataSize)
			trainVecs := vectors[:tc.trainSize]
			trainScales := scales[:tc.trainSize]

			b.ResetTimer()

			var memStats1, memStats2 runtime.MemStats
			
			for i := 0; i < b.N; i++ {
				runtime.GC()
				runtime.ReadMemStats(&memStats1)
				
				// Build index
				index := ivf.NewIVFIndex(tc.nlist, tc.nprobe)
				
				// Train
				start := time.Now()
				index.Train(trainVecs, trainScales)
				trainTime := time.Since(start)
				
				// Add vectors in batches
				addStart := time.Now()
				for j := 0; j < tc.dataSize; j += tc.batchSize {
					end := j + tc.batchSize
					if end > tc.dataSize {
						end = tc.dataSize
					}
					index.AddBatch(vectors[j:end], scales[j:end], ids[j:end])
				}
				addTime := time.Since(addStart)
				
				runtime.ReadMemStats(&memStats2)
				
				// Report detailed metrics
				if i == 0 {
					b.ReportMetric(trainTime.Seconds()*1000, "train_ms")
					b.ReportMetric(addTime.Seconds()*1000, "add_ms")
					b.ReportMetric(float64(memStats2.Alloc-memStats1.Alloc)/1024/1024, "alloc_MB")
					b.ReportMetric(float64(memStats2.TotalAlloc-memStats1.TotalAlloc)/1024/1024, "total_alloc_MB")
					b.ReportMetric(float64(memStats2.Mallocs-memStats1.Mallocs), "mallocs")
					b.ReportMetric(float64(tc.dataSize)/addTime.Seconds(), "vectors_per_sec")
					
					// Report index stats
					sizes := index.GetListSizes()
					minSize, maxSize := sizes[0], sizes[0]
					totalSize := 0
					for _, s := range sizes {
						if s < minSize { minSize = s }
						if s > maxSize { maxSize = s }
						totalSize += s
					}
					avgSize := float64(totalSize) / float64(len(sizes))
					b.ReportMetric(avgSize, "avg_list_size")
					b.ReportMetric(float64(maxSize-minSize)/avgSize*100, "list_imbalance_pct")
				}
			}
		})
	}
}

// BenchmarkIVFTrainingOnly focuses on just the training phase
func BenchmarkIVFTrainingOnly(b *testing.B) {
	testCases := []struct {
		name      string
		trainSize int
		nlist     int
	}{
		{"Train_1K_16list", 1000, 16},
		{"Train_5K_64list", 5000, 64},
		{"Train_10K_256list", 10000, 256},
		{"Train_20K_1024list", 20000, 1024},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			vectors, scales, _ := generateRandomVectors(tc.trainSize)

			b.ResetTimer()
			var memStats1, memStats2 runtime.MemStats

			for i := 0; i < b.N; i++ {
				runtime.GC()
				runtime.ReadMemStats(&memStats1)

				index := ivf.NewIVFIndex(tc.nlist, 8)
				index.Train(vectors, scales)

				runtime.ReadMemStats(&memStats2)

				if i == 0 {
					b.ReportMetric(float64(memStats2.Alloc-memStats1.Alloc)/1024/1024, "train_alloc_MB")
					b.ReportMetric(float64(memStats2.Mallocs-memStats1.Mallocs), "train_mallocs")
				}
			}
		})
	}
}

// BenchmarkIVFAddition focuses on just the vector addition phase
func BenchmarkIVFAddition(b *testing.B) {
	testCases := []struct {
		name      string
		dataSize  int
		trainSize int
		nlist     int
		batchSize int
	}{
		{"Add_10K_batch100", 10000, 1000, 64, 100},
		{"Add_10K_batch500", 10000, 1000, 64, 500},
		{"Add_10K_batch2K", 10000, 1000, 64, 2000},
		{"Add_50K_batch2K", 50000, 2000, 256, 2000},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			vectors, scales, ids := generateRandomVectors(tc.dataSize)
			
			// Pre-train the index
			index := ivf.NewIVFIndex(tc.nlist, 8)
			index.Train(vectors[:tc.trainSize], scales[:tc.trainSize])

			b.ResetTimer()
			var memStats1, memStats2 runtime.MemStats

			for i := 0; i < b.N; i++ {
				// Create fresh index for each iteration
				idx := ivf.NewIVFIndex(tc.nlist, 8)
				idx.Train(vectors[:tc.trainSize], scales[:tc.trainSize])
				
				runtime.GC()
				runtime.ReadMemStats(&memStats1)

				// Add vectors in batches
				for j := 0; j < tc.dataSize; j += tc.batchSize {
					end := j + tc.batchSize
					if end > tc.dataSize {
						end = tc.dataSize
					}
					idx.AddBatch(vectors[j:end], scales[j:end], ids[j:end])
				}

				runtime.ReadMemStats(&memStats2)

				if i == 0 {
					b.ReportMetric(float64(memStats2.Alloc-memStats1.Alloc)/1024/1024, "add_alloc_MB")
					b.ReportMetric(float64(memStats2.Mallocs-memStats1.Mallocs), "add_mallocs")
					b.ReportMetric(float64(tc.dataSize)/b.Elapsed().Seconds()*float64(b.N), "add_vectors_per_sec")
				}
			}
		})
	}
}

// BenchmarkBatchVsSingle compares batch vs single vector addition
func BenchmarkBatchVsSingle(b *testing.B) {
	dataSize := 10000
	trainSize := 1000
	nlist := 64
	
	vectors, scales, ids := generateRandomVectors(dataSize)
	trainVecs := vectors[:trainSize]
	trainScales := scales[:trainSize]

	b.Run("Single", func(b *testing.B) {
		var memStats1, memStats2 runtime.MemStats

		for i := 0; i < b.N; i++ {
			index := ivf.NewIVFIndex(nlist, 8)
			index.Train(trainVecs, trainScales)
			
			runtime.GC()
			runtime.ReadMemStats(&memStats1)

			// Add vectors one by one
			for j := 0; j < dataSize; j++ {
				index.Add(vectors[j], scales[j], ids[j])
			}

			runtime.ReadMemStats(&memStats2)

			if i == 0 {
				b.ReportMetric(float64(memStats2.Alloc-memStats1.Alloc)/1024/1024, "single_alloc_MB")
				b.ReportMetric(float64(memStats2.Mallocs-memStats1.Mallocs), "single_mallocs")
			}
		}
	})

	b.Run("Batch", func(b *testing.B) {
		var memStats1, memStats2 runtime.MemStats

		for i := 0; i < b.N; i++ {
			index := ivf.NewIVFIndex(nlist, 8)
			index.Train(trainVecs, trainScales)
			
			runtime.GC()
			runtime.ReadMemStats(&memStats1)

			// Add all vectors in one batch
			index.AddBatch(vectors, scales, ids)

			runtime.ReadMemStats(&memStats2)

			if i == 0 {
				b.ReportMetric(float64(memStats2.Alloc-memStats1.Alloc)/1024/1024, "batch_alloc_MB")
				b.ReportMetric(float64(memStats2.Mallocs-memStats1.Mallocs), "batch_mallocs")
			}
		}
	})
}

// BenchmarkMemoryProfile profiles memory usage patterns
func BenchmarkMemoryProfile(b *testing.B) {
	dataSize := 50000
	trainSize := 2000
	nlist := 256
	
	b.Run("MemoryProfile", func(b *testing.B) {
		vectors, scales, ids := generateRandomVectors(dataSize)
		
		for i := 0; i < b.N; i++ {
			var m1, m2, m3 runtime.MemStats
			
			runtime.GC()
			runtime.ReadMemStats(&m1)
			
			// Phase 1: Create index
			index := ivf.NewIVFIndex(nlist, 8)
			runtime.ReadMemStats(&m2)
			
			// Phase 2: Train
			index.Train(vectors[:trainSize], scales[:trainSize])
			runtime.ReadMemStats(&m3)
			
			// Phase 3: Add vectors
			index.AddBatch(vectors, scales, ids)
			var m4 runtime.MemStats
			runtime.ReadMemStats(&m4)
			
			if i == 0 {
				b.ReportMetric(float64(m2.Alloc-m1.Alloc)/1024, "create_alloc_KB")
				b.ReportMetric(float64(m3.Alloc-m2.Alloc)/1024, "train_alloc_KB") 
				b.ReportMetric(float64(m4.Alloc-m3.Alloc)/1024, "add_alloc_KB")
				b.ReportMetric(float64(m4.Alloc)/1024/1024, "total_alloc_MB")
			}
		}
	})
}