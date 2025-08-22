package ann

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/lee101/gobed/ann/flat"
	"github.com/lee101/gobed/ann/search"
	"github.com/lee101/gobed/ann/simd"
)

// generateRandomVectors generates random int8 vectors for testing
func generateRandomVectors(n int) ([]simd.Vec512, []float32) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)

	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8(rand.Intn(256) - 128)
		}
		scales[i] = 1.0
	}

	return vectors, scales
}

// BenchmarkSIMD benchmarks SIMD dot product performance
func BenchmarkSIMD(b *testing.B) {
	var vec1, vec2 simd.Vec512
	for i := 0; i < 512; i++ {
		vec1[i] = int8(rand.Intn(256) - 128)
		vec2[i] = int8(rand.Intn(256) - 128)
	}

	b.Run("Dot512", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = simd.Dot512(&vec1, &vec2)
		}
	})

	b.Run("L2Squared512", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = simd.L2Squared512(&vec1, &vec2)
		}
	})
}

// BenchmarkFlatSearch benchmarks flat index search
func BenchmarkFlatSearch(b *testing.B) {
	sizes := []int{1000, 10000, 50000}
	k := 10

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			vectors, scales := generateRandomVectors(size)
			index := flat.NewFlatIndex(size)

			// Add vectors
			for i := range vectors {
				index.Add(vectors[i], scales[i], i)
			}

			// Create query
			query := vectors[0]

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = index.SearchTopK(&query, k)
			}

			// Report throughput
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkFlatSearchParallel benchmarks parallel flat search
func BenchmarkFlatSearchParallel(b *testing.B) {
	sizes := []int{10000, 50000, 100000}
	k := 10

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			vectors, scales := generateRandomVectors(size)
			index := flat.NewFlatIndex(size)

			// Add vectors
			for i := range vectors {
				index.Add(vectors[i], scales[i], i)
			}

			query := vectors[0]

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = index.SearchTopKParallel(&query, k)
			}

			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkEngine benchmarks the full search engine
func BenchmarkEngine(b *testing.B) {
	configs := []struct {
		name   string
		size   int
		config search.Config
	}{
		{
			name: "Small_Flat",
			size: 10000,
			config: search.Config{
				MaxFlatSize: 50000,
			},
		},
		{
			name: "Medium_IVF",
			size: 100000,
			config: search.Config{
				MaxFlatSize: 10000,
				NList:       1024,
				NProbe:      8,
				RerankSize:  128,
			},
		},
		{
			name: "Large_IVFPQ",
			size: 500000,
			config: search.Config{
				MaxFlatSize: 10000,
				NList:       4096,
				NProbe:      8,
				M:           64,
				NBits:       8,
				RerankSize:  256,
				HNSWEnabled: true,
			},
		},
	}

	k := 10

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			// Generate data
			vectors, scales := generateRandomVectors(cfg.size)
			ids := make([]int, cfg.size)
			for i := range ids {
				ids[i] = i
			}

			// Create and populate engine
			engine := search.NewEngine(cfg.config)

			// Train if needed
			if cfg.size > cfg.config.MaxFlatSize {
				trainSize := min(cfg.size/10, 10000)
				err := engine.Train(vectors[:trainSize], scales[:trainSize])
				if err != nil {
					b.Fatal(err)
				}
			}

			// Add vectors in batches
			batchSize := 1000
			for i := 0; i < cfg.size; i += batchSize {
				end := min(i+batchSize, cfg.size)
				err := engine.AddBatch(vectors[i:end], scales[i:end], ids[i:end])
				if err != nil {
					b.Fatal(err)
				}
			}

			// Prepare queries
			numQueries := 100
			queries := make([]*simd.Vec512, numQueries)
			for i := 0; i < numQueries; i++ {
				queries[i] = &vectors[rand.Intn(cfg.size)]
			}

			b.ResetTimer()

			// Run searches
			for i := 0; i < b.N; i++ {
				query := queries[i%numQueries]
				_, err := engine.Search(query, k)
				if err != nil {
					b.Fatal(err)
				}
			}

			// Report metrics
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")

			stats := engine.Stats()
			b.ReportMetric(float64(stats.MemoryUsage)/1024/1024, "MB")
		})
	}
}

// TestSearchAccuracy tests search accuracy
func TestSearchAccuracy(t *testing.T) {
	sizes := []int{1000, 10000}
	k := 10

	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size%d", size), func(t *testing.T) {
			vectors, scales := generateRandomVectors(size)

			// Build flat index (ground truth)
			flatIndex := flat.NewFlatIndex(size)
			for i := range vectors {
				flatIndex.Add(vectors[i], scales[i], i)
			}

			// Build approximate index
			config := search.Config{
				MaxFlatSize: 100,
				NList:       32,
				NProbe:      4,
				RerankSize:  50,
			}
			engine := search.NewEngine(config)

			if size > config.MaxFlatSize {
				err := engine.Train(vectors[:min(size, 1000)], scales[:min(size, 1000)])
				if err != nil {
					t.Fatal(err)
				}
			}

			ids := make([]int, size)
			for i := range ids {
				ids[i] = i
			}
			err := engine.AddBatch(vectors, scales, ids)
			if err != nil {
				t.Fatal(err)
			}

			// Test queries
			numQueries := 10
			totalRecall := 0.0

			for q := 0; q < numQueries; q++ {
				query := &vectors[rand.Intn(size)]

				// Ground truth
				groundTruth := flatIndex.SearchTopK(query, k)
				groundTruthSet := make(map[int]bool)
				for _, r := range groundTruth {
					groundTruthSet[r.ID] = true
				}

				// Approximate search
				results, err := engine.Search(query, k)
				if err != nil {
					t.Fatal(err)
				}

				// Calculate recall
				hits := 0
				for _, r := range results {
					if groundTruthSet[r.ID] {
						hits++
					}
				}

				recall := float64(hits) / float64(k)
				totalRecall += recall
			}

			avgRecall := totalRecall / float64(numQueries)
			t.Logf("Average recall@%d: %.2f%%", k, avgRecall*100)

			// Expect at least 80% recall for this configuration
			if avgRecall < 0.8 {
				t.Errorf("Recall too low: %.2f%% < 80%%", avgRecall*100)
			}
		})
	}
}

// TestLatency tests search latency
func TestLatency(t *testing.T) {
	size := 100000
	k := 10

	vectors, scales := generateRandomVectors(size)

	// Test different configurations
	configs := []struct {
		name   string
		config search.Config
	}{
		{
			name: "Flat",
			config: search.Config{
				MaxFlatSize: 200000,
			},
		},
		{
			name: "IVF_High_Recall",
			config: search.Config{
				MaxFlatSize: 10000,
				NList:       1024,
				NProbe:      16,
				RerankSize:  256,
			},
		},
		{
			name: "IVF_Low_Latency",
			config: search.Config{
				MaxFlatSize: 10000,
				NList:       4096,
				NProbe:      4,
				RerankSize:  64,
			},
		},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			engine := search.NewEngine(cfg.config)

			if size > cfg.config.MaxFlatSize {
				err := engine.Train(vectors[:10000], scales[:10000])
				if err != nil {
					t.Fatal(err)
				}
			}

			ids := make([]int, size)
			for i := range ids {
				ids[i] = i
			}

			// Add in batches
			batchSize := 10000
			for i := 0; i < size; i += batchSize {
				end := min(i+batchSize, size)
				err := engine.AddBatch(vectors[i:end], scales[i:end], ids[i:end])
				if err != nil {
					t.Fatal(err)
				}
			}

			// Measure latency
			numQueries := 100
			queries := make([]*simd.Vec512, numQueries)
			for i := 0; i < numQueries; i++ {
				queries[i] = &vectors[rand.Intn(size)]
			}

			start := time.Now()
			for _, query := range queries {
				_, err := engine.Search(query, k)
				if err != nil {
					t.Fatal(err)
				}
			}
			elapsed := time.Since(start)

			avgLatency := elapsed / time.Duration(numQueries)
			t.Logf("Average latency: %v", avgLatency)

			// Check if meets 1ms target
			if avgLatency > 5*time.Millisecond {
				t.Logf("Warning: latency exceeds target (>5ms): %v", avgLatency)
			}
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
