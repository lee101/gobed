package gobed

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/lee101/gobed/ann/search"
	"github.com/lee101/gobed/ann/simd"
)

// BenchmarkMaxFlatSize tests different MaxFlatSize values to find optimal performance
func BenchmarkMaxFlatSize(b *testing.B) {
	// Test different dataset sizes
	datasetSizes := []int{
		100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000,
	}
	
	// Test different MaxFlatSize values
	maxFlatSizes := []int{
		100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000,
	}
	
	for _, dataSize := range datasetSizes {
		// Generate test data once for this dataset size
		vectors, scales := generateBenchmarkData(dataSize)
		queries := generateQueryData(100) // 100 test queries
		
		for _, maxFlat := range maxFlatSizes {
			// Skip if MaxFlatSize is too small for the dataset
			if maxFlat < dataSize/100 && dataSize > 5000 {
				continue
			}
			
			benchName := fmt.Sprintf("data_%d/maxflat_%d", dataSize, maxFlat)
			
			b.Run(benchName, func(b *testing.B) {
				config := search.Config{
					MaxFlatSize: maxFlat,
					NList:       minInt(dataSize/50, 4096), // Reasonable clustering
					NProbe:      8,                       // Standard probe count
					UseParallel: true,
				}
				
				// Create and populate index
				engine := search.NewEngine(config)
				
				// Train if needed (for IVF)
				if dataSize > maxFlat {
					trainSize := minInt(dataSize/10, 10000)
					err := engine.Train(vectors[:trainSize], scales[:trainSize])
					if err != nil {
						b.Fatalf("Training failed: %v", err)
					}
				}
				
				// Add all vectors
				ids := make([]int, dataSize)
				for i := range ids {
					ids[i] = i
				}
				err := engine.AddBatch(vectors, scales, ids)
				if err != nil {
					b.Fatalf("AddBatch failed: %v", err)
				}
				
				// Benchmark search
				b.ResetTimer()
				
				for i := 0; i < b.N; i++ {
					query := queries[i%len(queries)]
					_, _ = engine.Search(&query.vec, 10)
				}
				
				// Report search latency (calculated from benchmark iterations)
				
				if dataSize <= maxFlat {
					b.ReportMetric(100.0, "recall_%") // Exact search
				} else {
					// Estimate recall based on probe count
					recall := float64(config.NProbe) / float64(config.NList) * 100
					if recall > 100 {
						recall = 100
					}
					b.ReportMetric(recall, "est_recall_%")
				}
			})
		}
	}
}

// BenchmarkMaxFlatSizeTradeoff specifically tests the speed/accuracy tradeoff
func BenchmarkMaxFlatSizeTradeoff(b *testing.B) {
	testCases := []struct {
		dataSize int
		configs  []search.Config
	}{
		{
			dataSize: 5000,
			configs: []search.Config{
				{MaxFlatSize: 10000, UseParallel: true}, // Exact search
				{MaxFlatSize: 2000, NList: 100, NProbe: 8, UseParallel: true},
				{MaxFlatSize: 1000, NList: 100, NProbe: 8, UseParallel: true},
				{MaxFlatSize: 500, NList: 100, NProbe: 8, UseParallel: true},
			},
		},
		{
			dataSize: 20000,
			configs: []search.Config{
				{MaxFlatSize: 50000, UseParallel: true}, // Exact search
				{MaxFlatSize: 5000, NList: 400, NProbe: 8, UseParallel: true},
				{MaxFlatSize: 2000, NList: 400, NProbe: 8, UseParallel: true},
				{MaxFlatSize: 1000, NList: 400, NProbe: 8, UseParallel: true},
			},
		},
		{
			dataSize: 100000,
			configs: []search.Config{
				{MaxFlatSize: 5000, NList: 2000, NProbe: 16, UseParallel: true},
				{MaxFlatSize: 2000, NList: 2000, NProbe: 16, UseParallel: true},
				{MaxFlatSize: 1000, NList: 2000, NProbe: 8, UseParallel: true},
				{MaxFlatSize: 500, NList: 2000, NProbe: 4, UseParallel: true},
			},
		},
	}
	
	for _, tc := range testCases {
		vectors, scales := generateBenchmarkData(tc.dataSize)
		queries := generateQueryData(100)
		
		// Get ground truth with exact search
		exactEngine := search.NewEngine(search.Config{
			MaxFlatSize: tc.dataSize + 1,
			UseParallel: true,
		})
		
		ids := make([]int, tc.dataSize)
		for i := range ids {
			ids[i] = i
		}
		_ = exactEngine.AddBatch(vectors, scales, ids)
		
		// Get ground truth results
		groundTruth := make([][]int, len(queries))
		for i, q := range queries {
			results, _ := exactEngine.Search(&q.vec, 10)
			resultIDs := make([]int, len(results))
			for j, r := range results {
				resultIDs[j] = r.ID
			}
			groundTruth[i] = resultIDs
		}
		
		// Test each configuration
		for _, config := range tc.configs {
			benchName := fmt.Sprintf("size_%d/maxflat_%d", tc.dataSize, config.MaxFlatSize)
			
			b.Run(benchName, func(b *testing.B) {
				engine := search.NewEngine(config)
				
				// Train if needed
				if tc.dataSize > config.MaxFlatSize {
					trainSize := minInt(tc.dataSize/10, 10000)
					_ = engine.Train(vectors[:trainSize], scales[:trainSize])
				}
				
				_ = engine.AddBatch(vectors, scales, ids)
				
				// Measure recall
				totalRecall := 0.0
				for i, q := range queries {
					results, _ := engine.Search(&q.vec, 10)
					resultIDs := make([]int, len(results))
					for j, r := range results {
						resultIDs[j] = r.ID
					}
					recall := calculateRecall(resultIDs, groundTruth[i])
					totalRecall += recall
				}
				avgRecall := totalRecall / float64(len(queries))
				
				// Benchmark search speed
				b.ResetTimer()
				
				totalTime := time.Duration(0)
				for i := 0; i < b.N; i++ {
					query := queries[i%len(queries)]
					start := time.Now()
					_, _ = engine.Search(&query.vec, 10)
					totalTime += time.Since(start)
				}
				
				avgLatency := totalTime / time.Duration(b.N)
				b.ReportMetric(float64(avgLatency.Microseconds()), "μs/search")
				b.ReportMetric(avgRecall*100, "recall_%")
				b.ReportMetric(float64(1000000/avgLatency.Microseconds()), "qps")
			})
		}
	}
}

// BenchmarkOptimalDefaults tests what should be the default MaxFlatSize for common use cases
func BenchmarkOptimalDefaults(b *testing.B) {
	scenarios := []struct {
		name        string
		dataSize    int
		queryLoad   int // queries per second expected
		priority    string // "speed" or "accuracy"
	}{
		{"small_high_qps", 1000, 10000, "speed"},
		{"small_balanced", 1000, 1000, "accuracy"},
		{"medium_high_qps", 10000, 5000, "speed"},
		{"medium_balanced", 10000, 500, "accuracy"},
		{"large_high_qps", 50000, 1000, "speed"},
		{"large_balanced", 50000, 100, "accuracy"},
	}
	
	for _, scenario := range scenarios {
		vectors, scales := generateBenchmarkData(scenario.dataSize)
		queries := generateQueryData(100)
		
		// Test different MaxFlatSize values
		var configs []search.Config
		if scenario.priority == "speed" {
			configs = []search.Config{
				{MaxFlatSize: 500, NList: scenario.dataSize / 50, NProbe: 4, UseParallel: true},
				{MaxFlatSize: 1000, NList: scenario.dataSize / 50, NProbe: 4, UseParallel: true},
				{MaxFlatSize: 2000, NList: scenario.dataSize / 50, NProbe: 4, UseParallel: true},
			}
		} else {
			configs = []search.Config{
				{MaxFlatSize: 2000, NList: scenario.dataSize / 100, NProbe: 8, UseParallel: true},
				{MaxFlatSize: 5000, NList: scenario.dataSize / 100, NProbe: 8, UseParallel: true},
				{MaxFlatSize: 10000, NList: scenario.dataSize / 100, NProbe: 8, UseParallel: true},
			}
		}
		
		for _, config := range configs {
			benchName := fmt.Sprintf("%s/maxflat_%d", scenario.name, config.MaxFlatSize)
			
			b.Run(benchName, func(b *testing.B) {
				engine := search.NewEngine(config)
				
				if scenario.dataSize > config.MaxFlatSize {
					trainSize := minInt(scenario.dataSize/10, 10000)
					_ = engine.Train(vectors[:trainSize], scales[:trainSize])
				}
				
				ids := make([]int, scenario.dataSize)
				for i := range ids {
					ids[i] = i
				}
				_ = engine.AddBatch(vectors, scales, ids)
				
				// Simulate query load
				b.ResetTimer()
				
				start := time.Now()
				for i := 0; i < b.N; i++ {
					query := queries[i%len(queries)]
					_, _ = engine.Search(&query.vec, 10)
				}
				elapsed := time.Since(start)
				
				qps := float64(b.N) / elapsed.Seconds()
				latency := elapsed.Nanoseconds() / int64(b.N) / 1000 // microseconds
				
				b.ReportMetric(float64(latency), "μs/search")
				b.ReportMetric(qps, "qps")
				
				// Check if it meets requirements
				meetsRequirement := qps >= float64(scenario.queryLoad)
				if meetsRequirement {
					b.ReportMetric(1.0, "meets_qps_requirement")
				} else {
					b.ReportMetric(0.0, "meets_qps_requirement")
				}
			})
		}
	}
}

// Helper functions
func generateBenchmarkData(n int) ([]simd.Vec512, []float32) {
	rand.Seed(42)
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	
	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8(rand.Intn(256) - 128)
		}
		scales[i] = rand.Float32() * 2.0
	}
	
	return vectors, scales
}

type queryData struct {
	vec   simd.Vec512
	scale float32
}

func generateQueryData(n int) []queryData {
	rand.Seed(123)
	queries := make([]queryData, n)
	
	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			queries[i].vec[j] = int8(rand.Intn(256) - 128)
		}
		queries[i].scale = rand.Float32() * 2.0
	}
	
	return queries
}

func calculateRecall(results, groundTruth []int) float64 {
	if len(groundTruth) == 0 {
		return 0.0
	}
	
	truthSet := make(map[int]bool)
	for _, id := range groundTruth {
		truthSet[id] = true
	}
	
	correct := 0
	for _, id := range results {
		if truthSet[id] {
			correct++
		}
	}
	
	return float64(correct) / float64(len(groundTruth))
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}