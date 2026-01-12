//go:build cagra

package gobed

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

type ScaleBenchResult struct {
	Scale           string
	NumVectors      int
	IndexBuildMs    float64
	SearchLatencyUs float64
	BatchSearchMs   float64
	QPS             float64
	MemoryMB        int64
	VectorsPerSec   float64
}

func generateLargeScaleVectors(n int) ([]simd.Vec512, []float32) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8((i*j + j) % 256 - 128)
		}
		scales[i] = 0.5 + float32(i%100)/100.0
	}
	return vectors, scales
}

func generateQueryVectors(n int) ([]simd.Vec512, []float32) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8((i*17 + j*31) % 256 - 128)
		}
		scales[i] = 1.0
	}
	return vectors, scales
}

func TestLargeScaleCAGRABenchmark(t *testing.T) {
	if !isCAGRAAvailable() {
		t.Skip("CAGRA not available")
	}

	scales := []struct {
		name   string
		count  int
		config CAGRAConfig
	}{
		{"100K", 100000, FastCAGRAConfig()},
		{"500K", 500000, DefaultCAGRAConfig()},
		{"1M", 1000000, DefaultCAGRAConfig()},
		{"5M", 5000000, QualityCAGRAConfig()},
	}

	results := make([]ScaleBenchResult, 0)

	for _, s := range scales {
		t.Run(s.name, func(t *testing.T) {
			runtime.GC()
			var memBefore runtime.MemStats
			runtime.ReadMemStats(&memBefore)

			cfg := s.config
			cfg.MaxVectors = s.count + 1000
			cfg.CachePath = ""

			idx, err := NewCAGRAIndex(cfg)
			if err != nil {
				t.Fatalf("failed to create index: %v", err)
			}
			defer idx.Close()

			fmt.Printf("\n=== %s vectors ===\n", s.name)

			vectors, vecScales := generateLargeScaleVectors(s.count)

			buildStart := time.Now()
			if err := idx.BuildIndex(vectors, vecScales); err != nil {
				t.Fatalf("failed to build index: %v", err)
			}
			buildTime := time.Since(buildStart)
			vectorsPerSec := float64(s.count) / buildTime.Seconds()

			fmt.Printf("Index build: %v (%.0f vectors/sec)\n", buildTime, vectorsPerSec)

			var memAfter runtime.MemStats
			runtime.ReadMemStats(&memAfter)
			memUsedMB := int64(memAfter.Alloc-memBefore.Alloc) / 1024 / 1024

			queries, qScales := generateQueryVectors(100)

			var totalSearchTime time.Duration
			numSearches := 100
			for i := 0; i < numSearches; i++ {
				start := time.Now()
				_, err := idx.Search(queries[i%len(queries)], qScales[i%len(qScales)], 10)
				if err != nil {
					t.Fatalf("search failed: %v", err)
				}
				totalSearchTime += time.Since(start)
			}
			avgSearchUs := float64(totalSearchTime.Microseconds()) / float64(numSearches)
			qps := float64(numSearches) / totalSearchTime.Seconds()

			fmt.Printf("Search latency: %.2fus avg (%.0f QPS)\n", avgSearchUs, qps)

			batchStart := time.Now()
			_, err = idx.SearchBatch(queries, qScales, 10)
			if err != nil {
				t.Fatalf("batch search failed: %v", err)
			}
			batchTime := time.Since(batchStart)
			batchQPS := float64(len(queries)) / batchTime.Seconds()

			fmt.Printf("Batch search (100): %v (%.0f QPS)\n", batchTime, batchQPS)
			fmt.Printf("Memory: ~%d MB\n", memUsedMB)

			results = append(results, ScaleBenchResult{
				Scale:           s.name,
				NumVectors:      s.count,
				IndexBuildMs:    float64(buildTime.Milliseconds()),
				SearchLatencyUs: avgSearchUs,
				BatchSearchMs:   float64(batchTime.Milliseconds()),
				QPS:             qps,
				MemoryMB:        memUsedMB,
				VectorsPerSec:   vectorsPerSec,
			})

			stats := idx.GetStats()
			fmt.Printf("CAGRA stats: build=%.1fms, avg_search=%.3fms, searches=%d\n",
				stats.BuildTimeMs, stats.AvgSearchTimeMs, stats.SearchCount)
		})
	}

	fmt.Printf("\n=== SUMMARY ===\n")
	fmt.Printf("%-8s %-12s %-15s %-12s %-10s %-10s\n",
		"Scale", "Build(ms)", "Vec/sec", "Search(us)", "QPS", "Mem(MB)")
	for _, r := range results {
		fmt.Printf("%-8s %-12.0f %-15.0f %-12.2f %-10.0f %-10d\n",
			r.Scale, r.IndexBuildMs, r.VectorsPerSec, r.SearchLatencyUs, r.QPS, r.MemoryMB)
	}
}

func BenchmarkCAGRAIndexBuild(b *testing.B) {
	if !isCAGRAAvailable() {
		b.Skip("CAGRA not available")
	}

	sizes := []int{10000, 100000, 500000, 1000000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dk", size/1000), func(b *testing.B) {
			vectors, scales := generateLargeScaleVectors(size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cfg := FastCAGRAConfig()
				cfg.MaxVectors = size + 1000
				cfg.CachePath = ""

				idx, err := NewCAGRAIndex(cfg)
				if err != nil {
					b.Fatal(err)
				}

				if err := idx.BuildIndex(vectors, scales); err != nil {
					b.Fatal(err)
				}

				idx.Close()
			}
		})
	}
}

func BenchmarkCAGRASearch(b *testing.B) {
	if !isCAGRAAvailable() {
		b.Skip("CAGRA not available")
	}

	sizes := []int{100000, 500000, 1000000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dk", size/1000), func(b *testing.B) {
			cfg := FastCAGRAConfig()
			cfg.MaxVectors = size + 1000
			cfg.CachePath = ""

			idx, err := NewCAGRAIndex(cfg)
			if err != nil {
				b.Fatal(err)
			}
			defer idx.Close()

			vectors, scales := generateLargeScaleVectors(size)
			if err := idx.BuildIndex(vectors, scales); err != nil {
				b.Fatal(err)
			}

			queries, qScales := generateQueryVectors(1000)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := i % len(queries)
				_, err := idx.Search(queries[q], qScales[q], 10)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkCAGRABatchSearch(b *testing.B) {
	if !isCAGRAAvailable() {
		b.Skip("CAGRA not available")
	}

	cfg := FastCAGRAConfig()
	cfg.MaxVectors = 1000001
	cfg.CachePath = ""

	idx, err := NewCAGRAIndex(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	vectors, scales := generateTestVectors(1000000)
	if err := idx.BuildIndex(vectors, scales); err != nil {
		b.Fatal(err)
	}

	batchSizes := []int{10, 50, 100, 500}

	for _, bs := range batchSizes {
		b.Run(fmt.Sprintf("batch%d", bs), func(b *testing.B) {
			queries, qScales := generateQueryVectors(bs)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := idx.SearchBatch(queries, qScales, 10)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(bs), "queries/op")
		})
	}
}

func TestCAGRAMemoryProfile(t *testing.T) {
	if !isCAGRAAvailable() {
		t.Skip("CAGRA not available")
	}

	sizes := []int{100000, 500000, 1000000}

	fmt.Printf("\n=== Memory Profile ===\n")
	fmt.Printf("%-10s %-15s %-15s %-15s\n", "Vectors", "Heap(MB)", "Per-Vec(bytes)", "Overhead")

	for _, size := range sizes {
		runtime.GC()
		var before runtime.MemStats
		runtime.ReadMemStats(&before)

		cfg := FastCAGRAConfig()
		cfg.MaxVectors = size + 1000
		cfg.CachePath = ""

		idx, err := NewCAGRAIndex(cfg)
		if err != nil {
			t.Fatal(err)
		}

		vectors, scales := generateLargeScaleVectors(size)
		if err := idx.BuildIndex(vectors, scales); err != nil {
			t.Fatal(err)
		}

		runtime.GC()
		var after runtime.MemStats
		runtime.ReadMemStats(&after)

		heapMB := float64(after.Alloc-before.Alloc) / 1024 / 1024
		perVec := float64(after.Alloc-before.Alloc) / float64(size)
		rawBytes := float64(size * 512)
		overhead := (float64(after.Alloc-before.Alloc) - rawBytes) / rawBytes * 100

		fmt.Printf("%-10d %-15.1f %-15.1f %-15.1f%%\n", size, heapMB, perVec, overhead)

		idx.Close()
	}
}

func TestCAGRASearchLatencyDistribution(t *testing.T) {
	if !isCAGRAAvailable() {
		t.Skip("CAGRA not available")
	}

	cfg := FastCAGRAConfig()
	cfg.MaxVectors = 1000001
	cfg.CachePath = ""

	idx, err := NewCAGRAIndex(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	fmt.Printf("Building 1M vector index...\n")
	vectors, scales := generateTestVectors(1000000)
	if err := idx.BuildIndex(vectors, scales); err != nil {
		t.Fatal(err)
	}

	queries, qScales := generateQueryVectors(1000)

	latencies := make([]time.Duration, 1000)
	for i := 0; i < 1000; i++ {
		start := time.Now()
		_, err := idx.Search(queries[i], qScales[i], 10)
		if err != nil {
			t.Fatal(err)
		}
		latencies[i] = time.Since(start)
	}

	var total time.Duration
	var min, max time.Duration = latencies[0], latencies[0]
	for _, l := range latencies {
		total += l
		if l < min {
			min = l
		}
		if l > max {
			max = l
		}
	}

	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	fmt.Printf("\n=== Search Latency Distribution (1M vectors) ===\n")
	fmt.Printf("Min:  %v\n", min)
	fmt.Printf("Avg:  %v\n", total/1000)
	fmt.Printf("P50:  %v\n", sorted[500])
	fmt.Printf("P95:  %v\n", sorted[950])
	fmt.Printf("P99:  %v\n", sorted[990])
	fmt.Printf("Max:  %v\n", max)
	fmt.Printf("QPS:  %.0f\n", float64(1000)/total.Seconds())

	if sorted[950] > 2*time.Millisecond {
		t.Logf("Warning: P95 latency %.2fms exceeds 2ms target", float64(sorted[950].Microseconds())/1000)
	}
}

func TestCAGRAIndexingThroughput(t *testing.T) {
	if !isCAGRAAvailable() {
		t.Skip("CAGRA not available")
	}

	fmt.Printf("\n=== Indexing Throughput Test ===\n")

	sizes := []int{100000, 500000, 1000000, 2000000}

	for _, size := range sizes {
		vectors, scales := generateLargeScaleVectors(size)

		cfg := FastCAGRAConfig()
		cfg.MaxVectors = size + 1000
		cfg.CachePath = ""

		idx, err := NewCAGRAIndex(cfg)
		if err != nil {
			t.Fatal(err)
		}

		start := time.Now()
		if err := idx.BuildIndex(vectors, scales); err != nil {
			t.Fatal(err)
		}
		elapsed := time.Since(start)

		throughput := float64(size) / elapsed.Seconds()
		fmt.Printf("%dk vectors: %v (%.0f vec/sec)\n", size/1000, elapsed, throughput)

		idx.Close()
		runtime.GC()
	}
}
