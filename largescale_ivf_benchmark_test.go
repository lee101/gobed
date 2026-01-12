package gobed

import (
	"fmt"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/search"
	"github.com/lee101/gobed/pkg/ann/simd"
)

type IVFScaleBenchResult struct {
	Scale         string
	NumVectors    int
	IndexBuildMs  float64
	TrainMs       float64
	SearchUs      float64
	QPS           float64
	MemoryMB      int64
	VectorsPerSec float64
}

func generateIVFTestVectors(n int) ([]simd.Vec512, []float32, []int) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	ids := make([]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8((i*j + j) % 256 - 128)
		}
		scales[i] = 0.5 + float32(i%100)/100.0
		ids[i] = i
	}
	return vectors, scales, ids
}

func generateIVFQueryVectors(n int) []simd.Vec512 {
	vectors := make([]simd.Vec512, n)
	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8((i*17 + j*31) % 256 - 128)
		}
	}
	return vectors
}

func TestLargeScaleIVFBenchmark(t *testing.T) {
	scales := []struct {
		name    string
		count   int
		nlist   int
		nprobe  int
		maxFlat int
	}{
		{"100K", 100000, 256, 16, 0},
		{"500K", 500000, 1024, 32, 0},
		{"1M", 1000000, 2048, 64, 0},
	}

	results := make([]IVFScaleBenchResult, 0)

	for _, s := range scales {
		t.Run(s.name, func(t *testing.T) {
			runtime.GC()
			var memBefore runtime.MemStats
			runtime.ReadMemStats(&memBefore)

			config := search.Config{
				MaxFlatSize: s.maxFlat,
				NList:       s.nlist,
				NProbe:      s.nprobe,
				UseParallel: true,
			}

			engine := search.NewEngine(config)

			fmt.Printf("\n=== %s vectors (IVF nlist=%d nprobe=%d) ===\n", s.name, s.nlist, s.nprobe)

			vectors, vecScales, ids := generateIVFTestVectors(s.count)

			trainSize := s.nlist * 50
			if trainSize > s.count {
				trainSize = s.count / 2
			}

			trainStart := time.Now()
			if err := engine.Train(vectors[:trainSize], vecScales[:trainSize]); err != nil {
				t.Fatalf("train failed: %v", err)
			}
			trainTime := time.Since(trainStart)
			fmt.Printf("Train: %v (%d samples)\n", trainTime, trainSize)

			buildStart := time.Now()
			if err := engine.AddBatch(vectors, vecScales, ids); err != nil {
				t.Fatalf("add failed: %v", err)
			}
			buildTime := time.Since(buildStart)
			vectorsPerSec := float64(s.count) / buildTime.Seconds()
			fmt.Printf("Index build: %v (%.0f vectors/sec)\n", buildTime, vectorsPerSec)

			var memAfter runtime.MemStats
			runtime.ReadMemStats(&memAfter)
			memUsedMB := int64(memAfter.Alloc-memBefore.Alloc) / 1024 / 1024

			queries := generateIVFQueryVectors(100)

			latencies := make([]time.Duration, 100)
			for i := 0; i < 100; i++ {
				start := time.Now()
				_, err := engine.Search(&queries[i%len(queries)], 10)
				if err != nil {
					t.Fatalf("search failed: %v", err)
				}
				latencies[i] = time.Since(start)
			}

			var total time.Duration
			for _, l := range latencies {
				total += l
			}
			avgSearchUs := float64(total.Microseconds()) / 100.0
			qps := 100.0 / total.Seconds()

			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

			fmt.Printf("Search latency: avg=%.2fus p50=%v p95=%v p99=%v (%.0f QPS)\n",
				avgSearchUs, latencies[50], latencies[95], latencies[99], qps)
			fmt.Printf("Memory: ~%d MB\n", memUsedMB)

			results = append(results, IVFScaleBenchResult{
				Scale:         s.name,
				NumVectors:    s.count,
				IndexBuildMs:  float64(buildTime.Milliseconds()),
				TrainMs:       float64(trainTime.Milliseconds()),
				SearchUs:      avgSearchUs,
				QPS:           qps,
				MemoryMB:      memUsedMB,
				VectorsPerSec: vectorsPerSec,
			})
		})
	}

	fmt.Printf("\n=== IVF SUMMARY ===\n")
	fmt.Printf("%-8s %-10s %-12s %-12s %-10s %-10s\n",
		"Scale", "Train(ms)", "Build(ms)", "Search(us)", "QPS", "Mem(MB)")
	for _, r := range results {
		fmt.Printf("%-8s %-10.0f %-12.0f %-12.2f %-10.0f %-10d\n",
			r.Scale, r.TrainMs, r.IndexBuildMs, r.SearchUs, r.QPS, r.MemoryMB)
	}
}

func BenchmarkLargeScaleIVFIndexBuild(b *testing.B) {
	sizes := []int{10000, 50000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dk", size/1000), func(b *testing.B) {
			vectors, scales, ids := generateIVFTestVectors(size)

			nlist := size / 50
			if nlist > 2048 {
				nlist = 2048
			}
			if nlist < 64 {
				nlist = 64
			}

			config := search.Config{
				MaxFlatSize: 0,
				NList:       nlist,
				NProbe:      nlist / 8,
				UseParallel: true,
			}

			trainSize := nlist * 50
			if trainSize > size {
				trainSize = size / 2
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				engine := search.NewEngine(config)
				engine.Train(vectors[:trainSize], scales[:trainSize])
				engine.AddBatch(vectors, scales, ids)
			}
		})
	}
}

func BenchmarkLargeScaleIVFSearch(b *testing.B) {
	sizes := []int{100000, 500000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dk", size/1000), func(b *testing.B) {
			vectors, scales, ids := generateIVFTestVectors(size)
			queries := generateIVFQueryVectors(1000)

			nlist := size / 50
			if nlist > 2048 {
				nlist = 2048
			}

			config := search.Config{
				MaxFlatSize: 0,
				NList:       nlist,
				NProbe:      nlist / 8,
				UseParallel: true,
			}

			engine := search.NewEngine(config)
			trainSize := nlist * 50
			if trainSize > size {
				trainSize = size / 2
			}
			engine.Train(vectors[:trainSize], scales[:trainSize])
			engine.AddBatch(vectors, scales, ids)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := i % len(queries)
				_, err := engine.Search(&queries[q], 10)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestIVFMemoryProfile(t *testing.T) {
	sizes := []int{100000, 500000, 1000000}

	fmt.Printf("\n=== IVF Memory Profile ===\n")
	fmt.Printf("%-10s %-15s %-15s %-15s\n", "Vectors", "Heap(MB)", "Per-Vec(bytes)", "Overhead")

	for _, size := range sizes {
		runtime.GC()
		var before runtime.MemStats
		runtime.ReadMemStats(&before)

		vectors, scales, ids := generateIVFTestVectors(size)

		nlist := size / 50
		if nlist > 2048 {
			nlist = 2048
		}

		config := search.Config{
			MaxFlatSize: 0,
			NList:       nlist,
			NProbe:      nlist / 8,
			UseParallel: true,
		}

		engine := search.NewEngine(config)
		trainSize := nlist * 50
		if trainSize > size {
			trainSize = size / 2
		}
		engine.Train(vectors[:trainSize], scales[:trainSize])
		engine.AddBatch(vectors, scales, ids)

		runtime.GC()
		var after runtime.MemStats
		runtime.ReadMemStats(&after)

		heapMB := float64(after.Alloc-before.Alloc) / 1024 / 1024
		perVec := float64(after.Alloc-before.Alloc) / float64(size)
		rawBytes := float64(size * 512)
		overhead := (float64(after.Alloc-before.Alloc) - rawBytes) / rawBytes * 100

		fmt.Printf("%-10d %-15.1f %-15.1f %-15.1f%%\n", size, heapMB, perVec, overhead)
	}
}

func TestIVFSearchLatencyDistribution(t *testing.T) {
	size := 500000
	vectors, scales, ids := generateIVFTestVectors(size)
	queries := generateIVFQueryVectors(1000)

	nlist := 2048
	nprobe := 64

	config := search.Config{
		MaxFlatSize: 0,
		NList:       nlist,
		NProbe:      nprobe,
		UseParallel: true,
	}

	fmt.Printf("Building %dk vector IVF index (nlist=%d, nprobe=%d)...\n", size/1000, nlist, nprobe)

	engine := search.NewEngine(config)
	trainSize := nlist * 50
	engine.Train(vectors[:trainSize], scales[:trainSize])
	engine.AddBatch(vectors, scales, ids)

	latencies := make([]time.Duration, 1000)
	for i := 0; i < 1000; i++ {
		start := time.Now()
		_, err := engine.Search(&queries[i], 10)
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

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	fmt.Printf("\n=== IVF Search Latency Distribution (500K vectors) ===\n")
	fmt.Printf("Min:  %v\n", min)
	fmt.Printf("Avg:  %v\n", total/1000)
	fmt.Printf("P50:  %v\n", latencies[500])
	fmt.Printf("P95:  %v\n", latencies[950])
	fmt.Printf("P99:  %v\n", latencies[990])
	fmt.Printf("Max:  %v\n", max)
	fmt.Printf("QPS:  %.0f\n", float64(1000)/total.Seconds())
}
