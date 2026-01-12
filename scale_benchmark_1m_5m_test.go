package gobed

import (
	"fmt"
	"runtime"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/ivf"
	"github.com/lee101/gobed/pkg/ann/simd"
)

type ScaleResult struct {
	Scale            string
	NumVectors       int
	VectorGenTimeMs  float64
	IndexBuildTimeMs float64
	AvgSearchUs      float64
	P50SearchUs      float64
	P95SearchUs      float64
	P99SearchUs      float64
	BatchSearchMs    float64
	BatchQPS         float64
	SingleQPS        float64
	MemoryMB         float64
	BytesPerVector   float64
}

func generateScaleVectors(n int) ([]simd.Vec512, []float32, []int) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	ids := make([]int, n)

	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()
	chunkSize := (n + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < 512; j++ {
					vectors[i][j] = int8((i*j + j*17 + i*31) % 256 - 128)
				}
				scales[i] = 0.1 + float32(i%100)/500.0
				ids[i] = i
			}
		}(start, end)
	}
	wg.Wait()
	return vectors, scales, ids
}

func generateQueryVectorsScale(n int) ([]simd.Vec512, []float32) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8((i*23 + j*41) % 256 - 128)
		}
		scales[i] = 1.0
	}
	return vectors, scales
}

func TestScale1M5MBenchmark(t *testing.T) {
	scales := []struct {
		name  string
		count int
		nlist int
	}{
		{"10K", 10000, 100},
		{"100K", 100000, 316},
		{"500K", 500000, 707},
		{"1M", 1000000, 1000},
		{"2M", 2000000, 1414},
		{"5M", 5000000, 2236},
	}

	results := make([]ScaleResult, 0)
	numQueries := 1000
	topK := 10
	nprobe := 8

	for _, s := range scales {
		t.Run(s.name, func(t *testing.T) {
			result := ScaleResult{
				Scale:      s.name,
				NumVectors: s.count,
			}

			runtime.GC()
			var memBefore runtime.MemStats
			runtime.ReadMemStats(&memBefore)

			fmt.Printf("\n=== %s vectors (%d) ===\n", s.name, s.count)

			genStart := time.Now()
			vectors, vecScales, ids := generateScaleVectors(s.count)
			result.VectorGenTimeMs = float64(time.Since(genStart).Milliseconds())
			fmt.Printf("Vector generation: %.0fms (%.0f vec/sec)\n",
				result.VectorGenTimeMs, float64(s.count)/(result.VectorGenTimeMs/1000))

			idx := ivf.NewIVFIndex(s.nlist, nprobe)

			trainSize := min(s.count, s.nlist*50)
			trainStart := time.Now()
			idx.Train(vectors[:trainSize], vecScales[:trainSize])
			trainTime := time.Since(trainStart)
			fmt.Printf("IVF training (%d clusters): %v\n", s.nlist, trainTime)

			addStart := time.Now()
			idx.AddBatch(vectors, vecScales, ids)
			addTime := time.Since(addStart)
			result.IndexBuildTimeMs = float64(trainTime.Milliseconds()) + float64(addTime.Milliseconds())
			addThroughput := float64(s.count) / addTime.Seconds()
			fmt.Printf("Index build: %v (%.0f vec/sec)\n", addTime, addThroughput)

			var memAfter runtime.MemStats
			runtime.ReadMemStats(&memAfter)
			result.MemoryMB = float64(memAfter.Alloc-memBefore.Alloc) / 1024 / 1024
			result.BytesPerVector = float64(memAfter.Alloc-memBefore.Alloc) / float64(s.count)

			queries, _ := generateQueryVectorsScale(numQueries)

			for i := 0; i < 10; i++ {
				idx.Search(&queries[i], topK)
			}

			latencies := make([]time.Duration, numQueries)
			for i := 0; i < numQueries; i++ {
				start := time.Now()
				_ = idx.Search(&queries[i], topK)
				latencies[i] = time.Since(start)
			}

			sort.Slice(latencies, func(i, j int) bool {
				return latencies[i] < latencies[j]
			})

			var totalLatency time.Duration
			for _, l := range latencies {
				totalLatency += l
			}

			result.AvgSearchUs = float64(totalLatency.Microseconds()) / float64(numQueries)
			result.P50SearchUs = float64(latencies[numQueries/2].Microseconds())
			result.P95SearchUs = float64(latencies[numQueries*95/100].Microseconds())
			result.P99SearchUs = float64(latencies[numQueries*99/100].Microseconds())
			result.SingleQPS = float64(numQueries) / totalLatency.Seconds()

			fmt.Printf("Search latency: avg=%.1fus, p50=%.1fus, p95=%.1fus, p99=%.1fus\n",
				result.AvgSearchUs, result.P50SearchUs, result.P95SearchUs, result.P99SearchUs)
			fmt.Printf("Single-query QPS: %.0f\n", result.SingleQPS)

			batchStart := time.Now()
			for i := 0; i < numQueries; i++ {
				idx.Search(&queries[i], topK)
			}
			batchTime := time.Since(batchStart)
			result.BatchSearchMs = float64(batchTime.Milliseconds())
			result.BatchQPS = float64(numQueries) / batchTime.Seconds()
			fmt.Printf("Batch search (%d queries): %v (%.0f QPS)\n", numQueries, batchTime, result.BatchQPS)

			fmt.Printf("Memory: %.1f MB (%.1f bytes/vec)\n", result.MemoryMB, result.BytesPerVector)

			results = append(results, result)

			vectors = nil
			vecScales = nil
			ids = nil
			runtime.GC()
		})
	}

	fmt.Printf("\n" + "=" + "===========================================================================================================\n")
	fmt.Printf("SUMMARY\n")
	fmt.Printf("%-8s %-12s %-12s %-10s %-10s %-10s %-10s %-10s %-12s\n",
		"Scale", "Build(ms)", "AvgSearch", "P50(us)", "P95(us)", "P99(us)", "QPS", "Mem(MB)", "Bytes/Vec")
	fmt.Printf("%-8s %-12s %-12s %-10s %-10s %-10s %-10s %-10s %-12s\n",
		"-----", "---------", "---------", "-------", "-------", "-------", "---", "-------", "---------")

	for _, r := range results {
		fmt.Printf("%-8s %-12.0f %-12.1fus %-10.1f %-10.1f %-10.1f %-10.0f %-10.1f %-12.1f\n",
			r.Scale, r.IndexBuildTimeMs, r.AvgSearchUs, r.P50SearchUs, r.P95SearchUs, r.P99SearchUs, r.SingleQPS, r.MemoryMB, r.BytesPerVector)
	}
}

func TestBruteForceVsIVF(t *testing.T) {
	sizes := []int{10000, 100000, 500000}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("%dk", n/1000), func(t *testing.T) {
			vectors, scales, _ := generateScaleVectors(n)
			queries, _ := generateQueryVectorsScale(100)

			var bruteTotal time.Duration
			for i := 0; i < 10; i++ {
				start := time.Now()
				bruteForceSearch(&queries[i], vectors, scales, 10)
				bruteTotal += time.Since(start)
			}
			bruteAvg := bruteTotal / 10
			fmt.Printf("%dk brute-force: %v per query\n", n/1000, bruteAvg)

			nlist := int(float64(n) * 0.01)
			if nlist < 10 {
				nlist = 10
			}
			idx := ivf.NewIVFIndex(nlist, 8)
			idx.Train(vectors[:min(n, nlist*50)], scales[:min(n, nlist*50)])
			idx.AddBatch(vectors, scales, make([]int, n))

			for i := 0; i < 5; i++ {
				idx.Search(&queries[i], 10)
			}

			var ivfTotal time.Duration
			for i := 0; i < 100; i++ {
				start := time.Now()
				idx.Search(&queries[i%len(queries)], 10)
				ivfTotal += time.Since(start)
			}
			ivfAvg := ivfTotal / 100
			fmt.Printf("%dk IVF: %v per query (%.1fx speedup)\n", n/1000, ivfAvg, float64(bruteAvg)/float64(ivfAvg))
		})
	}
}

func bruteForceSearch(query *simd.Vec512, vectors []simd.Vec512, scales []float32, k int) []int {
	type scoredIdx struct {
		idx   int
		score int32
	}
	results := make([]scoredIdx, len(vectors))
	for i := range vectors {
		results[i] = scoredIdx{idx: i, score: simd.Dot512(query, &vectors[i])}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})
	topK := make([]int, k)
	for i := 0; i < k && i < len(results); i++ {
		topK[i] = results[i].idx
	}
	return topK
}

func TestParallelSearchThroughput(t *testing.T) {
	n := 500000
	vectors, scales, ids := generateScaleVectors(n)
	queries, _ := generateQueryVectorsScale(1000)

	nlist := 707
	idx := ivf.NewIVFIndex(nlist, 8)
	idx.Train(vectors[:nlist*50], scales[:nlist*50])
	idx.AddBatch(vectors, scales, ids)

	for _, numWorkers := range []int{1, 2, 4, 8, 16} {
		t.Run(fmt.Sprintf("workers_%d", numWorkers), func(t *testing.T) {
			numQueries := 10000
			queriesPerWorker := numQueries / numWorkers

			var wg sync.WaitGroup
			start := time.Now()

			for w := 0; w < numWorkers; w++ {
				wg.Add(1)
				go func(workerID int) {
					defer wg.Done()
					for i := 0; i < queriesPerWorker; i++ {
						qIdx := (workerID*queriesPerWorker + i) % len(queries)
						idx.Search(&queries[qIdx], 10)
					}
				}(w)
			}
			wg.Wait()

			elapsed := time.Since(start)
			qps := float64(numQueries) / elapsed.Seconds()
			fmt.Printf("%d workers: %v for %d queries (%.0f QPS)\n", numWorkers, elapsed, numQueries, qps)
		})
	}
}

func TestMemoryAllocationAnalysis(t *testing.T) {
	sizes := []int{100000, 500000, 1000000}

	fmt.Printf("\n=== Memory Allocation Analysis ===\n")
	fmt.Printf("%-10s %-15s %-15s %-15s %-15s\n",
		"Size", "Raw Vec(MB)", "With Scale(MB)", "IVF Overhead", "Total(MB)")

	for _, n := range sizes {
		runtime.GC()
		var m1 runtime.MemStats
		runtime.ReadMemStats(&m1)

		vectors, scales, ids := generateScaleVectors(n)

		runtime.GC()
		var m2 runtime.MemStats
		runtime.ReadMemStats(&m2)

		_ = float64(m2.Alloc-m1.Alloc) / 1024 / 1024

		nlist := int(float64(n) * 0.001)
		if nlist < 100 {
			nlist = 100
		}
		idx := ivf.NewIVFIndex(nlist, 8)
		idx.Train(vectors[:min(n, nlist*50)], scales[:min(n, nlist*50)])
		idx.AddBatch(vectors, scales, ids)

		runtime.GC()
		var m3 runtime.MemStats
		runtime.ReadMemStats(&m3)

		totalMem := float64(m3.Alloc-m1.Alloc) / 1024 / 1024
		ivfOverhead := float64(m3.Alloc-m2.Alloc) / 1024 / 1024

		rawVecMB := float64(n*512) / 1024 / 1024
		withScaleMB := float64(n*(512+4)) / 1024 / 1024

		fmt.Printf("%-10d %-15.1f %-15.1f %-15.1f %-15.1f\n",
			n, rawVecMB, withScaleMB, ivfOverhead, totalMem)

		vectors = nil
		scales = nil
		ids = nil
		runtime.GC()
	}
}

func TestIndexBuildScaling(t *testing.T) {
	fmt.Printf("\n=== Index Build Time Scaling ===\n")

	sizes := []int{10000, 50000, 100000, 250000, 500000, 1000000}

	for _, n := range sizes {
		vectors, scales, ids := generateScaleVectors(n)

		nlist := int(float64(n) * 0.001)
		if nlist < 100 {
			nlist = 100
		}

		start := time.Now()
		idx := ivf.NewIVFIndex(nlist, 8)
		idx.Train(vectors[:min(n, nlist*50)], scales[:min(n, nlist*50)])
		idx.AddBatch(vectors, scales, ids)
		elapsed := time.Since(start)

		throughput := float64(n) / elapsed.Seconds()
		fmt.Printf("%dk vectors: %v (%.0f vec/sec, nlist=%d)\n",
			n/1000, elapsed, throughput, nlist)

		vectors = nil
		runtime.GC()
	}
}

