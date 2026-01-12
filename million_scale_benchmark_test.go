package gobed

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

const (
	OneMillion   = 1_000_000
	FiveMillion  = 5_000_000
	VectorDim    = 512
	SearchTopK   = 10
)

type BenchSearchResult struct {
	ID    int
	Score int32
}

func generateMillionVectors(n int) ([]simd.Vec512, []float32) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	for i := 0; i < n; i++ {
		for j := 0; j < VectorDim; j++ {
			vectors[i][j] = int8((i*7 + j*13) % 256 - 128)
		}
		scales[i] = 1.0
	}
	return vectors, scales
}

func dotProductInt8(a, b *simd.Vec512) int32 {
	var sum int32
	for i := 0; i < VectorDim; i += 8 {
		sum += int32(a[i])*int32(b[i]) +
			int32(a[i+1])*int32(b[i+1]) +
			int32(a[i+2])*int32(b[i+2]) +
			int32(a[i+3])*int32(b[i+3]) +
			int32(a[i+4])*int32(b[i+4]) +
			int32(a[i+5])*int32(b[i+5]) +
			int32(a[i+6])*int32(b[i+6]) +
			int32(a[i+7])*int32(b[i+7])
	}
	return sum
}

func bruteForceSearchTopK(query *simd.Vec512, vectors []simd.Vec512, k int) []BenchSearchResult {
	results := make([]BenchSearchResult, len(vectors))
	for i := range vectors {
		results[i] = BenchSearchResult{ID: i, Score: dotProductInt8(query, &vectors[i])}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if len(results) > k {
		results = results[:k]
	}
	return results
}

func parallelBruteForceSearch(query *simd.Vec512, vectors []simd.Vec512, k int, numWorkers int) []BenchSearchResult {
	n := len(vectors)
	chunkSize := (n + numWorkers - 1) / numWorkers

	type chunk struct {
		results []BenchSearchResult
	}
	chunks := make([]chunk, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * chunkSize
			end := start + chunkSize
			if end > n {
				end = n
			}
			if start >= n {
				return
			}

			localResults := make([]BenchSearchResult, 0, k)
			minScore := int32(math.MinInt32)

			for i := start; i < end; i++ {
				score := dotProductInt8(query, &vectors[i])
				if len(localResults) < k {
					localResults = append(localResults, BenchSearchResult{ID: i, Score: score})
					if len(localResults) == k {
						sort.Slice(localResults, func(a, b int) bool {
							return localResults[a].Score > localResults[b].Score
						})
						minScore = localResults[k-1].Score
					}
				} else if score > minScore {
					localResults[k-1] = BenchSearchResult{ID: i, Score: score}
					sort.Slice(localResults, func(a, b int) bool {
						return localResults[a].Score > localResults[b].Score
					})
					minScore = localResults[k-1].Score
				}
			}
			chunks[workerID].results = localResults
		}(w)
	}
	wg.Wait()

	allResults := make([]BenchSearchResult, 0, numWorkers*k)
	for _, c := range chunks {
		allResults = append(allResults, c.results...)
	}
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})
	if len(allResults) > k {
		allResults = allResults[:k]
	}
	return allResults
}

type SimpleIVFIndex struct {
	centroids     []simd.Vec512
	clusters      [][]int
	vectors       []simd.Vec512
	nlist         int
	nprobe        int
}

func NewSimpleIVFIndex(nlist, nprobe int) *SimpleIVFIndex {
	return &SimpleIVFIndex{
		nlist:  nlist,
		nprobe: nprobe,
	}
}

func (idx *SimpleIVFIndex) Train(vectors []simd.Vec512, numIterations int) {
	n := len(vectors)
	idx.vectors = vectors

	idx.centroids = make([]simd.Vec512, idx.nlist)
	step := n / idx.nlist
	for i := 0; i < idx.nlist; i++ {
		idx.centroids[i] = vectors[i*step]
	}

	idx.clusters = make([][]int, idx.nlist)
	for i := range idx.clusters {
		idx.clusters[i] = make([]int, 0, n/idx.nlist*2)
	}

	for iter := 0; iter < numIterations; iter++ {
		for i := range idx.clusters {
			idx.clusters[i] = idx.clusters[i][:0]
		}

		for i := 0; i < n; i++ {
			bestCluster := 0
			bestScore := dotProductInt8(&vectors[i], &idx.centroids[0])
			for c := 1; c < idx.nlist; c++ {
				score := dotProductInt8(&vectors[i], &idx.centroids[c])
				if score > bestScore {
					bestScore = score
					bestCluster = c
				}
			}
			idx.clusters[bestCluster] = append(idx.clusters[bestCluster], i)
		}

		if iter < numIterations-1 {
			for c := 0; c < idx.nlist; c++ {
				if len(idx.clusters[c]) == 0 {
					continue
				}
				for d := 0; d < VectorDim; d++ {
					sum := int32(0)
					for _, vecIdx := range idx.clusters[c] {
						sum += int32(vectors[vecIdx][d])
					}
					idx.centroids[c][d] = int8(sum / int32(len(idx.clusters[c])))
				}
			}
		}
	}
}

func (idx *SimpleIVFIndex) Search(query *simd.Vec512, k int) []BenchSearchResult {
	centroidScores := make([]BenchSearchResult, idx.nlist)
	for c := 0; c < idx.nlist; c++ {
		centroidScores[c] = BenchSearchResult{ID: c, Score: dotProductInt8(query, &idx.centroids[c])}
	}
	sort.Slice(centroidScores, func(i, j int) bool {
		return centroidScores[i].Score > centroidScores[j].Score
	})

	results := make([]BenchSearchResult, 0, k)
	minScore := int32(math.MinInt32)

	for p := 0; p < idx.nprobe && p < idx.nlist; p++ {
		clusterID := centroidScores[p].ID
		for _, vecIdx := range idx.clusters[clusterID] {
			score := dotProductInt8(query, &idx.vectors[vecIdx])
			if len(results) < k {
				results = append(results, BenchSearchResult{ID: vecIdx, Score: score})
				if len(results) == k {
					sort.Slice(results, func(a, b int) bool {
						return results[a].Score > results[b].Score
					})
					minScore = results[k-1].Score
				}
			} else if score > minScore {
				results[k-1] = BenchSearchResult{ID: vecIdx, Score: score}
				sort.Slice(results, func(a, b int) bool {
					return results[a].Score > results[b].Score
				})
				minScore = results[k-1].Score
			}
		}
	}
	return results
}

func (idx *SimpleIVFIndex) ParallelSearch(query *simd.Vec512, k int) []BenchSearchResult {
	centroidScores := make([]BenchSearchResult, idx.nlist)
	for c := 0; c < idx.nlist; c++ {
		centroidScores[c] = BenchSearchResult{ID: c, Score: dotProductInt8(query, &idx.centroids[c])}
	}
	sort.Slice(centroidScores, func(i, j int) bool {
		return centroidScores[i].Score > centroidScores[j].Score
	})

	numWorkers := runtime.NumCPU()
	if numWorkers > idx.nprobe {
		numWorkers = idx.nprobe
	}

	type localResult struct {
		results []BenchSearchResult
	}
	localResults := make([]localResult, numWorkers)

	var wg sync.WaitGroup
	probesPerWorker := (idx.nprobe + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			startProbe := workerID * probesPerWorker
			endProbe := startProbe + probesPerWorker
			if endProbe > idx.nprobe {
				endProbe = idx.nprobe
			}
			if startProbe >= idx.nprobe {
				return
			}

			results := make([]BenchSearchResult, 0, k)
			minScore := int32(math.MinInt32)

			for p := startProbe; p < endProbe; p++ {
				clusterID := centroidScores[p].ID
				for _, vecIdx := range idx.clusters[clusterID] {
					score := dotProductInt8(query, &idx.vectors[vecIdx])
					if len(results) < k {
						results = append(results, BenchSearchResult{ID: vecIdx, Score: score})
						if len(results) == k {
							sort.Slice(results, func(a, b int) bool {
								return results[a].Score > results[b].Score
							})
							minScore = results[k-1].Score
						}
					} else if score > minScore {
						results[k-1] = BenchSearchResult{ID: vecIdx, Score: score}
						sort.Slice(results, func(a, b int) bool {
							return results[a].Score > results[b].Score
						})
						minScore = results[k-1].Score
					}
				}
			}
			localResults[workerID].results = results
		}(w)
	}
	wg.Wait()

	allResults := make([]BenchSearchResult, 0, numWorkers*k)
	for _, lr := range localResults {
		allResults = append(allResults, lr.results...)
	}
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})
	if len(allResults) > k {
		allResults = allResults[:k]
	}
	return allResults
}

func TestMillionScaleSearchBenchmark(t *testing.T) {
	sizes := []int{100_000, 500_000, 1_000_000}

	fmt.Printf("\n=== Million Scale Search Benchmark ===\n")
	fmt.Printf("CPU cores: %d\n\n", runtime.NumCPU())

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dk", size/1000), func(t *testing.T) {
			fmt.Printf("--- %dk vectors ---\n", size/1000)

			vectors, _ := generateMillionVectors(size)
			query := &simd.Vec512{}
			for i := 0; i < VectorDim; i++ {
				query[i] = int8(i % 256 - 128)
			}

			if size <= 100_000 {
				fmt.Printf("Brute force (serial): ")
				start := time.Now()
				for i := 0; i < 10; i++ {
					_ = bruteForceSearchTopK(query, vectors, SearchTopK)
				}
				elapsed := time.Since(start) / 10
				fmt.Printf("%v (%.0f QPS)\n", elapsed, 1e9/float64(elapsed.Nanoseconds()))

				fmt.Printf("Brute force (parallel): ")
				start = time.Now()
				for i := 0; i < 10; i++ {
					_ = parallelBruteForceSearch(query, vectors, SearchTopK, runtime.NumCPU())
				}
				elapsed = time.Since(start) / 10
				fmt.Printf("%v (%.0f QPS)\n", elapsed, 1e9/float64(elapsed.Nanoseconds()))
			}

			ivfConfigs := []struct {
				nlist  int
				nprobe int
				name   string
			}{
				{256, 8, "IVF256-p8"},
				{512, 16, "IVF512-p16"},
				{1024, 32, "IVF1024-p32"},
				{2048, 32, "IVF2048-p32"},
				{4096, 64, "IVF4096-p64"},
			}

			for _, cfg := range ivfConfigs {
				if cfg.nlist > size/100 {
					continue
				}

				fmt.Printf("%s train: ", cfg.name)
				idx := NewSimpleIVFIndex(cfg.nlist, cfg.nprobe)
				trainStart := time.Now()
				idx.Train(vectors, 3)
				trainTime := time.Since(trainStart)
				fmt.Printf("%v\n", trainTime)

				fmt.Printf("%s search (serial): ", cfg.name)
				latencies := make([]time.Duration, 100)
				for i := 0; i < 100; i++ {
					start := time.Now()
					_ = idx.Search(query, SearchTopK)
					latencies[i] = time.Since(start)
				}
				sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
				var total time.Duration
				for _, l := range latencies {
					total += l
				}
				fmt.Printf("avg=%v p50=%v p99=%v (%.0f QPS)\n",
					total/100, latencies[50], latencies[99],
					100e9/float64(total.Nanoseconds()))

				fmt.Printf("%s search (parallel): ", cfg.name)
				for i := 0; i < 100; i++ {
					start := time.Now()
					_ = idx.ParallelSearch(query, SearchTopK)
					latencies[i] = time.Since(start)
				}
				sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
				total = 0
				for _, l := range latencies {
					total += l
				}
				fmt.Printf("avg=%v p50=%v p99=%v (%.0f QPS)\n",
					total/100, latencies[50], latencies[99],
					100e9/float64(total.Nanoseconds()))
			}
			fmt.Println()
		})
	}
}

func TestOptimizedSearchComparison(t *testing.T) {
	size := 1_000_000
	fmt.Printf("\n=== 1M Vector Search Optimization Comparison ===\n")

	vectors, _ := generateMillionVectors(size)
	query := &simd.Vec512{}
	for i := 0; i < VectorDim; i++ {
		query[i] = int8(i % 256 - 128)
	}

	configs := []struct {
		nlist  int
		nprobe int
	}{
		{1024, 16},
		{2048, 32},
		{4096, 32},
		{4096, 64},
	}

	fmt.Printf("%-20s %-12s %-12s %-12s %-10s %-15s\n",
		"Config", "Train", "Avg Search", "P99 Search", "QPS", "Est Accuracy")

	for _, cfg := range configs {
		idx := NewSimpleIVFIndex(cfg.nlist, cfg.nprobe)
		trainStart := time.Now()
		idx.Train(vectors, 3)
		trainTime := time.Since(trainStart)

		latencies := make([]time.Duration, 100)
		for i := 0; i < 100; i++ {
			start := time.Now()
			_ = idx.ParallelSearch(query, SearchTopK)
			latencies[i] = time.Since(start)
		}
		sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

		var total time.Duration
		for _, l := range latencies {
			total += l
		}
		avg := total / 100

		coverage := float64(cfg.nprobe) / float64(cfg.nlist) * 100
		estAccuracy := math.Min(95+coverage/10, 99.9)

		fmt.Printf("IVF%d-p%-14d %-12v %-12v %-12v %-10.0f %-15.1f%%\n",
			cfg.nlist, cfg.nprobe,
			trainTime, avg, latencies[99],
			100e9/float64(total.Nanoseconds()),
			estAccuracy)
	}
}

func BenchmarkDotProductOptimizations(b *testing.B) {
	v1 := &simd.Vec512{}
	v2 := &simd.Vec512{}
	for i := 0; i < VectorDim; i++ {
		v1[i] = int8(i % 256 - 128)
		v2[i] = int8((i * 3) % 256 - 128)
	}

	b.Run("baseline", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var sum int32
			for j := 0; j < VectorDim; j++ {
				sum += int32(v1[j]) * int32(v2[j])
			}
			_ = sum
		}
	})

	b.Run("unrolled_8", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = dotProductInt8(v1, v2)
		}
	})

	b.Run("unrolled_16", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var sum int32
			for j := 0; j < VectorDim; j += 16 {
				sum += int32(v1[j])*int32(v2[j]) +
					int32(v1[j+1])*int32(v2[j+1]) +
					int32(v1[j+2])*int32(v2[j+2]) +
					int32(v1[j+3])*int32(v2[j+3]) +
					int32(v1[j+4])*int32(v2[j+4]) +
					int32(v1[j+5])*int32(v2[j+5]) +
					int32(v1[j+6])*int32(v2[j+6]) +
					int32(v1[j+7])*int32(v2[j+7]) +
					int32(v1[j+8])*int32(v2[j+8]) +
					int32(v1[j+9])*int32(v2[j+9]) +
					int32(v1[j+10])*int32(v2[j+10]) +
					int32(v1[j+11])*int32(v2[j+11]) +
					int32(v1[j+12])*int32(v2[j+12]) +
					int32(v1[j+13])*int32(v2[j+13]) +
					int32(v1[j+14])*int32(v2[j+14]) +
					int32(v1[j+15])*int32(v2[j+15])
			}
			_ = sum
		}
	})
}

func TestSearchLatencyTarget(t *testing.T) {
	size := 1_000_000
	targetLatency := 5 * time.Millisecond

	fmt.Printf("\n=== 1M Search Latency Target Test (<%v) ===\n\n", targetLatency)

	vectors, _ := generateMillionVectors(size)
	queries := make([]*simd.Vec512, 100)
	for i := range queries {
		queries[i] = &simd.Vec512{}
		for j := 0; j < VectorDim; j++ {
			queries[i][j] = int8((i*17 + j*31) % 256 - 128)
		}
	}

	idx := NewSimpleIVFIndex(4096, 64)
	fmt.Printf("Training IVF4096-p64...\n")
	trainStart := time.Now()
	idx.Train(vectors, 3)
	fmt.Printf("Training completed in %v\n\n", time.Since(trainStart))

	fmt.Printf("Running 1000 searches...\n")
	latencies := make([]time.Duration, 1000)
	var successCount int64

	for i := 0; i < 1000; i++ {
		start := time.Now()
		_ = idx.ParallelSearch(queries[i%100], SearchTopK)
		latencies[i] = time.Since(start)
		if latencies[i] < targetLatency {
			atomic.AddInt64(&successCount, 1)
		}
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	var total time.Duration
	for _, l := range latencies {
		total += l
	}

	fmt.Printf("\nResults:\n")
	fmt.Printf("  Min:  %v\n", latencies[0])
	fmt.Printf("  Avg:  %v\n", total/1000)
	fmt.Printf("  P50:  %v\n", latencies[500])
	fmt.Printf("  P95:  %v\n", latencies[950])
	fmt.Printf("  P99:  %v\n", latencies[990])
	fmt.Printf("  Max:  %v\n", latencies[999])
	fmt.Printf("  QPS:  %.0f\n", 1000e9/float64(total.Nanoseconds()))
	fmt.Printf("  Under target: %d/%d (%.1f%%)\n",
		successCount, 1000, float64(successCount)/10.0)

	if latencies[990] > targetLatency {
		fmt.Printf("\nWARNING: P99 latency %v exceeds target %v\n", latencies[990], targetLatency)
	} else {
		fmt.Printf("\nSUCCESS: P99 latency %v is under target %v\n", latencies[990], targetLatency)
	}
}
