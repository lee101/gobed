package gobed

import (
	"fmt"
	"runtime"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

type FastSearchResult struct {
	ID    int
	Score int32
}

func fastDotProduct(a, b *simd.Vec512) int32 {
	var sum int32
	for i := 0; i < 512; i += 8 {
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

type FastIVFIndex struct {
	centroids []simd.Vec512
	clusters  [][]int
	vectors   []simd.Vec512
	nlist     int
	nprobe    int
}

func (idx *FastIVFIndex) FastTrain(vectors []simd.Vec512, sampleRate float64) {
	n := len(vectors)
	idx.vectors = vectors

	sampleSize := int(float64(n) * sampleRate)
	if sampleSize < idx.nlist*10 {
		sampleSize = idx.nlist * 10
	}
	if sampleSize > n {
		sampleSize = n
	}

	step := n / idx.nlist
	idx.centroids = make([]simd.Vec512, idx.nlist)
	for i := 0; i < idx.nlist; i++ {
		idx.centroids[i] = vectors[(i*step)%n]
	}

	idx.clusters = make([][]int, idx.nlist)
	for i := range idx.clusters {
		idx.clusters[i] = make([]int, 0, n/idx.nlist*2)
	}

	sampleStep := n / sampleSize
	for iter := 0; iter < 2; iter++ {
		for i := range idx.clusters {
			idx.clusters[i] = idx.clusters[i][:0]
		}

		for s := 0; s < sampleSize; s++ {
			i := (s * sampleStep) % n
			bestCluster := 0
			bestScore := fastDotProduct(&vectors[i], &idx.centroids[0])
			for c := 1; c < idx.nlist; c++ {
				score := fastDotProduct(&vectors[i], &idx.centroids[c])
				if score > bestScore {
					bestScore = score
					bestCluster = c
				}
			}
			idx.clusters[bestCluster] = append(idx.clusters[bestCluster], i)
		}

		for c := 0; c < idx.nlist; c++ {
			if len(idx.clusters[c]) == 0 {
				continue
			}
			for d := 0; d < 512; d++ {
				sum := int32(0)
				for _, vecIdx := range idx.clusters[c] {
					sum += int32(vectors[vecIdx][d])
				}
				idx.centroids[c][d] = int8(sum / int32(len(idx.clusters[c])))
			}
		}
	}

	numWorkers := runtime.NumCPU()
	chunkSize := (n + numWorkers - 1) / numWorkers

	localClusters := make([][]int, numWorkers)
	for w := 0; w < numWorkers; w++ {
		localClusters[w] = make([]int, n)
	}

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
			for i := start; i < end; i++ {
				bestCluster := 0
				bestScore := fastDotProduct(&vectors[i], &idx.centroids[0])
				for c := 1; c < idx.nlist; c++ {
					score := fastDotProduct(&vectors[i], &idx.centroids[c])
					if score > bestScore {
						bestScore = score
						bestCluster = c
					}
				}
				localClusters[workerID][i-start] = bestCluster
			}
		}(w)
	}
	wg.Wait()

	for i := range idx.clusters {
		idx.clusters[i] = idx.clusters[i][:0]
	}
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		for i := start; i < end; i++ {
			cluster := localClusters[w][i-start]
			idx.clusters[cluster] = append(idx.clusters[cluster], i)
		}
	}
}

func (idx *FastIVFIndex) Search(query *simd.Vec512, k int) []FastSearchResult {
	centroidScores := make([]FastSearchResult, idx.nlist)
	for c := 0; c < idx.nlist; c++ {
		centroidScores[c] = FastSearchResult{ID: c, Score: fastDotProduct(query, &idx.centroids[c])}
	}
	sort.Slice(centroidScores, func(i, j int) bool {
		return centroidScores[i].Score > centroidScores[j].Score
	})

	results := make([]FastSearchResult, 0, k*2)
	for p := 0; p < idx.nprobe && p < idx.nlist; p++ {
		clusterID := centroidScores[p].ID
		for _, vecIdx := range idx.clusters[clusterID] {
			score := fastDotProduct(query, &idx.vectors[vecIdx])
			results = append(results, FastSearchResult{ID: vecIdx, Score: score})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if len(results) > k {
		results = results[:k]
	}
	return results
}

func TestFastMillionSearch(t *testing.T) {
	sizes := []int{500_000, 1_000_000}

	fmt.Printf("\n=== Fast Million Scale Search ===\n")
	fmt.Printf("CPU: %d cores\n\n", runtime.NumCPU())

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dk", size/1000), func(t *testing.T) {
			fmt.Printf("--- %dk vectors ---\n", size/1000)

			vectors := make([]simd.Vec512, size)
			for i := 0; i < size; i++ {
				for j := 0; j < 512; j++ {
					vectors[i][j] = int8((i*7 + j*13) % 256 - 128)
				}
			}

			query := &simd.Vec512{}
			for i := 0; i < 512; i++ {
				query[i] = int8(i % 256 - 128)
			}

			configs := []struct {
				nlist  int
				nprobe int
			}{
				{1024, 16},
				{1024, 32},
				{2048, 32},
				{2048, 64},
			}

			for _, cfg := range configs {
				idx := &FastIVFIndex{nlist: cfg.nlist, nprobe: cfg.nprobe}

				trainStart := time.Now()
				idx.FastTrain(vectors, 0.1)
				trainTime := time.Since(trainStart)

				latencies := make([]time.Duration, 100)
				for i := 0; i < 100; i++ {
					start := time.Now()
					_ = idx.Search(query, 10)
					latencies[i] = time.Since(start)
				}
				sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

				var total time.Duration
				for _, l := range latencies {
					total += l
				}
				avg := total / 100
				qps := 100e9 / float64(total.Nanoseconds())

				avgClusterSize := size / cfg.nlist
				vectorsSearched := avgClusterSize * cfg.nprobe

				fmt.Printf("IVF%d-p%d: train=%v avg=%v p99=%v QPS=%.0f (search ~%dk vecs)\n",
					cfg.nlist, cfg.nprobe, trainTime, avg, latencies[99], qps, vectorsSearched/1000)
			}
			fmt.Println()
		})
	}
}

func TestTargetLatencyFast(t *testing.T) {
	size := 1_000_000
	target := 5 * time.Millisecond

	fmt.Printf("\n=== 1M Target Latency Test (<%v) ===\n\n", target)

	vectors := make([]simd.Vec512, size)
	for i := 0; i < size; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8((i*7 + j*13) % 256 - 128)
		}
	}

	queries := make([]*simd.Vec512, 100)
	for q := range queries {
		queries[q] = &simd.Vec512{}
		for j := 0; j < 512; j++ {
			queries[q][j] = int8((q*17 + j*31) % 256 - 128)
		}
	}

	idx := &FastIVFIndex{nlist: 2048, nprobe: 32}

	fmt.Printf("Training IVF2048-p32 (10%% sample)...\n")
	trainStart := time.Now()
	idx.FastTrain(vectors, 0.1)
	fmt.Printf("Training: %v\n\n", time.Since(trainStart))

	fmt.Printf("Running 1000 searches...\n")
	latencies := make([]time.Duration, 1000)
	successCount := 0

	for i := 0; i < 1000; i++ {
		start := time.Now()
		_ = idx.Search(queries[i%100], 10)
		latencies[i] = time.Since(start)
		if latencies[i] < target {
			successCount++
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
	fmt.Printf("  Under %v: %d/%d (%.1f%%)\n", target, successCount, 1000, float64(successCount)/10.0)

	if latencies[990] <= target {
		fmt.Printf("\nSUCCESS: P99 %v <= target %v\n", latencies[990], target)
	} else {
		fmt.Printf("\nNOT MET: P99 %v > target %v\n", latencies[990], target)
	}
}
