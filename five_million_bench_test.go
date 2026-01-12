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

type FiveMillionResult struct {
	ID    int
	Score int32
}

func fmDot(a, b *simd.Vec512) int32 {
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

type FiveMillionIVF struct {
	centroids       []simd.Vec512
	clusters        [][]int
	vectors         []simd.Vec512
	nlist           int
	nprobe          int
	maxPerCluster   int
}

func (idx *FiveMillionIVF) Train(vectors []simd.Vec512) {
	n := len(vectors)
	idx.vectors = vectors
	numWorkers := runtime.NumCPU()

	step := n / idx.nlist
	idx.centroids = make([]simd.Vec512, idx.nlist)
	for i := 0; i < idx.nlist; i++ {
		idx.centroids[i] = vectors[i*step]
	}

	idx.clusters = make([][]int, idx.nlist)
	for i := range idx.clusters {
		idx.clusters[i] = make([]int, 0, n/idx.nlist*2)
	}

	sampleSize := idx.nlist * 30
	if sampleSize > n/20 {
		sampleSize = n / 20
	}
	sampleStep := n / sampleSize

	for iter := 0; iter < 2; iter++ {
		for i := range idx.clusters {
			idx.clusters[i] = idx.clusters[i][:0]
		}

		for s := 0; s < sampleSize; s++ {
			i := (s * sampleStep) % n
			bestCluster := 0
			bestScore := fmDot(&vectors[i], &idx.centroids[0])
			for c := 1; c < idx.nlist; c++ {
				score := fmDot(&vectors[i], &idx.centroids[c])
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

	chunkSize := (n + numWorkers - 1) / numWorkers
	assignments := make([]int, n)

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
				bestScore := fmDot(&vectors[i], &idx.centroids[0])
				for c := 1; c < idx.nlist; c++ {
					score := fmDot(&vectors[i], &idx.centroids[c])
					if score > bestScore {
						bestScore = score
						bestCluster = c
					}
				}
				assignments[i] = bestCluster
			}
		}(w)
	}
	wg.Wait()

	for i := range idx.clusters {
		idx.clusters[i] = idx.clusters[i][:0]
	}
	for i, c := range assignments {
		idx.clusters[c] = append(idx.clusters[c], i)
	}
}

func (idx *FiveMillionIVF) Search(query *simd.Vec512, k int) []FiveMillionResult {
	centroidScores := make([]FiveMillionResult, idx.nlist)
	for c := 0; c < idx.nlist; c++ {
		centroidScores[c] = FiveMillionResult{ID: c, Score: fmDot(query, &idx.centroids[c])}
	}
	sort.Slice(centroidScores, func(i, j int) bool {
		return centroidScores[i].Score > centroidScores[j].Score
	})

	numWorkers := idx.nprobe
	if numWorkers > runtime.NumCPU() {
		numWorkers = runtime.NumCPU()
	}

	type localRes struct {
		results []FiveMillionResult
	}
	localResults := make([]localRes, numWorkers)

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

			results := make([]FiveMillionResult, 0, k)

			for p := startProbe; p < endProbe; p++ {
				clusterID := centroidScores[p].ID
				cluster := idx.clusters[clusterID]

				limit := len(cluster)
				if idx.maxPerCluster > 0 && limit > idx.maxPerCluster {
					limit = idx.maxPerCluster
				}

				for i := 0; i < limit; i++ {
					vecIdx := cluster[i]
					score := fmDot(query, &idx.vectors[vecIdx])
					results = append(results, FiveMillionResult{ID: vecIdx, Score: score})
				}
			}

			if len(results) > k {
				sort.Slice(results, func(a, b int) bool {
					return results[a].Score > results[b].Score
				})
				results = results[:k]
			}
			localResults[workerID].results = results
		}(w)
	}
	wg.Wait()

	allResults := make([]FiveMillionResult, 0, numWorkers*k)
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

func TestFiveMillionSearch(t *testing.T) {
	size := 5_000_000
	target := 10 * time.Millisecond

	fmt.Printf("\n=== 5M Vector Search (target <%v) ===\n", target)
	fmt.Printf("CPU: %d cores\n", runtime.NumCPU())
	fmt.Printf("Allocating %d vectors (%.1f GB)...\n\n", size, float64(size*512)/1e9)

	allocStart := time.Now()
	vectors := make([]simd.Vec512, size)
	for i := 0; i < size; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8((i*7 + j*13) % 256 - 128)
		}
	}
	fmt.Printf("Allocation: %v\n\n", time.Since(allocStart))

	queries := make([]*simd.Vec512, 100)
	for q := range queries {
		queries[q] = &simd.Vec512{}
		for j := 0; j < 512; j++ {
			queries[q][j] = int8((q*17 + j*31) % 256 - 128)
		}
	}

	configs := []struct {
		nlist         int
		nprobe        int
		maxPerCluster int
	}{
		{4096, 16, 2000},
		{8192, 16, 1000},
		{8192, 32, 500},
		{16384, 16, 500},
		{16384, 32, 300},
	}

	fmt.Printf("%-25s %-12s %-10s %-10s %-10s %-8s\n",
		"Config", "Train", "Avg", "P50", "P99", "QPS")

	for _, cfg := range configs {
		idx := &FiveMillionIVF{
			nlist:         cfg.nlist,
			nprobe:        cfg.nprobe,
			maxPerCluster: cfg.maxPerCluster,
		}

		trainStart := time.Now()
		idx.Train(vectors)
		trainTime := time.Since(trainStart)

		latencies := make([]time.Duration, 500)
		for i := 0; i < 500; i++ {
			start := time.Now()
			_ = idx.Search(queries[i%100], 10)
			latencies[i] = time.Since(start)
		}
		sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

		var total time.Duration
		for _, l := range latencies {
			total += l
		}
		avg := total / 500
		qps := 500e9 / float64(total.Nanoseconds())

		status := ""
		if latencies[495] <= target {
			status = " OK"
		}

		fmt.Printf("IVF%d-p%d-max%-13d %-12v %-10v %-10v %-10v %-8.0f%s\n",
			cfg.nlist, cfg.nprobe, cfg.maxPerCluster,
			trainTime.Round(time.Second), avg.Round(time.Microsecond),
			latencies[250].Round(time.Microsecond), latencies[495].Round(time.Microsecond), qps, status)
	}
}
