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

type OptSearchResult struct {
	ID    int
	Score int32
}

func optDot(a, b *simd.Vec512) int32 {
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

type OptIVFIndex struct {
	centroids       []simd.Vec512
	clusters        [][]int
	vectors         []simd.Vec512
	nlist           int
	nprobe          int
	maxPerCluster   int
	parallelSearch  bool
}

func NewOptIVFIndex(nlist, nprobe, maxPerCluster int) *OptIVFIndex {
	return &OptIVFIndex{
		nlist:          nlist,
		nprobe:         nprobe,
		maxPerCluster:  maxPerCluster,
		parallelSearch: true,
	}
}

func (idx *OptIVFIndex) Train(vectors []simd.Vec512) {
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

	sampleSize := idx.nlist * 50
	if sampleSize > n/10 {
		sampleSize = n / 10
	}
	sampleStep := n / sampleSize

	for iter := 0; iter < 2; iter++ {
		for i := range idx.clusters {
			idx.clusters[i] = idx.clusters[i][:0]
		}

		for s := 0; s < sampleSize; s++ {
			i := (s * sampleStep) % n
			bestCluster := 0
			bestScore := optDot(&vectors[i], &idx.centroids[0])
			for c := 1; c < idx.nlist; c++ {
				score := optDot(&vectors[i], &idx.centroids[c])
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
				bestScore := optDot(&vectors[i], &idx.centroids[0])
				for c := 1; c < idx.nlist; c++ {
					score := optDot(&vectors[i], &idx.centroids[c])
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

func (idx *OptIVFIndex) Search(query *simd.Vec512, k int) []OptSearchResult {
	centroidScores := make([]OptSearchResult, idx.nlist)
	for c := 0; c < idx.nlist; c++ {
		centroidScores[c] = OptSearchResult{ID: c, Score: optDot(query, &idx.centroids[c])}
	}
	sort.Slice(centroidScores, func(i, j int) bool {
		return centroidScores[i].Score > centroidScores[j].Score
	})

	if !idx.parallelSearch {
		return idx.searchSequential(query, k, centroidScores)
	}
	return idx.searchParallel(query, k, centroidScores)
}

func (idx *OptIVFIndex) searchSequential(query *simd.Vec512, k int, centroidScores []OptSearchResult) []OptSearchResult {
	results := make([]OptSearchResult, 0, k*2)

	for p := 0; p < idx.nprobe && p < idx.nlist; p++ {
		clusterID := centroidScores[p].ID
		cluster := idx.clusters[clusterID]

		limit := len(cluster)
		if idx.maxPerCluster > 0 && limit > idx.maxPerCluster {
			limit = idx.maxPerCluster
		}

		for i := 0; i < limit; i++ {
			vecIdx := cluster[i]
			score := optDot(query, &idx.vectors[vecIdx])
			results = append(results, OptSearchResult{ID: vecIdx, Score: score})
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

func (idx *OptIVFIndex) searchParallel(query *simd.Vec512, k int, centroidScores []OptSearchResult) []OptSearchResult {
	numWorkers := idx.nprobe
	if numWorkers > runtime.NumCPU() {
		numWorkers = runtime.NumCPU()
	}

	type localRes struct {
		results []OptSearchResult
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

			results := make([]OptSearchResult, 0, k)

			for p := startProbe; p < endProbe; p++ {
				clusterID := centroidScores[p].ID
				cluster := idx.clusters[clusterID]

				limit := len(cluster)
				if idx.maxPerCluster > 0 && limit > idx.maxPerCluster {
					limit = idx.maxPerCluster
				}

				for i := 0; i < limit; i++ {
					vecIdx := cluster[i]
					score := optDot(query, &idx.vectors[vecIdx])
					results = append(results, OptSearchResult{ID: vecIdx, Score: score})
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

	allResults := make([]OptSearchResult, 0, numWorkers*k)
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

func TestOptimizedMillionSearch(t *testing.T) {
	size := 1_000_000
	target := 5 * time.Millisecond

	fmt.Printf("\n=== Optimized 1M Search (target <%v) ===\n", target)
	fmt.Printf("CPU: %d cores\n\n", runtime.NumCPU())

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

	configs := []struct {
		nlist         int
		nprobe        int
		maxPerCluster int
		parallel      bool
		name          string
	}{
		{2048, 16, 0, true, "IVF2048-p16-par"},
		{2048, 16, 1000, true, "IVF2048-p16-max1k-par"},
		{4096, 16, 0, true, "IVF4096-p16-par"},
		{4096, 16, 500, true, "IVF4096-p16-max500-par"},
		{4096, 32, 500, true, "IVF4096-p32-max500-par"},
		{8192, 16, 250, true, "IVF8192-p16-max250-par"},
		{8192, 32, 250, true, "IVF8192-p32-max250-par"},
	}

	fmt.Printf("%-25s %-10s %-10s %-10s %-10s %-8s\n",
		"Config", "Train", "Avg", "P50", "P99", "QPS")

	for _, cfg := range configs {
		idx := NewOptIVFIndex(cfg.nlist, cfg.nprobe, cfg.maxPerCluster)
		idx.parallelSearch = cfg.parallel

		trainStart := time.Now()
		idx.Train(vectors)
		trainTime := time.Since(trainStart)

		latencies := make([]time.Duration, 1000)
		for i := 0; i < 1000; i++ {
			start := time.Now()
			_ = idx.Search(queries[i%100], 10)
			latencies[i] = time.Since(start)
		}
		sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

		var total time.Duration
		for _, l := range latencies {
			total += l
		}
		avg := total / 1000
		qps := 1000e9 / float64(total.Nanoseconds())

		status := ""
		if latencies[990] <= target {
			status = " OK"
		}

		fmt.Printf("%-25s %-10v %-10v %-10v %-10v %-8.0f%s\n",
			cfg.name, trainTime.Round(time.Second), avg.Round(time.Microsecond),
			latencies[500].Round(time.Microsecond), latencies[990].Round(time.Microsecond), qps, status)
	}
}

func TestBestConfig1M(t *testing.T) {
	size := 1_000_000
	target := 5 * time.Millisecond

	fmt.Printf("\n=== Best Config for 1M (target <%v) ===\n\n", target)

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

	idx := NewOptIVFIndex(8192, 32, 200)
	idx.parallelSearch = true

	fmt.Printf("Training IVF8192-p32-max200...\n")
	trainStart := time.Now()
	idx.Train(vectors)
	fmt.Printf("Training: %v\n\n", time.Since(trainStart))

	clusterSizes := make([]int, len(idx.clusters))
	for i, c := range idx.clusters {
		clusterSizes[i] = len(c)
	}
	sort.Ints(clusterSizes)
	fmt.Printf("Cluster sizes: min=%d median=%d max=%d\n\n",
		clusterSizes[0], clusterSizes[len(clusterSizes)/2], clusterSizes[len(clusterSizes)-1])

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
		fmt.Printf("Need more clusters or lower maxPerCluster\n")
	}
}
