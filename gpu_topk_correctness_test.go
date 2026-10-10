//go:build gpu && cuda

package gobed

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"testing"
	"time"
)

func requireGPUOptimizationTests(t *testing.T) {
	t.Helper()
	if os.Getenv("GOBED_GPU_OPT_TESTS") != "1" {
		t.Skip("set GOBED_GPU_OPT_TESTS=1 to enable GPU optimization tests")
	}
	if !IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}
}

func equalSearchResults(aIdx []int32, aScores []float32, bIdx []int32, bScores []float32) bool {
	if len(aIdx) != len(bIdx) || len(aScores) != len(bScores) {
		return false
	}
	for i := range aIdx {
		if aIdx[i] != bIdx[i] || aScores[i] != bScores[i] {
			return false
		}
	}
	return true
}

// int8BruteForceTopK computes the reference ranking on the CPU using the same
// scoring the kernel uses: dot(query, vec) * queryScale * vecScale.
func int8BruteForceTopK(vectors [][]int8, scales []float32, query []int8, queryScale float32, k int) ([]int32, []float32) {
	type sv struct {
		idx   int32
		score float32
	}
	all := make([]sv, len(vectors))
	for i, v := range vectors {
		var sum int32
		for j := range v {
			sum += int32(v[j]) * int32(query[j])
		}
		all[i] = sv{int32(i), float32(sum) * queryScale * scales[i]}
	}
	sort.SliceStable(all, func(a, b int) bool { return all[a].score > all[b].score })
	if k > len(all) {
		k = len(all)
	}
	idx := make([]int32, k)
	sc := make([]float32, k)
	for i := 0; i < k; i++ {
		idx[i] = all[i].idx
		sc[i] = all[i].score
	}
	return idx, sc
}

// TestGPUTopKMatchesCPU guards the persistent-scratch-buffer rewrite of
// thrust_topk: reusing d_sort_idx/d_sort_keys across queries must not leak
// state between searches or change the ranking.
func TestGPUTopKMatchesCPU(t *testing.T) {
	if !IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}
	const (
		dim = 128
		n   = 4096
		k   = 32
	)
	idxr, err := NewGPUIndexer(IndexConfig{VectorDim: dim})
	if err != nil {
		t.Skipf("could not create GPU indexer (likely out of device memory): %v", err)
	}
	defer idxr.Close()

	rng := rand.New(rand.NewSource(7))
	vectors := make([][]int8, n)
	scales := make([]float32, n)
	for i := range vectors {
		v := make([]int8, dim)
		for j := range v {
			v[j] = int8(rng.Intn(255) - 127)
		}
		vectors[i] = v
		scales[i] = 0.5 + rng.Float32()
	}
	if err := idxr.AddVectors(vectors, scales); err != nil {
		t.Fatalf("AddVectors: %v", err)
	}

	// Several distinct queries in a row: a stale-scratch bug would show up as
	// the second query returning the first query's ranking.
	for q := 0; q < 8; q++ {
		query := make([]int8, dim)
		for j := range query {
			query[j] = int8(rng.Intn(255) - 127)
		}
		queryScale := float32(0.25 + rng.Float32())

		gotIdx, gotScores, err := idxr.Search(query, queryScale, k)
		if err != nil {
			t.Fatalf("query %d: GPU search failed: %v", q, err)
		}
		if len(gotIdx) != k || len(gotScores) != k {
			t.Fatalf("query %d: expected %d results, got %d/%d", q, k, len(gotIdx), len(gotScores))
		}

		wantIdx, wantScores := int8BruteForceTopK(vectors, scales, query, queryScale, k)

		// Compare scores (ties may reorder equal-scoring ids, so scores are the
		// stable contract; the top-1 id must still match).
		for i := 0; i < k; i++ {
			if math.Abs(float64(gotScores[i]-wantScores[i])) > 1e-2*math.Max(1, math.Abs(float64(wantScores[i]))) {
				t.Fatalf("query %d rank %d: score %v, want %v (idx %d vs %d)",
					q, i, gotScores[i], wantScores[i], gotIdx[i], wantIdx[i])
			}
		}
		if gotIdx[0] != wantIdx[0] {
			t.Fatalf("query %d: top-1 idx %d, want %d", q, gotIdx[0], wantIdx[0])
		}
		// Scores must be non-increasing
		for i := 1; i < k; i++ {
			if gotScores[i] > gotScores[i-1]+1e-3 {
				t.Fatalf("query %d: results not sorted at rank %d: %v > %v", q, i, gotScores[i], gotScores[i-1])
			}
		}
	}
}

// TestGPUSearchRepeatableAcrossManyCalls hammers the reused buffers to make
// sure repeated searches stay consistent (this path used to cudaMalloc twice
// per query, so it is also the regression test for the allocation removal).
func TestGPUSearchRepeatableAcrossManyCalls(t *testing.T) {
	if !IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}
	const (
		dim = 128
		n   = 2048
		k   = 16
	)
	idxr, err := NewGPUIndexer(IndexConfig{VectorDim: dim})
	if err != nil {
		t.Skipf("could not create GPU indexer: %v", err)
	}
	defer idxr.Close()

	rng := rand.New(rand.NewSource(11))
	vectors := make([][]int8, n)
	scales := make([]float32, n)
	for i := range vectors {
		v := make([]int8, dim)
		for j := range v {
			v[j] = int8(rng.Intn(255) - 127)
		}
		vectors[i] = v
		scales[i] = 1.0
	}
	if err := idxr.AddVectors(vectors, scales); err != nil {
		t.Fatalf("AddVectors: %v", err)
	}

	query := make([]int8, dim)
	for j := range query {
		query[j] = int8(rng.Intn(255) - 127)
	}

	firstIdx, firstScores, err := idxr.Search(query, 1.0, k)
	if err != nil {
		t.Fatalf("initial search: %v", err)
	}
	for i := 0; i < 200; i++ {
		gotIdx, gotScores, err := idxr.Search(query, 1.0, k)
		if err != nil {
			t.Fatalf("search %d: %v", i, err)
		}
		for j := 0; j < k; j++ {
			if gotIdx[j] != firstIdx[j] || gotScores[j] != firstScores[j] {
				t.Fatalf("search %d diverged at rank %d: (%d,%v) vs (%d,%v)",
					i, j, gotIdx[j], gotScores[j], firstIdx[j], firstScores[j])
			}
		}
	}
}

func TestGPUReserveShrinkMatchesGeometricGrowth(t *testing.T) {
	requireGPUOptimizationTests(t)
	const (
		dim = 128
		n   = 8192
		k   = 17
	)

	rng := rand.New(rand.NewSource(23))
	vectors := make([][]int8, n)
	scales := make([]float32, n)
	for i := range vectors {
		vectors[i] = make([]int8, dim)
		for j := range vectors[i] {
			vectors[i][j] = int8(rng.Intn(255) - 127)
		}
		scales[i] = 0.5 + rng.Float32()
	}
	query := make([]int8, dim)
	for i := range query {
		query[i] = int8(rng.Intn(255) - 127)
	}

	unreserved, err := NewGPUIndexer(IndexConfig{VectorDim: dim})
	if err != nil {
		t.Skipf("could not create unreserved index: %v", err)
	}
	defer unreserved.Close()
	if err := unreserved.AddVectors(vectors, scales); err != nil {
		t.Fatalf("unreserved AddVectors: %v", err)
	}
	beforeShrink := unreserved.GetAllocatedMemoryUsage()
	wantIdx, wantScores, err := unreserved.Search(query, 0.75, k)
	if err != nil {
		t.Fatalf("unreserved search: %v", err)
	}
	if err := unreserved.ShrinkToFit(); err != nil {
		t.Fatalf("unreserved ShrinkToFit: %v", err)
	}
	afterShrink := unreserved.GetAllocatedMemoryUsage()
	if afterShrink >= beforeShrink || beforeShrink-afterShrink < afterShrink/2 {
		t.Fatalf("shrink was not material: before=%d after=%d", beforeShrink, afterShrink)
	}

	reserved, err := NewGPUIndexer(IndexConfig{VectorDim: dim})
	if err != nil {
		t.Skipf("could not create reserved index: %v", err)
	}
	defer reserved.Close()
	if err := reserved.Reserve(n); err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	reservedBytes := reserved.GetAllocatedMemoryUsage()
	if err := reserved.Reserve(n / 2); err != nil {
		t.Fatalf("smaller Reserve: %v", err)
	}
	if got := reserved.GetAllocatedMemoryUsage(); got != reservedBytes {
		t.Fatalf("smaller Reserve changed allocation: before=%d after=%d", reservedBytes, got)
	}
	if err := reserved.AddVectors(vectors, scales); err != nil {
		t.Fatalf("reserved AddVectors: %v", err)
	}
	if err := reserved.ShrinkToFit(); err != nil {
		t.Fatalf("reserved ShrinkToFit: %v", err)
	}
	gotIdx, gotScores, err := reserved.Search(query, 0.75, k)
	if err != nil {
		t.Fatalf("reserved search: %v", err)
	}
	if !equalSearchResults(gotIdx, gotScores, wantIdx, wantScores) {
		t.Fatal("reserved and unreserved indexes returned different search results")
	}

	wantBytes := uint64(n * (dim + 4))
	if got := reserved.GetAllocatedMemoryUsage(); got != wantBytes {
		t.Fatalf("reserved allocation=%d, want exact %d", got, wantBytes)
	}
}

func TestGPUConcurrentTokenSearchMatchesUnbuffered(t *testing.T) {
	requireGPUOptimizationTests(t)
	const (
		dim       = 64
		vocabSize = 257
		n         = 4096
		k         = 7
		queries   = 12
	)

	idx, err := NewGPUIndexer(IndexConfig{
		VectorDim: dim,
		VocabSize: vocabSize,
		EmbedDim:  dim,
	})
	if err != nil {
		t.Skipf("could not create GPU index: %v", err)
	}
	defer idx.Close()

	rng := rand.New(rand.NewSource(29))
	embeddings := make([]float32, vocabSize*dim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*2 - 1
	}
	if err := idx.LoadEmbeddings(embeddings); err != nil {
		t.Fatalf("LoadEmbeddings: %v", err)
	}

	vectors := make([][]int8, n)
	scales := make([]float32, n)
	for i := range vectors {
		vectors[i] = make([]int8, dim)
		for j := range vectors[i] {
			vectors[i][j] = int8(rng.Intn(255) - 127)
		}
		scales[i] = 0.25 + rng.Float32()
	}
	if err := idx.Reserve(n); err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	if err := idx.AddVectors(vectors, scales); err != nil {
		t.Fatalf("AddVectors: %v", err)
	}
	if err := idx.ShrinkToFit(); err != nil {
		t.Fatalf("ShrinkToFit: %v", err)
	}

	tokenQueries := make([][]int32, queries)
	for i := range tokenQueries {
		tokenQueries[i] = make([]int32, 24+i%5)
		for j := range tokenQueries[i] {
			tokenQueries[i][j] = int32(rng.Intn(vocabSize))
		}
	}

	checkConcurrent := func(label string) {
		wantIndices := make([][]int32, queries)
		wantScores := make([][]float32, queries)
		for i := range tokenQueries {
			wantIndices[i], wantScores[i], err = idx.searchWithTokensUnbuffered(tokenQueries[i], k)
			if err != nil {
				t.Fatalf("%s unbuffered query %d: %v", label, i, err)
			}
		}

		var wg sync.WaitGroup
		errs := make(chan error, queries*8)
		for round := 0; round < 8; round++ {
			for q := range tokenQueries {
				wg.Add(1)
				go func(queryIndex int) {
					defer wg.Done()
					gotIdx, gotScores, searchErr := idx.SearchWithTokens(tokenQueries[queryIndex], k)
					if searchErr != nil {
						errs <- searchErr
						return
					}
					if !equalSearchResults(
						gotIdx, gotScores, wantIndices[queryIndex], wantScores[queryIndex]) {
						errs <- fmt.Errorf("%s query %d differed from unbuffered result", label, queryIndex)
					}
				}(q)
			}
		}
		wg.Wait()
		close(errs)
		for searchErr := range errs {
			t.Fatal(searchErr)
		}
	}

	checkConcurrent("float embeddings")

	int8Embeddings := make([]int8, vocabSize*dim)
	embeddingScales := make([]float32, vocabSize)
	for i := range int8Embeddings {
		int8Embeddings[i] = int8(rng.Intn(255) - 127)
	}
	for i := range embeddingScales {
		embeddingScales[i] = 1
	}
	if err := idx.LoadEmbeddingsInt8(int8Embeddings, embeddingScales); err != nil {
		t.Fatalf("LoadEmbeddingsInt8: %v", err)
	}
	checkConcurrent("int8 embeddings")
}

func TestGPUOneMillionMemoryAndTokenLatency(t *testing.T) {
	if os.Getenv("GOBED_GPU_OPT_BENCH") != "1" {
		t.Skip("set GOBED_GPU_OPT_BENCH=1 to enable the 1M-vector GPU measurement")
	}
	requireGPUOptimizationTests(t)
	const (
		dim       = 512
		vocabSize = 512
		n         = 1_000_000
		batchSize = 10_000
		k         = 7
		samples   = 21
	)

	idx, err := NewGPUIndexer(IndexConfig{
		VectorDim: dim,
		VocabSize: vocabSize,
		EmbedDim:  dim,
	})
	if err != nil {
		t.Skipf("could not create GPU index: %v", err)
	}
	defer idx.Close()

	embeddings := make([]float32, vocabSize*dim)
	for i := range embeddings {
		embeddings[i] = float32((i%257)-128) / 128
	}
	if err := idx.LoadEmbeddings(embeddings); err != nil {
		t.Fatalf("LoadEmbeddings: %v", err)
	}
	if err := idx.Reserve(2 * n); err != nil {
		t.Skipf("insufficient free VRAM for 1M-vector measurement: %v", err)
	}
	for base := 0; base < n; base += batchSize {
		vectors := make([][]int8, batchSize)
		scales := make([]float32, batchSize)
		for i := range vectors {
			vectors[i] = make([]int8, dim)
			for j := range vectors[i] {
				vectors[i][j] = int8((base + i + j*17) % 127)
			}
			scales[i] = 1
		}
		if err := idx.AddVectors(vectors, scales); err != nil {
			t.Fatalf("AddVectors at %d: %v", base, err)
		}
	}

	beforeShrink := idx.GetAllocatedMemoryUsage()
	if err := idx.ShrinkToFit(); err != nil {
		t.Fatalf("ShrinkToFit: %v", err)
	}
	afterShrink := idx.GetAllocatedMemoryUsage()
	t.Logf("1M-vector allocation: before=%d after=%d saved=%d",
		beforeShrink, afterShrink, beforeShrink-afterShrink)

	tokens := []int32{1, 17, 31, 63, 127, 255, 511}
	measure := func(search func() error) time.Duration {
		durations := make([]time.Duration, samples)
		for i := range durations {
			start := time.Now()
			if err := search(); err != nil {
				t.Fatal(err)
			}
			durations[i] = time.Since(start)
		}
		sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
		return durations[len(durations)/2]
	}

	_, _, err = idx.SearchWithTokens(tokens, k)
	if err != nil {
		t.Fatalf("buffer warmup: %v", err)
	}
	unbuffered := measure(func() error {
		_, _, searchErr := idx.searchWithTokensUnbuffered(tokens, k)
		return searchErr
	})
	buffered := measure(func() error {
		_, _, searchErr := idx.SearchWithTokens(tokens, k)
		return searchErr
	})
	t.Logf("single-query median: unbuffered=%s buffered=%s", unbuffered, buffered)
}
