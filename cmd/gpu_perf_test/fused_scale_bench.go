//go:build fusedcagra

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

type BenchConfig struct {
	DatasetSize int
	NumQueries  int
	TopK        int
	BatchSize   int
}

type BenchResult struct {
	DatasetSize   int
	BuildTimeMs   float64
	AvgLatencyUs  float64
	P50LatencyUs  float64
	P95LatencyUs  float64
	P99LatencyUs  float64
	SingleQPS     float64
	BatchQPS      float64
	NDCG10        float64
	Recall10      float64
	MemoryMB      float64
	VecsPerSecBld float64
}

func main() {
	var maxSize int
	var numQueries int
	var quick bool

	flag.IntVar(&maxSize, "n", 100000, "max dataset size")
	flag.IntVar(&numQueries, "q", 100, "number of queries")
	flag.BoolVar(&quick, "quick", false, "quick mode")
	flag.Parse()

	if quick {
		maxSize = 10000
		numQueries = 50
	}

	fmt.Println("=== Fused CAGRA GPU Scale Benchmark ===")
	fmt.Printf("Max dataset: %d, Queries: %d\n\n", maxSize, numQueries)

	model, err := gobed.LoadInt8Model512()
	if err != nil {
		fmt.Printf("Failed to load model: %v\n", err)
		return
	}

	fmt.Println("Generating dataset...")
	vectors, scales := generateDataset(model, maxSize)
	queryVecs, queryScales := generateQueries(model, numQueries)
	queryTokens := generateQueryTokens(numQueries)

	sizes := []int{1000, 5000, 10000, 25000, 50000, 100000}
	if maxSize > 100000 {
		sizes = append(sizes, 250000, 500000, 1000000)
	}

	validSizes := make([]int, 0)
	for _, s := range sizes {
		if s <= maxSize {
			validSizes = append(validSizes, s)
		}
	}

	results := make([]BenchResult, 0)

	for _, size := range validSizes {
		fmt.Printf("\n--- Testing dataset size: %d ---\n", size)

		fmt.Println("Computing ground truth (brute force)...")
		groundTruth := computeGroundTruth(vectors[:size], scales[:size], queryVecs, queryScales, 10)

		result := runFusedBench(model, vectors[:size], scales[:size], queryTokens, groundTruth, size, numQueries)
		results = append(results, result)
	}

	printResults(results)
}

func runFusedBench(model *gobed.Int8EmbeddingModel512, vectors []simd.Vec512, scales []float32,
	queryTokens [][]uint16, groundTruth [][]int, size, numQueries int) BenchResult {

	embedWeights, embedScales := model.GetInt8Weights()

	config := gobed.FusedCAGRAConfig{
		VocabSize:   len(embedScales),
		EmbedDim:    512,
		MaxVectors:  size,
		TopK:        10,
		GraphDegree: 32,
	}

	engine, err := gobed.NewFusedCAGRAEngine(config)
	if err != nil {
		fmt.Printf("Failed to create engine: %v\n", err)
		return BenchResult{DatasetSize: size}
	}
	defer engine.Close()

	buildStart := time.Now()
	if err := engine.BuildIndex(embedWeights, embedScales, vectors, scales); err != nil {
		fmt.Printf("Failed to build: %v\n", err)
		return BenchResult{DatasetSize: size}
	}
	buildTime := time.Since(buildStart)

	for i := 0; i < 10; i++ {
		_, _ = engine.Search(queryTokens[i%len(queryTokens)])
	}

	latencies := make([]float64, numQueries)
	searchResults := make([][]int, numQueries)

	for i := 0; i < numQueries; i++ {
		start := time.Now()
		results, err := engine.Search(queryTokens[i])
		latency := time.Since(start)
		latencies[i] = float64(latency.Microseconds())

		if err != nil {
			continue
		}
		ids := make([]int, len(results))
		for j, r := range results {
			ids[j] = r.ID
		}
		searchResults[i] = ids
	}

	sort.Float64s(latencies)
	avgLatency := average(latencies)
	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)
	p99 := percentile(latencies, 0.99)

	batchStart := time.Now()
	batchSize := 32
	numBatches := (numQueries + batchSize - 1) / batchSize
	for b := 0; b < numBatches; b++ {
		start := b * batchSize
		end := min(start+batchSize, numQueries)
		batch := queryTokens[start:end]
		maxLen := 0
		for _, t := range batch {
			if len(t) > maxLen {
				maxLen = len(t)
			}
		}
		_, _ = engine.SearchBatch(batch, maxLen+5)
	}
	batchTime := time.Since(batchStart)
	batchQPS := float64(numQueries) / batchTime.Seconds()

	ndcg10 := computeNDCG10(searchResults, groundTruth)
	recall10 := computeRecall(searchResults, groundTruth, 10)

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	result := BenchResult{
		DatasetSize:   size,
		BuildTimeMs:   float64(buildTime.Milliseconds()),
		AvgLatencyUs:  avgLatency,
		P50LatencyUs:  p50,
		P95LatencyUs:  p95,
		P99LatencyUs:  p99,
		SingleQPS:     1000000.0 / avgLatency,
		BatchQPS:      batchQPS,
		NDCG10:        ndcg10,
		Recall10:      recall10,
		MemoryMB:      float64(memStats.Alloc) / (1024 * 1024),
		VecsPerSecBld: float64(size) / buildTime.Seconds(),
	}

	fmt.Printf("Build: %.0fms (%.0f vec/s), Latency: %.0fus (p95: %.0fus), QPS: %.0f, NDCG@10: %.4f\n",
		result.BuildTimeMs, result.VecsPerSecBld, result.AvgLatencyUs, result.P95LatencyUs, result.SingleQPS, result.NDCG10)

	return result
}

func generateDataset(model *gobed.Int8EmbeddingModel512, size int) ([]simd.Vec512, []float32) {
	vectors := make([]simd.Vec512, size)
	scales := make([]float32, size)

	numWorkers := runtime.NumCPU()
	chunkSize := (size + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * chunkSize
			end := min(start+chunkSize, size)

			for i := start; i < end; i++ {
				tokens := tokensForIndex(i)
				floatVec, err := model.EmbedTokens(tokens)
				if err != nil {
					continue
				}
				int8Vec, scale := quantizeToInt8(floatVec)
				copy(vectors[i][:], int8Vec)
				scales[i] = scale
			}
		}(w)
	}
	wg.Wait()

	return vectors, scales
}

func generateQueries(model *gobed.Int8EmbeddingModel512, numQueries int) ([]simd.Vec512, []float32) {
	queries := make([]simd.Vec512, numQueries)
	scales := make([]float32, numQueries)

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < numQueries; i++ {
		tokens := []int16{int16(rng.Intn(gobed.Int8VocabSize)), int16(rng.Intn(gobed.Int8VocabSize))}
		floatVec, err := model.EmbedTokens(tokens)
		if err != nil {
			continue
		}
		int8Vec, scale := quantizeToInt8(floatVec)
		copy(queries[i][:], int8Vec)
		scales[i] = scale
	}

	return queries, scales
}

func generateQueryTokens(numQueries int) [][]uint16 {
	tokens := make([][]uint16, numQueries)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < numQueries; i++ {
		tokens[i] = []uint16{uint16(rng.Intn(gobed.Int8VocabSize)), uint16(rng.Intn(gobed.Int8VocabSize))}
	}
	return tokens
}

func computeGroundTruth(vectors []simd.Vec512, scales []float32, queries []simd.Vec512, queryScales []float32, topK int) [][]int {
	numQueries := len(queries)
	groundTruth := make([][]int, numQueries)

	numWorkers := runtime.NumCPU()
	chunkSize := (numQueries + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * chunkSize
			end := min(start+chunkSize, numQueries)

			type scored struct {
				id    int
				score float32
			}

			for q := start; q < end; q++ {
				scores := make([]scored, len(vectors))
				for i := 0; i < len(vectors); i++ {
					s := dotProductInt8(queries[q][:], vectors[i][:]) * queryScales[q] * scales[i]
					scores[i] = scored{id: i, score: s}
				}
				sort.Slice(scores, func(i, j int) bool {
					return scores[i].score > scores[j].score
				})
				topIDs := make([]int, topK)
				for i := 0; i < topK && i < len(scores); i++ {
					topIDs[i] = scores[i].id
				}
				groundTruth[q] = topIDs
			}
		}(w)
	}
	wg.Wait()

	return groundTruth
}

func dotProductInt8(a, b []int8) float32 {
	var sum int32
	for i := 0; i < len(a); i++ {
		sum += int32(a[i]) * int32(b[i])
	}
	return float32(sum)
}

func computeNDCG10(predicted [][]int, groundTruth [][]int) float64 {
	if len(predicted) != len(groundTruth) {
		return 0
	}

	var totalNDCG float64
	for q := 0; q < len(predicted); q++ {
		relevance := make(map[int]float64)
		for rank, id := range groundTruth[q] {
			relevance[id] = float64(len(groundTruth[q]) - rank)
		}

		dcg := 0.0
		for i, id := range predicted[q] {
			if rel, ok := relevance[id]; ok {
				dcg += rel / math.Log2(float64(i)+2)
			}
		}

		idcg := 0.0
		rels := make([]float64, 0, len(relevance))
		for _, rel := range relevance {
			rels = append(rels, rel)
		}
		sort.Float64s(rels)
		for i := len(rels) - 1; i >= 0 && (len(rels)-1-i) < 10; i-- {
			rank := len(rels) - 1 - i
			idcg += rels[i] / math.Log2(float64(rank)+2)
		}

		if idcg > 0 {
			totalNDCG += dcg / idcg
		}
	}

	return totalNDCG / float64(len(predicted))
}

func computeRecall(predicted [][]int, groundTruth [][]int, k int) float64 {
	if len(predicted) != len(groundTruth) {
		return 0
	}

	var totalRecall float64
	for q := 0; q < len(predicted); q++ {
		gtSet := make(map[int]struct{})
		for _, id := range groundTruth[q] {
			gtSet[id] = struct{}{}
		}

		hits := 0
		for _, id := range predicted[q] {
			if _, ok := gtSet[id]; ok {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(min(k, len(groundTruth[q])))
	}

	return totalRecall / float64(len(predicted))
}

func tokensForIndex(idx int) []int16 {
	vocab := gobed.Int8VocabSize
	tokens := []int16{int16(idx % vocab)}
	if idx >= vocab {
		tokens = append(tokens, int16((idx*31)%vocab))
	}
	if idx >= 2*vocab {
		tokens = append(tokens, int16((idx*17)%vocab))
	}
	return tokens
}

func quantizeToInt8(vec []float32) ([]int8, float32) {
	maxAbs := float32(0)
	for _, v := range vec {
		if abs := float32(math.Abs(float64(v))); abs > maxAbs {
			maxAbs = abs
		}
	}
	scale := maxAbs / 127.0
	if scale == 0 {
		scale = 1.0
	}

	out := make([]int8, len(vec))
	inv := 1 / scale
	for i, v := range vec {
		q := int(math.Round(float64(v * inv)))
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		out[i] = int8(q)
	}
	return out, scale
}

func average(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}

func printResults(results []BenchResult) {
	fmt.Println("\n=== SCALE BENCHMARK RESULTS ===")
	fmt.Printf("%-10s %-10s %-12s %-10s %-10s %-10s %-10s %-10s\n",
		"Size", "Build(ms)", "Vec/s", "Lat(us)", "P95(us)", "QPS", "NDCG@10", "Recall@10")
	fmt.Println(repeatStr("-", 90))

	for _, r := range results {
		fmt.Printf("%-10d %-10.0f %-12.0f %-10.0f %-10.0f %-10.0f %-10.4f %-10.4f\n",
			r.DatasetSize, r.BuildTimeMs, r.VecsPerSecBld, r.AvgLatencyUs, r.P95LatencyUs, r.SingleQPS, r.NDCG10, r.Recall10)
	}
}

func repeatStr(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
