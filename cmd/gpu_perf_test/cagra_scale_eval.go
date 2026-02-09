//go:build cagra

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

type EvalConfig struct {
	DatasetSize  int
	NumQueries   int
	TopK         int
	GraphDegree  int
	MaxIters     int
	BatchSize    int
	WarmupQueries int
}

type EvalResult struct {
	Config       EvalConfig
	BuildTimeMs  float64
	AvgLatencyUs float64
	P50LatencyUs float64
	P95LatencyUs float64
	P99LatencyUs float64
	QPS          float64
	BatchQPS     float64
	NDCG10       float64
	Recall10     float64
	MemoryMB     float64
}

type bruteForceResult struct {
	ID    int
	Score float32
}

func main() {
	var datasetSize int
	var numQueries int
	var graphDegree int
	var maxIters int
	var quick bool

	flag.IntVar(&datasetSize, "n", 100000, "dataset size")
	flag.IntVar(&numQueries, "q", 100, "number of queries")
	flag.IntVar(&graphDegree, "degree", 64, "CAGRA graph degree")
	flag.IntVar(&maxIters, "iters", 64, "CAGRA max iterations")
	flag.BoolVar(&quick, "quick", false, "quick mode with smaller dataset")
	flag.Parse()

	if quick {
		datasetSize = 10000
		numQueries = 50
	}

	fmt.Println("=== CAGRA GPU Scale Evaluation ===")
	fmt.Printf("Dataset: %d vectors, Queries: %d, TopK: 10\n", datasetSize, numQueries)
	fmt.Printf("CAGRA Config: degree=%d, iters=%d\n\n", graphDegree, maxIters)

	model, err := gobed.LoadInt8Model512()
	if err != nil {
		fmt.Printf("Failed to load model: %v\n", err)
		return
	}

	fmt.Println("Generating dataset...")
	vectors, scales := generateDataset(model, datasetSize)
	queryVecs, queryScales := generateQueries(model, numQueries)

	fmt.Println("Computing brute-force ground truth...")
	groundTruth := computeGroundTruth(vectors, scales, queryVecs, queryScales, 10)

	configs := []EvalConfig{
		{DatasetSize: datasetSize, NumQueries: numQueries, TopK: 10, GraphDegree: 32, MaxIters: 32, BatchSize: 32, WarmupQueries: 10},
		{DatasetSize: datasetSize, NumQueries: numQueries, TopK: 10, GraphDegree: 64, MaxIters: 64, BatchSize: 32, WarmupQueries: 10},
		{DatasetSize: datasetSize, NumQueries: numQueries, TopK: 10, GraphDegree: 128, MaxIters: 128, BatchSize: 32, WarmupQueries: 10},
	}

	if graphDegree != 64 {
		configs = []EvalConfig{
			{DatasetSize: datasetSize, NumQueries: numQueries, TopK: 10, GraphDegree: graphDegree, MaxIters: maxIters, BatchSize: 32, WarmupQueries: 10},
		}
	}

	results := make([]EvalResult, 0, len(configs))
	for _, cfg := range configs {
		result := runEval(cfg, vectors, scales, queryVecs, queryScales, groundTruth)
		results = append(results, result)
	}

	fmt.Println("\n=== Results Summary ===")
	fmt.Printf("%-12s %-10s %-12s %-12s %-12s %-10s %-10s %-10s\n",
		"Config", "Build(ms)", "Latency(us)", "P95(us)", "QPS", "BatchQPS", "NDCG@10", "Recall@10")
	fmt.Println(repeatStr("-", 100))

	for _, r := range results {
		fmt.Printf("deg=%3d/it=%3d %8.1f %10.1f %10.1f %10.0f %10.0f %9.4f %9.4f\n",
			r.Config.GraphDegree, r.Config.MaxIters,
			r.BuildTimeMs, r.AvgLatencyUs, r.P95LatencyUs, r.QPS, r.BatchQPS, r.NDCG10, r.Recall10)
	}
}

func runEval(cfg EvalConfig, vectors []simd.Vec512, scales []float32, queryVecs []simd.Vec512, queryScales []float32, groundTruth [][]int) EvalResult {
	fmt.Printf("\nEvaluating: degree=%d, iters=%d\n", cfg.GraphDegree, cfg.MaxIters)

	config := gobed.CAGRAConfig{
		MaxVectors:    cfg.DatasetSize,
		VectorDim:     512,
		GraphDegree:   cfg.GraphDegree,
		MaxIterations: cfg.MaxIters,
		CachePath:     "",
	}

	index, err := gobed.NewCAGRAIndex(config)
	if err != nil {
		fmt.Printf("  Failed to create index: %v\n", err)
		return EvalResult{Config: cfg}
	}
	defer index.Close()

	buildStart := time.Now()
	if err := index.BuildIndex(vectors[:cfg.DatasetSize], scales[:cfg.DatasetSize]); err != nil {
		fmt.Printf("  Failed to build index: %v\n", err)
		return EvalResult{Config: cfg}
	}
	buildTime := time.Since(buildStart)

	for i := 0; i < cfg.WarmupQueries; i++ {
		_, _ = index.Search(queryVecs[i%len(queryVecs)], queryScales[i%len(queryScales)], cfg.TopK)
	}

	latencies := make([]float64, cfg.NumQueries)
	cagraResults := make([][]int, cfg.NumQueries)

	for i := 0; i < cfg.NumQueries; i++ {
		start := time.Now()
		results, err := index.Search(queryVecs[i], queryScales[i], cfg.TopK)
		latency := time.Since(start)
		latencies[i] = float64(latency.Microseconds())

		if err != nil {
			fmt.Printf("  Search failed: %v\n", err)
			continue
		}

		ids := make([]int, len(results))
		for j, r := range results {
			ids[j] = r.ID
		}
		cagraResults[i] = ids
	}

	sort.Float64s(latencies)
	avgLatency := average(latencies)
	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)
	p99 := percentile(latencies, 0.99)

	batchStart := time.Now()
	numBatches := (cfg.NumQueries + cfg.BatchSize - 1) / cfg.BatchSize
	for b := 0; b < numBatches; b++ {
		start := b * cfg.BatchSize
		end := min(start+cfg.BatchSize, cfg.NumQueries)
		_, _ = index.SearchBatch(queryVecs[start:end], queryScales[start:end], cfg.TopK)
	}
	batchTime := time.Since(batchStart)
	batchQPS := float64(cfg.NumQueries) / batchTime.Seconds()

	ndcg10 := computeNDCG10(cagraResults, groundTruth)
	recall10 := computeRecall(cagraResults, groundTruth, cfg.TopK)

	stats := index.GetStats()

	result := EvalResult{
		Config:       cfg,
		BuildTimeMs:  float64(buildTime.Milliseconds()),
		AvgLatencyUs: avgLatency,
		P50LatencyUs: p50,
		P95LatencyUs: p95,
		P99LatencyUs: p99,
		QPS:          1000000.0 / avgLatency,
		BatchQPS:     batchQPS,
		NDCG10:       ndcg10,
		Recall10:     recall10,
		MemoryMB:     float64(stats.MemoryBytes) / (1024 * 1024),
	}

	fmt.Printf("  Build: %.1fms, Latency: %.1fus (p95: %.1fus), QPS: %.0f, NDCG@10: %.4f\n",
		result.BuildTimeMs, result.AvgLatencyUs, result.P95LatencyUs, result.QPS, result.NDCG10)

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

			for q := start; q < end; q++ {
				scores := make([]bruteForceResult, len(vectors))
				for i := 0; i < len(vectors); i++ {
					score := dotProductInt8(queries[q][:], vectors[i][:]) * queryScales[q] * scales[i]
					scores[i] = bruteForceResult{ID: i, Score: score}
				}
				sort.Slice(scores, func(i, j int) bool {
					return scores[i].Score > scores[j].Score
				})
				topIDs := make([]int, topK)
				for i := 0; i < topK && i < len(scores); i++ {
					topIDs[i] = scores[i].ID
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
