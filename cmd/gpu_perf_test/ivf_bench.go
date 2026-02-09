//go:build cgo

package main

/*
#cgo CFLAGS: -I../../
#cgo LDFLAGS: -L../../ -lfused_cagra_opt -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm
#include <stdlib.h>
#include <stdint.h>

void* create_opt_context(
    int8_t* embed_weights,
    float* embed_scales,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales,
    int num_vectors,
    int top_k);

void build_ivf_index(void* context);

void opt_search(
    void* context,
    int8_t* query_batch,
    float* query_scales,
    int batch_size,
    float* output_distances,
    int* output_indices);

void destroy_opt_context(void* context);
*/
import "C"
import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"
	"unsafe"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	var maxSize int
	var numQueries int

	flag.IntVar(&maxSize, "n", 100000, "max dataset size")
	flag.IntVar(&numQueries, "q", 100, "number of queries")
	flag.Parse()

	fmt.Println("=== IVF GPU Search Benchmark ===")
	fmt.Printf("Max dataset: %d, Queries: %d\n\n", maxSize, numQueries)

	model, err := gobed.LoadInt8Model512()
	if err != nil {
		fmt.Printf("Failed to load model: %v\n", err)
		return
	}

	fmt.Println("Generating dataset...")
	vectors, scales := generateDataset(model, maxSize)
	queryVecs, queryScales := generateQueries(model, numQueries)

	sizes := []int{10000, 25000, 50000, 100000}
	if maxSize > 100000 {
		sizes = append(sizes, 250000, 500000)
	}

	validSizes := make([]int, 0)
	for _, s := range sizes {
		if s <= maxSize {
			validSizes = append(validSizes, s)
		}
	}

	for _, size := range validSizes {
		fmt.Printf("\n--- Testing dataset size: %d ---\n", size)

		fmt.Println("Computing ground truth...")
		groundTruth := computeGroundTruth(vectors[:size], scales[:size], queryVecs, queryScales, 10)

		runIVFBench(vectors[:size], scales[:size], queryVecs, queryScales, groundTruth, size, numQueries)
	}
}

func runIVFBench(vectors []simd.Vec512, scales []float32, queryVecs []simd.Vec512, queryScales []float32,
	groundTruth [][]int, size, numQueries int) {

	// Flatten vectors for C
	flatVectors := make([]int8, size*512)
	for i := 0; i < size; i++ {
		for j := 0; j < 512; j++ {
			flatVectors[i*512+j] = vectors[i][j]
		}
	}

	// Create context
	embedWeights := make([]int8, 30522*512)
	embedScales := make([]float32, 30522)
	for i := range embedScales {
		embedScales[i] = 1.0
	}

	buildStart := time.Now()
	ctx := C.create_opt_context(
		(*C.int8_t)(unsafe.Pointer(&embedWeights[0])),
		(*C.float)(unsafe.Pointer(&embedScales[0])),
		C.int(30522),
		C.int(512),
		(*C.int8_t)(unsafe.Pointer(&flatVectors[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int(size),
		C.int(10),
	)

	C.build_ivf_index(ctx)
	buildTime := time.Since(buildStart)

	// Warmup
	for i := 0; i < 10; i++ {
		queryBatch := make([]int8, 512)
		for j := 0; j < 512; j++ {
			queryBatch[j] = queryVecs[i%numQueries][j]
		}
		qScales := []float32{queryScales[i%numQueries]}
		dists := make([]float32, 10)
		indices := make([]int32, 10)
		C.opt_search(ctx,
			(*C.int8_t)(unsafe.Pointer(&queryBatch[0])),
			(*C.float)(unsafe.Pointer(&qScales[0])),
			C.int(1),
			(*C.float)(unsafe.Pointer(&dists[0])),
			(*C.int)(unsafe.Pointer(&indices[0])))
	}

	// Benchmark
	latencies := make([]float64, numQueries)
	searchResults := make([][]int, numQueries)

	for i := 0; i < numQueries; i++ {
		queryBatch := make([]int8, 512)
		for j := 0; j < 512; j++ {
			queryBatch[j] = queryVecs[i][j]
		}
		qScales := []float32{queryScales[i]}
		dists := make([]float32, 10)
		indices := make([]int32, 10)

		start := time.Now()
		C.opt_search(ctx,
			(*C.int8_t)(unsafe.Pointer(&queryBatch[0])),
			(*C.float)(unsafe.Pointer(&qScales[0])),
			C.int(1),
			(*C.float)(unsafe.Pointer(&dists[0])),
			(*C.int)(unsafe.Pointer(&indices[0])))
		latencies[i] = float64(time.Since(start).Microseconds())

		ids := make([]int, 10)
		for j := 0; j < 10; j++ {
			ids[j] = int(indices[j])
		}
		searchResults[i] = ids
	}

	C.destroy_opt_context(ctx)

	sort.Float64s(latencies)
	avgLatency := average(latencies)
	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)
	p99 := percentile(latencies, 0.99)

	ndcg10 := computeNDCG10(searchResults, groundTruth)
	recall10 := computeRecall(searchResults, groundTruth, 10)

	fmt.Printf("IVF Results:\n")
	fmt.Printf("  Build: %.0fms\n", float64(buildTime.Milliseconds()))
	fmt.Printf("  Latency: %.0fus (p50: %.0fus, p95: %.0fus, p99: %.0fus)\n",
		avgLatency, p50, p95, p99)
	fmt.Printf("  QPS: %.0f\n", 1000000.0/avgLatency)
	fmt.Printf("  NDCG@10: %.4f, Recall@10: %.4f\n", ndcg10, recall10)
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
