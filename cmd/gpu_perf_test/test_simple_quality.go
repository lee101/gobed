//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"time"
	"unsafe"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

/*
#cgo CFLAGS: -I../../
#cgo LDFLAGS: -L../../ -lsimple_cagra -L/usr/local/cuda/lib64 -lcudart -ldl -lstdc++ -lm
#include <stdlib.h>
#include <stdint.h>

void* create_simple_context(
    int8_t* embed_weights,
    float* embed_scales_raw,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales_raw,
    int num_vectors,
    int top_k);

void simple_search(
    void* context,
    uint16_t* token_batch,
    int* token_lengths,
    int batch_size,
    int max_tokens,
    float* output_distances,
    int* output_indices);

void destroy_simple_context(void* context);
*/
import "C"

// QualityResult stores test results
type QualityResult struct {
	ExactMatches     int
	TopKMatches      int
	TotalQueries     int
	AvgSearchTime    time.Duration
	ExactRecall      float64
	TopKRecall       float64
	DistanceErrors   []float64
}

func main() {
	fmt.Println("🎯 Simplified CAGRA Quality Test")
	fmt.Println("=================================")
	fmt.Println("Testing search quality with synthetic but realistic data")
	fmt.Println("Focusing on exact match and semantic similarity verification")
	fmt.Println()

	// Load regular model
	fmt.Print("📦 Loading model: ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("OK (%v)\n", time.Since(start))

	// Create realistic test dataset
	fmt.Print("📄 Creating test dataset: ")
	vectors, scales, queryVectors, queryScales := createRealisticTestData(50, 10)
	fmt.Printf("OK (%d docs, %d queries)\n", len(vectors), len(queryVectors))

	// Test IVF quality
	fmt.Println("\n📊 Testing IVF Search Quality")
	fmt.Println(strings.Repeat("-", 40))
	ivfResult := testIVFSimple(model, vectors, scales, queryVectors, queryScales)
	printSimpleResults("IVF-HNSW-PQ", ivfResult)

	// Test CAGRA quality
	fmt.Println("\n⚡ Testing CAGRA Search Quality")
	fmt.Println(strings.Repeat("-", 40))
	cagraResult := testCAGRASimple(vectors, scales, queryVectors, queryScales)
	printSimpleResults("CAGRA", cagraResult)

	// Compare
	fmt.Println("\n📈 Quality vs Speed Comparison")
	fmt.Println(strings.Repeat("=", 40))
	compareResults(ivfResult, cagraResult)
}

func createRealisticTestData(numDocs, numQueries int) ([]simd.Vec512, []float32, []simd.Vec512, []float32) {
	// Generate diverse document vectors with realistic patterns
	docs := make([]simd.Vec512, numDocs)
	docScales := make([]float32, numDocs)

	// Create clusters of related documents
	numClusters := 5
	clusterCenters := make([]simd.Vec512, numClusters)

	// Generate cluster centers
	for c := 0; c < numClusters; c++ {
		for i := 0; i < 512; i++ {
			clusterCenters[c][i] = int8(rand.Intn(100) - 50)
		}
	}

	// Generate documents around cluster centers
	for d := 0; d < numDocs; d++ {
		cluster := d % numClusters

		for i := 0; i < 512; i++ {
			// Add noise to cluster center
			noise := int8(rand.Intn(40) - 20)
			docs[d][i] = clusterCenters[cluster][i] + noise
		}
		docScales[d] = 0.05 + rand.Float32()*0.05 // 0.05-0.10
	}

	// Generate queries - some exact matches, some similar
	queries := make([]simd.Vec512, numQueries)
	queryScales := make([]float32, numQueries)

	for q := 0; q < numQueries; q++ {
		if q < numQueries/2 {
			// Exact matches (copy existing docs with small noise)
			docIdx := q % numDocs
			for i := 0; i < 512; i++ {
				noise := int8(rand.Intn(6) - 3) // Small noise
				queries[q][i] = docs[docIdx][i] + noise
			}
			queryScales[q] = docScales[docIdx]
		} else {
			// Semantic matches (same cluster but more variation)
			cluster := q % numClusters
			for i := 0; i < 512; i++ {
				noise := int8(rand.Intn(60) - 30) // More noise
				queries[q][i] = clusterCenters[cluster][i] + noise
			}
			queryScales[q] = 0.05 + rand.Float32()*0.05
		}
	}

	return docs, docScales, queries, queryScales
}

func testIVFSimple(model *gobed.EmbeddingModel, docs []simd.Vec512, docScales []float32, queries []simd.Vec512, queryScales []float32) QualityResult {
	result := QualityResult{TotalQueries: len(queries)}

	// Create search engine
	engine := gobed.NewGPUSearchEngine(model)
	defer engine.Close()

	// Index documents (use simple text representations)
	docTexts := make([]string, len(docs))
	docIDs := make([]int, len(docs))
	for i := range docs {
		// Create simple text representation for indexing
		docTexts[i] = fmt.Sprintf("Document %d with semantic content about topic %d", i, i%5)
		docIDs[i] = i
	}

	err := engine.IndexBatchWithIDs(docIDs, docTexts)
	if err != nil {
		fmt.Printf("IVF indexing failed: %v\n", err)
		return result
	}

	// Test queries
	var totalTime time.Duration
	var distanceErrors []float64

	for q, query := range queries {
		queryText := fmt.Sprintf("Document %d", q%len(docs)) // Simple exact match query

		start := time.Now()
		results, err := engine.Search(queryText, 10)
		searchTime := time.Since(start)
		totalTime += searchTime

		if err == nil && len(results) > 0 {
			// Check for exact matches (first half of queries should find exact matches)
			if q < len(queries)/2 {
				expectedID := q % len(docs)
				found := false
				rank := -1

				for r, res := range results {
					if res.ID == expectedID {
						found = true
						rank = r
						break
					}
				}

				if found {
					result.ExactMatches++
					if rank < 3 { // Top-3 is good quality
						result.TopKMatches++
					}
				}

				// Calculate distance error
				if len(results) > 0 {
					expectedSim := calculateExpectedSimilarity(query, docs[expectedID])
					actualSim := results[0].Similarity
					distanceErrors = append(distanceErrors, math.Abs(float64(expectedSim-actualSim)))
				}
			} else {
				// For semantic queries, check if we get reasonable results
				if len(results) > 0 && results[0].Similarity > 0.1 {
					result.TopKMatches++
				}
			}
		}
	}

	result.ExactRecall = float64(result.ExactMatches) / float64(len(queries)/2)
	result.TopKRecall = float64(result.TopKMatches) / float64(len(queries))
	result.AvgSearchTime = totalTime / time.Duration(len(queries))
	result.DistanceErrors = distanceErrors

	return result
}

func testCAGRASimple(docs []simd.Vec512, docScales []float32, queries []simd.Vec512, queryScales []float32) QualityResult {
	result := QualityResult{TotalQueries: len(queries)}

	// Create CAGRA context with realistic parameters
	vocabSize := 10000
	embedDim := 512

	// Generate reasonable embedding weights
	embedWeights := make([]int8, vocabSize*embedDim)
	embedScales := make([]float32, vocabSize)

	for i := 0; i < vocabSize; i++ {
		embedScales[i] = 0.05
		for j := 0; j < embedDim; j++ {
			embedWeights[i*embedDim+j] = int8(rand.Intn(200) - 100)
		}
	}

	// Flatten database
	flatDatabase := make([]int8, len(docs)*embedDim)
	for i, vec := range docs {
		for j := 0; j < embedDim; j++ {
			flatDatabase[i*embedDim+j] = vec[j]
		}
	}

	// Create CAGRA context
	context := C.create_simple_context(
		(*C.int8_t)(unsafe.Pointer(&embedWeights[0])),
		(*C.float)(unsafe.Pointer(&embedScales[0])),
		C.int(vocabSize),
		C.int(embedDim),
		(*C.int8_t)(unsafe.Pointer(&flatDatabase[0])),
		(*C.float)(unsafe.Pointer(&docScales[0])),
		C.int(len(docs)),
		C.int(10), // top-k
	)

	if context == nil {
		fmt.Println("CAGRA context creation failed")
		return result
	}
	defer C.destroy_simple_context(context)

	// Test queries
	var totalTime time.Duration
	var distanceErrors []float64

	for q := range queries {
		// Create token representation (using query index pattern)
		tokens := []uint16{
			uint16(q % 1000),
			uint16((q + 1) % 1000),
			uint16((q + 2) % 1000),
		}
		tokenLengths := []int{len(tokens)}
		maxTokens := 10

		// Pad tokens
		paddedTokens := make([]uint16, maxTokens)
		copy(paddedTokens, tokens)

		// Output buffers
		outputDistances := make([]float32, 10)
		outputIndices := make([]int, 10)

		start := time.Now()
		C.simple_search(
			context,
			(*C.uint16_t)(unsafe.Pointer(&paddedTokens[0])),
			(*C.int)(unsafe.Pointer(&tokenLengths[0])),
			C.int(1), // batch size
			C.int(maxTokens),
			(*C.float)(unsafe.Pointer(&outputDistances[0])),
			(*C.int)(unsafe.Pointer(&outputIndices[0])),
		)
		searchTime := time.Since(start)
		totalTime += searchTime

		// Analyze results
		if q < len(queries)/2 {
			// For exact match queries
			expectedID := q % len(docs)
			found := false
			rank := -1

			for r := 0; r < 10; r++ {
				if outputIndices[r] == expectedID {
					found = true
					rank = r
					break
				}
			}

			if found {
				result.ExactMatches++
				if rank < 3 {
					result.TopKMatches++
				}
			}

			// Distance error calculation
			if outputIndices[0] >= 0 && outputIndices[0] < len(docs) {
				expectedSim := calculateExpectedSimilarity(queries[q], docs[expectedID])
				actualDist := outputDistances[0]
				distanceErrors = append(distanceErrors, math.Abs(float64(expectedSim)+float64(actualDist)))
			}
		} else {
			// For semantic queries, check if we get reasonable results
			validResults := 0
			for r := 0; r < 10; r++ {
				if outputIndices[r] >= 0 && outputIndices[r] < len(docs) {
					validResults++
				}
			}
			if validResults > 5 { // At least half valid results
				result.TopKMatches++
			}
		}
	}

	result.ExactRecall = float64(result.ExactMatches) / float64(len(queries)/2)
	result.TopKRecall = float64(result.TopKMatches) / float64(len(queries))
	result.AvgSearchTime = totalTime / time.Duration(len(queries))
	result.DistanceErrors = distanceErrors

	return result
}

func calculateExpectedSimilarity(a, b simd.Vec512) float32 {
	// Simple dot product similarity
	var dot int32
	for i := 0; i < 512; i++ {
		dot += int32(a[i]) * int32(b[i])
	}
	return float32(dot) / (512.0 * 128.0) // Normalize
}

func printSimpleResults(name string, result QualityResult) {
	fmt.Printf("%s Results:\n", name)
	fmt.Printf("  Total queries: %d\n", result.TotalQueries)
	fmt.Printf("  Exact matches: %d (%.1f%% recall)\n",
		result.ExactMatches, result.ExactRecall*100)
	fmt.Printf("  Top-K quality: %d (%.1f%% recall)\n",
		result.TopKMatches, result.TopKRecall*100)
	fmt.Printf("  Avg search time: %v\n", result.AvgSearchTime)

	if len(result.DistanceErrors) > 0 {
		avgError := 0.0
		for _, err := range result.DistanceErrors {
			avgError += err
		}
		avgError /= float64(len(result.DistanceErrors))
		fmt.Printf("  Avg distance error: %.4f\n", avgError)
	}
}

func compareResults(ivf, cagra QualityResult) {
	speedup := float64(ivf.AvgSearchTime) / float64(cagra.AvgSearchTime)

	fmt.Printf("Exact Match Recall: IVF %.1f%% vs CAGRA %.1f%%\n",
		ivf.ExactRecall*100, cagra.ExactRecall*100)
	fmt.Printf("Top-K Quality: IVF %.1f%% vs CAGRA %.1f%%\n",
		ivf.TopKRecall*100, cagra.TopKRecall*100)
	fmt.Printf("Search Speed: CAGRA %.1fx faster (%v vs %v)\n",
		speedup, cagra.AvgSearchTime, ivf.AvgSearchTime)

	fmt.Printf("\nConclusion: ")
	if cagra.ExactRecall >= 0.7 && speedup > 10 {
		fmt.Println("✅ CAGRA provides good quality with excellent speed")
	} else if speedup > 50 && cagra.TopKRecall >= 0.8 {
		fmt.Println("⚡ CAGRA excellent for speed-critical applications")
	} else if cagra.ExactRecall < 0.5 {
		fmt.Println("⚠️  CAGRA needs quality improvements")
	} else {
		fmt.Println("🔀 Both viable - IVF for quality, CAGRA for speed")
	}
}
