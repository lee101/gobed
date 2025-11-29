//go:build cagra && gpu

package gobed

import (
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

func TestCAGRAGPUSearchQuality(t *testing.T) {
	if !IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}
	if !isCAGRAAvailable() {
		t.Skip("CAGRA not available")
	}

	model, err := LoadInt8Model512()
	if err != nil {
		t.Skipf("int8 model unavailable: %v", err)
	}

	datasetTexts := []string{
		"machine learning transforms data science",
		"deep learning advances computer vision research",
		"healthy vegan recipes for home cooking",
		"financial markets and stock trading strategies",
		"classical music history and famous composers",
		"quantum computing algorithms for qubits",
		"natural language processing with transformers",
		"reinforcement learning for robotics control",
		"cybersecurity best practices for cloud infrastructure",
		"biotechnology breakthroughs in gene editing",
		"ancient history of the roman empire",
		"renewable energy storage with solar and wind",
	}

	dbVectors, dbScales, dbFloatVectors := buildCAGRATestDataset(t, model, datasetTexts)

	config := DefaultCAGRAConfig()
	config.MaxVectors = len(datasetTexts)
	config.VectorDim = Int8EmbeddingDim
	config.CachePath = ""

	index, err := NewCAGRAIndex(config)
	if err != nil {
		t.Fatalf("failed to create CAGRA index: %v", err)
	}
	defer index.Close()

	if err := index.BuildIndex(dbVectors, dbScales); err != nil {
		t.Fatalf("failed to build CAGRA index: %v", err)
	}

	type queryCase struct {
		text          string
		expectedTopID int
	}

	queryCases := []queryCase{
		{datasetTexts[0], 0},
		{"computer vision with deep neural networks", 1},
		{"plant-based cooking and vegan meals", 2},
		{"stock market technical analysis", 3},
		{"baroque era classical composers", 4},
		{"advanced quantum algorithms", 5},
		{"transformer architectures for language tasks", 6},
		{"robotics policies from reinforcement learning", 7},
		{"cloud security best practices", 8},
		{"crispr and gene editing advances", 9},
		{"roman empire ancient history", 10},
		{"solar and wind renewable storage", 11},
	}

	k := 10
    expectedLen := cgMin(k, len(datasetTexts))

	queryVectors := make([]simd.Vec512, len(queryCases))
	queryScales := make([]float32, len(queryCases))
	baselineScores := make([][]float32, len(queryCases))
	relevanceGrades := make([]map[int]float64, len(queryCases))

	for i, qc := range queryCases {
		int8Embedding, err := model.EmbedInt8(qc.text)
		if err != nil {
			t.Fatalf("failed to embed query %q: %v", qc.text, err)
		}
		var vec simd.Vec512
		copy(vec[:], int8Embedding.Vector)
		queryVectors[i] = vec
		queryScales[i] = int8Embedding.Scale

		tokens, err := model.Tokenize(qc.text)
		if err != nil {
			t.Fatalf("failed to tokenize query %q: %v", qc.text, err)
		}
		floatVec, err := model.EmbedTokens(tokens)
		if err != nil {
			t.Fatalf("failed to embed tokens for query %q: %v", qc.text, err)
		}

		scores := make([]float32, len(dbFloatVectors))
		for j := range dbFloatVectors {
			scores[j] = dotProduct(floatVec, dbFloatVectors[j])
		}
		baselineScores[i] = scores
		relevanceGrades[i] = buildRelevanceGrades(scores, k)
	}

	singleResults := make([][]SearchResult, len(queryCases))
	totalSingleNDCG := 0.0

	t.Run("SingleQueryAccuracy", func(t *testing.T) {
		for i, qc := range queryCases {
			start := time.Now()
			results, err := index.Search(queryVectors[i], queryScales[i], k)
			latency := time.Since(start)
			if err != nil {
				t.Fatalf("search failed for query %q: %v", qc.text, err)
			}
			if len(results) != expectedLen {
				t.Fatalf("expected %d results, got %d", expectedLen, len(results))
			}
			singleResults[i] = results

			expectedTop := topKIndices(baselineScores[i], expectedLen)
			gotTop := extractIDs(results, expectedLen)

			if qc.expectedTopID >= 0 && !containsInt(gotTop, qc.expectedTopID) {
				t.Errorf("query %q expected dataset index %d in top results, got %v", qc.text, qc.expectedTopID, gotTop)
			}
			if gotTop[0] != expectedTop[0] {
				t.Errorf("query %q top-1 mismatch: got %d, expected %d", qc.text, gotTop[0], expectedTop[0])
			}
			if !sameIDSet(gotTop, expectedTop) {
				t.Errorf("query %q top-%d set mismatch: got %v, expected %v", qc.text, expectedLen, gotTop, expectedTop)
			}

			for _, res := range results {
				expectedScore := baselineScores[i][res.ID]
				if !withinSimilarityTolerance(expectedScore, res.Similarity) {
					t.Errorf("query %q result %d similarity mismatch: expected %.6f got %.6f",
						qc.text, res.ID, expectedScore, res.Similarity)
				}
			}

			ndcg := computeNDCGFromResults(results, relevanceGrades[i], k)
			totalSingleNDCG += ndcg

			if latency > 15*time.Millisecond {
				t.Logf("query %q latency %v exceeds 15ms target", qc.text, latency)
			} else {
				t.Logf("query %q latency %v", qc.text, latency)
			}
		}
	})

	avgSingleNDCG := totalSingleNDCG / float64(len(queryCases))
	if avgSingleNDCG < 0.9 {
		t.Errorf("average single-query NDCG@10 too low: got %.3f, want >= 0.90", avgSingleNDCG)
	}
	t.Logf("Average single-query NDCG@10: %.3f", avgSingleNDCG)

	t.Run("BatchSearchConsistency", func(t *testing.T) {
		for i := range singleResults {
			if singleResults[i] == nil {
				t.Fatalf("missing single search baseline for query %d", i)
			}
		}

		batchResults, err := index.SearchBatch(queryVectors, queryScales, k)
		if err != nil {
			t.Fatalf("batch search failed: %v", err)
		}
		if len(batchResults) != len(queryCases) {
			t.Fatalf("expected %d batch result sets, got %d", len(queryCases), len(batchResults))
		}

		var totalBatchNDCG float64

		for i, qc := range queryCases {
			results := batchResults[i]
			if len(results) != expectedLen {
				t.Fatalf("query %q batch expected %d results, got %d", qc.text, expectedLen, len(results))
			}

			expectedTop := topKIndices(baselineScores[i], expectedLen)
			gotTop := extractIDs(results, expectedLen)

			if !sameIDSet(gotTop, expectedTop) {
				t.Errorf("query %q batch top-%d set mismatch: got %v, expected %v", qc.text, expectedLen, gotTop, expectedTop)
			}

			single := singleResults[i]
			for j := range results {
				if results[j].ID != single[j].ID {
					t.Errorf("query %q batch vs single mismatch at rank %d: batch=%d single=%d",
						qc.text, j, results[j].ID, single[j].ID)
				}
				expectedScore := baselineScores[i][results[j].ID]
				if !withinSimilarityTolerance(expectedScore, results[j].Similarity) {
					t.Errorf("query %q batch result %d similarity mismatch: expected %.6f got %.6f",
						qc.text, results[j].ID, expectedScore, results[j].Similarity)
				}
			}

			totalBatchNDCG += computeNDCGFromResults(results, relevanceGrades[i], k)
		}

		avgBatchNDCG := totalBatchNDCG / float64(len(queryCases))
		if avgBatchNDCG < 0.9 {
			t.Errorf("average batch-query NDCG@10 too low: got %.3f, want >= 0.90", avgBatchNDCG)
		}
		t.Logf("Average batch-query NDCG@10: %.3f", avgBatchNDCG)
	})

	stats := index.GetStats()
	if !stats.IsBuilt {
		t.Error("expected CAGRA index to be marked as built")
	}
	if stats.NumVectors != len(datasetTexts) {
		t.Errorf("expected %d vectors in stats, got %d", len(datasetTexts), stats.NumVectors)
	}
	if stats.SearchCount < int64(len(queryCases)) {
		t.Errorf("expected at least %d searches recorded, got %d", len(queryCases), stats.SearchCount)
	}
}

func buildCAGRATestDataset(tb testing.TB, model *Int8EmbeddingModel512, texts []string) ([]simd.Vec512, []float32, [][]float32) {
	tb.Helper()

	vectors := make([]simd.Vec512, len(texts))
	scales := make([]float32, len(texts))
	floatVectors := make([][]float32, len(texts))

	for i, text := range texts {
		int8Embedding, err := model.EmbedInt8(text)
		if err != nil {
			tb.Fatalf("failed to embed dataset text %q: %v", text, err)
		}
		var vec simd.Vec512
		copy(vec[:], int8Embedding.Vector)
		vectors[i] = vec
		scales[i] = int8Embedding.Scale
		floatVectors[i] = dequantizeVector(int8Embedding.Vector, int8Embedding.Scale)
	}

	return vectors, scales, floatVectors
}

func BenchmarkCAGRAGPUSearch(b *testing.B) {
	if !IsCUDAAvailable() {
		b.Skip("CUDA not available")
	}
	if !isCAGRAAvailable() {
		b.Skip("CAGRA not available")
	}

	model, err := LoadInt8Model512()
	if err != nil {
		b.Skipf("int8 model unavailable: %v", err)
	}

	datasetTexts := []string{
		"machine learning transforms data science",
		"deep learning advances computer vision research",
		"healthy vegan recipes for home cooking",
		"financial markets and stock trading strategies",
		"classical music history and famous composers",
		"quantum computing algorithms for qubits",
		"natural language processing with transformers",
		"reinforcement learning for robotics control",
		"cybersecurity best practices for cloud infrastructure",
		"biotechnology breakthroughs in gene editing",
		"ancient history of the roman empire",
		"renewable energy storage with solar and wind",
	}

	dbVectors, dbScales, dbFloatVectors := buildCAGRATestDataset(b, model, datasetTexts)

	config := DefaultCAGRAConfig()
	config.MaxVectors = len(datasetTexts)
	config.VectorDim = Int8EmbeddingDim
	config.CachePath = ""

	index, err := NewCAGRAIndex(config)
	if err != nil {
		b.Fatalf("failed to create CAGRA index: %v", err)
	}
	defer index.Close()

	if err := index.BuildIndex(dbVectors, dbScales); err != nil {
		b.Fatalf("failed to build CAGRA index: %v", err)
	}

	queryPhrases := []string{
		"transformers for language",
		"quantum computing algorithms",
		"vegan cooking at home",
		"financial trading strategies",
		"robotics reinforcement learning control",
		"cloud infrastructure security best practices",
		"gene editing breakthroughs",
		"roman empire history lessons",
		"renewable solar wind storage",
	}

	k := 10
	totalNDCG := 0.0

	queryVectors := make([]simd.Vec512, len(queryPhrases))
	queryScales := make([]float32, len(queryPhrases))
	relevanceGrades := make([]map[int]float64, len(queryPhrases))

	for i, text := range queryPhrases {
		int8Embedding, err := model.EmbedInt8(text)
		if err != nil {
			b.Fatalf("failed to embed query %q: %v", text, err)
		}
		copy(queryVectors[i][:], int8Embedding.Vector)
		queryScales[i] = int8Embedding.Scale

		tokens, err := model.Tokenize(text)
		if err != nil {
			b.Fatalf("failed to tokenize query %q: %v", text, err)
		}
		floatVec, err := model.EmbedTokens(tokens)
		if err != nil {
			b.Fatalf("failed to embed tokens for query %q: %v", text, err)
		}

		scores := make([]float32, len(dbFloatVectors))
		for j := range dbFloatVectors {
			scores[j] = dotProduct(floatVec, dbFloatVectors[j])
		}
		relevanceGrades[i] = buildRelevanceGrades(scores, k)
	}

	// Warmup to stabilize latency measurements.
	for i := range queryVectors {
		if _, err := index.Search(queryVectors[i], queryScales[i], k); err != nil {
			b.Fatalf("warmup search failed: %v", err)
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		q := i % len(queryVectors)
		results, err := index.Search(queryVectors[q], queryScales[q], k)
		if err != nil {
			b.Fatalf("search failed: %v", err)
		}
		totalNDCG += computeNDCGFromResults(results, relevanceGrades[q], k)
	}

	totalQueries := float64(b.N)
	if b.Elapsed().Seconds() > 0 {
		b.ReportMetric(totalQueries/b.Elapsed().Seconds(), "queries/sec")
		b.ReportMetric(b.Elapsed().Seconds()/totalQueries*1000, "ms/query")
	}
	if b.N > 0 {
		b.ReportMetric(totalNDCG/float64(b.N), "ndcg@10")
	}
}

func BenchmarkCAGRAGPUBatchSearch(b *testing.B) {
	if !IsCUDAAvailable() {
		b.Skip("CUDA not available")
	}
	if !isCAGRAAvailable() {
		b.Skip("CAGRA not available")
	}

	model, err := LoadInt8Model512()
	if err != nil {
		b.Skipf("int8 model unavailable: %v", err)
	}

	datasetTexts := []string{
		"machine learning transforms data science",
		"deep learning advances computer vision research",
		"healthy vegan recipes for home cooking",
		"financial markets and stock trading strategies",
		"classical music history and famous composers",
		"quantum computing algorithms for qubits",
		"natural language processing with transformers",
		"reinforcement learning for robotics control",
		"cybersecurity best practices for cloud infrastructure",
		"biotechnology breakthroughs in gene editing",
		"ancient history of the roman empire",
		"renewable energy storage with solar and wind",
	}

	dbVectors, dbScales, dbFloatVectors := buildCAGRATestDataset(b, model, datasetTexts)

	config := DefaultCAGRAConfig()
	config.MaxVectors = len(datasetTexts)
	config.VectorDim = Int8EmbeddingDim
	config.CachePath = ""

	index, err := NewCAGRAIndex(config)
	if err != nil {
		b.Fatalf("failed to create CAGRA index: %v", err)
	}
	defer index.Close()

	if err := index.BuildIndex(dbVectors, dbScales); err != nil {
		b.Fatalf("failed to build CAGRA index: %v", err)
	}

	baseQueries := []string{
		"language transformer models",
		"quantum computing research",
		"vegan cooking recipes",
		"financial stock analysis",
		"reinforcement learning robotics",
		"classical composers history",
		"deep learning vision",
		"machine learning data science",
		"cloud cybersecurity architecture",
		"gene editing biotechnology",
		"roman empire wars",
		"renewable storage batteries",
	}

	k := 10

	precomputedVecs := make([]simd.Vec512, len(baseQueries))
	precomputedScales := make([]float32, len(baseQueries))
	precomputedRelevance := make([]map[int]float64, len(baseQueries))

	for i, text := range baseQueries {
		int8Embedding, err := model.EmbedInt8(text)
		if err != nil {
			b.Fatalf("failed to embed batch query %q: %v", text, err)
		}
		copy(precomputedVecs[i][:], int8Embedding.Vector)
		precomputedScales[i] = int8Embedding.Scale

		tokens, err := model.Tokenize(text)
		if err != nil {
			b.Fatalf("failed to tokenize batch query %q: %v", text, err)
		}
		floatVec, err := model.EmbedTokens(tokens)
		if err != nil {
			b.Fatalf("failed to embed tokens for batch query %q: %v", text, err)
		}

		scores := make([]float32, len(dbFloatVectors))
		for j := range dbFloatVectors {
			scores[j] = dotProduct(floatVec, dbFloatVectors[j])
		}
		precomputedRelevance[i] = buildRelevanceGrades(scores, k)
	}

	batchSizes := []int{8, 32, 64}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("batch_%d", batchSize), func(b *testing.B) {
			batchVecs := make([]simd.Vec512, batchSize)
			batchScales := make([]float32, batchSize)
			batchRelevance := make([]map[int]float64, batchSize)
			for i := 0; i < batchSize; i++ {
				src := i % len(precomputedVecs)
				batchVecs[i] = precomputedVecs[src]
				batchScales[i] = precomputedScales[src]
				batchRelevance[i] = precomputedRelevance[src]
			}

			// Warmup batch execution.
			for i := 0; i < 3; i++ {
				if _, err := index.SearchBatch(batchVecs, batchScales, k); err != nil {
					b.Fatalf("warmup batch search failed: %v", err)
				}
			}

			totalNDCG := 0.0

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				results, err := index.SearchBatch(batchVecs, batchScales, k)
				if err != nil {
					b.Fatalf("batch search failed: %v", err)
				}
				for q := range results {
					totalNDCG += computeNDCGFromResults(results[q], batchRelevance[q], k)
				}
			}

			totalQueries := float64(b.N * batchSize)
			if b.Elapsed().Seconds() > 0 {
				b.ReportMetric(totalQueries/b.Elapsed().Seconds(), "queries/sec")
				b.ReportMetric(float64(batchSize), "batch_size")
				b.ReportMetric(b.Elapsed().Seconds()/float64(b.N)*1000, "ms/batch")
			}
			if b.N > 0 && batchSize > 0 {
				b.ReportMetric(totalNDCG/float64(b.N*batchSize), "ndcg@10")
			}
		})
	}
}

func buildRelevanceGrades(baseline []float32, k int) map[int]float64 {
    top := topKIndices(baseline, cgMin(k, len(baseline)))
	if len(top) == 0 {
		return map[int]float64{}
	}
	grades := make(map[int]float64, len(top))
	for i, id := range top {
		grades[id] = float64(len(top) - i)
	}
	return grades
}

func computeNDCGFromResults(results []SearchResult, relevance map[int]float64, k int) float64 {
	if len(results) == 0 || len(relevance) == 0 {
		return 0
	}

    actual := make([]float64, 0, cgMin(k, len(results)))
	for i := 0; i < len(results) && i < k; i++ {
		actual = append(actual, relevance[results[i].ID])
	}

	if len(actual) == 0 {
		return 0
	}

	ideal := make([]float64, 0, len(relevance))
	for _, grade := range relevance {
		if grade > 0 {
			ideal = append(ideal, grade)
		}
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(ideal)))
	if len(ideal) > len(actual) {
		ideal = ideal[:len(actual)]
	}

	return computeNDCG(actual, ideal)
}

func computeNDCG(actual, ideal []float64) float64 {
	if len(actual) == 0 || len(ideal) == 0 {
		return 0
	}
	dcg := discountedGain(actual)
	idcg := discountedGain(ideal)
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func discountedGain(grades []float64) float64 {
	var sum float64
	for i, grade := range grades {
		if grade <= 0 {
			continue
		}
		sum += grade / math.Log2(float64(i)+2)
	}
	return sum
}

// Local helpers for test expectations (avoid name collisions)
func dotProduct(a, b []float32) float32 {
    if len(a) != len(b) {
        panic("dotProduct length mismatch")
    }
    var sum float32
    for i := range a {
        sum += a[i] * b[i]
    }
    return sum
}

func topKIndices(scores []float32, k int) []int {
    if k > len(scores) { k = len(scores) }
    idx := make([]int, len(scores))
    for i := range idx { idx[i] = i }
    sort.Slice(idx, func(i, j int) bool {
        if scores[idx[i]] == scores[idx[j]] { return idx[i] < idx[j] }
        return scores[idx[i]] > scores[idx[j]]
    })
    return idx[:k]
}

func extractIDs(results []SearchResult, k int) []int {
    if k > len(results) { k = len(results) }
    ids := make([]int, k)
    for i := 0; i < k; i++ { ids[i] = results[i].ID }
    return ids
}

func sameIDSet(a, b []int) bool {
    if len(a) != len(b) { return false }
    a2 := append([]int(nil), a...)
    b2 := append([]int(nil), b...)
    sort.Ints(a2); sort.Ints(b2)
    for i := range a2 { if a2[i] != b2[i] { return false } }
    return true
}

func containsInt(slice []int, target int) bool {
    for _, v := range slice { if v == target { return true } }
    return false
}

func withinSimilarityTolerance(expected, actual float32) bool {
    diff := float32(math.Abs(float64(expected - actual)))
    base := float32(math.Abs(float64(expected)))
    tol := 0.02*base + 1e-3
    return diff <= tol
}

func cgMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}
