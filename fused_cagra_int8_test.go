//go:build fusedcagra
// +build fusedcagra

package gobed

import (
	"math"
	"sort"
	"testing"

	"github.com/lee101/gobed/pkg/ann/simd"
)

func TestFusedCAGRAInt8BatchQuality(t *testing.T) {
	if testing.Short() {
		t.Skip("int8 fused CAGRA quality test requires full model and GPU bindings")
	}

	model, err := LoadInt8Model512()
	if err != nil {
		t.Skipf("int8 model not available: %v", err)
	}

	if len(model.embeddings) == 0 || len(model.scales) == 0 {
		t.Fatalf("int8 model loaded but embeddings/scales are empty")
	}

	cfg := FusedCAGRAConfig{
		VocabSize:   len(model.embeddings),
		EmbedDim:    Int8EmbeddingDim,
		MaxVectors:  32,
		TopK:        3,
		GraphDegree: 32,
	}

	engine, err := NewFusedCAGRAEngine(cfg)
	if err != nil {
		t.Skipf("fused CAGRA engine unavailable: %v", err)
	}
	t.Cleanup(func() { engine.Close() })

	embedWeights := flattenEmbeddingsForTest(t, model.embeddings)

	datasetTexts := []string{
		"machine learning transforms data science",
		"deep learning advances computer vision research",
		"healthy vegan recipes for home cooking",
		"financial markets and stock trading strategies",
		"classical music history and famous composers",
		"quantum computing algorithms for qubits",
		"natural language processing with transformers",
		"reinforcement learning for robotics control",
	}

	dbVectors := make([]simd.Vec512, len(datasetTexts))
	dbScales := make([]float32, len(datasetTexts))
	dbFloatVectors := make([][]float32, len(datasetTexts))

	for i, text := range datasetTexts {
		result, err := model.EmbedInt8(text)
		if err != nil {
			t.Fatalf("failed to embed dataset text %q: %v", text, err)
		}
		if len(result.Vector) != Int8EmbeddingDim {
			t.Fatalf("unexpected embedding dimension for %q: got %d", text, len(result.Vector))
		}
		copy(dbVectors[i][:], result.Vector)
		dbScales[i] = result.Scale
		dbFloatVectors[i] = dequantizeVector(result.Vector, result.Scale)
	}

	if err := engine.BuildIndex(embedWeights, model.scales, dbVectors, dbScales); err != nil {
		t.Fatalf("failed to build fused CAGRA index: %v", err)
	}

	queryCases := []struct {
		text          string
		expectedTopID int
	}{
		{datasetTexts[0], 0},
		{"computer vision with deep neural networks", 1},
		{"plant-based cooking and vegan meals", 2},
		{"stock market technical analysis", 3},
		{"baroque era classical composers", 4},
		{"advanced quantum algorithms", 5},
		{"transformer architectures for language tasks", 6},
		{"robotics policies from reinforcement learning", 7},
	}

	tokenBatch := make([][]uint16, len(queryCases))
	tokenBatchInt16 := make([][]int16, len(queryCases))
	maxTokens := 0

	for i, qc := range queryCases {
		tokens, err := model.Tokenize(qc.text)
		if err != nil {
			t.Fatalf("failed to tokenize query %q: %v", qc.text, err)
		}
		if len(tokens) == 0 {
			t.Fatalf("tokenizer produced zero tokens for query %q", qc.text)
		}
		tokenBatchInt16[i] = append([]int16(nil), tokens...)

		uintTokens := make([]uint16, len(tokens))
		for j, token := range tokens {
			if token < 0 {
				t.Fatalf("unexpected negative token ID %d for query %q", token, qc.text)
			}
			uintTokens[j] = uint16(token)
		}
		tokenBatch[i] = uintTokens
		if len(tokens) > maxTokens {
			maxTokens = len(tokens)
		}
	}

	results, err := engine.SearchBatch(tokenBatch, maxTokens)
	if err != nil {
		t.Fatalf("fused batch search failed: %v", err)
	}

	if len(results) != len(queryCases) {
		t.Fatalf("expected %d result sets, got %d", len(queryCases), len(results))
	}

	for i, qc := range queryCases {
		if len(results[i]) == 0 {
			t.Fatalf("no results returned for query %q", qc.text)
		}

		queryVec, err := model.EmbedTokens(tokenBatchInt16[i])
		if err != nil {
			t.Fatalf("failed to embed tokens for query %q: %v", qc.text, err)
		}

		baselineScores := make([]float32, len(dbFloatVectors))
		for j := range dbFloatVectors {
    baselineScores[j] = fusedDotProduct(queryVec, dbFloatVectors[j])
		}

    baselineTop := fusedTopKIndices(baselineScores, engine.topK)
    fusedTop := fusedExtractIDs(results[i], engine.topK)

		if fusedTop[0] != baselineTop[0] {
			t.Errorf("query %q top result mismatch: fused=%d baseline=%d", qc.text, fusedTop[0], baselineTop[0])
		}

    if !fusedSameIDSet(fusedTop, baselineTop) {
			t.Errorf("query %q top-%d set mismatch: fused=%v baseline=%v", qc.text, engine.topK, fusedTop, baselineTop)
		}

    if qc.expectedTopID >= 0 && fusedTop[0] != qc.expectedTopID && !fusedContainsInt(fusedTop, qc.expectedTopID) {
			t.Errorf("query %q expected dataset index %d in top results, got %v", qc.text, qc.expectedTopID, fusedTop)
		}

		for j := 0; j < len(results[i]) && j < len(baselineTop); j++ {
			idx := results[i][j].ID
			expected := baselineScores[idx]
			actual := results[i][j].Similarity
            if !fusedWithinSimilarityTolerance(expected, actual) {
				t.Errorf("query %q result %d similarity mismatch: expected %.6f got %.6f",
					qc.text, idx, expected, actual)
			}
		}

		single, err := engine.Search(tokenBatch[i])
		if err != nil {
			t.Fatalf("single search failed for query %q: %v", qc.text, err)
		}
		if len(single) != len(results[i]) {
			t.Fatalf("single search returned %d results, batch returned %d", len(single), len(results[i]))
		}
		for j := range single {
			if single[j].ID != results[i][j].ID {
				t.Errorf("single vs batch result mismatch for query %q at rank %d: single=%d batch=%d",
					qc.text, j, single[j].ID, results[i][j].ID)
			}
		}
	}
}

func flattenEmbeddingsForTest(t *testing.T, embeddings [][]int8) []int8 {
	t.Helper()
	if len(embeddings) == 0 {
		t.Fatalf("no embeddings to flatten")
	}
	flat := make([]int8, len(embeddings)*Int8EmbeddingDim)
	offset := 0
	for i, row := range embeddings {
		if len(row) != Int8EmbeddingDim {
			t.Fatalf("embedding row %d has dimension %d, expected %d", i, len(row), Int8EmbeddingDim)
		}
		copy(flat[offset:offset+Int8EmbeddingDim], row)
		offset += Int8EmbeddingDim
	}
	return flat
}

func dequantizeVector(vec []int8, scale float32) []float32 {
	floatVec := make([]float32, len(vec))
	for i, v := range vec {
		floatVec[i] = float32(v) * scale
	}
	return floatVec
}

func fusedDotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("dotProduct length mismatch")
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func fusedTopKIndices(scores []float32, k int) []int {
    k = fusedMinIntLocal(k, len(scores))
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		if scores[indices[i]] == scores[indices[j]] {
			return indices[i] < indices[j]
		}
		return scores[indices[i]] > scores[indices[j]]
	})
	return indices[:k]
}

func fusedExtractIDs(results []SearchResult, k int) []int {
    k = fusedMinIntLocal(k, len(results))
	ids := make([]int, k)
	for i := 0; i < k; i++ {
		ids[i] = results[i].ID
	}
	return ids
}

func fusedSameIDSet(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	aCopy := append([]int(nil), a...)
	bCopy := append([]int(nil), b...)
	sort.Ints(aCopy)
	sort.Ints(bCopy)
	for i := range aCopy {
		if aCopy[i] != bCopy[i] {
			return false
		}
	}
	return true
}

func fusedContainsInt(slice []int, target int) bool {
	for _, v := range slice {
		if v == target {
			return true
		}
	}
	return false
}

func fusedWithinSimilarityTolerance(expected, actual float32) bool {
	diff := float32(math.Abs(float64(expected - actual)))
	base := float32(math.Abs(float64(expected)))
	tolerance := 0.02*base + 1e-3
	return diff <= tolerance
}

func fusedMinIntLocal(a, b int) int {
	if a < b {
		return a
	}
	return b
}
