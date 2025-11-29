package metrics

import (
	"math"
	"sort"
)

// ComputeNDCG calculates the Normalized Discounted Cumulative Gain at rank k.
// ranking contains ordered result identifiers, while relevance maps identifiers
// to non-negative relevance grades. When k <= 0 the cutoff defaults to the
// number of available relevance grades.
func ComputeNDCG[T comparable](ranking []T, relevance map[T]float64, k int) float64 {
	if len(relevance) == 0 {
		return 0
	}

	positive := make([]float64, 0, len(relevance))
	for _, score := range relevance {
		if score > 0 {
			positive = append(positive, score)
		}
	}
	if len(positive) == 0 {
		return 0
	}

	sort.Slice(positive, func(i, j int) bool {
		return positive[i] > positive[j]
	})

	if k <= 0 || k > len(positive) {
		k = len(positive)
	}

	actual := make([]float64, k)
	for i := 0; i < len(ranking) && i < k; i++ {
		if rel, ok := relevance[ranking[i]]; ok && rel > 0 {
			actual[i] = rel
		}
	}

	ideal := positive[:k]

	dcg := discountedGain(actual)
	idcg := discountedGain(ideal)
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

// NDCGAtK deduplicates the ranking sequence and computes NDCG at the given cutoff.
// This is useful when the ranking may contain multiple hits for the same logical
// document (e.g. chunked search results). The first occurrence is kept to preserve
// ordering and significance.
func NDCGAtK[T comparable](ranking []T, relevance map[T]float64, k int) float64 {
	if k == 0 || len(ranking) == 0 || len(relevance) == 0 {
		return 0
	}

	unique := make([]T, 0, min(k, len(ranking)))
	seen := make(map[T]struct{}, len(ranking))
	for _, item := range ranking {
		if _, ok := seen[item]; ok {
			continue
		}
		seen[item] = struct{}{}
		unique = append(unique, item)
		if len(unique) == k {
			break
		}
	}

	return ComputeNDCG(unique, relevance, k)
}

func discountedGain(relevances []float64) float64 {
	total := 0.0
	for i, rel := range relevances {
		if rel <= 0 {
			continue
		}
		total += rel / math.Log2(float64(i)+2)
	}
	return total
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
