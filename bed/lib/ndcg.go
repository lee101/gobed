package lib

import "math"

// NDCG computes Normalized Discounted Cumulative Gain at K.
// relevances maps doc ID to relevance score (higher = more relevant).
// predicted is the ordered list of predicted doc IDs.
func NDCG(relevances map[int]float64, predicted []int, k int) float64 {
	if k <= 0 || len(predicted) == 0 {
		return 0
	}
	if k > len(predicted) {
		k = len(predicted)
	}

	// DCG for predicted ranking
	dcg := 0.0
	for i := 0; i < k; i++ {
		rel, exists := relevances[predicted[i]]
		if !exists {
			rel = 0
		}
		dcg += (math.Pow(2, rel) - 1) / math.Log2(float64(i+2))
	}

	// Ideal DCG (sort by relevance descending)
	type idRelPair struct {
		id  int
		rel float64
	}
	ideal := make([]idRelPair, 0, len(relevances))
	for id, rel := range relevances {
		ideal = append(ideal, idRelPair{id, rel})
	}
	// sort descending by relevance
	for i := 0; i < len(ideal); i++ {
		for j := i + 1; j < len(ideal); j++ {
			if ideal[j].rel > ideal[i].rel {
				ideal[i], ideal[j] = ideal[j], ideal[i]
			}
		}
	}

	idcg := 0.0
	for i := 0; i < min(k, len(ideal)); i++ {
		idcg += (math.Pow(2, ideal[i].rel) - 1) / math.Log2(float64(i+2))
	}

	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

// NDCGAtK computes NDCG@K for multiple queries.
// goldRelevances[i] is the relevance map for query i.
// predicted[i] is the predicted ranking for query i.
func NDCGAtK(goldRelevances []map[int]float64, predicted [][]int, k int) float64 {
	if len(goldRelevances) == 0 || len(predicted) == 0 {
		return 0
	}
	total := 0.0
	count := 0
	for i := 0; i < min(len(goldRelevances), len(predicted)); i++ {
		total += NDCG(goldRelevances[i], predicted[i], k)
		count++
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}
