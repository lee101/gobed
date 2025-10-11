package metrics

import "testing"

func TestComputeNDCGPerfectRanking(t *testing.T) {
	relevance := map[int]float64{
		1: 3,
		2: 2,
		3: 1,
	}
	ranking := []int{1, 2, 3}

	ndcg := ComputeNDCG(ranking, relevance, 3)
	if ndcg < 0.999 {
		t.Fatalf("expected NDCG close to 1.0, got %.4f", ndcg)
	}
}

func TestComputeNDCGImperfectRanking(t *testing.T) {
	relevance := map[string]float64{
		"relA": 3,
		"relB": 2,
		"relC": 1,
	}
	ranking := []string{"relB", "relA", "other"}

	ndcg := ComputeNDCG(ranking, relevance, 3)
	if ndcg <= 0 || ndcg >= 1 {
		t.Fatalf("expected partial NDCG between 0 and 1, got %.4f", ndcg)
	}
}

func TestComputeNDCGRankingShorterThanK(t *testing.T) {
	relevance := map[int]float64{
		10: 4,
		20: 3,
		30: 2,
		40: 1,
	}
	ranking := []int{30, 10}

	ndcg := ComputeNDCG(ranking, relevance, 5)
	if ndcg <= 0 {
		t.Fatalf("expected positive NDCG, got %.4f", ndcg)
	}
}

func TestComputeNDCGNoRelevance(t *testing.T) {
	relevance := map[int]float64{}
	ranking := []int{1, 2, 3}

	if ndcg := ComputeNDCG(ranking, relevance, 3); ndcg != 0 {
		t.Fatalf("expected zero NDCG when no relevances, got %.4f", ndcg)
	}
}

func TestNDCGAtKHandlesDuplicates(t *testing.T) {
	relevance := map[string]float64{
		"a": 3,
		"b": 2,
		"c": 1,
	}

	ranking := []string{"a", "a", "b", "c"}

	score := NDCGAtK(ranking, relevance, 3)
	if score <= 0 {
		t.Fatalf("expected positive NDCG with duplicates removed, got %.4f", score)
	}

	scoreAll := NDCGAtK(ranking, relevance, 10)
	if scoreAll < score {
		t.Fatalf("expected NDCG with larger k to be >= smaller k, got %.4f vs %.4f", scoreAll, score)
	}
}
