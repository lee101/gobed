package hnsw

import (
	"math/rand"
	"sort"
	"testing"

	"github.com/lee101/gobed/pkg/ann/simd"
)

func randomVecs(n int, seed int64) []simd.Vec512 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]simd.Vec512, n)
	for i := range v {
		for j := range v[i] {
			v[i][j] = int8(rng.Intn(255) - 127)
		}
	}
	return v
}

func TestSearchRecallAndRestore(t *testing.T) {
	vecs := randomVecs(1500, 1)
	h := NewHNSW(16, 100)
	for i := range vecs {
		h.Add(vecs[i], 1, i)
	}
	queries := randomVecs(50, 2)
	hit := 0
	for _, q := range queries {
		ids := make([]int, len(vecs))
		for i := range ids {
			ids[i] = i
		}
		sort.Slice(ids, func(a, b int) bool {
			return simd.L2Squared512(&q, &vecs[ids[a]]) < simd.L2Squared512(&q, &vecs[ids[b]])
		})
		truth := map[int]bool{}
		for _, id := range ids[:10] {
			truth[id] = true
		}
		res := h.Search(&q, 10)
		for i, r := range res {
			if truth[r.ID] {
				hit++
			}
			if i > 0 && r.Distance < res[i-1].Distance {
				t.Fatalf("results not sorted nearest first")
			}
		}
	}
	recall := float64(hit) / float64(10*len(queries))
	t.Logf("recall@10=%.3f", recall)
	if recall < 0.9 {
		t.Fatalf("recall too low: %.3f", recall)
	}

	scales := make([]float32, len(vecs))
	r := NewHNSW(16, 100)
	r.Restore(h.Export(), vecs, scales)
	for _, q := range queries {
		a, b := h.Search(&q, 10), r.Search(&q, 10)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("restored graph differs")
			}
		}
	}
}
