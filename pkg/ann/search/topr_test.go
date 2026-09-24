package search

import (
	"math/rand"
	"sort"
	"sync"
	"testing"

	"github.com/lee101/gobed/pkg/ann/ivf"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// v1.0.4 implementations kept for equivalence checks.
func oldCollect(e *Engine, clusters []int) []int {
	set := make(map[int]bool)
	for _, c := range clusters {
		if c >= 0 && c < len(e.ivfIndex.Lists) {
			for _, idx := range e.ivfIndex.Lists[c] {
				set[idx] = true
			}
		}
	}
	out := make([]int, 0, len(set))
	for idx := range set {
		out = append(out, idx)
	}
	return out
}

type oldPair struct {
	idx   int
	score int32
}

func oldScoreExact(e *Engine, q *simd.Vec512, cands []int, R int) []int {
	if len(cands) == 0 {
		return nil
	}
	scores := make([]oldPair, len(cands))
	for i, idx := range cands {
		scores[i] = oldPair{idx, simd.Dot512(q, &e.rawVectors[idx])}
	}
	if R > len(scores) {
		R = len(scores)
	}
	for i := 0; i < R; i++ {
		m := i
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[m].score {
				m = j
			}
		}
		scores[i], scores[m] = scores[m], scores[i]
	}
	top := make([]int, R)
	for i := range top {
		top[i] = scores[i].idx
	}
	return top
}

func oldRerank(e *Engine, q *simd.Vec512, cands []int, k int) []SearchResult {
	if len(cands) == 0 {
		return nil
	}
	scores := make([]oldPair, len(cands))
	for i, idx := range cands {
		scores[i] = oldPair{idx, simd.Dot512(q, &e.rawVectors[idx])}
	}
	for i := 0; i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}
	if k > len(scores) {
		k = len(scores)
	}
	res := make([]SearchResult, k)
	for i := range res {
		res[i] = SearchResult{ID: e.ids[scores[i].idx], Score: float32(scores[i].score), Distance: float32(-scores[i].score)}
	}
	return res
}

func oldSearchClusters(e *Engine, q *simd.Vec512, clusters []int, k int) []SearchResult {
	c := oldCollect(e, clusters)
	return oldRerank(e, q, oldScoreExact(e, q, c, e.config.RerankSize), k)
}

// testEngine builds an IVF-shaped engine; small levels forces many score ties.
func testEngine(rng *rand.Rand, n, nlist, levels int) *Engine {
	e := &Engine{config: Config{RerankSize: 256}}
	e.rawVectors = make([]simd.Vec512, n)
	e.ids = make([]int, n)
	for i := range e.rawVectors {
		for d := 0; d < 512; d++ {
			e.rawVectors[i][d] = int8(rng.Intn(2*levels+1) - levels)
		}
		e.ids[i] = 1000000 + i
	}
	e.size = n
	e.ivfIndex = &ivf.IVFIndex{Lists: make([][]int, nlist), ListLocks: make([]sync.RWMutex, nlist)}
	for i := 0; i < n; i++ {
		c := rng.Intn(nlist)
		e.ivfIndex.Lists[c] = append(e.ivfIndex.Lists[c], i)
	}
	return e
}

func randQuery(rng *rand.Rand, levels int) *simd.Vec512 {
	var q simd.Vec512
	for d := 0; d < 512; d++ {
		q[d] = int8(rng.Intn(2*levels+1) - levels)
	}
	return &q
}

// sparseEngine makes ties very likely: vectors have one nonzero dim.
func sparseEngine(rng *rand.Rand, n, nlist int) *Engine {
	e := testEngine(rng, n, nlist, 0)
	for i := range e.rawVectors {
		e.rawVectors[i][rng.Intn(4)] = int8(rng.Intn(3))
	}
	return e
}

func refSearch(e *Engine, q *simd.Vec512, clusters []int, k int) []SearchResult {
	var cands []int
	seen := map[int]bool{}
	for _, c := range clusters {
		if c < 0 || c >= len(e.ivfIndex.Lists) {
			continue
		}
		for _, idx := range e.ivfIndex.Lists[c] {
			if !seen[idx] {
				seen[idx] = true
				cands = append(cands, idx)
			}
		}
	}
	ps := make([]oldPair, len(cands))
	for i, idx := range cands {
		ps[i] = oldPair{idx, simd.Dot512(q, &e.rawVectors[idx])}
	}
	sort.SliceStable(ps, func(a, b int) bool { return ps[a].score > ps[b].score })
	n := min(min(k, e.config.RerankSize), len(ps))
	if n <= 0 {
		return nil
	}
	res := make([]SearchResult, n)
	for i := range res {
		res[i] = SearchResult{ID: e.ids[ps[i].idx], Score: float32(ps[i].score), Distance: float32(-ps[i].score)}
	}
	return res
}

func TestSearchClustersEquivalence(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 300; trial++ {
		var e *Engine
		levels := []int{1, 3, 100}[trial%3]
		if trial%4 == 0 {
			e = sparseEngine(rng, 50+rng.Intn(3000), 1+rng.Intn(16))
		} else {
			e = testEngine(rng, 50+rng.Intn(3000), 1+rng.Intn(16), levels)
		}
		e.config.RerankSize = []int{0, 1, 7, 256, 5000}[rng.Intn(5)]
		nl := len(e.ivfIndex.Lists)
		clusters := make([]int, 1+rng.Intn(4))
		for i := range clusters {
			clusters[i] = rng.Intn(nl+2) - 1 // includes invalid and duplicate clusters
		}
		k := 1 + rng.Intn(300)
		q := randQuery(rng, levels)

		got := e.searchClusters(q, clusters, k)
		want := refSearch(e, q, clusters, k)
		old := oldSearchClusters(e, q, clusters, k)
		if len(got) != len(want) || len(got) != len(old) {
			t.Fatalf("trial %d: len got=%d ref=%d old=%d", trial, len(got), len(want), len(old))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d pos %d: got %+v ref %+v", trial, i, got[i], want[i])
			}
			if got[i].Score != old[i].Score {
				t.Fatalf("trial %d pos %d: score %v old %v", trial, i, got[i].Score, old[i].Score)
			}
		}
		// Old candidate order was map order, so ties are only comparable as sets
		// strictly above the cutoff score.
		if len(got) > 0 {
			cut := got[len(got)-1].Score
			a, b := map[int]bool{}, map[int]bool{}
			for i := range got {
				if got[i].Score > cut {
					a[got[i].ID] = true
				}
				if old[i].Score > cut {
					b[old[i].ID] = true
				}
			}
			for id := range a {
				if !b[id] || len(a) != len(b) {
					t.Fatalf("trial %d: id sets differ", trial)
				}
			}
		}
	}
}

func TestTopRFloatStable(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 500; trial++ {
		n := rng.Intn(2000)
		R := rng.Intn(300)
		xs := make([]float32, n)
		for i := range xs {
			xs[i] = float32(rng.Intn(1 + trial%50))
		}
		var h []scored[float32]
		for i, x := range xs {
			h = topRPush(h, R, scored[float32]{s: -x, pos: int32(i)})
		}
		topRSort(h)
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return xs[idx[a]] < xs[idx[b]] })
		m := min(R, n)
		if len(h) != m {
			t.Fatalf("len %d want %d", len(h), m)
		}
		for i := 0; i < m; i++ {
			if int(h[i].pos) != idx[i] {
				t.Fatalf("trial %d pos %d: got %d want %d", trial, i, h[i].pos, idx[i])
			}
		}
	}
}

func TestSearchClustersNoAllocsBeyondResult(t *testing.T) {
	if raceEnabled {
		t.Skip("sync.Pool drops items under -race")
	}
	rng := rand.New(rand.NewSource(3))
	e := testEngine(rng, 20000, 64, 100)
	q := randQuery(rng, 100)
	cl := []int{3, 9}
	e.searchClusters(q, cl, 10)
	a := testing.AllocsPerRun(50, func() { e.searchClusters(q, cl, 10) })
	if a > 1 {
		t.Fatalf("allocs/op = %v, want <= 1", a)
	}
}

func benchSetup() (*Engine, []*simd.Vec512, []int) {
	rng := rand.New(rand.NewSource(4))
	// Mirrors cutedsl prod: ~814k docs / 64 lists, nprobe 2 => ~25k candidates.
	e := testEngine(rng, 25600, 2, 100)
	e.config.RerankSize = 256
	qs := make([]*simd.Vec512, 64)
	for i := range qs {
		qs[i] = randQuery(rng, 100)
	}
	return e, qs, []int{0, 1}
}

func BenchmarkSearchClustersOld(b *testing.B) {
	e, qs, cl := benchSetup()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		oldSearchClusters(e, qs[i%len(qs)], cl, 20)
	}
}

func BenchmarkSearchClustersNew(b *testing.B) {
	e, qs, cl := benchSetup()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		e.searchClusters(qs[i%len(qs)], cl, 20)
	}
}
