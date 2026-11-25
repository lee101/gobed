package search

import (
	"bufio"
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// quantize mirrors gobed's embedding quantizer: symmetric absmax/127.
func quantize(v []float32) (simd.Vec512, float32) {
	var absMax float32
	for _, x := range v {
		if x < 0 {
			x = -x
		}
		if x > absMax {
			absMax = x
		}
	}
	scale := absMax / 127
	if scale == 0 {
		scale = 1
	}
	var out simd.Vec512
	for i := range out {
		out[i] = int8(math.Round(float64(v[i] / scale)))
	}
	return out, scale
}

// clusteredVectors draws n points around `topics` gaussian centres with a
// spread that leaves neighbouring topics overlapping, like text embeddings.
func clusteredVectors(n, topics int, seed int64) ([]simd.Vec512, []float32) {
	rng := rand.New(rand.NewSource(seed))
	centres := make([][]float32, topics)
	for c := range centres {
		centres[c] = make([]float32, 512)
		for j := range centres[c] {
			centres[c][j] = float32(rng.NormFloat64())
		}
	}
	vecs := make([]simd.Vec512, n)
	scales := make([]float32, n)
	buf := make([]float32, 512)
	for i := range vecs {
		c := centres[rng.Intn(topics)]
		for j := range buf {
			buf[j] = c[j] + float32(rng.NormFloat64())*0.8
		}
		vecs[i], scales[i] = quantize(buf)
	}
	return vecs, scales
}

// loadVectorFile reads uint32 n then n x (512 int8 + float32 scale), little endian.
func loadVectorFile(t testing.TB, path string) ([]simd.Vec512, []float32) {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		t.Fatal(err)
	}
	vecs := make([]simd.Vec512, n)
	scales := make([]float32, n)
	for i := range vecs {
		if err := binary.Read(r, binary.LittleEndian, &vecs[i]); err != nil {
			t.Fatal(err)
		}
		if err := binary.Read(r, binary.LittleEndian, &scales[i]); err != nil {
			t.Fatal(err)
		}
	}
	return vecs, scales
}

// benchQueries perturbs random dataset points so queries are near, but not
// identical to, indexed vectors.
func benchQueries(vecs []simd.Vec512, nq int, seed int64) []simd.Vec512 {
	rng := rand.New(rand.NewSource(seed))
	qs := make([]simd.Vec512, nq)
	for i := range qs {
		src := &vecs[rng.Intn(len(vecs))]
		for j := range qs[i] {
			v := int(src[j]) + rng.Intn(31) - 15
			qs[i][j] = int8(max(-127, min(127, v)))
		}
	}
	return qs
}

// bruteTopK is the exact int8 dot-product ranking the flat index uses.
func bruteTopK(vecs []simd.Vec512, q *simd.Vec512, k int) []int {
	type sc struct {
		id int
		s  int32
	}
	all := make([]sc, len(vecs))
	for i := range vecs {
		all[i] = sc{i, simd.Dot512(q, &vecs[i])}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
	out := make([]int, k)
	for i := range out {
		out[i] = all[i].id
	}
	return out
}

func recallAt(e *Engine, vecs []simd.Vec512, qs []simd.Vec512, k int) float64 {
	hit := 0
	for i := range qs {
		truth := map[int]bool{}
		for _, id := range bruteTopK(vecs, &qs[i], k) {
			truth[id] = true
		}
		res, _ := e.Search(&qs[i], k)
		for _, r := range res {
			if truth[r.ID] {
				hit++
			}
		}
	}
	return float64(hit) / float64(k*len(qs))
}

// prodAuxConfig is what gobed.SearchEngine builds for netwrck's aux engines
// (AutoMode=false, MaxExactSearchSize=50000, NumClusters unset): the config
// that took 41m to train 61894 audio assets in production.
func prodAuxConfig() Config {
	return Config{MaxFlatSize: 50000, M: 64, NBits: 8, UseParallel: true}
}

// TestColdStartBench times a cold build of a ~62k x 512 int8 index and its
// recall@10 against brute force. Opt-in: GOBED_COLDSTART=1. Set
// GOBED_BENCH_VECTORS to a file written in loadVectorFile's format to use
// real embeddings instead of synthetic clusters.
func TestColdStartBench(t *testing.T) {
	if os.Getenv("GOBED_COLDSTART") == "" {
		t.Skip("set GOBED_COLDSTART=1")
	}
	var vecs []simd.Vec512
	var scales []float32
	if p := os.Getenv("GOBED_BENCH_VECTORS"); p != "" {
		vecs, scales = loadVectorFile(t, p)
	} else {
		vecs, scales = clusteredVectors(61894, 600, 1)
	}
	ids := make([]int, len(vecs))
	for i := range ids {
		ids[i] = i
	}
	qs := benchQueries(vecs, 200, 2)

	configs := map[string]Config{"prod-aux": prodAuxConfig()}
	if os.Getenv("GOBED_COLDSTART_HNSW") != "" {
		c := DefaultConfig()
		c.MaxFlatSize = 50000
		configs["default-hnsw"] = c
	}
	for name, cfg := range configs {
		e := NewEngine(cfg)
		start := time.Now()
		if err := e.AddBatch(vecs, scales, ids); err != nil {
			t.Fatal(err)
		}
		total := time.Since(start)
		tt := e.LastTrainTimings()
		t.Logf("%s n=%d: total=%v ivf=%v pq=%v hnsw=%v add=%v nlist=%d",
			name, len(vecs), total, tt.IVF, tt.PQ, tt.HNSW,
			total-tt.IVF-tt.PQ-tt.HNSW, e.Stats().NLists)
		qStart := time.Now()
		for i := range qs {
			e.Search(&qs[i], 10)
		}
		perQuery := time.Since(qStart) / time.Duration(len(qs))
		t.Logf("%s recall@10=%.4f search=%v/query", name, recallAt(e, vecs, qs, 10), perQuery)

		if save := os.Getenv("GOBED_COLDSTART_SAVE"); save != "" {
			saveStart := time.Now()
			if err := e.SaveFile(save); err != nil {
				t.Fatal(err)
			}
			saved := time.Since(saveStart)
			loaded := NewEngine(cfg)
			loadStart := time.Now()
			if err := loaded.LoadFile(save); err != nil {
				t.Fatal(err)
			}
			t.Logf("%s save=%v load=%v recall@10(loaded)=%.4f", name, saved, time.Since(loadStart), recallAt(loaded, vecs, qs, 10))
		}
	}
}
