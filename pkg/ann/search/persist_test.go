package search

import (
	"bytes"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func buildEngine(t *testing.T, cfg Config, n int) (*Engine, []int) {
	t.Helper()
	vecs, scales := clusteredVectors(n, 40, 7)
	ids := make([]int, n)
	for i := range ids {
		ids[i] = 1000 + i
	}
	e := NewEngine(cfg)
	if err := e.AddBatch(vecs, scales, ids); err != nil {
		t.Fatal(err)
	}
	return e, ids
}

func sameResults(t *testing.T, a, b *Engine, n int) {
	t.Helper()
	vecs, _ := clusteredVectors(n, 40, 7)
	for _, q := range benchQueries(vecs, 25, 3) {
		ra, err := a.Search(&q, 10)
		if err != nil {
			t.Fatal(err)
		}
		rb, err := b.Search(&q, 10)
		if err != nil {
			t.Fatal(err)
		}
		if len(ra) != len(rb) || len(ra) == 0 {
			t.Fatalf("result counts differ: %d vs %d", len(ra), len(rb))
		}
		for i := range ra {
			if ra[i] != rb[i] {
				t.Fatalf("result %d differs: %+v vs %+v", i, ra[i], rb[i])
			}
		}
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	cases := map[string]Config{
		"ivf-pq":      {MaxFlatSize: 1000, M: 64, UseParallel: true},
		"ivf-pq-hnsw": {MaxFlatSize: 1000, M: 64, HNSWEnabled: true, HNSWM: 8, HNSWEfC: 50, UseParallel: true},
		"flat":        {MaxFlatSize: 50000, UseParallel: true},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			n := 17000
			orig, _ := buildEngine(t, cfg, n)
			var buf bytes.Buffer
			if err := orig.Save(&buf); err != nil {
				t.Fatal(err)
			}
			loaded := NewEngine(cfg)
			start := time.Now()
			if err := loaded.Load(bytes.NewReader(buf.Bytes())); err != nil {
				t.Fatal(err)
			}
			t.Logf("load %d bytes in %v", buf.Len(), time.Since(start))
			if loaded.trained != orig.trained || loaded.size != orig.size {
				t.Fatalf("state differs")
			}
			if (orig.ivfIndex == nil) != (loaded.ivfIndex == nil) || (orig.pq == nil) != (loaded.pq == nil) || (orig.hnswRouter == nil) != (loaded.hnswRouter == nil) {
				t.Fatalf("components differ")
			}
			if loaded.ContentHash() != orig.ContentHash() {
				t.Fatalf("content hash differs")
			}
			sameResults(t, orig, loaded, n)
		})
	}
}

func TestLoadConfigMismatch(t *testing.T) {
	cfg := Config{MaxFlatSize: 1000, M: 64, UseParallel: true}
	orig, _ := buildEngine(t, cfg, 4000)
	path := filepath.Join(t.TempDir(), "idx.bin")
	if err := orig.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	other := cfg
	other.NList = 64
	e := NewEngine(other)
	if err := e.LoadFile(path); !errors.Is(err, ErrIndexMismatch) {
		t.Fatalf("want ErrIndexMismatch, got %v", err)
	}
	if e.size != 0 || e.trained {
		t.Fatalf("engine modified by failed load")
	}
}

func TestLoadCorrupt(t *testing.T) {
	cfg := Config{MaxFlatSize: 1000, M: 64, UseParallel: true}
	orig, _ := buildEngine(t, cfg, 4000)
	var buf bytes.Buffer
	if err := orig.Save(&buf); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()
	for _, mutate := range []func([]byte) []byte{
		func(b []byte) []byte { c := append([]byte(nil), b...); c[len(c)/2] ^= 1; return c },
		func(b []byte) []byte { return b[:len(b)-10] },
		func(b []byte) []byte { return []byte("not an index at all") },
	} {
		e := NewEngine(cfg)
		if err := e.Load(bytes.NewReader(mutate(data))); !errors.Is(err, ErrIndexCorrupt) {
			t.Fatalf("want ErrIndexCorrupt, got %v", err)
		}
	}
}

func TestAutoNList(t *testing.T) {
	for _, c := range []struct{ n, want int }{{61894, 995}, {1000, 125}, {100, 12}, {10_000_000, 4096}} {
		if got := autoNList(c.n); got != c.want {
			t.Errorf("autoNList(%d)=%d want %d", c.n, got, c.want)
		}
	}
}
