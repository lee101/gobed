//go:build cagra && gpu

package cagra_test

import (
    "testing"
    gobed "github.com/lee101/gobed"
    "github.com/lee101/gobed/pkg/ann/simd"
)

// Synthetic sanity: build 16 orthogonal-ish vectors; query one; expect self at top.
func TestCAGRASynthetic_SelfNearest(t *testing.T) {
    if !gobed.IsCUDAAvailable() {
        t.Skip("CUDA not available")
    }

    cfg := gobed.DefaultCAGRAConfig()
    cfg.MaxVectors = 16
    cfg.VectorDim = gobed.Int8EmbeddingDim
    cfg.CachePath = ""
    idx, err := gobed.NewCAGRAIndex(cfg)
    if err != nil { t.Skipf("CAGRA unavailable: %v", err) }
    defer idx.Close()

    vecs := make([]simd.Vec512, 16)
    scales := make([]float32, 16)
    for i := 0; i < 16; i++ {
        for d := 0; d < 512; d++ {
            if d%16 == i { vecs[i][d] = 127 } else { vecs[i][d] = 0 }
        }
        scales[i] = 1.0
    }
    if err := idx.BuildIndex(vecs, scales); err != nil {
        t.Fatalf("build index: %v", err)
    }

    for i := 0; i < 16; i++ {
        q := vecs[i]
        res, err := idx.Search(q, scales[i], 3)
        if err != nil { t.Fatalf("search: %v", err) }
        if len(res) == 0 { t.Fatalf("no results") }
        if res[0].ID != i { t.Fatalf("top-1 mismatch: got %d want %d", res[0].ID, i) }
        if res[0].Similarity <= 0 { t.Fatalf("similarity not positive: %.3f", res[0].Similarity) }
    }
}

