//go:build cagra && gpu

package gobed

import (
    "runtime"
    "sort"
    "sync"
    "testing"

    "github.com/lee101/gobed/pkg/ann/simd"
)

// TestCAGRAPipelineParallelIndex validates that parallel int8 embedding + CAGRA index
// building produces sane and consistent results, and batch search aligns with a
// serial pipeline on a small synthetic dataset.
func TestCAGRAPipelineParallelIndex(t *testing.T) {
    if !IsCUDAAvailable() {
        t.Skip("CUDA not available")
    }
    if !isCAGRAAvailable() {
        t.Skip("CAGRA/cuVS not available")
    }

    model, err := LoadInt8Model512()
    if err != nil {
        t.Skipf("int8 model unavailable: %v", err)
    }

    // Build synthetic dataset
    texts := make([]string, 256)
    for i := range texts {
        // Repeatable topical strings
        topicA := i % 8
        topicB := (i / 8) % 8
        texts[i] = 
            "topic A:" + string(rune('a'+topicA)) + " " +
            "topic B:" + string(rune('a'+topicB)) + " " +
            "fast semantic GPU search with cagra fused"
    }

    // Embed serially as baseline
    serialVecs := make([]simd.Vec512, len(texts))
    serialScales := make([]float32, len(texts))
    for i, s := range texts {
        r, err := model.EmbedInt8(s)
        if err != nil { t.Fatalf("serial embed failed: %v", err) }
        copy(serialVecs[i][:], r.Vector)
        serialScales[i] = r.Scale
    }

    // Embed in parallel
    parVecs := make([]simd.Vec512, len(texts))
    parScales := make([]float32, len(texts))

    type job struct{ idx int; text string }
    jobs := make(chan job, len(texts))
    var wg sync.WaitGroup
    workers := runtime.NumCPU()
    if workers > 8 { workers = 8 }
    for w := 0; w < workers; w++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for j := range jobs {
                r, err := model.EmbedInt8(j.text)
                if err != nil { t.Errorf("parallel embed failed: %v", err); continue }
                copy(parVecs[j.idx][:], r.Vector)
                parScales[j.idx] = r.Scale
            }
        }()
    }
    for i, s := range texts { jobs <- job{i, s} }
    close(jobs)
    wg.Wait()

    // Spot-check deterministic equality across pipelines (int8 + scale)
    for i := 0; i < 8; i++ { // sample first 8
        if parScales[i] != serialScales[i] {
            t.Fatalf("scale mismatch at %d: parallel %.6f vs serial %.6f", i, parScales[i], serialScales[i])
        }
        for d := 0; d < 512; d++ {
            if parVecs[i][d] != serialVecs[i][d] {
                t.Fatalf("vec mismatch at %d dim %d: %d vs %d", i, d, parVecs[i][d], serialVecs[i][d])
            }
        }
    }

    // Build CAGRA index with parallel embeddings
    cfg := DefaultCAGRAConfig()
    cfg.MaxVectors = len(texts)
    cfg.VectorDim  = Int8EmbeddingDim
    cfg.CachePath  = "" // in-memory only for this test

    idx, err := NewCAGRAIndex(cfg)
    if err != nil { t.Fatalf("create CAGRA index: %v", err) }
    defer idx.Close()

    if err := idx.BuildIndex(parVecs, parScales); err != nil {
        t.Fatalf("build CAGRA index: %v", err)
    }

    // Create a few queries and compare to serial float baseline
    queries := []string{
        "topic A:a GPU search",
        "topic B:c semantic",
        "cagra fused kernels",
    }

    k := 5
    for _, q := range queries {
        // int8 query for CAGRA
        qr, err := model.EmbedInt8(q)
        if err != nil { t.Fatalf("embed query: %v", err) }
        var qv simd.Vec512; copy(qv[:], qr.Vector)

        got, err := idx.Search(qv, qr.Scale, k)
        if err != nil { t.Fatalf("cagra search failed: %v", err) }
        if len(got) == 0 { t.Fatalf("no results") }

        // Float baseline (serial) for alignment check
        tokens, err := model.Tokenize(q)
        if err != nil { t.Fatalf("tokenize: %v", err) }
        qf, err := model.EmbedTokens(tokens)
        if err != nil { t.Fatalf("embed tokens: %v", err) }
        scores := make([]float32, len(texts))
        for i := range texts {
            // dequantize serial embeddings for baseline score
            dv := make([]float32, 512)
            for d := 0; d < 512; d++ {
                dv[d] = float32(serialVecs[i][d]) * serialScales[i]
            }
        scores[i] = parDotProduct(qf, dv)
        }
        top := parTopKIndices(scores, k)

        // Compare ID sets (order may differ rarely on ties)
        gotIDs := make([]int, len(got))
        for i := range got { gotIDs[i] = got[i].ID }
        if !parSameIDSet(gotIDs, top) {
            t.Fatalf("top-%d set mismatch: cagra=%v baseline=%v", k, gotIDs, top)
        }
    }
}

// Utility helpers duplicated locally to avoid export churn
func parDotProduct(a, b []float32) float32 {
    if len(a) != len(b) { panic("dotProduct length mismatch") }
    var s float32
    for i := range a { s += a[i] * b[i] }
    return s
}
func parTopKIndices(scores []float32, k int) []int {
    if k > len(scores) { k = len(scores) }
    idx := make([]int, len(scores))
    for i := range idx { idx[i] = i }
    sort.Slice(idx, func(i, j int) bool {
        if scores[idx[i]] == scores[idx[j]] { return idx[i] < idx[j] }
        return scores[idx[i]] > scores[idx[j]]
    })
    return idx[:k]
}
func parSameIDSet(a, b []int) bool {
    if len(a) != len(b) { return false }
    a2 := append([]int(nil), a...)
    b2 := append([]int(nil), b...)
    sort.Ints(a2); sort.Ints(b2)
    for i := range a2 { if a2[i] != b2[i] { return false } }
    return true
}
