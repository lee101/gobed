//go:build cagra && gpu

package cagra_test

import (
    "math"
    "sort"
    "testing"
    "time"

    gobed "github.com/lee101/gobed"
    "github.com/lee101/gobed/pkg/ann/simd"
)

// Sanity/quality check for gobed CAGRA index using dot-product baseline
func TestCAGRAGPUSearchQuality_Subpkg(t *testing.T) {
    if !gobed.IsCUDAAvailable() {
        t.Skip("CUDA not available")
    }
    // Try to create model and index; skip on failure
    model, err := gobed.LoadInt8Model512()
    if err != nil {
        t.Skipf("int8 model unavailable: %v", err)
    }

    datasetTexts := []string{
        "machine learning transforms data science",
        "deep learning advances computer vision research",
        "healthy vegan recipes for home cooking",
        "financial markets and stock trading strategies",
        "classical music history and famous composers",
        "quantum computing algorithms for qubits",
        "natural language processing with transformers",
        "reinforcement learning for robotics control",
        "cybersecurity best practices for cloud infrastructure",
        "biotechnology breakthroughs in gene editing",
        "ancient history of the roman empire",
        "renewable energy storage with solar and wind",
    }

    dbVectors := make([]simd.Vec512, len(datasetTexts))
    dbScales := make([]float32, len(datasetTexts))
    dbFloat := make([][]float32, len(datasetTexts))
    for i, s := range datasetTexts {
        r, err := model.EmbedInt8(s)
        if err != nil { t.Fatalf("embed dataset[%d]: %v", i, err) }
        copy(dbVectors[i][:], r.Vector)
        dbScales[i] = r.Scale
        dbFloat[i] = dequant(r.Vector, r.Scale)
    }

    cfg := gobed.DefaultCAGRAConfig()
    cfg.MaxVectors = len(datasetTexts)
    cfg.VectorDim = gobed.Int8EmbeddingDim
    cfg.CachePath = ""
    idx, err := gobed.NewCAGRAIndex(cfg)
    if err != nil { t.Skipf("CAGRA unavailable: %v", err) }
    defer idx.Close()
    if err := idx.BuildIndex(dbVectors, dbScales); err != nil {
        t.Fatalf("build index: %v", err)
    }

    queries := []string{
        "transformers for language",
        "quantum computing algorithms",
        "vegan cooking at home",
        "financial trading strategies",
    }
    k := 10

    var sumNDCG float64
    for _, q := range queries {
        qr, err := model.EmbedInt8(q)
        if err != nil { t.Fatalf("embed query: %v", err) }
        var qv simd.Vec512; copy(qv[:], qr.Vector)

        start := time.Now()
        got, err := idx.Search(qv, qr.Scale, k)
        if err != nil { t.Fatalf("search failed: %v", err) }
        _ = time.Since(start)
        if len(got) == 0 { t.Fatalf("no results") }

        // Float baseline
        toks, _ := model.Tokenize(q)
        qf, _ := model.EmbedTokens(toks)
        scores := make([]float32, len(dbFloat))
        for i := range dbFloat { scores[i] = dot(qf, dbFloat[i]) }

        expIDs := topK(scores, min(k, len(scores)))
        gotIDs := ids(got, min(k, len(got)))

        if !sameSet(gotIDs, expIDs) {
            t.Errorf("top-%d set mismatch: got=%v exp=%v", k, gotIDs, expIDs)
        }

        // NDCG vs graded baseline
        rel := grades(scores, k)
        nd := ndcg(got, rel, k)
        sumNDCG += nd
    }

    avg := sumNDCG / float64(len(queries))
    if avg < 0.9 {
        t.Errorf("avg NDCG@10 too low: %.3f", avg)
    }
}

// Helpers (module-local)
func dequant(v []int8, s float32) []float32 { f := make([]float32, len(v)); for i, x := range v { f[i] = float32(x) * s }; return f }
func dot(a, b []float32) float32 { var s float32; for i := range a { s += a[i]*b[i] }; return s }
func topK(scores []float32, k int) []int { if k > len(scores) { k = len(scores) }; idx := make([]int, len(scores)); for i := range idx { idx[i]=i }; sort.Slice(idx, func(i,j int) bool { if scores[idx[i]]==scores[idx[j]] {return idx[i]<idx[j]}; return scores[idx[i]]>scores[idx[j]] }); return idx[:k] }
func ids(res []gobed.SearchResult, k int) []int { if k>len(res){k=len(res)}; out:=make([]int,k); for i:=0;i<k;i++{out[i]=res[i].ID}; return out }
func sameSet(a,b []int) bool { if len(a)!=len(b){return false}; a2:=append([]int(nil),a...); b2:=append([]int(nil),b...); sort.Ints(a2); sort.Ints(b2); for i:=range a2{ if a2[i]!=b2[i]{return false}}; return true }
func grades(base []float32, k int) map[int]float64 { top := topK(base, min(k,len(base))); g:=make(map[int]float64,len(top)); for i,id:= range top { g[id]=float64(len(top)-i) }; return g }
func ndcg(res []gobed.SearchResult, rel map[int]float64, k int) float64 { if len(res)==0||len(rel)==0 {return 0}; a:=make([]float64,0,min(k,len(res))); for i:=0;i<len(res)&&i<k;i++{a=append(a,rel[res[i].ID])}; if len(a)==0{return 0}; ideal:=make([]float64,0,len(rel)); for _,g:=range rel{ if g>0{ideal=append(ideal,g)} }; sort.Sort(sort.Reverse(sort.Float64Slice(ideal))); if len(ideal)>len(a){ideal=ideal[:len(a)]}; return dcg(a)/dcg(ideal) }
func dcg(g []float64) float64 { if len(g)==0{return 0}; var s float64; for i,x := range g { if x<=0 {continue}; s += x / math.Log2(float64(i)+2) }; return s }
func min(a,b int) int { if a<b { return a }; return b }
