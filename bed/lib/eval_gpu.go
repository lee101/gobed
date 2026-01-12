//go:build gpu

package lib

import (
    "time"
    "sort"
    "math"

    gobed "github.com/lee101/gobed"
    "github.com/lee101/gobed/pkg/ann/simd"
)

type EvalGPUResult struct {
    K, NumQueries int
    // Search-only timings (precomputed query embeddings)
    P50SearchMs float64
    P95SearchMs float64
    // End-to-end timings (embedding + search per query)
    P50EndToEndMs float64
    P95EndToEndMs float64
    QPS          float64 // for search-only path
    RecallAtK    float64
    NDCGAtK      float64
}

// RunEvalGPU builds a GPU CAGRA-style index and evaluates search-only and end-to-end latencies.
func RunEvalGPU(model *gobed.EmbeddingModel, docs []gobed.Document, base EvalConfig, graph gobed.GPUCagraConfig) (EvalGPUResult, error) {
    if base.K <= 0 { base.K = 10 }
    if base.NumQueries <= 0 { base.NumQueries = 200 }
    if base.Warmup < 0 { base.Warmup = 0 }

    // Precompute doc embeddings
    vecs := make([]simd.Vec512, len(docs))
    scales := make([]float32, len(docs))
    ids := make([]int, len(docs))
    for i, d := range docs {
        emb, err := model.EmbedInt8(d.Text)
        if err != nil { return EvalGPUResult{}, err }
        copy(vecs[i][:], emb.Vector)
        scales[i] = emb.Scale
        ids[i] = d.ID
    }

    // Build GPU CAGRA-like index
    gi, err := gobed.NewGPUCagraIndexer(graph)
    if err != nil { return EvalGPUResult{}, err }
    defer gi.Close()
    if err := gi.Build(vecs, scales, ids); err != nil { return EvalGPUResult{}, err }

    // Prepare queries
    nQ := min(base.NumQueries, len(docs))
    qvecs := make([]simd.Vec512, nQ)
    qscales := make([]float32, nQ)
    for i := 0; i < nQ; i++ {
        emb, err := model.EmbedInt8(docs[i].Text)
        if err != nil { return EvalGPUResult{}, err }
        copy(qvecs[i][:], emb.Vector)
        qscales[i] = emb.Scale
    }

    // gold via brute-force
    goldIDs := make([][]int, nQ)
    for i := 0; i < nQ; i++ {
        goldIDs[i] = bruteForceTopKIDs(&qvecs[i], qscales[i], vecs, scales, base.K)
    }

    // Warmup
    for i := 0; i < min(base.Warmup, nQ); i++ {
        _, _, _ = gi.Search(&qvecs[i], qscales[i], base.K)
    }

    // Search-only timings
    lat := make([]float64, 0, nQ)
    var correct float64
    goldRels := make([]map[int]float64, nQ)
    predictions := make([][]int, nQ)

    // Build relevance maps from gold scores
    for i := 0; i < nQ; i++ {
        _, goldScores := bruteForceTopK(&qvecs[i], qscales[i], vecs, scales, base.K)
        goldRels[i] = make(map[int]float64)
        maxScore := float32(0)
        for _, s := range goldScores {
            if s > maxScore { maxScore = s }
        }
        for j, id := range goldIDs[i] {
            if maxScore > 0 {
                goldRels[i][id] = float64(goldScores[j] / maxScore * 3)
            } else {
                goldRels[i][id] = 3.0
            }
        }
    }

    tStart := time.Now()
    for i := 0; i < nQ; i++ {
        t0 := time.Now()
        gotIDs, _, err := gi.Search(&qvecs[i], qscales[i], base.K)
        if err != nil { return EvalGPUResult{}, err }
        lat = append(lat, float64(time.Since(t0).Microseconds())/1000.0)
        // recall
        pred := make([]int, len(gotIDs))
        for j := range gotIDs { pred[j] = int(gotIDs[j]) }
        predictions[i] = pred
        correct += recallAtK(goldIDs[i], pred)
    }
    qps := float64(nQ) / time.Since(tStart).Seconds()
    ndcg := NDCGAtK(goldRels, predictions, base.K)

    // End-to-end timings (re-embed per query)
    e2e := make([]float64, 0, nQ)
    for i := 0; i < nQ; i++ {
        t0 := time.Now()
        emb, err := model.EmbedInt8(docs[i].Text)
        if err != nil { return EvalGPUResult{}, err }
        var q simd.Vec512
        copy(q[:], emb.Vector)
        _, _, err = gi.Search(&q, emb.Scale, base.K)
        if err != nil { return EvalGPUResult{}, err }
        e2e = append(e2e, float64(time.Since(t0).Microseconds())/1000.0)
    }

    return EvalGPUResult{
        K: base.K, NumQueries: nQ,
        P50SearchMs: percentile(lat, 50), P95SearchMs: percentile(lat, 95),
        P50EndToEndMs: percentile(e2e, 50), P95EndToEndMs: percentile(e2e, 95),
        QPS: qps, RecallAtK: correct/float64(nQ),
        NDCGAtK: ndcg,
    }, nil
}

func percentile(xs []float64, p int) float64 {
    if len(xs) == 0 { return 0 }
    ys := append([]float64(nil), xs...)
    sort.Float64s(ys)
    idx := int(math.Ceil(float64(p)/100.0*float64(len(ys)))) - 1
    if idx < 0 { idx = 0 }
    if idx >= len(ys) { idx = len(ys)-1 }
    return ys[idx]
}

