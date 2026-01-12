package lib

import (
    "math"
    "sort"
    "time"

    gobed "github.com/lee101/gobed"
    "github.com/lee101/gobed/ann/simd"
)

// EvalConfig captures basic evaluation knobs.
type EvalConfig struct {
    K           int
    NumQueries  int
    Warmup      int
}

// EvalResult summarizes latency and recall metrics.
type EvalResult struct {
    K               int
    NumQueries      int
    P50LatencyMs    float64
    P95LatencyMs    float64
    QPS             float64
    RecallAtK       float64
    NDCGAtK         float64
}

// RunEval builds an index with provided documents, then evaluates search latency and recall@K
// against a brute-force baseline on CPU. Embeddings are computed once for the baseline/queries.
func RunEval(model *gobed.EmbeddingModel, docs []gobed.Document, cfg EvalConfig) (EvalResult, error) {
    if cfg.K <= 0 { cfg.K = 10 }
    if cfg.NumQueries <= 0 { cfg.NumQueries = 200 }
    if cfg.Warmup < 0 { cfg.Warmup = 0 }

    // Precompute embeddings for brute-force baseline
    vecs := make([]simd.Vec512, len(docs))
    scales := make([]float32, len(docs))
    for i, d := range docs {
        emb, err := model.EmbedInt8(d.Text)
        if err != nil { return EvalResult{}, err }
        copy(vecs[i][:], emb.Vector)
        scales[i] = emb.Scale
    }

    // Build index (CPU path; GPU/CAGRA can be wired via a different indexer)
    vcfg := gobed.DefaultVectorIndexConfig()
    vcfg.NList = 1024
    vcfg.NProbe = 8
    vcfg.UsePQ = true
    vcfg.UseHNSW = true
    idx := gobed.NewVectorIndex(model, vcfg)
    if err := idx.AddDocuments(docs); err != nil { return EvalResult{}, err }

    // Build query set (reuse first NumQueries documents as queries)
    nQ := min(cfg.NumQueries, len(docs))
    queries := make([]simd.Vec512, nQ)
    qscales := make([]float32, nQ)
    for i := 0; i < nQ; i++ {
        emb, err := model.EmbedInt8(docs[i].Text)
        if err != nil { return EvalResult{}, err }
        copy(queries[i][:], emb.Vector)
        qscales[i] = emb.Scale
    }

    // Warmup
    for i := 0; i < min(cfg.Warmup, nQ); i++ {
        _, _ = bruteForceTopK(&queries[i], qscales[i], vecs, scales, cfg.K)
    }

    // Measure latency + recall + NDCG
    latencies := make([]float64, 0, nQ)
    var correct float64
    goldRels := make([]map[int]float64, nQ)
    predictions := make([][]int, nQ)

    start := time.Now()
    for i := 0; i < nQ; i++ {
        // gold via brute force
        goldIDs, goldScores := bruteForceTopK(&queries[i], qscales[i], vecs, scales, cfg.K)

        // Build relevance map for NDCG (normalize scores to 0-3 scale)
        goldRels[i] = make(map[int]float64)
        maxScore := float32(0)
        for _, s := range goldScores {
            if s > maxScore { maxScore = s }
        }
        for j, id := range goldIDs {
            if maxScore > 0 {
                goldRels[i][id] = float64(goldScores[j] / maxScore * 3)
            } else {
                goldRels[i][id] = 3.0
            }
        }

        // index search
        t0 := time.Now()
        res, err := idx.Search(docs[i].Text, cfg.K)
        if err != nil { return EvalResult{}, err }
        latencies = append(latencies, float64(time.Since(t0).Microseconds())/1000.0)

        // compute recall@K and collect predictions
        predIDs := make([]int, len(res))
        for j := range res { predIDs[j] = res[j].ID }
        predictions[i] = predIDs
        correct += recallAtK(goldIDs, predIDs)
    }
    total := time.Since(start)
    qps := float64(nQ) / total.Seconds()
    p50, p95 := percentile(latencies, 50), percentile(latencies, 95)
    ndcg := NDCGAtK(goldRels, predictions, cfg.K)

    return EvalResult{
        K: cfg.K, NumQueries: nQ,
        P50LatencyMs: p50, P95LatencyMs: p95,
        QPS: qps, RecallAtK: correct/float64(nQ),
        NDCGAtK: ndcg,
    }, nil
}

func bruteForceTopK(q *simd.Vec512, qScale float32, vecs []simd.Vec512, scales []float32, k int) ([]int, []float32) {
    type pair struct{ id int; s float32 }
    scores := make([]pair, len(vecs))
    for i := range vecs {
        // dot over int8 * int8 scaled
        s := float32(simd.Dot512(q, &vecs[i])) * qScale * scales[i]
        scores[i] = pair{id: i, s: s}
    }
    sort.Slice(scores, func(i,j int) bool { return scores[i].s > scores[j].s })
    if k > len(scores) { k = len(scores) }
    ids := make([]int, k)
    vals := make([]float32, k)
    for i := 0; i < k; i++ { ids[i] = scores[i].id; vals[i] = scores[i].s }
    return ids, vals
}

func bruteForceTopKIDs(q *simd.Vec512, qScale float32, vecs []simd.Vec512, scales []float32, k int) []int {
    ids, _ := bruteForceTopK(q, qScale, vecs, scales, k)
    return ids
}

func recallAtK(gold, pred []int) float64 {
    m := make(map[int]struct{}, len(gold))
    for _, id := range gold { m[id] = struct{}{} }
    var hit float64
    for _, id := range pred {
        if _, ok := m[id]; ok { hit++ }
    }
    return hit / float64(len(gold))
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

func min(a,b int) int { if a<b { return a }; return b }
