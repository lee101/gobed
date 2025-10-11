//go:build legacy
// +build legacy

package main

import (
    "flag"
    "fmt"
    "math/rand"
    "os"
    "time"

    "github.com/lee101/gobed"
    "github.com/lee101/gobed/pkg/ann/simd"
)

func generateDB(numDocs, dim int) ([]simd.Vec512, []float32) {
    db := make([]simd.Vec512, numDocs)
    scales := make([]float32, numDocs)
    for i := 0; i < numDocs; i++ {
        for d := 0; d < dim; d++ {
            db[i][d] = int8(rand.Intn(200) - 100)
        }
        scales[i] = 0.05 + rand.Float32()*0.05
    }
    return db, scales
}

func main() {
    rand.Seed(42)
    var (
        numDocs    = flag.Int("docs", 110000, "number of documents")
        numQueries = flag.Int("queries", 100, "number of queries")
        k          = flag.Int("k", 5, "top-k results")
        vocabSize  = flag.Int("vocab", 0, "vocab size (0=match docs)")
    )
    flag.Parse()

    dim := 512
    db, dbScales := generateDB(*numDocs, dim)

    // Embed weights: to guarantee exact@1, set vocab to numDocs and set weights[i] == db[i], scales[i] == dbScale[i]
    vsize := *vocabSize
    if vsize <= 0 { vsize = *numDocs }
    embedWeights := make([]int8, vsize*dim)
    embedScales := make([]float32, vsize)
    for i := 0; i < vsize && i < *numDocs; i++ {
        for d := 0; d < dim; d++ {
            embedWeights[i*dim+d] = db[i][d]
        }
        embedScales[i] = dbScales[i]
    }
    for i := *numDocs; i < vsize; i++ {
        embedScales[i] = 0.08
    }

    // Build fused engine
    cfg := gobed.DefaultFusedCAGRAConfig()
    cfg.VocabSize = vsize
    cfg.EmbedDim = dim
    cfg.TopK = *k

    engine, err := gobed.NewFusedCAGRAEngine(cfg)
    if err != nil { fmt.Fprintln(os.Stderr, "engine init error:", err); os.Exit(1) }

    t0 := time.Now()
    if err := engine.BuildIndex(embedWeights, embedScales, db, dbScales); err != nil {
        fmt.Fprintln(os.Stderr, "build error:", err); os.Exit(1)
    }
    buildS := time.Since(t0)

    // Prepare queries: first half exact (single token of doc id), second half similar (token id of a random doc in same range)
    maxTokens := 4
    tokenBatch := make([][]uint16, *numQueries)
    for i := 0; i < *numQueries; i++ {
        if i < *numQueries/2 {
            id := i % *numDocs
            tokenBatch[i] = []uint16{uint16(id)}
        } else {
            id := rand.Intn(*numDocs)
            tokenBatch[i] = []uint16{uint16(id)}
        }
    }

    // Run batch
    t1 := time.Now()
    results, err := engine.SearchBatch(tokenBatch, maxTokens)
    if err != nil { fmt.Fprintln(os.Stderr, "search error:", err); os.Exit(1) }
    elapsed := time.Since(t1)
    avgMs := float64(elapsed.Microseconds()) / 1000.0 / float64(*numQueries)

    // Evaluate
    exactTop1 := 0
    clusterHits := 0 // proxy unused here; we just check top-2 different
    perQueryDuplicates := 0
    globalDup := true
    for i := 0; i < *numQueries; i++ {
        if i == 0 { /* baseline */ } else {
            same := true
            if len(results[i]) != len(results[0]) { same = false }
            for j := range results[i] {
                if results[i][j].ID != results[0][j].ID { same = false; break }
            }
            if !same { globalDup = false }
        }
        // per-query dup ids
        seen := make(map[int]bool)
        for _, r := range results[i] { if seen[r.ID] { perQueryDuplicates++; break } ; seen[r.ID] = true }
        if i < *numQueries/2 {
            id := i % *numDocs
            if len(results[i]) > 0 && results[i][0].ID == id { exactTop1++ }
        } else {
            if len(results[i]) > 1 && results[i][0].ID != results[i][1].ID { clusterHits++ }
        }
    }

    fmt.Printf("Fused CAGRA large eval\n")
    fmt.Printf("  Build: %v\n", buildS)
    fmt.Printf("  Queries: %d in %v (avg %.3f ms, %.0f QPS)\n", *numQueries, elapsed, avgMs, float64(*numQueries)/elapsed.Seconds())
    fmt.Printf("  Exact@1: %d/%d\n", exactTop1, *numQueries/2)
    if globalDup { fmt.Println("  ❌ All queries returned identical results") } else { fmt.Println("  ✓ Per-query diversity") }
    if perQueryDuplicates > 0 { fmt.Printf("  ❌ Queries with duplicate IDs in top-K: %d\n", perQueryDuplicates) } else { fmt.Println("  ✓ No duplicate IDs within top-K") }
}

