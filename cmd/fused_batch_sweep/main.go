//go:build legacy
// +build legacy

package main

import (
    "fmt"
    "math/rand"
    "time"

    "github.com/lee101/gobed"
    "github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
    rand.Seed(42)
    // Large DB to stress GPU
    docs := 110000
    dim := 512

    // Build DB and vocab tied 1:1 for exact@1 potential
    dbRaw, dbScales := generateDB(docs, dim)
    embedWeights, embedScales := mirrorAsVocab(dbRaw, dbScales)
    // Convert to simd.Vec512
    var db [] [512]int8
    db = make([][512]int8, docs)
    for i := 0; i < docs; i++ { for d := 0; d < dim; d++ { db[i][d] = int8(dbRaw[i][d]) } }

    cfg := gobed.DefaultFusedCAGRAConfig()
    cfg.VocabSize = docs
    cfg.EmbedDim = dim
    cfg.TopK = 5
    engine, err := gobed.NewFusedCAGRAEngine(cfg)
    if err != nil { panic(err) }
    // Gobed type is simd.Vec512; reuse array-compatible memory
    // Convert to gobed's simd.Vec512 type via copy to avoid unsafe tricks
    simdDB := make([]simd.Vec512, docs)
    for i := 0; i < docs; i++ { copy(simdDB[i][:], db[i][:]) }
    if err := engine.BuildIndex(embedWeights, embedScales, simdDB, dbScales); err != nil { panic(err) }

    // Sweep batch sizes
    batches := []int{16, 32, 64, 128, 256, 512, 1024}
    fmt.Println("batch\tavg_ms\tQPS\texact@1\tuniqOK\tmonoOK")
    for _, bs := range batches {
        tokenBatch := make([][]uint16, bs)
        exact := 0
        uniqOK := 0
        monoOK := 0
        for i := 0; i < bs; i++ {
            id := rand.Intn(docs)
            tokenBatch[i] = []uint16{uint16(id)}
        }
        t0 := time.Now()
        res, err := engine.SearchBatch(tokenBatch, 4)
        if err != nil { panic(err) }
        elapsed := time.Since(t0)
        for i := 0; i < bs; i++ {
            id := int(tokenBatch[i][0])
            if len(res[i]) > 0 && res[i][0].ID == id { exact++ }
            // Uniqueness within top-K
            seen := make(map[int]bool)
            dup := false
            for _, r := range res[i] { if seen[r.ID] { dup = true; break }; seen[r.ID] = true }
            if !dup { uniqOK++ }
            // Distances monotonic (ascending)
            mono := true
            for j := 1; j < len(res[i]); j++ {
                if res[i][j].Similarity < res[i][j-1].Similarity { mono = false; break }
            }
            if mono { monoOK++ }
        }
        avg := float64(elapsed.Microseconds()) / 1000.0 / float64(bs)
        qps := float64(bs) / elapsed.Seconds()
        fmt.Printf("%d\t%.3f\t%.0f\t%d/%d\t%d/%d\t%d/%d\n", bs, avg, qps, exact, bs, uniqOK, bs, monoOK, bs)
    }
}

// helpers duplicated locally to avoid imports bloat
func generateDB(numDocs, dim int) ([][512]byte, []float32) {
    db := make([][512]byte, numDocs)
    scales := make([]float32, numDocs)
    for i := 0; i < numDocs; i++ {
        var v [512]byte
        for d := 0; d < dim; d++ { v[d] = byte(rand.Intn(200) - 100) }
        db[i] = v
        scales[i] = 0.05 + rand.Float32()*0.05
    }
    return db, scales
}

func mirrorAsVocab(db [][512]byte, scales []float32) ([]int8, []float32) {
    vsize := len(db)
    dim := 512
    weights := make([]int8, vsize*dim)
    vscales := make([]float32, vsize)
    for i := 0; i < vsize; i++ {
        off := i * dim
        for d := 0; d < dim; d++ { weights[off+d] = int8(db[i][d]) }
        vscales[i] = scales[i]
    }
    return weights, vscales
}
