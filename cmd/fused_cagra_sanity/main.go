//go:build legacy
// +build legacy

package main

import (
    "fmt"
    "math/rand"
    "os"
    "time"

    "github.com/lee101/gobed"
    "github.com/lee101/gobed/pkg/ann/simd"
)

// Sanity test for the fused CAGRA kernel:
// - Ensures per-query diversity (no identical neighbor lists)
// - Ensures exact matches rank #1
// - Ensures cluster-similar results appear in top-K
func main() {
    rand.Seed(42)

    // Synthetic dataset configuration
    const (
        vocabSize   = 512
        embedDim    = 512
        numVectors  = 200
        topK        = 10
        exactCount  = 10  // number of exact-match docs/queries
        numClusters = 5
    )

    // Generate embedding weights and scales
    embedWeights := make([]int8, vocabSize*embedDim)
    embedScales := make([]float32, vocabSize)
    for t := 0; t < vocabSize; t++ {
        embedScales[t] = 0.05 + rand.Float32()*0.05 // 0.05-0.10
        base := t * embedDim
        for j := 0; j < embedDim; j++ {
            embedWeights[base+j] = int8(rand.Intn(200) - 100) // [-100,100)
        }
    }

    // Database vectors and per-vector scales
    db := make([]simd.Vec512, numVectors)
    dbScales := make([]float32, numVectors)
    docCluster := make([]int, numVectors)

    // First exactCount docs: exact copies of tokens 0..exactCount-1
    for i := 0; i < exactCount; i++ {
        base := i * embedDim
        for j := 0; j < embedDim; j++ {
            db[i][j] = embedWeights[base+j]
        }
        dbScales[i] = embedScales[i]
        docCluster[i] = i % numClusters
    }

    // Remaining docs: clustered around selected token embeddings with noise
    for i := exactCount; i < numVectors; i++ {
        cluster := i % numClusters
        tok := cluster // use token index equal to cluster id
        base := tok * embedDim
        for j := 0; j < embedDim; j++ {
            noise := int8(rand.Intn(40) - 20)
            db[i][j] = embedWeights[base+j] + noise
        }
        dbScales[i] = embedScales[tok]
        docCluster[i] = cluster
    }

    // Build fused CAGRA engine
    cfg := gobed.DefaultFusedCAGRAConfig()
    cfg.VocabSize = vocabSize
    cfg.EmbedDim = embedDim
    cfg.TopK = 10
    engine, err := gobed.NewFusedCAGRAEngine(cfg)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to create fused CAGRA engine: %v\n", err)
        os.Exit(1)
    }

    if err := engine.BuildIndex(embedWeights, embedScales, db, dbScales); err != nil {
        fmt.Fprintf(os.Stderr, "BuildIndex failed: %v\n", err)
        os.Exit(1)
    }

    // Construct queries
    // - First exactCount queries: single token matching exact docs
    // - Next queries: one per cluster to test semantic similarity
    var tokenBatch [][]uint16
    var expectedExact []int
    var expectedCluster []int

    for i := 0; i < exactCount; i++ {
        tokenBatch = append(tokenBatch, []uint16{uint16(i)})
        expectedExact = append(expectedExact, i) // expect doc i at rank 1
        expectedCluster = append(expectedCluster, i%numClusters)
    }

    // Add cluster queries (numClusters of them)
    for c := 0; c < numClusters; c++ {
        // Use token representing cluster c
        tokenBatch = append(tokenBatch, []uint16{uint16(c)})
        expectedExact = append(expectedExact, -1) // not an exact query
        expectedCluster = append(expectedCluster, c)
    }

    // Run batch search
    start := time.Now()
    results, err := engine.SearchBatch(tokenBatch, 4)
    took := time.Since(start)
    if err != nil {
        fmt.Fprintf(os.Stderr, "SearchBatch failed: %v\n", err)
        os.Exit(1)
    }

    fmt.Fprintf(os.Stderr, "Ran %d queries in %v (%.3f ms/query)\n", len(tokenBatch), took, float64(took.Microseconds())/1000.0/float64(len(tokenBatch)))

    // 1) Duplicate bug check: ensure not all neighbor lists are identical
    allSame := true
    for i := 1; i < len(results); i++ {
        if !sameTopK(results[0], results[i]) {
            allSame = false
            break
        }
    }
    if allSame {
        fmt.Println("❌ CRITICAL BUG: All queries return identical results")
        os.Exit(2)
    } else {
        fmt.Println("✓ Per-query diversity: results differ across queries")
    }

    // 2) Exact match verification for the first set of queries
    exactPass := true
    for i := 0; i < exactCount; i++ {
        if len(results[i]) == 0 || results[i][0].ID != expectedExact[i] {
            exactPass = false
            fmt.Printf("Exact query %d expected doc %d at #1, got %v\n", i, expectedExact[i], results[i])
        }
    }
    if exactPass {
        fmt.Println("✓ Exact matches ranked #1 for exact queries")
    } else {
        fmt.Println("❌ Exact match test failed for some queries")
        os.Exit(3)
    }

    // 3) Cluster similarity check: for cluster queries, ensure top-3 contain docs from that cluster
    clusterPass := true
    for qi := exactCount; qi < len(results); qi++ {
        want := expectedCluster[qi]
        hit := false
        limit := 3
        if limit > len(results[qi]) {
            limit = len(results[qi])
        }
        for r := 0; r < limit; r++ {
            id := results[qi][r].ID
            if id >= 0 && id < numVectors && docCluster[id] == want {
                hit = true
                break
            }
        }
        if !hit {
            clusterPass = false
            fmt.Printf("Cluster query %d expected cluster %d in top-%d, got %v\n", qi, want, limit, results[qi])
        }
    }
    if clusterPass {
        fmt.Println("✓ Cluster similarity: similar docs appear in top-3")
    } else {
        fmt.Println("❌ Cluster similarity test failed for some queries")
        os.Exit(4)
    }

    // 4) Intra-query duplicate check: ensure each query's top-K contain unique IDs
    intraDupPass := true
    for qi := range results {
        seen := make(map[int]bool)
        dup := false
        for _, r := range results[qi] {
            if seen[r.ID] {
                dup = true
                break
            }
            seen[r.ID] = true
        }
        if dup {
            intraDupPass = false
            fmt.Printf("Query %d contains duplicate IDs in top-K: %v\n", qi, results[qi])
        }
    }
    if intraDupPass {
        fmt.Println("✓ No duplicates within individual query results")
    } else {
        fmt.Println("❌ Found duplicates within some query results")
        os.Exit(5)
    }

    fmt.Println("\nAll fused CAGRA sanity checks passed.")
}

func sameTopK(a, b []gobed.SearchResult) bool {
    if len(a) != len(b) {
        return false
    }
    for i := range a {
        if a[i].ID != b[i].ID {
            return false
        }
    }
    return true
}
