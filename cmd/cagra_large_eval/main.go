//go:build legacy
// +build legacy

package main

import (
    "flag"
    "fmt"
    "log"
    "math/rand"
    "os"
    "time"

    "github.com/lee101/gobed"
    "github.com/lee101/gobed/pkg/ann/simd"
)

// generateClusteredData creates numDocs int8 vectors with per-vector scales across numClusters.
func generateClusteredData(numDocs, numClusters, dim int) ([]simd.Vec512, []float32, []int) {
    centers := make([]simd.Vec512, numClusters)
    for c := 0; c < numClusters; c++ {
        for i := 0; i < dim; i++ {
            centers[c][i] = int8(rand.Intn(200) - 100)
        }
    }

    docs := make([]simd.Vec512, numDocs)
    scales := make([]float32, numDocs)
    clusters := make([]int, numDocs)

    for d := 0; d < numDocs; d++ {
        c := d % numClusters
        clusters[d] = c
        for i := 0; i < dim; i++ {
            noise := int8(rand.Intn(40) - 20)
            v := int16(centers[c][i]) + int16(noise)
            if v > 127 {
                v = 127
            } else if v < -128 {
                v = -128
            }
            docs[d][i] = int8(v)
        }
        scales[d] = 0.05 + rand.Float32()*0.05
    }
    return docs, scales, clusters
}

// quantizeScale computes a symmetric int8 scale for a vector
func quantizeScale(v *simd.Vec512) float32 {
    maxAbs := float32(0)
    for i := 0; i < 512; i++ {
        a := float32(v[i])
        if a < 0 { a = -a }
        if a > maxAbs {
            maxAbs = a
        }
    }
    if maxAbs == 0 {
        return 1.0
    }
    return maxAbs / 127.0
}

func arraysEqual(a, b []gobed.SearchResult) bool {
    if len(a) != len(b) { return false }
    for i := range a {
        if a[i].ID != b[i].ID { return false }
    }
    return true
}

func main() {
    rand.Seed(42)

    var (
        numDocs    = flag.Int("docs", 110000, "number of documents")
        numQueries = flag.Int("queries", 100, "number of queries")
        clusters   = flag.Int("clusters", 100, "number of clusters")
        k          = flag.Int("k", 5, "top-k results")
        useCache   = flag.Bool("cache", true, "use CAGRA cache if available")
    )
    flag.Parse()

    fmt.Printf("CAGRA large-scale eval: docs=%d queries=%d clusters=%d k=%d\n", *numDocs, *numQueries, *clusters, *k)

    // Prepare data
    start := time.Now()
    docs, scales, docClusters := generateClusteredData(*numDocs, *clusters, 512)
    genTime := time.Since(start)
    fmt.Printf("Generated dataset in %v\n", genTime)

    // Configure CAGRA
    cfg := gobed.FastCAGRAConfig()
    cfg.MaxVectors = *numDocs
    cfg.VectorDim = 512
    cfg.CachePath = gobed.BuildCAGRACachePath("large_eval", cfg.VectorDim, cfg.GraphDegree, cfg.MaxVectors)

    index, err := gobed.NewCAGRAIndex(cfg)
    if err != nil {
        log.Fatalf("failed to create CAGRA index: %v", err)
    }
    defer index.Close()

    // Try load from cache
    if *useCache {
        if _, err := os.Stat(cfg.CachePath); err == nil {
            fmt.Printf("Loading index from cache: %s\n", cfg.CachePath)
            if err := index.LoadFromCache(); err != nil {
                fmt.Printf("Cache load failed, rebuilding: %v\n", err)
            } else {
                fmt.Println("Loaded from cache.")
            }
        }
    }

    // Build if not built
    if !index.GetStats().IsBuilt {
        fmt.Println("Building CAGRA index...")
        startBuild := time.Now()
        if err := index.BuildIndex(docs, scales); err != nil {
            log.Fatalf("build failed: %v", err)
        }
        fmt.Printf("Build done in %v\n", time.Since(startBuild))
    }

    // Prepare queries: half exact, half cluster-style
    queries := make([]simd.Vec512, *numQueries)
    queryScales := make([]float32, *numQueries)
    expectedID := make([]int, *numQueries)
    expectedCluster := make([]int, *numQueries)

    half := *numQueries / 2
    for i := 0; i < half; i++ {
        id := rand.Intn(*numDocs)
        queries[i] = docs[id]
        queryScales[i] = scales[id]
        expectedID[i] = id
        expectedCluster[i] = docClusters[id]
    }

    // cluster queries: build around a random doc's cluster with noise
    for i := half; i < *numQueries; i++ {
        baseID := rand.Intn(*numDocs)
        c := docClusters[baseID]
        expectedID[i] = -1
        expectedCluster[i] = c
        // derive from a doc in cluster c with a bit more noise
        var q simd.Vec512
        for d := 0; d < 512; d++ {
            noise := int8(rand.Intn(60) - 30)
            v := int16(docs[baseID][d]) + int16(noise)
            if v > 127 { v = 127 } else if v < -128 { v = -128 }
            q[d] = int8(v)
        }
        queries[i] = q
        queryScales[i] = quantizeScale(&q)
    }

    // Execute and evaluate
    dupAll := true
    exactPass := 0
    clusterHits := 0
    intraDupCount := 0
    nonMonotonic := 0

    for i := 0; i < *numQueries; i++ {
        res, err := index.Search(queries[i], queryScales[i], *k)
        if err != nil {
            log.Fatalf("search failed: %v", err)
        }
        if i == 0 {
            // nothing
        } else if !arraysEqual(res, mustSearch(index, queries[0], queryScales[0], *k)) {
            dupAll = false
        }

        // Per-query duplicate check
        seen := make(map[int]bool)
        for _, r := range res {
            if seen[r.ID] { intraDupCount++; break }
            seen[r.ID] = true
        }

        // Distances should be non-decreasing (lower is better)
        for j := 1; j < len(res); j++ {
            if res[j].Similarity < res[j-1].Similarity { // Similarity holds distances per wrapper
                nonMonotonic++
                break
            }
        }

        if i < half {
            if len(res) > 0 && res[0].ID == expectedID[i] {
                exactPass++
            }
        } else {
            // cluster relevance in top-3
            limit := 3
            if limit > len(res) { limit = len(res) }
            wanted := expectedCluster[i]
            found := false
            for r := 0; r < limit; r++ {
                id := res[r].ID
                if id >= 0 && id < *numDocs && docClusters[id] == wanted {
                    found = true
                    break
                }
            }
            if found { clusterHits++ }
        }
    }

    fmt.Println("\n=== Evaluation ===")
    if dupAll {
        fmt.Println("❌ All queries returned identical results (duplicate bug)")
    } else {
        fmt.Println("✓ Per-query diversity")
    }
    fmt.Printf("Exact match top-1: %d/%d\n", exactPass, half)
    fmt.Printf("Cluster relevance in top-3: %d/%d\n", clusterHits, *numQueries-half)
    if intraDupCount > 0 {
        fmt.Printf("❌ Queries with duplicate IDs in top-K: %d/%d\n", intraDupCount, *numQueries)
    } else {
        fmt.Println("✓ No duplicate IDs within top-K")
    }
    if nonMonotonic > 0 {
        fmt.Printf("❌ Queries with non-monotonic distances: %d/%d\n", nonMonotonic, *numQueries)
    } else {
        fmt.Println("✓ Distances monotonic (ascending)")
    }
}

func mustSearch(index *gobed.CAGRAIndex, q simd.Vec512, s float32, k int) []gobed.SearchResult {
    res, err := index.Search(q, s, k)
    if err != nil { log.Fatal(err) }
    return res
}
