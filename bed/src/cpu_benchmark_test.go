package src

import (
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func BenchmarkSIMDDot512(b *testing.B) {
	var v1, v2 simd.Vec512
	for i := 0; i < 512; i++ {
		v1[i] = int8(rand.Intn(256) - 128)
		v2[i] = int8(rand.Intn(256) - 128)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simd.Dot512(&v1, &v2)
	}
}

func BenchmarkFastInvSqrt(b *testing.B) {
	vals := make([]float32, 1000)
	for i := range vals {
		vals[i] = rand.Float32()*1000 + 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastInvSqrt(vals[i%1000])
	}
}

func BenchmarkSearchEngine_Search(b *testing.B) {
	engine, err := NewOptimizedSearchEngine()
	if err != nil {
		b.Skip("Model not available")
	}

	// Generate synthetic documents
	numDocs := 10000
	docs := make([]Document, numDocs)
	for i := 0; i < numDocs; i++ {
		docs[i] = Document{
			Path:       fmt.Sprintf("/test/file%d.go", i),
			LineNumber: i % 100,
			Content:    fmt.Sprintf("func test%d() { return %d }", i, i*17),
		}
	}

	if err := engine.BatchAddDocuments(docs, runtime.NumCPU()); err != nil {
		b.Fatalf("BatchAdd failed: %v", err)
	}

	queries := []string{
		"function that returns value",
		"test function implementation",
		"error handling code",
		"database connection",
		"user authentication",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.Search(queries[i%len(queries)], 10, 0.5)
	}
}

func BenchmarkSearchEngine_BatchIndex(b *testing.B) {
	for _, numDocs := range []int{100, 1000, 5000} {
		b.Run(fmt.Sprintf("docs=%d", numDocs), func(b *testing.B) {
			docs := make([]Document, numDocs)
			for i := 0; i < numDocs; i++ {
				docs[i] = Document{
					Path:       fmt.Sprintf("/test/file%d.go", i),
					LineNumber: i % 100,
					Content:    fmt.Sprintf("func handler%d() { process(%d) }", i, i),
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				engine, _ := NewOptimizedSearchEngine()
				engine.BatchAddDocuments(docs, runtime.NumCPU())
			}
		})
	}
}

func TestSearchEngineScaleBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping scale benchmark in short mode")
	}

	sizes := []int{1000, 5000, 10000, 25000, 50000}

	fmt.Println("\n=== CPU SIMD Search Engine Scale Benchmark ===")
	fmt.Printf("%-10s %-12s %-12s %-10s %-10s %-10s\n",
		"Size", "Index(ms)", "Docs/sec", "Search(us)", "P95(us)", "QPS")
	fmt.Println(repeatStr("-", 70))

	for _, size := range sizes {
		engine, err := NewOptimizedSearchEngine()
		if err != nil {
			t.Skipf("Model not available: %v", err)
		}

		// Generate docs
		docs := make([]Document, size)
		for i := 0; i < size; i++ {
			docs[i] = Document{
				Path:       fmt.Sprintf("/test/file%d.go", i),
				LineNumber: i % 500,
				Content:    fmt.Sprintf("func process%d() { handle%d(%d) }", i, i%100, i*7),
			}
		}

		// Index
		indexStart := time.Now()
		engine.BatchAddDocuments(docs, runtime.NumCPU())
		indexTime := time.Since(indexStart)
		docsPerSec := float64(size) / indexTime.Seconds()

		// Search benchmark
		numQueries := 100
		latencies := make([]float64, numQueries)
		queries := []string{
			"function that processes data",
			"error handling routine",
			"database query builder",
			"authentication middleware",
			"request handler function",
		}

		for i := 0; i < numQueries; i++ {
			start := time.Now()
			engine.Search(queries[i%len(queries)], 10, 0.5)
			latencies[i] = float64(time.Since(start).Microseconds())
		}

		sort.Float64s(latencies)
		avgLatency := average(latencies)
		p95 := latencies[int(float64(len(latencies))*0.95)]
		qps := 1000000.0 / avgLatency

		fmt.Printf("%-10d %-12.0f %-12.0f %-10.0f %-10.0f %-10.0f\n",
			size, float64(indexTime.Milliseconds()), docsPerSec, avgLatency, p95, qps)
	}
}

func TestSIMDDotProductAccuracy(t *testing.T) {
	// Verify SIMD produces correct results
	var v1, v2 simd.Vec512

	for i := 0; i < 512; i++ {
		v1[i] = int8(i % 128)
		v2[i] = int8((i * 3) % 128)
	}

	simdResult := simd.Dot512(&v1, &v2)

	// Scalar reference
	var scalarResult int32
	for i := 0; i < 512; i++ {
		scalarResult += int32(v1[i]) * int32(v2[i])
	}

	if simdResult != scalarResult {
		t.Errorf("SIMD mismatch: got %d, want %d", simdResult, scalarResult)
	}
}

func TestParallelSearchScaling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping parallel scaling test in short mode")
	}

	engine, err := NewOptimizedSearchEngine()
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}

	// Generate 50k docs
	numDocs := 50000
	docs := make([]Document, numDocs)
	for i := 0; i < numDocs; i++ {
		docs[i] = Document{
			Path:       fmt.Sprintf("/test/file%d.go", i),
			LineNumber: i % 500,
			Content:    fmt.Sprintf("func handler%d() { return process(%d) }", i, i%1000),
		}
	}

	engine.BatchAddDocuments(docs, runtime.NumCPU())

	fmt.Println("\n=== Parallel Search Scaling ===")
	fmt.Printf("Documents: %d\n", numDocs)

	numQueries := 50
	query := "function that handles request"

	start := time.Now()
	for i := 0; i < numQueries; i++ {
		engine.Search(query, 10, 0.5)
	}
	elapsed := time.Since(start)

	avgLatency := float64(elapsed.Microseconds()) / float64(numQueries)
	qps := float64(numQueries) / elapsed.Seconds()

	fmt.Printf("Avg latency: %.0f us\n", avgLatency)
	fmt.Printf("QPS: %.0f\n", qps)
	fmt.Printf("CPU cores used: %d\n", runtime.NumCPU())
}

func average(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func repeatStr(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}

func BenchmarkFastBedSearcher_Index(b *testing.B) {
	for _, numFiles := range []int{10, 50, 100} {
		b.Run(fmt.Sprintf("files=%d", numFiles), func(b *testing.B) {
			// This would need actual files - skip for now
			b.Skip("Requires filesystem setup")
		})
	}
}

func TestMemoryUsage(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}

	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	engine, err := NewOptimizedSearchEngine()
	if err != nil {
		t.Skip("Model not available")
	}

	numDocs := 50000
	docs := make([]Document, numDocs)
	for i := 0; i < numDocs; i++ {
		docs[i] = Document{
			Path:       fmt.Sprintf("/test/file%d.go", i),
			LineNumber: i % 500,
			Content:    fmt.Sprintf("func process%d() { handle(%d) }", i, i),
		}
	}

	engine.BatchAddDocuments(docs, runtime.NumCPU())

	runtime.GC()
	runtime.ReadMemStats(&m2)

	allocMB := float64(m2.Alloc-m1.Alloc) / (1024 * 1024)
	bytesPerDoc := float64(m2.Alloc-m1.Alloc) / float64(numDocs)

	fmt.Printf("\n=== Memory Usage ===\n")
	fmt.Printf("Documents: %d\n", numDocs)
	fmt.Printf("Total allocated: %.1f MB\n", allocMB)
	fmt.Printf("Per document: %.0f bytes\n", bytesPerDoc)
	fmt.Printf("Embedding storage: %.1f MB (512 dims * 4 bytes * %d docs)\n",
		float64(512*4*numDocs)/(1024*1024), numDocs)
}

func BenchmarkEmbedding(b *testing.B) {
	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		b.Skip("Model not available")
	}
	defer model.Close()

	texts := []string{
		"func handleRequest(w http.ResponseWriter, r *http.Request)",
		"if err != nil { return nil, err }",
		"type Config struct { Port int; Host string }",
		"for i := 0; i < len(items); i++ { process(items[i]) }",
		"go func() { for msg := range ch { handle(msg) } }()",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		emb, release := model.EmbedFast(texts[i%len(texts)])
		_ = emb
		release()
	}
}
