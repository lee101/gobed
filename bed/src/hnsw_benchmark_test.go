package src

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"testing"
	"time"
)

func TestHNSWvsBruteForce(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping HNSW benchmark in short mode")
	}

	sizes := []int{1000, 5000, 10000, 25000, 50000}

	fmt.Println("\n=== HNSW vs Brute Force Comparison ===")
	fmt.Printf("%-10s %-12s %-12s %-10s %-10s %-10s %-10s\n",
		"Size", "BF(us)", "HNSW(us)", "Speedup", "BF QPS", "HNSW QPS", "Recall@10")
	fmt.Println(repeatStr("-", 80))

	for _, size := range sizes {
		// Create searchers
		bfEngine, err := NewOptimizedSearchEngine()
		if err != nil {
			t.Skipf("Model not available: %v", err)
		}

		hnswEngine, err := NewHNSWSearcher()
		if err != nil {
			t.Skipf("HNSW searcher not available: %v", err)
		}

		// Generate docs
		docs := make([]Document, size)
		for i := 0; i < size; i++ {
			docs[i] = Document{
				Path:       fmt.Sprintf("/test/file%d.go", i),
				LineNumber: i % 500,
				Content:    fmt.Sprintf("func handler%d() { process%d(%d) }", i, i%100, i*7),
			}
		}

		// Index both
		bfEngine.BatchAddDocuments(docs, runtime.NumCPU())
		hnswEngine.BatchAddDocuments(docs, runtime.NumCPU())

		// Benchmark queries
		numQueries := 50
		queries := []string{
			"function that handles requests",
			"error handling routine",
			"database connection pool",
			"authentication middleware",
			"process data handler",
		}

		bfLatencies := make([]float64, numQueries)
		hnswLatencies := make([]float64, numQueries)
		var totalRecall float64

		for i := 0; i < numQueries; i++ {
			q := queries[i%len(queries)]

			// Brute force
			start := time.Now()
			bfResults, _ := bfEngine.Search(q, 10, 0.0)
			bfLatencies[i] = float64(time.Since(start).Microseconds())

			// HNSW
			start = time.Now()
			hnswResults, _ := hnswEngine.Search(q, 10, 0.0)
			hnswLatencies[i] = float64(time.Since(start).Microseconds())

			// Compute recall
			bfSet := make(map[int]bool)
			for _, r := range bfResults {
				bfSet[r.Document.ID] = true
			}
			hits := 0
			for _, r := range hnswResults {
				if bfSet[r.Document.ID] {
					hits++
				}
			}
			if len(bfResults) > 0 {
				totalRecall += float64(hits) / float64(len(bfResults))
			}
		}

		sort.Float64s(bfLatencies)
		sort.Float64s(hnswLatencies)

		bfAvg := avg(bfLatencies)
		hnswAvg := avg(hnswLatencies)
		speedup := bfAvg / hnswAvg
		recall := totalRecall / float64(numQueries)

		fmt.Printf("%-10d %-12.0f %-12.0f %-10.1fx %-10.0f %-10.0f %-10.2f\n",
			size, bfAvg, hnswAvg, speedup,
			1000000.0/bfAvg, 1000000.0/hnswAvg, recall)

		hnswEngine.Close()
	}
}

func TestHNSWScaling(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}

	hnswEngine, err := NewHNSWSearcher()
	if err != nil {
		t.Skipf("HNSW not available: %v", err)
	}
	defer hnswEngine.Close()

	// Build index incrementally and measure
	fmt.Println("\n=== HNSW Scaling Test ===")
	fmt.Printf("%-10s %-12s %-12s %-10s\n", "Docs", "Build(ms)", "Search(us)", "QPS")
	fmt.Println(repeatStr("-", 50))

	checkpoints := []int{1000, 5000, 10000, 25000, 50000}
	currentSize := 0

	for _, checkpoint := range checkpoints {
		// Add docs up to checkpoint
		docsToAdd := checkpoint - currentSize
		docs := make([]Document, docsToAdd)
		for i := 0; i < docsToAdd; i++ {
			idx := currentSize + i
			docs[i] = Document{
				Path:       fmt.Sprintf("/test/file%d.go", idx),
				LineNumber: idx % 500,
				Content:    fmt.Sprintf("func process%d() { handle%d(%d) }", idx, idx%100, idx*7),
			}
		}

		buildStart := time.Now()
		hnswEngine.BatchAddDocuments(docs, runtime.NumCPU())
		buildTime := time.Since(buildStart)
		currentSize = checkpoint

		// Search benchmark
		numQueries := 50
		query := "function that processes data"
		latencies := make([]float64, numQueries)

		for i := 0; i < numQueries; i++ {
			start := time.Now()
			hnswEngine.Search(query, 10, 0.0)
			latencies[i] = float64(time.Since(start).Microseconds())
		}

		sort.Float64s(latencies)
		avgLatency := avg(latencies)
		qps := 1000000.0 / avgLatency

		fmt.Printf("%-10d %-12.0f %-12.0f %-10.0f\n",
			checkpoint, float64(buildTime.Milliseconds()), avgLatency, qps)
	}
}

func TestHNSWRecallVsEf(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}

	size := 10000
	hnswEngine, err := NewHNSWSearcher()
	if err != nil {
		t.Skipf("HNSW not available: %v", err)
	}
	defer hnswEngine.Close()

	bfEngine, err := NewOptimizedSearchEngine()
	if err != nil {
		t.Skipf("BF not available: %v", err)
	}

	docs := make([]Document, size)
	for i := 0; i < size; i++ {
		docs[i] = Document{
			Path:       fmt.Sprintf("/test/file%d.go", i),
			LineNumber: i % 500,
			Content:    fmt.Sprintf("func handler%d() { process%d(%d) }", i, i%100, i*7),
		}
	}

	bfEngine.BatchAddDocuments(docs, runtime.NumCPU())
	hnswEngine.BatchAddDocuments(docs, runtime.NumCPU())

	fmt.Println("\n=== HNSW Recall vs EF (10k docs) ===")
	fmt.Printf("%-10s %-12s %-12s\n", "EF", "Recall@10", "Latency(us)")
	fmt.Println(repeatStr("-", 40))

	efValues := []int{10, 20, 50, 100, 200, 500}
	numQueries := 50
	queries := []string{
		"function handler process",
		"error handling code",
		"database query builder",
		"authentication check",
		"data processing routine",
	}

	for _, ef := range efValues {
		hnswEngine.hnsw.SetEf(ef)

		var totalRecall float64
		var totalLatency float64

		for i := 0; i < numQueries; i++ {
			q := queries[i%len(queries)]

			bfResults, _ := bfEngine.Search(q, 10, 0.0)

			start := time.Now()
			hnswResults, _ := hnswEngine.Search(q, 10, 0.0)
			totalLatency += float64(time.Since(start).Microseconds())

			bfSet := make(map[int]bool)
			for _, r := range bfResults {
				bfSet[r.Document.ID] = true
			}
			hits := 0
			for _, r := range hnswResults {
				if bfSet[r.Document.ID] {
					hits++
				}
			}
			if len(bfResults) > 0 {
				totalRecall += float64(hits) / float64(len(bfResults))
			}
		}

		avgRecall := totalRecall / float64(numQueries)
		avgLatency := totalLatency / float64(numQueries)

		fmt.Printf("%-10d %-12.2f %-12.0f\n", ef, avgRecall, avgLatency)
	}
}

func TestHNSWBuildSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}

	sizes := []int{1000, 5000, 10000, 25000}

	fmt.Println("\n=== HNSW Build Speed ===")
	fmt.Printf("%-10s %-12s %-12s\n", "Size", "Time(ms)", "Docs/sec")
	fmt.Println(repeatStr("-", 40))

	for _, size := range sizes {
		hnswEngine, err := NewHNSWSearcher()
		if err != nil {
			t.Skipf("HNSW not available: %v", err)
		}

		docs := make([]Document, size)
		for i := 0; i < size; i++ {
			docs[i] = Document{
				Path:       fmt.Sprintf("/test/file%d.go", i),
				LineNumber: i % 500,
				Content:    fmt.Sprintf("func handler%d() { process%d(%d) }", i, i%100, i*7),
			}
		}

		start := time.Now()
		hnswEngine.BatchAddDocuments(docs, runtime.NumCPU())
		elapsed := time.Since(start)

		docsPerSec := float64(size) / elapsed.Seconds()
		fmt.Printf("%-10d %-12.0f %-12.0f\n", size, float64(elapsed.Milliseconds()), docsPerSec)

		hnswEngine.Close()
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	hnswEngine, err := NewHNSWSearcher()
	if err != nil {
		b.Skip("HNSW not available")
	}
	defer hnswEngine.Close()

	docs := make([]Document, 10000)
	for i := 0; i < 10000; i++ {
		docs[i] = Document{
			Path:       fmt.Sprintf("/test/file%d.go", i),
			LineNumber: i % 500,
			Content:    fmt.Sprintf("func handler%d() { process%d(%d) }", i, i%100, i*7),
		}
	}
	hnswEngine.BatchAddDocuments(docs, runtime.NumCPU())

	queries := []string{
		"function handler process",
		"error handling code",
		"database query",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hnswEngine.Search(queries[i%len(queries)], 10, 0.5)
	}
}

func avg(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func computeNDCG(predicted, groundTruth []SearchMatch) float64 {
	if len(groundTruth) == 0 {
		return 0
	}

	relevance := make(map[int]float64)
	for rank, r := range groundTruth {
		relevance[r.Document.ID] = float64(len(groundTruth) - rank)
	}

	dcg := 0.0
	for i, r := range predicted {
		if rel, ok := relevance[r.Document.ID]; ok {
			dcg += rel / math.Log2(float64(i)+2)
		}
	}

	idcg := 0.0
	for i := 0; i < len(groundTruth); i++ {
		idcg += float64(len(groundTruth)-i) / math.Log2(float64(i)+2)
	}

	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}
