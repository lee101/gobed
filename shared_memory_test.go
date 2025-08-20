package gobed

import (
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/lee101/gobed/ann/simd"
)

func TestSharedMemoryIndex(t *testing.T) {
	// Setup
	testPath := "/tmp/gobed_test_shared_" + fmt.Sprint(time.Now().UnixNano())
	defer os.RemoveAll(testPath)

	model, err := LoadModel()
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	t.Run("CreateAndClose", func(t *testing.T) {
		config := SharedMemoryConfig{
			BasePath:    testPath + "_create",
			MaxVectors:  100,
			CreateIfNew: true,
		}

		idx, err := NewSharedMemoryIndex(config)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}

		if err := idx.Close(); err != nil {
			t.Errorf("Failed to close index: %v", err)
		}
	})

	t.Run("AddAndSearch", func(t *testing.T) {
		config := SharedMemoryConfig{
			BasePath:    testPath + "_search",
			MaxVectors:  100,
			CreateIfNew: true,
		}

		idx, err := NewSharedMemoryIndex(config)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}
		defer idx.Close()

		// Add test vectors
		texts := []string{
			"machine learning algorithms",
			"deep neural networks",
			"computer vision systems",
			"natural language processing",
			"reinforcement learning",
		}

		for i, text := range texts {
			embedding, err := model.EmbedInt8(text)
			if err != nil {
				t.Errorf("Failed to embed text %d: %v", i, err)
				continue
			}

			var vec simd.Vec512
			copy(vec[:], embedding.Vector)

			if err := idx.AddVector(&vec, embedding.Scale, i); err != nil {
				t.Errorf("Failed to add vector %d: %v", i, err)
			}
		}

		// Verify count
		stats := idx.Stats()
		if stats.NumVectors != uint64(len(texts)) {
			t.Errorf("Expected %d vectors, got %d", len(texts), stats.NumVectors)
		}

		// Test search
		query := "deep learning neural networks"
		embedding, _ := model.EmbedInt8(query)
		var queryVec simd.Vec512
		copy(queryVec[:], embedding.Vector)

		results := idx.SearchTopK(&queryVec, 3)
		if len(results) != 3 {
			t.Errorf("Expected 3 results, got %d", len(results))
		}

		// Check that most similar result is reasonable
		if results[0].ID != 1 && results[0].ID != 4 {
			t.Logf("Top result ID %d might not be the most relevant", results[0].ID)
		}
	})

	t.Run("ReadOnlyAccess", func(t *testing.T) {
		basePath := testPath + "_readonly"
		
		// First create and populate index
		writeConfig := SharedMemoryConfig{
			BasePath:    basePath,
			MaxVectors:  50,
			CreateIfNew: true,
			ReadOnly:    false,
		}

		writeIdx, err := NewSharedMemoryIndex(writeConfig)
		if err != nil {
			t.Fatalf("Failed to create write index: %v", err)
		}

		// Add some vectors
		for i := 0; i < 10; i++ {
			text := fmt.Sprintf("test document %d", i)
			embedding, _ := model.EmbedInt8(text)
			var vec simd.Vec512
			copy(vec[:], embedding.Vector)
			writeIdx.AddVector(&vec, embedding.Scale, i)
		}
		writeIdx.Sync()
		writeIdx.Close()

		// Open as read-only
		readConfig := SharedMemoryConfig{
			BasePath:    basePath,
			ReadOnly:    true,
			CreateIfNew: false,
		}

		readIdx, err := NewSharedMemoryIndex(readConfig)
		if err != nil {
			t.Fatalf("Failed to open read-only index: %v", err)
		}
		defer readIdx.Close()

		// Verify can read
		stats := readIdx.Stats()
		if stats.NumVectors != 10 {
			t.Errorf("Expected 10 vectors in read-only index, got %d", stats.NumVectors)
		}

		// Try to write (should fail)
		var vec simd.Vec512
		err = readIdx.AddVector(&vec, 1.0, 100)
		if err == nil {
			t.Error("Expected error when writing to read-only index")
		}
	})

	t.Run("ConcurrentSearch", func(t *testing.T) {
		config := SharedMemoryConfig{
			BasePath:    testPath + "_concurrent",
			MaxVectors:  1000,
			CreateIfNew: true,
			CacheSize:   100,
		}

		idx, err := NewSharedMemoryIndex(config)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}
		defer idx.Close()

		// Add vectors
		numVectors := 100
		for i := 0; i < numVectors; i++ {
			text := fmt.Sprintf("document about topic %d", i%10)
			embedding, _ := model.EmbedInt8(text)
			var vec simd.Vec512
			copy(vec[:], embedding.Vector)
			idx.AddVector(&vec, embedding.Scale, i)
		}
		idx.Sync()

		// Concurrent searches
		numGoroutines := 20
		numSearches := 50
		var wg sync.WaitGroup

		query := "document about topic 5"
		embedding, _ := model.EmbedInt8(query)
		var queryVec simd.Vec512
		copy(queryVec[:], embedding.Vector)

		start := time.Now()
		for g := 0; g < numGoroutines; g++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for s := 0; s < numSearches; s++ {
					results := idx.SearchTopK(&queryVec, 10)
					if len(results) == 0 {
						t.Errorf("Goroutine %d: No results returned", id)
					}
				}
			}(g)
		}

		wg.Wait()
		elapsed := time.Since(start)

		totalSearches := numGoroutines * numSearches
		qps := float64(totalSearches) / elapsed.Seconds()
		t.Logf("Concurrent search: %d searches in %v (%.0f QPS)", totalSearches, elapsed, qps)

		// Verify search count
		finalStats := idx.Stats()
		if finalStats.TotalSearches < uint64(totalSearches) {
			t.Errorf("Expected at least %d searches, got %d", totalSearches, finalStats.TotalSearches)
		}
	})

	t.Run("BatchSearch", func(t *testing.T) {
		config := SharedMemoryConfig{
			BasePath:    testPath + "_batch",
			MaxVectors:  500,
			CreateIfNew: true,
		}

		idx, err := NewSharedMemoryIndex(config)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}
		defer idx.Close()

		// Add vectors
		for i := 0; i < 50; i++ {
			text := fmt.Sprintf("batch test document %d", i)
			embedding, _ := model.EmbedInt8(text)
			var vec simd.Vec512
			copy(vec[:], embedding.Vector)
			idx.AddVector(&vec, embedding.Scale, i)
		}

		// Prepare batch queries
		queries := make([]*simd.Vec512, 10)
		for i := 0; i < 10; i++ {
			text := fmt.Sprintf("batch query %d", i)
			embedding, _ := model.EmbedInt8(text)
			vec := &simd.Vec512{}
			copy(vec[:], embedding.Vector)
			queries[i] = vec
		}

		// Batch search
		start := time.Now()
		results := idx.BatchSearch(queries, 5)
		elapsed := time.Since(start)

		if len(results) != len(queries) {
			t.Errorf("Expected %d result sets, got %d", len(queries), len(results))
		}

		for i, res := range results {
			if len(res) == 0 {
				t.Errorf("Query %d returned no results", i)
			}
		}

		t.Logf("Batch search: %d queries in %v", len(queries), elapsed)
	})

	t.Run("MemoryBounds", func(t *testing.T) {
		config := SharedMemoryConfig{
			BasePath:    testPath + "_bounds",
			MaxVectors:  10,
			CreateIfNew: true,
		}

		idx, err := NewSharedMemoryIndex(config)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}
		defer idx.Close()

		// Fill to capacity
		for i := 0; i < 10; i++ {
			var vec simd.Vec512
			vec[0] = int8(i)
			if err := idx.AddVector(&vec, 1.0, i); err != nil {
				t.Errorf("Failed to add vector %d: %v", i, err)
			}
		}

		// Try to exceed capacity
		var vec simd.Vec512
		err = idx.AddVector(&vec, 1.0, 100)
		if err == nil {
			t.Error("Expected error when exceeding capacity")
		}
	})

	t.Run("GetVector", func(t *testing.T) {
		config := SharedMemoryConfig{
			BasePath:    testPath + "_get",
			MaxVectors:  100,
			CreateIfNew: true,
			CacheSize:   10,
		}

		idx, err := NewSharedMemoryIndex(config)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}
		defer idx.Close()

		// Add known vector
		var testVec simd.Vec512
		for i := 0; i < 512; i++ {
			testVec[i] = int8(i % 128)
		}
		idx.AddVector(&testVec, 2.5, 42)

		// Retrieve vector
		retrieved, err := idx.GetVector(0)
		if err != nil {
			t.Fatalf("Failed to get vector: %v", err)
		}

		// Verify content
		for i := 0; i < 512; i++ {
			if retrieved[i] != testVec[i] {
				t.Errorf("Vector mismatch at index %d: expected %d, got %d", 
					i, testVec[i], retrieved[i])
				break
			}
		}

		// Test out of bounds
		_, err = idx.GetVector(100)
		if err == nil {
			t.Error("Expected error for out of bounds index")
		}
	})
}

func TestSharedMemoryPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	testPath := "/tmp/gobed_perf_test_" + fmt.Sprint(time.Now().UnixNano())
	defer os.RemoveAll(testPath)

	model, err := LoadModel()
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	config := SharedMemoryConfig{
		BasePath:    testPath,
		MaxVectors:  10000,
		CreateIfNew: true,
		CacheSize:   1000,
	}

	idx, err := NewSharedMemoryIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer idx.Close()

	// Benchmark indexing
	t.Run("IndexingPerformance", func(t *testing.T) {
		numDocs := 1000
		start := time.Now()

		for i := 0; i < numDocs; i++ {
			text := fmt.Sprintf("performance test document number %d with some content", i)
			embedding, _ := model.EmbedInt8(text)
			var vec simd.Vec512
			copy(vec[:], embedding.Vector)
			idx.AddVector(&vec, embedding.Scale, i)
		}
		idx.Sync()

		elapsed := time.Since(start)
		docsPerSec := float64(numDocs) / elapsed.Seconds()
		t.Logf("Indexed %d documents in %v (%.0f docs/sec)", numDocs, elapsed, docsPerSec)
	})

	// Benchmark search
	t.Run("SearchPerformance", func(t *testing.T) {
		query := "performance test query"
		embedding, _ := model.EmbedInt8(query)
		var queryVec simd.Vec512
		copy(queryVec[:], embedding.Vector)

		// Warmup
		for i := 0; i < 10; i++ {
			idx.SearchTopK(&queryVec, 10)
		}

		// Benchmark
		numSearches := 100
		start := time.Now()
		for i := 0; i < numSearches; i++ {
			results := idx.SearchTopK(&queryVec, 10)
			if len(results) == 0 {
				t.Error("No results returned")
			}
		}
		elapsed := time.Since(start)

		avgLatency := elapsed / time.Duration(numSearches)
		qps := float64(numSearches) / elapsed.Seconds()
		t.Logf("Search performance: %v avg latency, %.0f QPS", avgLatency, qps)
	})

	// Memory stats
	stats := idx.Stats()
	t.Logf("Memory usage: %.2f MB for %d vectors", stats.MemoryUsageMB, stats.NumVectors)
}

func BenchmarkSharedMemorySearch(b *testing.B) {
	testPath := "/tmp/gobed_bench_" + fmt.Sprint(time.Now().UnixNano())
	defer os.RemoveAll(testPath)

	model, _ := LoadModel()
	config := SharedMemoryConfig{
		BasePath:    testPath,
		MaxVectors:  1000,
		CreateIfNew: true,
	}

	idx, _ := NewSharedMemoryIndex(config)
	defer idx.Close()

	// Add vectors
	for i := 0; i < 100; i++ {
		text := fmt.Sprintf("benchmark document %d", i)
		embedding, _ := model.EmbedInt8(text)
		var vec simd.Vec512
		copy(vec[:], embedding.Vector)
		idx.AddVector(&vec, embedding.Scale, i)
	}

	query := "benchmark query"
	embedding, _ := model.EmbedInt8(query)
	var queryVec simd.Vec512
	copy(queryVec[:], embedding.Vector)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			idx.SearchTopK(&queryVec, 10)
		}
	})
}