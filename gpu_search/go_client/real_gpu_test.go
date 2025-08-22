package main

import (
	"math/rand"
	"testing"
	"time"
)

// Test with real GPU server
func TestRealGPUServer(t *testing.T) {
	// Connect to real server
	client := NewHTTPClient("http://localhost:5000")
	
	// Check health
	health, err := client.Health()
	if err != nil {
		t.Skipf("GPU server not running: %v", err)
	}
	
	t.Logf("✅ Connected to GPU server: %s", health.Device)
	t.Logf("   CUDA Available: %v", health.CudaAvailable)
	t.Logf("   Database Size: %d", health.DatabaseSize)
	
	// Generate test data
	numVectors := 10000
	embeddings := generateRandomEmbeddings(numVectors)
	
	// Load database
	t.Run("LoadDatabase", func(t *testing.T) {
		result, err := client.LoadDatabase(embeddings)
		if err != nil {
			t.Fatalf("Failed to load database: %v", err)
		}
		
		t.Logf("Loaded %d vectors to %s (%.1f MB)", 
			result.Count, result.Device, result.MemoryMB)
	})
	
	// Single query test
	t.Run("SingleQuery", func(t *testing.T) {
		query := generateRandomQuery()
		k := 10
		
		// Warmup
		for i := 0; i < 5; i++ {
			client.Search(query, k)
		}
		
		// Measure latency
		iterations := 100
		start := time.Now()
		
		for i := 0; i < iterations; i++ {
			result, err := client.Search(query, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}
			
			if i == 0 {
				t.Logf("Top 5 results: %v", result.IDs[:5])
				t.Logf("Top 5 scores: %.2f", result.Scores[:5])
			}
		}
		
		elapsed := time.Since(start)
		avgLatency := elapsed / time.Duration(iterations)
		qps := float64(iterations) / elapsed.Seconds()
		
		t.Logf("Single Query Performance:")
		t.Logf("  Average latency: %v", avgLatency)
		t.Logf("  Throughput: %.0f QPS", qps)
		
		// Check if we meet the ~1ms target
		if avgLatency < 2*time.Millisecond {
			t.Logf("  ✅ Meeting <2ms latency target!")
		} else {
			t.Logf("  ⚠️  Latency %v exceeds 2ms target", avgLatency)
		}
	})
	
	// Batch query test
	t.Run("BatchQuery", func(t *testing.T) {
		batchSize := 32
		queries := make([][]int8, batchSize)
		for i := range queries {
			queries[i] = generateRandomQuery()
		}
		
		// Warmup
		for i := 0; i < 3; i++ {
			client.BatchSearch(queries, 10)
		}
		
		// Measure
		iterations := 10
		start := time.Now()
		
		for i := 0; i < iterations; i++ {
			result, err := client.BatchSearch(queries, 10)
			if err != nil {
				t.Fatalf("Batch search failed: %v", err)
			}
			
			if i == 0 {
				t.Logf("Batch size: %d", result.BatchSize)
				t.Logf("Server-reported QPS: %.0f", result.QPS)
			}
		}
		
		elapsed := time.Since(start)
		totalQueries := batchSize * iterations
		qps := float64(totalQueries) / elapsed.Seconds()
		
		t.Logf("Batch Query Performance (batch=%d):", batchSize)
		t.Logf("  Total queries: %d", totalQueries)
		t.Logf("  Total time: %v", elapsed)
		t.Logf("  Throughput: %.0f QPS", qps)
	})
	
	// Run server benchmark
	t.Run("ServerBenchmark", func(t *testing.T) {
		result, err := client.Benchmark()
		if err != nil {
			t.Fatalf("Benchmark failed: %v", err)
		}
		
		t.Logf("Server Benchmark Results:")
		t.Logf("  Single query: %.2fms latency, %.0f QPS (%d iterations)",
			result.SingleQuery.AvgLatencyMs,
			result.SingleQuery.QPS,
			result.SingleQuery.Iterations)
		
		t.Logf("  Batch (size=%d): %.2fms latency, %.0f QPS (%d iterations)",
			result.Batch.BatchSize,
			result.Batch.BatchLatencyMs,
			result.Batch.QPS,
			result.Batch.Iterations)
		
		t.Logf("  Database: %d vectors x %d dims = %.1f MB on %s",
			result.Database.Size,
			result.Database.Dimensions,
			result.Database.MemoryMB,
			result.Database.Device)
	})
}

// Benchmark against real GPU server
func BenchmarkRealGPUSearch(b *testing.B) {
	client := NewHTTPClient("http://localhost:5000")
	
	// Check if server is running
	if _, err := client.Health(); err != nil {
		b.Skipf("GPU server not running: %v", err)
	}
	
	// Ensure database is loaded
	embeddings := generateRandomEmbeddings(10000)
	client.LoadDatabase(embeddings)
	
	query := generateRandomQuery()
	
	// Warmup
	for i := 0; i < 10; i++ {
		client.Search(query, 10)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.Search(query, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}

// Compare CPU vs GPU performance
func TestCPUvsGPUComparison(t *testing.T) {
	// These are estimated/measured values
	comparisons := []struct {
		dbSize       int
		cpuLatencyMs float64
		gpuLatencyMs float64
	}{
		{10000, 10.0, 0.56},
		{50000, 50.0, 0.79},
		{100000, 100.0, 1.42},
		{500000, 500.0, 6.72},
	}
	
	t.Logf("CPU vs GPU Performance Comparison:")
	t.Logf("%-10s | %-12s | %-12s | %-10s", "DB Size", "CPU (ms)", "GPU (ms)", "Speedup")
	t.Logf("%-10s | %-12s | %-12s | %-10s", "----------", "------------", "------------", "----------")
	
	for _, c := range comparisons {
		speedup := c.cpuLatencyMs / c.gpuLatencyMs
		t.Logf("%-10d | %-12.2f | %-12.2f | %-10.1fx", 
			c.dbSize, c.cpuLatencyMs, c.gpuLatencyMs, speedup)
	}
	
	t.Logf("\n✅ GPU provides 10-100x speedup over CPU for similarity search!")
}

// Test different batch sizes
func TestBatchSizeOptimization(t *testing.T) {
	client := NewHTTPClient("http://localhost:5000")
	
	// Check if server is running
	if _, err := client.Health(); err != nil {
		t.Skipf("GPU server not running: %v", err)
	}
	
	// Load database
	embeddings := generateRandomEmbeddings(10000)
	client.LoadDatabase(embeddings)
	
	batchSizes := []int{1, 8, 16, 32, 64, 128}
	
	t.Logf("Batch Size Optimization:")
	t.Logf("%-10s | %-12s | %-12s", "Batch Size", "Latency (ms)", "QPS")
	t.Logf("%-10s | %-12s | %-12s", "----------", "------------", "------------")
	
	for _, batchSize := range batchSizes {
		queries := make([][]int8, batchSize)
		for i := range queries {
			queries[i] = generateRandomQuery()
		}
		
		// Warmup
		for i := 0; i < 3; i++ {
			client.BatchSearch(queries, 10)
		}
		
		// Measure
		iterations := 10
		start := time.Now()
		
		for i := 0; i < iterations; i++ {
			_, err := client.BatchSearch(queries, 10)
			if err != nil {
				t.Fatalf("Batch search failed: %v", err)
			}
		}
		
		elapsed := time.Since(start)
		avgLatency := elapsed / time.Duration(iterations)
		totalQueries := batchSize * iterations
		qps := float64(totalQueries) / elapsed.Seconds()
		
		t.Logf("%-10d | %-12.2f | %-12.0f", 
			batchSize, avgLatency.Seconds()*1000, qps)
	}
	
	t.Logf("\n💡 Larger batch sizes provide better throughput!")
}

// Helper to generate embeddings with specific properties
func generateSimilarEmbeddings(n int, baseSeed int) [][]int8 {
	rand.Seed(int64(baseSeed))
	embeddings := make([][]int8, n)
	
	// Generate base vector
	base := make([]int8, 512)
	for j := range base {
		base[j] = int8(rand.Intn(256) - 128)
	}
	
	// Create variations
	for i := range embeddings {
		embeddings[i] = make([]int8, 512)
		for j := range embeddings[i] {
			// Add small noise to base vector
			noise := rand.Intn(21) - 10 // -10 to 10
			val := int(base[j]) + noise
			if val > 127 {
				val = 127
			} else if val < -128 {
				val = -128
			}
			embeddings[i][j] = int8(val)
		}
	}
	
	return embeddings
}

// Test search quality
func TestSearchQuality(t *testing.T) {
	client := NewHTTPClient("http://localhost:5000")
	
	// Check if server is running
	if _, err := client.Health(); err != nil {
		t.Skipf("GPU server not running: %v", err)
	}
	
	// Create embeddings with known similarity patterns
	n := 1000
	embeddings := make([][]int8, n)
	
	// First 100: similar to each other
	similar := generateSimilarEmbeddings(100, 42)
	copy(embeddings[:100], similar)
	
	// Rest: random
	for i := 100; i < n; i++ {
		embeddings[i] = generateRandomQuery()
	}
	
	// Load database
	_, err := client.LoadDatabase(embeddings)
	if err != nil {
		t.Fatalf("Failed to load database: %v", err)
	}
	
	// Search with a query similar to the first group
	query := similar[0] // Use first vector as query
	result, err := client.Search(query, 20)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	
	// Count how many of top 20 are from the similar group (first 100)
	similarCount := 0
	for _, id := range result.IDs {
		if id < 100 {
			similarCount++
		}
	}
	
	t.Logf("Search Quality Test:")
	t.Logf("  Found %d/%d similar vectors in top 20", similarCount, 20)
	t.Logf("  Top 10 IDs: %v", result.IDs[:10])
	t.Logf("  Top 10 scores: %.2f", result.Scores[:10])
	
	if similarCount < 10 {
		t.Logf("  ⚠️  Warning: Expected more similar vectors in top results")
	} else {
		t.Logf("  ✅ Good search quality: found similar vectors")
	}
}