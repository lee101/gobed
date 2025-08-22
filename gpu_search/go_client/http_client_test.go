package main

import (
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// Test helpers
func generateRandomEmbeddings(n int) [][]int8 {
	rand.Seed(time.Now().UnixNano())
	embeddings := make([][]int8, n)
	for i := range embeddings {
		embeddings[i] = make([]int8, 512)
		for j := range embeddings[i] {
			embeddings[i][j] = int8(rand.Intn(256) - 128)
		}
	}
	return embeddings
}

func generateRandomQuery() []int8 {
	query := make([]int8, 512)
	for i := range query {
		query[i] = int8(rand.Intn(256) - 128)
	}
	return query
}

// Mock server for testing
func createMockGPUServer() *httptest.Server {
	mux := http.NewServeMux()
	
	var database [][]int8
	
	// Health endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		resp := HealthResponse{
			Status:         "healthy",
			Device:         "cuda",
			CudaAvailable:  true,
			DatabaseLoaded: database != nil,
			DatabaseSize:   len(database),
		}
		json.NewEncoder(w).Encode(resp)
	})
	
	// Load endpoint
	mux.HandleFunc("/load", func(w http.ResponseWriter, r *http.Request) {
		var req LoadRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		database = req.Embeddings
		
		resp := LoadResponse{
			Status:   "loaded",
			Count:    len(database),
			Shape:    []int{len(database), 512},
			Device:   "cuda",
			MemoryMB: float64(len(database)*512) / 1e6,
		}
		json.NewEncoder(w).Encode(resp)
	})
	
	// Search endpoint
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		var req SearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		if database == nil {
			http.Error(w, "Database not loaded", http.StatusBadRequest)
			return
		}
		
		// Mock search results
		k := req.K
		if k > len(database) {
			k = len(database)
		}
		
		ids := make([]int, k)
		scores := make([]float32, k)
		for i := 0; i < k; i++ {
			ids[i] = i
			scores[i] = float32(100 - i*10)
		}
		
		resp := SearchResponse{
			IDs:          ids,
			Scores:       scores,
			SearchTimeMs: 0.5,
			K:            k,
		}
		json.NewEncoder(w).Encode(resp)
	})
	
	// Batch search endpoint
	mux.HandleFunc("/batch_search", func(w http.ResponseWriter, r *http.Request) {
		var req BatchSearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		if database == nil {
			http.Error(w, "Database not loaded", http.StatusBadRequest)
			return
		}
		
		k := req.K
		if k > len(database) {
			k = len(database)
		}
		
		batchSize := len(req.Queries)
		batchIDs := make([][]int, batchSize)
		batchScores := make([][]float32, batchSize)
		
		for b := 0; b < batchSize; b++ {
			ids := make([]int, k)
			scores := make([]float32, k)
			for i := 0; i < k; i++ {
				ids[i] = i
				scores[i] = float32(100 - i*10)
			}
			batchIDs[b] = ids
			batchScores[b] = scores
		}
		
		resp := BatchSearchResponse{
			BatchIDs:     batchIDs,
			BatchScores:  batchScores,
			BatchSize:    batchSize,
			SearchTimeMs: float64(batchSize) * 0.5,
			QPS:          float64(batchSize) * 2000,
			K:            k,
		}
		json.NewEncoder(w).Encode(resp)
	})
	
	// Benchmark endpoint
	mux.HandleFunc("/benchmark", func(w http.ResponseWriter, r *http.Request) {
		if database == nil {
			http.Error(w, "Database not loaded", http.StatusBadRequest)
			return
		}
		
		result := BenchmarkResult{}
		result.SingleQuery.AvgLatencyMs = 0.5
		result.SingleQuery.QPS = 2000
		result.SingleQuery.Iterations = 100
		
		result.Batch.BatchSize = 32
		result.Batch.BatchLatencyMs = 1.0
		result.Batch.QPS = 32000
		result.Batch.Iterations = 10
		
		result.Database.Size = len(database)
		result.Database.Dimensions = 512
		result.Database.Device = "cuda"
		result.Database.MemoryMB = float64(len(database)*512) / 1e6
		
		json.NewEncoder(w).Encode(result)
	})
	
	// Clear endpoint
	mux.HandleFunc("/clear", func(w http.ResponseWriter, r *http.Request) {
		database = nil
		json.NewEncoder(w).Encode(map[string]string{"status": "cleared"})
	})
	
	return httptest.NewServer(mux)
}

// Tests
func TestHTTPClient_Health(t *testing.T) {
	server := createMockGPUServer()
	defer server.Close()
	
	client := NewHTTPClient(server.URL)
	
	health, err := client.Health()
	if err != nil {
		t.Fatalf("Health check failed: %v", err)
	}
	
	if health.Status != "healthy" {
		t.Errorf("Expected healthy status, got %s", health.Status)
	}
	
	if !health.CudaAvailable {
		t.Error("Expected CUDA to be available")
	}
}

func TestHTTPClient_LoadDatabase(t *testing.T) {
	server := createMockGPUServer()
	defer server.Close()
	
	client := NewHTTPClient(server.URL)
	
	embeddings := generateRandomEmbeddings(1000)
	
	result, err := client.LoadDatabase(embeddings)
	if err != nil {
		t.Fatalf("Failed to load database: %v", err)
	}
	
	if result.Count != 1000 {
		t.Errorf("Expected 1000 embeddings, got %d", result.Count)
	}
	
	if result.Device != "cuda" {
		t.Errorf("Expected cuda device, got %s", result.Device)
	}
}

func TestHTTPClient_Search(t *testing.T) {
	server := createMockGPUServer()
	defer server.Close()
	
	client := NewHTTPClient(server.URL)
	
	// Load database first
	embeddings := generateRandomEmbeddings(1000)
	_, err := client.LoadDatabase(embeddings)
	if err != nil {
		t.Fatalf("Failed to load database: %v", err)
	}
	
	// Perform search
	query := generateRandomQuery()
	k := 10
	
	result, err := client.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	
	if len(result.IDs) != k {
		t.Errorf("Expected %d results, got %d", k, len(result.IDs))
	}
	
	if len(result.Scores) != k {
		t.Errorf("Expected %d scores, got %d", k, len(result.Scores))
	}
	
	// Check that scores are sorted (descending)
	for i := 1; i < len(result.Scores); i++ {
		if result.Scores[i] > result.Scores[i-1] {
			t.Errorf("Scores not sorted: scores[%d]=%f > scores[%d]=%f", 
				i, result.Scores[i], i-1, result.Scores[i-1])
		}
	}
	
	t.Logf("Search completed in %.2fms", result.SearchTimeMs)
}

func TestHTTPClient_BatchSearch(t *testing.T) {
	server := createMockGPUServer()
	defer server.Close()
	
	client := NewHTTPClient(server.URL)
	
	// Load database
	embeddings := generateRandomEmbeddings(1000)
	_, err := client.LoadDatabase(embeddings)
	if err != nil {
		t.Fatalf("Failed to load database: %v", err)
	}
	
	// Prepare batch queries
	batchSize := 16
	queries := make([][]int8, batchSize)
	for i := range queries {
		queries[i] = generateRandomQuery()
	}
	
	k := 5
	result, err := client.BatchSearch(queries, k)
	if err != nil {
		t.Fatalf("Batch search failed: %v", err)
	}
	
	if result.BatchSize != batchSize {
		t.Errorf("Expected batch size %d, got %d", batchSize, result.BatchSize)
	}
	
	if len(result.BatchIDs) != batchSize {
		t.Errorf("Expected %d result sets, got %d", batchSize, len(result.BatchIDs))
	}
	
	// Check each batch result
	for i, ids := range result.BatchIDs {
		if len(ids) != k {
			t.Errorf("Batch %d: expected %d results, got %d", i, k, len(ids))
		}
	}
	
	t.Logf("Batch search completed in %.2fms, QPS: %.0f", result.SearchTimeMs, result.QPS)
}

func TestHTTPClient_Benchmark(t *testing.T) {
	server := createMockGPUServer()
	defer server.Close()
	
	client := NewHTTPClient(server.URL)
	
	// Load database
	embeddings := generateRandomEmbeddings(10000)
	_, err := client.LoadDatabase(embeddings)
	if err != nil {
		t.Fatalf("Failed to load database: %v", err)
	}
	
	// Run benchmark
	result, err := client.Benchmark()
	if err != nil {
		t.Fatalf("Benchmark failed: %v", err)
	}
	
	t.Logf("Single query: %.2fms latency, %.0f QPS", 
		result.SingleQuery.AvgLatencyMs, result.SingleQuery.QPS)
	
	t.Logf("Batch (size=%d): %.2fms latency, %.0f QPS", 
		result.Batch.BatchSize, result.Batch.BatchLatencyMs, result.Batch.QPS)
	
	t.Logf("Database: %d vectors, %.1f MB on %s", 
		result.Database.Size, result.Database.MemoryMB, result.Database.Device)
	
	// Performance assertions
	if result.SingleQuery.AvgLatencyMs > 10 {
		t.Logf("Warning: Single query latency %.2fms exceeds target of 10ms", 
			result.SingleQuery.AvgLatencyMs)
	}
}

func TestHTTPClient_Clear(t *testing.T) {
	server := createMockGPUServer()
	defer server.Close()
	
	client := NewHTTPClient(server.URL)
	
	// Load then clear
	embeddings := generateRandomEmbeddings(100)
	_, err := client.LoadDatabase(embeddings)
	if err != nil {
		t.Fatalf("Failed to load database: %v", err)
	}
	
	err = client.Clear()
	if err != nil {
		t.Fatalf("Failed to clear database: %v", err)
	}
	
	// Check that database is cleared
	health, err := client.Health()
	if err != nil {
		t.Fatalf("Health check failed: %v", err)
	}
	
	if health.DatabaseLoaded {
		t.Error("Database should not be loaded after clear")
	}
	
	if health.DatabaseSize != 0 {
		t.Errorf("Database size should be 0 after clear, got %d", health.DatabaseSize)
	}
}

func BenchmarkHTTPClient_Search(b *testing.B) {
	server := createMockGPUServer()
	defer server.Close()
	
	client := NewHTTPClient(server.URL)
	
	// Load database
	embeddings := generateRandomEmbeddings(10000)
	client.LoadDatabase(embeddings)
	
	query := generateRandomQuery()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.Search(query, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}

func BenchmarkHTTPClient_BatchSearch(b *testing.B) {
	server := createMockGPUServer()
	defer server.Close()
	
	client := NewHTTPClient(server.URL)
	
	// Load database
	embeddings := generateRandomEmbeddings(10000)
	client.LoadDatabase(embeddings)
	
	// Prepare batch
	queries := make([][]int8, 32)
	for i := range queries {
		queries[i] = generateRandomQuery()
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.BatchSearch(queries, 10)
		if err != nil {
			b.Fatalf("Batch search failed: %v", err)
		}
	}
	
	b.SetBytes(int64(len(queries) * 512))
}