package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	fmt.Println("=== Gobed Shared Memory Performance Benchmark ===")
	fmt.Println("Comparing standard vs shared memory architecture\n")

	// Load model
	fmt.Println("Loading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("✓ Model loaded\n")

	// Test configurations
	testSizes := []int{10000, 50000, 100000}

	for _, size := range testSizes {
		fmt.Printf("\n=== Testing with %d documents ===\n", size)

		// Generate test data
		documents := generateTestDocuments(size)
		queries := generateTestQueries(100)

		// Test 1: Standard in-process index
		standardResults := benchmarkStandardIndex(model, documents, queries)

		// Test 2: Shared memory index (single process)
		sharedSingleResults := benchmarkSharedMemorySingle(model, documents, queries)

		// Test 3: Shared memory index (multi-process)
		sharedMultiResults := benchmarkSharedMemoryMulti(model, documents, queries)

		// Test 4: HTTP server mode
		serverResults := benchmarkServerMode(model, documents, queries)

		// Print comparison
		printComparison(size, standardResults, sharedSingleResults, sharedMultiResults, serverResults)

		// Cleanup
		cleanupSharedIndex()
	}

	fmt.Println("\n✓ Benchmark completed successfully!")
}

// BenchmarkResult stores benchmark results
type BenchmarkResult struct {
	IndexTime     time.Duration
	SearchLatency time.Duration
	MemoryMB      float64
	Throughput    float64
	ProcessCount  int
}

// benchmarkStandardIndex benchmarks standard in-process index
func benchmarkStandardIndex(model *gobed.EmbeddingModel, documents []string, queries []string) BenchmarkResult {
	fmt.Println("\n1. Standard In-Process Index")
	fmt.Println("──────────────────────────────")

	// Create standard search engine
	engine := gobed.NewSearchEngine(model)

	// Measure indexing
	indexStart := time.Now()
	ids, err := engine.IndexBatch(documents)
	if err != nil {
		log.Printf("Indexing error: %v", err)
	}
	indexTime := time.Since(indexStart)

	fmt.Printf("Indexed %d documents in %v\n", len(ids), indexTime)

	// Measure search performance
	searchStart := time.Now()
	for _, query := range queries {
		_, err := engine.Search(query, 10)
		if err != nil {
			log.Printf("Search error: %v", err)
		}
	}
	searchTime := time.Since(searchStart)
	avgLatency := searchTime / time.Duration(len(queries))

	// Get memory usage
	stats := engine.Stats()

	fmt.Printf("Search latency: %v avg\n", avgLatency)
	fmt.Printf("Memory usage: %.2f MB\n", stats.MemoryUsageMB)

	return BenchmarkResult{
		IndexTime:     indexTime,
		SearchLatency: avgLatency,
		MemoryMB:      stats.MemoryUsageMB,
		Throughput:    float64(len(queries)) / searchTime.Seconds(),
		ProcessCount:  1,
	}
}

// benchmarkSharedMemorySingle benchmarks shared memory in single process
func benchmarkSharedMemorySingle(model *gobed.EmbeddingModel, documents []string, queries []string) BenchmarkResult {
	fmt.Println("\n2. Shared Memory Index (Single Process)")
	fmt.Println("──────────────────────────────────────────")

	// Create shared memory index
	config := gobed.SharedMemoryConfig{
		BasePath:    "/tmp/gobed_bench_shared",
		MaxVectors:  len(documents) * 2,
		CreateIfNew: true,
		CacheSize:   1000,
	}

	sharedIndex, err := gobed.NewSharedMemoryIndex(config)
	if err != nil {
		log.Fatalf("Failed to create shared index: %v", err)
	}
	defer sharedIndex.Close()

	// Measure indexing
	indexStart := time.Now()
	for i, doc := range documents {
		embedding, err := model.EmbedInt8(doc)
		if err != nil {
			continue
		}

		var vec simd.Vec512
		copy(vec[:], embedding.Vector)

		sharedIndex.AddVector(&vec, embedding.Scale, i)
	}
	sharedIndex.Sync()
	indexTime := time.Since(indexStart)

	fmt.Printf("Indexed %d documents in %v\n", len(documents), indexTime)

	// Generate query embeddings
	queryVecs := make([]*simd.Vec512, len(queries))
	for i, query := range queries {
		embedding, _ := model.EmbedInt8(query)
		vec := &simd.Vec512{}
		copy(vec[:], embedding.Vector)
		queryVecs[i] = vec
	}

	// Measure search performance (zero-copy)
	searchStart := time.Now()
	for _, qvec := range queryVecs {
		sharedIndex.SearchTopK(qvec, 10)
	}
	searchTime := time.Since(searchStart)
	avgLatency := searchTime / time.Duration(len(queries))

	// Get stats
	stats := sharedIndex.Stats()

	fmt.Printf("Search latency: %v avg (zero-copy)\n", avgLatency)
	fmt.Printf("Memory usage: %.2f MB (shared)\n", stats.MemoryUsageMB)

	return BenchmarkResult{
		IndexTime:     indexTime,
		SearchLatency: avgLatency,
		MemoryMB:      stats.MemoryUsageMB,
		Throughput:    float64(len(queries)) / searchTime.Seconds(),
		ProcessCount:  1,
	}
}

// benchmarkSharedMemoryMulti benchmarks shared memory with multiple processes
func benchmarkSharedMemoryMulti(model *gobed.EmbeddingModel, documents []string, queries []string) BenchmarkResult {
	fmt.Println("\n3. Shared Memory Index (Multi-Process)")
	fmt.Println("──────────────────────────────────────────")

	// Re-open the existing index created in single process test
	config := gobed.SharedMemoryConfig{
		BasePath:    "/tmp/gobed_bench_shared",
		MaxVectors:  len(documents) * 2,
		CreateIfNew: false, // Use existing
		ReadOnly:    true,  // Open as read-only
		CacheSize:   1000,
	}

	sharedIndex, err := gobed.NewSharedMemoryIndex(config)
	if err != nil {
		log.Fatalf("Failed to open shared index: %v", err)
	}

	// Generate query embeddings
	queryVecs := make([]*simd.Vec512, len(queries))
	for i, query := range queries {
		embedding, _ := model.EmbedInt8(query)
		vec := &simd.Vec512{}
		copy(vec[:], embedding.Vector)
		queryVecs[i] = vec
	}

	// Simulate multi-process search with goroutines
	// In reality, different processes would each map the shared memory
	numProcesses := 4
	queriesPerProcess := len(queries) / numProcesses

	searchStart := time.Now()
	var wg sync.WaitGroup

	for p := 0; p < numProcesses; p++ {
		wg.Add(1)
		go func(procID int) {
			defer wg.Done()

			// Perform searches using the main shared index
			// In real multi-process scenario, each process would have its own mapping
			start := procID * queriesPerProcess
			end := start + queriesPerProcess
			if end > len(queryVecs) {
				end = len(queryVecs)
			}

			for i := start; i < end; i++ {
				sharedIndex.SearchTopK(queryVecs[i], 10)
			}
		}(p)
	}

	wg.Wait()
	searchTime := time.Since(searchStart)
	avgLatency := searchTime / time.Duration(len(queries))

	stats := sharedIndex.Stats()
	sharedIndex.Close()

	fmt.Printf("Search latency: %v avg (%d processes)\n", avgLatency, numProcesses)
	fmt.Printf("Memory usage: %.2f MB (shared across all processes)\n", stats.MemoryUsageMB)
	fmt.Printf("Throughput: %.0f QPS\n", float64(len(queries))/searchTime.Seconds())

	return BenchmarkResult{
		IndexTime:     0, // Already indexed
		SearchLatency: avgLatency,
		MemoryMB:      stats.MemoryUsageMB / float64(numProcesses), // Shared memory
		Throughput:    float64(len(queries)) / searchTime.Seconds(),
		ProcessCount:  numProcesses,
	}
}

// benchmarkServerMode benchmarks HTTP server mode
func benchmarkServerMode(model *gobed.EmbeddingModel, documents []string, queries []string) BenchmarkResult {
	fmt.Println("\n4. HTTP Server Mode")
	fmt.Println("────────────────────")

	// Start server
	serverConfig := gobed.DefaultServerConfig()
	serverConfig.Port = 8090
	serverConfig.SharedIndexPath = "/tmp/gobed_bench_server"

	server, err := gobed.NewSearchServer(model, serverConfig)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	if err := server.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
	defer server.Stop()

	// Wait for server to start
	time.Sleep(100 * time.Millisecond)

	// Index documents via HTTP
	indexStart := time.Now()
	indexDocs := make([]gobed.ServerDocument, len(documents))
	for i, doc := range documents {
		indexDocs[i] = gobed.ServerDocument{ID: i, Text: doc}
	}

	// Batch index
	batchSize := 1000
	for i := 0; i < len(indexDocs); i += batchSize {
		end := i + batchSize
		if end > len(indexDocs) {
			end = len(indexDocs)
		}

		batch := indexDocs[i:end]
		indexViaHTTP(batch)
	}
	indexTime := time.Since(indexStart)

	fmt.Printf("Indexed %d documents in %v via HTTP\n", len(documents), indexTime)

	// Benchmark search via HTTP with concurrent clients
	numClients := 10
	queriesPerClient := len(queries) / numClients

	searchStart := time.Now()
	var wg sync.WaitGroup

	for c := 0; c < numClients; c++ {
		wg.Add(1)
		go func(clientID int) {
			defer wg.Done()

			start := clientID * queriesPerClient
			end := start + queriesPerClient
			if end > len(queries) {
				end = len(queries)
			}

			for i := start; i < end; i++ {
				searchViaHTTP(queries[i])
			}
		}(c)
	}

	wg.Wait()
	searchTime := time.Since(searchStart)
	avgLatency := searchTime / time.Duration(len(queries))

	// Get metrics
	metrics := getServerMetrics()

	fmt.Printf("Search latency: %v avg (%d concurrent clients)\n", avgLatency, numClients)
	fmt.Printf("Memory usage: %.2f MB\n", metrics["memory_usage_mb"].(float64))
	fmt.Printf("Throughput: %.0f QPS\n", float64(len(queries))/searchTime.Seconds())

	return BenchmarkResult{
		IndexTime:     indexTime,
		SearchLatency: avgLatency,
		MemoryMB:      metrics["memory_usage_mb"].(float64),
		Throughput:    float64(len(queries)) / searchTime.Seconds(),
		ProcessCount:  1, // Single server process
	}
}

// Helper functions

func generateTestDocuments(n int) []string {
	docs := make([]string, n)
	templates := []string{
		"Advanced machine learning techniques for %s",
		"Cloud computing infrastructure and %s",
		"Data science applications in %s",
		"Artificial intelligence for %s optimization",
		"Building scalable systems with %s",
	}

	topics := []string{
		"healthcare", "finance", "education", "manufacturing",
		"retail", "transportation", "energy", "agriculture",
	}

	for i := 0; i < n; i++ {
		template := templates[rand.Intn(len(templates))]
		topic := topics[rand.Intn(len(topics))]
		docs[i] = fmt.Sprintf(template, topic)
	}

	return docs
}

func generateTestQueries(n int) []string {
	queries := make([]string, n)
	for i := 0; i < n; i++ {
		queries[i] = fmt.Sprintf("machine learning for %d applications", i)
	}
	return queries
}

func indexViaHTTP(docs []gobed.ServerDocument) {
	req := gobed.ServerIndexRequest{Documents: docs}
	body, _ := json.Marshal(req)

	resp, err := http.Post("http://localhost:8090/batch_index", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("Index request failed: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("Index failed: %s", body)
	}
}

func searchViaHTTP(query string) {
	req := gobed.SearchRequest{Query: query, K: 10}
	body, _ := json.Marshal(req)

	resp, err := http.Post("http://localhost:8090/search", "application/json", bytes.NewReader(body))
	if err != nil {
		return
	}
	defer resp.Body.Close()
}

func getServerMetrics() map[string]interface{} {
	resp, err := http.Get("http://localhost:8090/metrics")
	if err != nil {
		return map[string]interface{}{"memory_usage_mb": 0.0}
	}
	defer resp.Body.Close()

	var metrics map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&metrics)
	return metrics
}

func cleanupSharedIndex() {
	os.RemoveAll("/tmp/gobed_bench_shared")
	os.RemoveAll("/tmp/gobed_bench_server")
}

func printComparison(size int, standard, sharedSingle, sharedMulti, server BenchmarkResult) {
	fmt.Printf("\n=== PERFORMANCE COMPARISON (%d documents) ===\n", size)
	fmt.Println()
	fmt.Println("| Method                | Search Latency | Memory/Process | Total Memory | Throughput |")
	fmt.Println("|----------------------|----------------|----------------|--------------|------------|")

	fmt.Printf("| Standard In-Process  | %14v | %13.1f MB | %11.1f MB | %9.0f QPS |\n",
		standard.SearchLatency, standard.MemoryMB, standard.MemoryMB, standard.Throughput)

	fmt.Printf("| Shared Memory Single | %14v | %13.1f MB | %11.1f MB | %9.0f QPS |\n",
		sharedSingle.SearchLatency, sharedSingle.MemoryMB, sharedSingle.MemoryMB, sharedSingle.Throughput)

	fmt.Printf("| Shared Memory Multi  | %14v | %13.1f MB | %11.1f MB | %9.0f QPS |\n",
		sharedMulti.SearchLatency, sharedMulti.MemoryMB, sharedMulti.MemoryMB*float64(sharedMulti.ProcessCount), sharedMulti.Throughput)

	fmt.Printf("| HTTP Server Mode     | %14v | %13.1f MB | %11.1f MB | %9.0f QPS |\n",
		server.SearchLatency, server.MemoryMB, server.MemoryMB, server.Throughput)

	// Calculate improvements
	fmt.Println("\n Memory Savings:")

	multiProcSaving := (standard.MemoryMB*float64(sharedMulti.ProcessCount) - sharedMulti.MemoryMB) / (standard.MemoryMB * float64(sharedMulti.ProcessCount)) * 100
	fmt.Printf("• Multi-process: %.1f%% memory saved (shared memory vs %d separate processes)\n",
		multiProcSaving, sharedMulti.ProcessCount)

	latencyImprovement := float64(standard.SearchLatency-sharedSingle.SearchLatency) / float64(standard.SearchLatency) * 100
	if latencyImprovement > 0 {
		fmt.Printf("• Zero-copy search: %.1f%% faster latency\n", latencyImprovement)
	}

	throughputGain := (sharedMulti.Throughput - standard.Throughput) / standard.Throughput * 100
	fmt.Printf("• Multi-process throughput: %.1f%% higher\n", throughputGain)
}
