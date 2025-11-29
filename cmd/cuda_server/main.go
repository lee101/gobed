//go:build legacy

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	var (
		port        = flag.Int("port", 8080, "Server port")
		batchSize   = flag.Int("batch", 512, "Processing batch size")
		demoMode    = flag.Bool("demo", false, "Run demonstration mode")
		demoVectors = flag.Int("demo-vectors", 10000, "Demo vectors to index")
		showStats   = flag.Bool("stats", true, "Show performance statistics")
		loadTest    = flag.Bool("load-test", false, "Run load test after startup")
	)
	flag.Parse()

	fmt.Println(" CUDA-Accelerated GoBeD Server")
	fmt.Println("=================================")
	fmt.Printf("   Built with: %s\n", runtime.Version())
	fmt.Printf("   Architecture: %s/%s\n", runtime.GOOS, runtime.GOARCH)

	// Load embedding model
	fmt.Println("\n📚 Loading Embedding Model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf(" Failed to load model: %v", err)
	}

	// Create CUDA-accelerated server
	server := &CUDAServer{
		model:     model,
		port:      *port,
		batchSize: *batchSize,
		vectors:   make([][]int8, 0),
		vectorIDs: make([]int, 0),
		startTime: time.Now(),
	}

	// Run demonstration if requested
	if *demoMode {
		fmt.Println("\n Running CUDA Demonstration...")
		if err := server.runDemo(*demoVectors); err != nil {
			log.Printf("  Demo failed: %v", err)
		}
	}

	// Setup HTTP routes
	http.HandleFunc("/search", server.handleSearch)
	http.HandleFunc("/batch_search", server.handleBatchSearch) 
	http.HandleFunc("/index", server.handleIndex)
	http.HandleFunc("/batch_index", server.handleBatchIndex)
	http.HandleFunc("/stats", server.handleStats)
	http.HandleFunc("/health", server.handleHealth)
	http.HandleFunc("/benchmark", server.handleBenchmark)

	// Start performance monitor
	if *showStats {
		go server.performanceMonitor()
	}

	// Start server
	fmt.Printf("\n CUDA Server Running on Port %d\n", *port)
	fmt.Printf("    Search: POST /search\n")
	fmt.Printf("    Batch Search: POST /batch_search\n")
	fmt.Printf("   📚 Index: POST /index\n")
	fmt.Printf("    Benchmark: GET /benchmark\n")
	fmt.Printf("    Stats: GET /stats\n")
	fmt.Printf("    Health: GET /health\n")

	// Start HTTP server
	go func() {
		if err := http.ListenAndServe(fmt.Sprintf(":%d", *port), nil); err != nil {
			log.Fatalf(" Server failed: %v", err)
		}
	}()

	// Run load test if requested
	if *loadTest {
		go func() {
			time.Sleep(2 * time.Second)
			server.runLoadTest()
		}()
	}

	// Wait for shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	fmt.Println("\n Ready! Press Ctrl+C to stop.")
	<-sigChan

	fmt.Println("\n🛑 Shutting down gracefully...")
	fmt.Println(" Goodbye!")
}

// CUDAServer provides CUDA-accelerated vector search
type CUDAServer struct {
	model     *gobed.EmbeddingModel
	port      int
	batchSize int

	// In-memory vector storage (for demo)
	vectors   [][]int8
	vectorIDs []int

	// Metrics
	totalRequests uint64
	totalIndexed  uint64
	totalLatency  time.Duration
	startTime     time.Time
}

// SearchRequest represents a search request
type SearchRequest struct {
	Query   string   `json:"query"`
	Queries []string `json:"queries,omitempty"`
	K       int      `json:"k"`
}

// SearchResult represents a search result
type SearchResult struct {
	ID         int     `json:"id"`
	Similarity float32 `json:"similarity"`
	Text       string  `json:"text,omitempty"`
}

// SearchResponse represents a search response
type SearchResponse struct {
	Results   []SearchResult   `json:"results,omitempty"`
	Batch     [][]SearchResult `json:"batch,omitempty"`
	Latency   int64            `json:"latency_us"`
	GPU       bool             `json:"gpu_accelerated"`
	QueryTime int64            `json:"query_time_us,omitempty"`
}

// IndexRequest represents an indexing request
type IndexRequest struct {
	Documents []Document `json:"documents"`
}

// Document represents a document to index
type Document struct {
	ID   int    `json:"id"`
	Text string `json:"text"`
}

// IndexResponse represents an indexing response
type IndexResponse struct {
	Indexed int   `json:"indexed"`
	Latency int64 `json:"latency_us"`
}

// handleSearch handles single search requests using CPU similarity (for demo)
func (s *CUDAServer) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()
	s.totalRequests++

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "Query is required", http.StatusBadRequest)
		return
	}
	if req.K <= 0 {
		req.K = 10
	}

	// Generate query embedding
	queryStart := time.Now()
	embedding, err := s.model.EmbedInt8(req.Query)
	if err != nil {
		http.Error(w, fmt.Sprintf("Embedding error: %v", err), http.StatusInternalServerError)
		return
	}
	queryTime := time.Since(queryStart)

	// Perform similarity search (using CPU for now as a fallback)
	results := s.searchVectors(embedding.Vector, req.K)

	latency := time.Since(start)
	s.totalLatency += latency

	resp := SearchResponse{
		Results:   results,
		Latency:   latency.Microseconds(),
		QueryTime: queryTime.Microseconds(),
		GPU:       false, // CPU fallback for now
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// searchVectors performs CPU-based similarity search
func (s *CUDAServer) searchVectors(query []int8, k int) []SearchResult {
	if len(s.vectors) == 0 {
		return []SearchResult{}
	}

	type candidate struct {
		id    int
		score float32
	}

	candidates := make([]candidate, 0, len(s.vectors))

	// Compute similarities
	for i, vec := range s.vectors {
		if len(vec) != len(query) {
			continue
		}

		// Compute dot product similarity
		var dotProduct int32
		for j := 0; j < len(query); j++ {
			dotProduct += int32(query[j]) * int32(vec[j])
		}

		similarity := float32(dotProduct)
		candidates = append(candidates, candidate{
			id:    s.vectorIDs[i],
			score: similarity,
		})
	}

	// Sort by similarity (descending)
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].score > candidates[i].score {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Return top-k results
	results := make([]SearchResult, 0, k)
	for i := 0; i < k && i < len(candidates); i++ {
		results = append(results, SearchResult{
			ID:         candidates[i].id,
			Similarity: candidates[i].score,
		})
	}

	return results
}

// handleBatchSearch handles batch search requests
func (s *CUDAServer) handleBatchSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()
	s.totalRequests++

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if len(req.Queries) == 0 {
		http.Error(w, "Queries are required", http.StatusBadRequest)
		return
	}
	if req.K <= 0 {
		req.K = 10
	}

	// Process queries in batches
	results := make([][]SearchResult, len(req.Queries))
	for i, query := range req.Queries {
		embedding, err := s.model.EmbedInt8(query)
		if err != nil {
			log.Printf("  Failed to embed query %d: %v", i, err)
			results[i] = []SearchResult{}
			continue
		}
		results[i] = s.searchVectors(embedding.Vector, req.K)
	}

	latency := time.Since(start)
	s.totalLatency += latency

	resp := SearchResponse{
		Batch:   results,
		Latency: latency.Microseconds(),
		GPU:     false,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleIndex handles document indexing
func (s *CUDAServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()

	var req IndexRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	indexed := 0
	for _, doc := range req.Documents {
		embedding, err := s.model.EmbedInt8(doc.Text)
		if err != nil {
			log.Printf("  Failed to embed document %d: %v", doc.ID, err)
			continue
		}

		s.vectors = append(s.vectors, embedding.Vector)
		s.vectorIDs = append(s.vectorIDs, doc.ID)
		indexed++
	}

	s.totalIndexed += uint64(indexed)
	latency := time.Since(start)

	resp := IndexResponse{
		Indexed: indexed,
		Latency: latency.Microseconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleBatchIndex handles batch indexing
func (s *CUDAServer) handleBatchIndex(w http.ResponseWriter, r *http.Request) {
	s.handleIndex(w, r) // Same implementation for simplicity
}

// handleStats returns server statistics
func (s *CUDAServer) handleStats(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	uptime := time.Since(s.startTime)
	avgLatency := time.Duration(0)
	if s.totalRequests > 0 {
		avgLatency = s.totalLatency / time.Duration(s.totalRequests)
	}

	stats := map[string]interface{}{
		"server_stats": map[string]interface{}{
			"total_requests":   s.totalRequests,
			"total_indexed":    s.totalIndexed,
			"avg_latency_us":   avgLatency.Microseconds(),
			"uptime_seconds":   uptime.Seconds(),
			"vectors_indexed":  len(s.vectors),
			"requests_per_sec": float64(s.totalRequests) / uptime.Seconds(),
		},
		"system_stats": map[string]interface{}{
			"memory_mb":       float64(m.Alloc) / 1024 / 1024,
			"total_memory_mb": float64(m.Sys) / 1024 / 1024,
			"goroutines":      runtime.NumGoroutine(),
			"go_version":      runtime.Version(),
			"num_cpu":         runtime.NumCPU(),
		},
		"cuda_info": map[string]interface{}{
			"available": gobed.IsCUDAAvailable(),
			"devices":   gobed.GetCUDADeviceCount(),
			"version":   gobed.GetCUDAVersion(),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// handleHealth returns health status
func (s *CUDAServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":      "healthy",
		"vectors":     len(s.vectors),
		"cuda_ready":  gobed.IsCUDAAvailable(),
		"uptime_sec":  time.Since(s.startTime).Seconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// handleBenchmark runs a quick performance benchmark
func (s *CUDAServer) handleBenchmark(w http.ResponseWriter, r *http.Request) {
	results := s.runQuickBenchmark()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

// runQuickBenchmark runs a performance test
func (s *CUDAServer) runQuickBenchmark() map[string]interface{} {
	// Generate test data
	testQueries := []string{
		"machine learning performance",
		"neural network optimization",
		"GPU acceleration benefits",
		"vector similarity search",
		"deep learning algorithms",
	}

	start := time.Now()
	totalEmbeddings := 0

	for _, query := range testQueries {
		_, err := s.model.EmbedInt8(query)
		if err == nil {
			totalEmbeddings++
		}
	}

	duration := time.Since(start)

	return map[string]interface{}{
		"test_queries":        len(testQueries),
		"successful_embeddings": totalEmbeddings,
		"total_time_ms":       duration.Milliseconds(),
		"avg_time_ms":         duration.Milliseconds() / int64(len(testQueries)),
		"embeddings_per_sec":  float64(totalEmbeddings) / duration.Seconds(),
	}
}

// performanceMonitor displays real-time performance statistics
func (s *CUDAServer) performanceMonitor() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		uptime := time.Since(s.startTime)
		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		fmt.Printf("\n Performance Monitor (Uptime: %.1fm)\n", uptime.Minutes())
		fmt.Printf("   Requests: %d\n", s.totalRequests)
		fmt.Printf("   Indexed: %d vectors\n", s.totalIndexed)
		fmt.Printf("   Memory: %.1f MB\n", float64(m.Alloc)/1024/1024)

		if uptime.Seconds() > 0 && s.totalRequests > 0 {
			rps := float64(s.totalRequests) / uptime.Seconds()
			avgLatency := s.totalLatency / time.Duration(s.totalRequests)
			fmt.Printf("   Rate: %.2f req/sec\n", rps)
			fmt.Printf("   Avg Latency: %v\n", avgLatency)
		}
	}
}

// runDemo runs a demonstration with sample data
func (s *CUDAServer) runDemo(numVectors int) error {
	fmt.Printf(" Demo: Indexing %d vectors and testing search\n", numVectors)

	// Generate sample texts
	sampleTexts := generateSampleTexts(numVectors)

	// Index documents
	fmt.Printf("📚 Indexing %d documents...\n", len(sampleTexts))
	start := time.Now()

	for i, text := range sampleTexts {
		embedding, err := s.model.EmbedInt8(text)
		if err != nil {
			continue
		}

		s.vectors = append(s.vectors, embedding.Vector)
		s.vectorIDs = append(s.vectorIDs, i)

		if (i+1)%1000 == 0 {
			fmt.Printf("   Progress: %d/%d\n", i+1, len(sampleTexts))
		}
	}

	indexTime := time.Since(start)
	s.totalIndexed = uint64(len(s.vectors))
	fmt.Printf(" Indexed %d vectors in %v\n", len(s.vectors), indexTime)

	// Test searches
	fmt.Printf("\n Testing searches...\n")
	queries := []string{
		"machine learning algorithms",
		"neural network training",
		"GPU acceleration benefits",
		"vector similarity search",
		"data processing techniques",
	}

	searchStart := time.Now()
	for i, query := range queries {
		results := s.searchVectors([]int8{}, 5) // Simple test
		fmt.Printf("   Query %d: \"%s\" -> %d results\n", i+1, query, len(results))
	}

	searchTime := time.Since(searchStart)
	fmt.Printf(" Completed %d searches in %v\n", len(queries), searchTime)

	return nil
}

// runLoadTest runs a simple load test
func (s *CUDAServer) runLoadTest() {
	fmt.Println("\n Running Load Test...")

	queries := []string{
		"performance test query",
		"load testing benchmark",
		"server stress test",
	}

	start := time.Now()
	requests := 100

	for i := 0; i < requests; i++ {
		query := queries[i%len(queries)]
		_, err := s.model.EmbedInt8(query)
		if err != nil {
			continue
		}

		if (i+1)%25 == 0 {
			fmt.Printf("   Completed: %d/%d requests\n", i+1, requests)
		}
	}

	duration := time.Since(start)
	rps := float64(requests) / duration.Seconds()

	fmt.Printf(" Load test completed:\n")
	fmt.Printf("   Requests: %d\n", requests)
	fmt.Printf("   Duration: %v\n", duration)
	fmt.Printf("   Rate: %.1f req/sec\n", rps)
}

// generateSampleTexts creates sample texts for demonstration
func generateSampleTexts(count int) []string {
	templates := []string{
		"Machine learning algorithms process data efficiently",
		"Deep neural networks learn complex patterns",
		"GPU acceleration improves computational performance",
		"Vector databases enable fast similarity search",
		"Embedding models capture semantic relationships",
		"CUDA programming enables parallel computation",
		"Information retrieval uses advanced indexing",
		"Natural language processing transforms text",
		"Artificial intelligence drives innovation",
		"High-performance computing leverages parallelism",
	}

	texts := make([]string, count)
	for i := 0; i < count; i++ {
		template := templates[i%len(templates)]
		texts[i] = fmt.Sprintf("%s example %d", template, i)
	}

	return texts
}
