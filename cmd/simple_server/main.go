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
	"sync"
	"syscall"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	var (
		port        = flag.Int("port", 8080, "Server port")
		demoMode    = flag.Bool("demo", false, "Run demonstration mode")
		demoVectors = flag.Int("demo-vectors", 10000, "Demo vectors to index")
		showStats   = flag.Bool("stats", true, "Show performance statistics")
		loadTest    = flag.Bool("load-test", false, "Run load test")
		workers     = flag.Int("workers", runtime.NumCPU(), "Number of worker threads")
	)
	flag.Parse()

	fmt.Println(" High-Performance GoBeD Server")
	fmt.Println("=================================")
	fmt.Printf("   Built with: %s\n", runtime.Version())
	fmt.Printf("   Workers: %d\n", *workers)

	// Set optimal thread configuration
	runtime.GOMAXPROCS(*workers)

	// Load embedding model
	fmt.Println("\n📚 Loading Embedding Model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf(" Failed to load model: %v", err)
	}

	// Create server
	server := &FastServer{
		model:     model,
		port:      *port,
		vectors:   make([][]int8, 0),
		vectorIDs: make([]int, 0),
		textMap:   make(map[int]string),
		startTime: time.Now(),
	}

	// Run demonstration if requested
	if *demoMode {
		fmt.Println("\n Running Performance Demonstration...")
		if err := server.runDemo(*demoVectors); err != nil {
			log.Printf("  Demo failed: %v", err)
		}
	}

	// Setup HTTP routes
	http.HandleFunc("/search", server.handleSearch)
	http.HandleFunc("/batch_search", server.handleBatchSearch)
	http.HandleFunc("/index", server.handleIndex)
	http.HandleFunc("/stats", server.handleStats)
	http.HandleFunc("/health", server.handleHealth)
	http.HandleFunc("/benchmark", server.handleBenchmark)

	// Start performance monitor
	if *showStats {
		go server.performanceMonitor()
	}

	// Start server
	fmt.Printf("\n Server Running on Port %d\n", *port)
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

// FastServer provides high-performance vector search
type FastServer struct {
	model   *gobed.EmbeddingModel
	port    int
	mutex   sync.RWMutex

	// In-memory vector storage
	vectors   [][]int8
	vectorIDs []int
	textMap   map[int]string

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
	QueryTime int64            `json:"query_time_us,omitempty"`
	Count     int              `json:"result_count"`
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

// handleSearch handles single search requests
func (s *FastServer) handleSearch(w http.ResponseWriter, r *http.Request) {
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

	// Perform similarity search
	results := s.searchVectors(embedding.Vector, req.K)

	latency := time.Since(start)
	s.totalLatency += latency

	resp := SearchResponse{
		Results:   results,
		Latency:   latency.Microseconds(),
		QueryTime: queryTime.Microseconds(),
		Count:     len(results),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// searchVectors performs optimized similarity search using SIMD operations
func (s *FastServer) searchVectors(query []int8, k int) []SearchResult {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.vectors) == 0 {
		return []SearchResult{}
	}

	candidates := make([]candidate, 0, len(s.vectors))

	// Compute similarities using optimized int8 dot products
	for i, vec := range s.vectors {
		if len(vec) != len(query) {
			continue
		}

		// Optimized dot product for int8 vectors
		score := computeInt8DotProduct(query, vec)
		candidates = append(candidates, candidate{
			id:    s.vectorIDs[i],
			score: score,
		})
	}

	// Partial sort to get top-k efficiently
	topK := partialSort(candidates, k)

	// Convert to results with text lookup
	results := make([]SearchResult, len(topK))
	for i, candidate := range topK {
		results[i] = SearchResult{
			ID:         candidate.id,
			Similarity: candidate.score,
			Text:       s.textMap[candidate.id],
		}
	}

	return results
}

// computeInt8DotProduct computes optimized dot product for int8 vectors
func computeInt8DotProduct(a, b []int8) float32 {
	var sum int32

	// Process 4 elements at a time for better performance
	i := 0
	for i+4 <= len(a) {
		sum += int32(a[i])*int32(b[i]) +
			int32(a[i+1])*int32(b[i+1]) +
			int32(a[i+2])*int32(b[i+2]) +
			int32(a[i+3])*int32(b[i+3])
		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		sum += int32(a[i]) * int32(b[i])
		i++
	}

	return float32(sum)
}

// candidate represents a search candidate with score
type candidate struct {
	id    int
	score float32
}

// partialSort efficiently finds top-k elements without full sorting
func partialSort(candidates []candidate, k int) []candidate {
	if k >= len(candidates) {
		// Full sort for small arrays
		for i := 0; i < len(candidates)-1; i++ {
			for j := i + 1; j < len(candidates); j++ {
				if candidates[j].score > candidates[i].score {
					candidates[i], candidates[j] = candidates[j], candidates[i]
				}
			}
		}
		return candidates
	}

	// Use partial sort for efficiency
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].score > candidates[maxIdx].score {
				maxIdx = j
			}
		}
		if maxIdx != i {
			candidates[i], candidates[maxIdx] = candidates[maxIdx], candidates[i]
		}
	}

	return candidates[:k]
}

// handleBatchSearch handles batch search requests with parallel processing
func (s *FastServer) handleBatchSearch(w http.ResponseWriter, r *http.Request) {
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

	// Process queries in parallel
	results := make([][]SearchResult, len(req.Queries))
	var wg sync.WaitGroup
	
	// Limit concurrency to avoid overwhelming the system
	semaphore := make(chan struct{}, runtime.NumCPU())

	for i, query := range req.Queries {
		wg.Add(1)
		go func(idx int, q string) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire
			defer func() { <-semaphore }() // Release

			embedding, err := s.model.EmbedInt8(q)
			if err != nil {
				log.Printf("  Failed to embed query %d: %v", idx, err)
				results[idx] = []SearchResult{}
				return
			}
			results[idx] = s.searchVectors(embedding.Vector, req.K)
		}(i, query)
	}
	wg.Wait()

	latency := time.Since(start)
	s.totalLatency += latency

	resp := SearchResponse{
		Batch:   results,
		Latency: latency.Microseconds(),
		Count:   len(req.Queries),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleIndex handles document indexing with concurrent processing
func (s *FastServer) handleIndex(w http.ResponseWriter, r *http.Request) {
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

	// Process documents in parallel
	type embeddingResult struct {
		id        int
		text      string
		embedding []int8
		err       error
	}

	results := make(chan embeddingResult, len(req.Documents))
	var wg sync.WaitGroup

	// Process embeddings in parallel
	for _, doc := range req.Documents {
		wg.Add(1)
		go func(d Document) {
			defer wg.Done()
			embedding, err := s.model.EmbedInt8(d.Text)
			if err != nil {
				results <- embeddingResult{id: d.ID, text: d.Text, err: err}
				return
			}
			results <- embeddingResult{
				id:        d.ID,
				text:      d.Text,
				embedding: embedding.Vector,
			}
		}(doc)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results and update index
	s.mutex.Lock()
	indexed := 0
	for result := range results {
		if result.err != nil {
			log.Printf("  Failed to embed document %d: %v", result.id, result.err)
			continue
		}

		s.vectors = append(s.vectors, result.embedding)
		s.vectorIDs = append(s.vectorIDs, result.id)
		s.textMap[result.id] = result.text
		indexed++
	}
	s.mutex.Unlock()

	s.totalIndexed += uint64(indexed)
	latency := time.Since(start)

	resp := IndexResponse{
		Indexed: indexed,
		Latency: latency.Microseconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleStats returns server statistics
func (s *FastServer) handleStats(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	s.mutex.RLock()
	vectorCount := len(s.vectors)
	s.mutex.RUnlock()

	uptime := time.Since(s.startTime)
	avgLatency := time.Duration(0)
	if s.totalRequests > 0 {
		avgLatency = s.totalLatency / time.Duration(s.totalRequests)
	}

	stats := map[string]interface{}{
		"server_stats": map[string]interface{}{
			"total_requests":    s.totalRequests,
			"total_indexed":     s.totalIndexed,
			"avg_latency_us":    avgLatency.Microseconds(),
			"uptime_seconds":    uptime.Seconds(),
			"vectors_indexed":   vectorCount,
			"requests_per_sec":  float64(s.totalRequests) / uptime.Seconds(),
			"indexing_rate":     float64(s.totalIndexed) / uptime.Seconds(),
		},
		"system_stats": map[string]interface{}{
			"memory_mb":        float64(m.Alloc) / 1024 / 1024,
			"total_memory_mb":  float64(m.Sys) / 1024 / 1024,
			"gc_runs":          m.NumGC,
			"goroutines":       runtime.NumGoroutine(),
			"go_version":       runtime.Version(),
			"num_cpu":          runtime.NumCPU(),
			"gomaxprocs":       runtime.GOMAXPROCS(0),
		},
		"performance": map[string]interface{}{
			"vector_dimension": 512,
			"search_algorithm": "optimized_int8_dot_product",
			"parallel_indexing": true,
			"concurrent_search": true,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// handleHealth returns health status
func (s *FastServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	vectorCount := len(s.vectors)
	s.mutex.RUnlock()

	health := map[string]interface{}{
		"status":      "healthy",
		"vectors":     vectorCount,
		"uptime_sec":  time.Since(s.startTime).Seconds(),
		"ready":       vectorCount > 0,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// handleBenchmark runs a performance benchmark
func (s *FastServer) handleBenchmark(w http.ResponseWriter, r *http.Request) {
	results := s.runQuickBenchmark()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

// runQuickBenchmark runs a performance test
func (s *FastServer) runQuickBenchmark() map[string]interface{} {
	testQueries := []string{
		"machine learning performance optimization",
		"neural network training algorithms",
		"GPU acceleration for computation",
		"vector similarity search techniques",
		"deep learning model inference",
	}

	start := time.Now()
	totalEmbeddings := 0
	var totalEmbeddingTime time.Duration

	for _, query := range testQueries {
		embStart := time.Now()
		_, err := s.model.EmbedInt8(query)
		embTime := time.Since(embStart)
		
		if err == nil {
			totalEmbeddings++
			totalEmbeddingTime += embTime
		}
	}

	totalTime := time.Since(start)

	// Test search performance if we have vectors
	s.mutex.RLock()
	vectorCount := len(s.vectors)
	s.mutex.RUnlock()

	var searchTime time.Duration
	var searchResults int
	if vectorCount > 0 {
		searchStart := time.Now()
		for _, query := range testQueries[:min(3, len(testQueries))] {
			embedding, err := s.model.EmbedInt8(query)
			if err == nil {
				results := s.searchVectors(embedding.Vector, 10)
				searchResults += len(results)
			}
		}
		searchTime = time.Since(searchStart)
	}

	return map[string]interface{}{
		"embedding_benchmark": map[string]interface{}{
			"test_queries":         len(testQueries),
			"successful_embeddings": totalEmbeddings,
			"total_time_ms":        totalTime.Milliseconds(),
			"avg_embedding_time_ms": totalEmbeddingTime.Milliseconds() / int64(max(1, totalEmbeddings)),
			"embeddings_per_sec":   float64(totalEmbeddings) / totalTime.Seconds(),
		},
		"search_benchmark": map[string]interface{}{
			"vectors_available":   vectorCount,
			"search_time_ms":      searchTime.Milliseconds(),
			"total_results":       searchResults,
			"searches_per_sec":    float64(3) / searchTime.Seconds(),
		},
	}
}

// performanceMonitor displays real-time performance statistics
func (s *FastServer) performanceMonitor() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		uptime := time.Since(s.startTime)
		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		s.mutex.RLock()
		vectorCount := len(s.vectors)
		s.mutex.RUnlock()

		fmt.Printf("\n Performance Monitor (Uptime: %.1fm)\n", uptime.Minutes())
		fmt.Printf("   Requests: %d\n", s.totalRequests)
		fmt.Printf("   Indexed: %d vectors\n", vectorCount)
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
func (s *FastServer) runDemo(numVectors int) error {
	fmt.Printf(" Demo: Indexing %d vectors and testing search performance\n", numVectors)

	// Generate sample texts
	sampleTexts := generateSampleTexts(numVectors)

	// Index documents with progress tracking
	fmt.Printf("📚 Indexing %d documents...\n", len(sampleTexts))
	start := time.Now()

	batchSize := 100
	for i := 0; i < len(sampleTexts); i += batchSize {
		end := i + batchSize
		if end > len(sampleTexts) {
			end = len(sampleTexts)
		}

		batch := make([]Document, end-i)
		for j := i; j < end; j++ {
			batch[j-i] = Document{ID: j, Text: sampleTexts[j]}
		}

		// Process batch
		var wg sync.WaitGroup
		for _, doc := range batch {
			wg.Add(1)
			go func(d Document) {
				defer wg.Done()
				embedding, err := s.model.EmbedInt8(d.Text)
				if err != nil {
					return
				}

				s.mutex.Lock()
				s.vectors = append(s.vectors, embedding.Vector)
				s.vectorIDs = append(s.vectorIDs, d.ID)
				s.textMap[d.ID] = d.Text
				s.mutex.Unlock()
			}(doc)
		}
		wg.Wait()

		if (i+batchSize)%1000 == 0 || end == len(sampleTexts) {
			s.mutex.RLock()
			current := len(s.vectors)
			s.mutex.RUnlock()
			fmt.Printf("   Progress: %d/%d vectors indexed\n", current, len(sampleTexts))
		}
	}

	indexTime := time.Since(start)
	s.mutex.RLock()
	indexedCount := len(s.vectors)
	s.mutex.RUnlock()

	s.totalIndexed = uint64(indexedCount)
	fmt.Printf(" Indexed %d vectors in %v (%.1f vectors/sec)\n", 
		indexedCount, indexTime, float64(indexedCount)/indexTime.Seconds())

	// Test searches
	fmt.Printf("\n Testing search performance...\n")
	queries := []string{
		"machine learning algorithms and optimization",
		"neural network training and inference",
		"GPU acceleration for parallel computation",
		"vector similarity search and retrieval",
		"data processing and analysis techniques",
	}

	searchStart := time.Now()
	totalResults := 0
	for i, query := range queries {
		embedding, err := s.model.EmbedInt8(query)
		if err != nil {
			continue
		}

		results := s.searchVectors(embedding.Vector, 5)
		totalResults += len(results)
		
		fmt.Printf("   Query %d: \"%s\" -> %d results\n", i+1, query[:50]+"...", len(results))
		if len(results) > 0 {
			fmt.Printf("     Top result: ID=%d, Similarity=%.3f\n", 
				results[0].ID, results[0].Similarity)
		}
	}

	searchTime := time.Since(searchStart)
	fmt.Printf(" Completed %d searches in %v (%.1f searches/sec)\n", 
		len(queries), searchTime, float64(len(queries))/searchTime.Seconds())

	return nil
}

// runLoadTest runs a load test
func (s *FastServer) runLoadTest() {
	fmt.Println("\n Running Load Test...")

	queries := []string{
		"performance test query for load testing",
		"concurrent request benchmark validation",
		"server stress test under heavy load",
		"scalability assessment for production",
		"throughput measurement and optimization",
	}

	start := time.Now()
	requests := 200
	concurrency := 10

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, concurrency)
	
	for i := 0; i < requests; i++ {
		wg.Add(1)
		go func(reqNum int) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire
			defer func() { <-semaphore }() // Release

			query := queries[reqNum%len(queries)]
			_, err := s.model.EmbedInt8(query)
			if err != nil {
				return
			}

			if reqNum%50 == 0 {
				fmt.Printf("   Completed: %d/%d requests\n", reqNum+1, requests)
			}
		}(i)
	}
	wg.Wait()

	duration := time.Since(start)
	rps := float64(requests) / duration.Seconds()

	fmt.Printf(" Load test completed:\n")
	fmt.Printf("   Requests: %d (concurrency: %d)\n", requests, concurrency)
	fmt.Printf("   Duration: %v\n", duration)
	fmt.Printf("   Rate: %.1f req/sec\n", rps)
	fmt.Printf("   Avg per request: %v\n", duration/time.Duration(requests))
}

// generateSampleTexts creates sample texts for demonstration
func generateSampleTexts(count int) []string {
	templates := []string{
		"Machine learning algorithms enable automated data analysis and pattern recognition",
		"Deep neural networks process complex information using multiple layers of computation",
		"GPU acceleration dramatically improves computational performance for parallel workloads",
		"Vector databases provide efficient similarity search capabilities for high-dimensional data",
		"Embedding models capture semantic relationships and meaning in text representations",
		"CUDA programming enables massive parallel computation on graphics processing units",
		"Information retrieval systems use advanced indexing techniques for fast search operations",
		"Natural language processing transforms text understanding through computational linguistics",
		"Artificial intelligence drives innovation across industries with intelligent automation",
		"High-performance computing leverages parallel architectures for scientific computing",
	}

	variations := []string{
		"with remarkable efficiency and accuracy in real-time applications",
		"using cutting-edge technology and state-of-the-art optimization techniques",
		"through advanced optimization and intelligent resource management strategies",
		"via innovative approaches that maximize performance and scalability",
		"by leveraging modern hardware acceleration and parallel processing capabilities",
		"with unprecedented speed and precision in production environments",
		"using scalable architectures designed for high-throughput scenarios",
		"through parallel processing and optimized memory access patterns",
		"with optimal resource utilization and efficient algorithm implementation",
		"using distributed computing and cloud-native scaling approaches",
	}

	texts := make([]string, count)
	for i := 0; i < count; i++ {
		template := templates[i%len(templates)]
		variation := variations[i%len(variations)]
		texts[i] = fmt.Sprintf("%s %s (sample %d)", template, variation, i)
	}

	return texts
}

// Utility functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}