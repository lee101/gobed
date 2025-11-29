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
		gpuDevice   = flag.Int("gpu", 0, "GPU device ID") 
		batchSize   = flag.Int("batch", 512, "GPU batch size")
		demoMode    = flag.Bool("demo", false, "Run demonstration mode")
		demoVectors = flag.Int("demo-vectors", 10000, "Demo vectors to index")
		showStats   = flag.Bool("stats", true, "Show performance statistics")
	)
	flag.Parse()

	fmt.Println(" Simple GPU-Accelerated GoBeD Server")
	fmt.Println("======================================")

	// Check GPU availability
	if !gobed.IsCUDAAvailable() {
		log.Fatal(" CUDA is not available. Please ensure NVIDIA drivers and CUDA are installed.")
	}

	gpuCount := gobed.GetCUDADeviceCount()
	cudaVersion := gobed.GetCUDAVersion()

	fmt.Printf(" CUDA Available: %s\n", cudaVersion)
	fmt.Printf("   GPU Devices: %d\n", gpuCount)
	fmt.Printf("   Using Device: %d\n", *gpuDevice)

	if *gpuDevice >= gpuCount {
		log.Fatalf(" Invalid GPU device %d. Available devices: 0-%d", *gpuDevice, gpuCount-1)
	}

	// Load embedding model
	fmt.Println("\n📚 Loading Embedding Model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf(" Failed to load model: %v", err)
	}

	// Initialize GPU indexer
	fmt.Println("\n Initializing GPU Indexer...")
	gpuConfig := gobed.IndexConfig{
		VectorDim:        512,
		NumSubquantizers: 8,
		CodebookSize:     256,
		IVFClusters:      1024,
		ProbeLists:       64,
		RerankK:          1000,
		DeviceID:         *gpuDevice,
	}

	gpuIndexer, err := gobed.NewGPUIndexer(gpuConfig)
	if err != nil {
		log.Fatalf(" Failed to create GPU indexer: %v", err)
	}
	defer gpuIndexer.Close()

	// Create simple HTTP server
	server := &SimpleGPUServer{
		model:      model,
		gpuIndexer: gpuIndexer,
		port:       *port,
		batchSize:  *batchSize,
	}

	// Run demonstration if requested
	if *demoMode {
		fmt.Println("\n Running GPU Demonstration...")
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

	// Start performance monitor
	if *showStats {
		go server.performanceMonitor()
	}

	// Start server
	fmt.Printf("\n GPU Server Running on Port %d\n", *port)
	fmt.Printf("    Search: POST /search\n")
	fmt.Printf("    Batch Search: POST /batch_search\n")
	fmt.Printf("   📚 Index: POST /index\n")
	fmt.Printf("    Stats: GET /stats\n")
	fmt.Printf("    Health: GET /health\n")

	// Start HTTP server
	go func() {
		if err := http.ListenAndServe(fmt.Sprintf(":%d", *port), nil); err != nil {
			log.Fatalf(" Server failed: %v", err)
		}
	}()

	// Wait for shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	fmt.Println("\n Ready! Press Ctrl+C to stop.")
	<-sigChan

	fmt.Println("\n🛑 Shutting down gracefully...")
	fmt.Println(" Goodbye!")
}

// SimpleGPUServer provides a lightweight GPU-accelerated server
type SimpleGPUServer struct {
	model      *gobed.EmbeddingModel
	gpuIndexer *gobed.GPUIndexer
	port       int
	batchSize  int

	// Simple metrics
	totalRequests uint64
	totalIndexed  uint64
	startTime     time.Time
}

// SearchRequest represents a search request
type SearchRequest struct {
	Query   string   `json:"query"`
	Queries []string `json:"queries,omitempty"`
	K       int      `json:"k"`
}

// SearchResponse represents a search response
type SearchResponse struct {
	Results []gobed.SearchResult   `json:"results,omitempty"`
	Batch   [][]gobed.SearchResult `json:"batch,omitempty"`
	Latency int64                  `json:"latency_us"`
	GPU     bool                   `json:"gpu_accelerated"`
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
func (s *SimpleGPUServer) handleSearch(w http.ResponseWriter, r *http.Request) {
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

	// Generate embedding
	embedding, err := s.model.EmbedInt8(req.Query)
	if err != nil {
		http.Error(w, fmt.Sprintf("Embedding error: %v", err), http.StatusInternalServerError)
		return
	}

	// Perform GPU search
	results, err := s.gpuIndexer.Search(embedding.Vector, req.K)
	if err != nil {
		http.Error(w, fmt.Sprintf("Search error: %v", err), http.StatusInternalServerError)
		return
	}

	latency := time.Since(start)

	resp := SearchResponse{
		Results: results,
		Latency: latency.Microseconds(),
		GPU:     true,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleBatchSearch handles batch search requests
func (s *SimpleGPUServer) handleBatchSearch(w http.ResponseWriter, r *http.Request) {
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

	// Generate embeddings
	embeddings := make([][]int8, len(req.Queries))
	for i, query := range req.Queries {
		embedding, err := s.model.EmbedInt8(query)
		if err != nil {
			log.Printf("  Failed to embed query %d: %v", i, err)
			continue
		}
		embeddings[i] = embedding.Vector
	}

	// Perform GPU batch search
	results, err := s.gpuIndexer.BatchSearch(embeddings, req.K)
	if err != nil {
		http.Error(w, fmt.Sprintf("Batch search error: %v", err), http.StatusInternalServerError)
		return
	}

	latency := time.Since(start)

	resp := SearchResponse{
		Batch:   results,
		Latency: latency.Microseconds(),
		GPU:     true,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleIndex handles document indexing
func (s *SimpleGPUServer) handleIndex(w http.ResponseWriter, r *http.Request) {
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

	// Process documents in batches
	batchSize := s.batchSize
	indexed := 0

	for i := 0; i < len(req.Documents); i += batchSize {
		end := i + batchSize
		if end > len(req.Documents) {
			end = len(req.Documents)
		}

		batch := req.Documents[i:end]
		batchEmbeddings := make([][]int8, 0, len(batch))

		// Generate embeddings for batch
		for _, doc := range batch {
			embedding, err := s.model.EmbedInt8(doc.Text)
			if err != nil {
				log.Printf("  Failed to embed document %d: %v", doc.ID, err)
				continue
			}
			batchEmbeddings = append(batchEmbeddings, embedding.Vector)
			indexed++
		}

		// Add to GPU index
		if len(batchEmbeddings) > 0 {
			if err := s.gpuIndexer.AddVectors(batchEmbeddings); err != nil {
				log.Printf("  Failed to index batch: %v", err)
			}
		}
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

// handleBatchIndex handles batch indexing (same as handleIndex for simplicity)
func (s *SimpleGPUServer) handleBatchIndex(w http.ResponseWriter, r *http.Request) {
	s.handleIndex(w, r)
}

// handleStats returns server statistics
func (s *SimpleGPUServer) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := s.gpuIndexer.GetStats()
	
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	uptime := time.Since(s.startTime)

	response := map[string]interface{}{
		"gpu_stats": stats,
		"server_stats": map[string]interface{}{
			"total_requests": s.totalRequests,
			"total_indexed":  s.totalIndexed,
			"uptime_sec":     uptime.Seconds(),
			"memory_mb":      float64(m.Alloc) / 1024 / 1024,
			"goroutines":     runtime.NumGoroutine(),
		},
		"system_info": map[string]interface{}{
			"cuda_version":   gobed.GetCUDAVersion(),
			"gpu_devices":    gobed.GetCUDADeviceCount(),
			"go_version":     runtime.Version(),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleHealth returns health status
func (s *SimpleGPUServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	stats := s.gpuIndexer.GetStats()

	health := map[string]interface{}{
		"status":      "healthy",
		"gpu_memory":  stats.GPUMemoryMB,
		"vectors":     stats.NumVectors,
		"index_ready": stats.IndexBuilt,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// performanceMonitor displays performance statistics
func (s *SimpleGPUServer) performanceMonitor() {
	s.startTime = time.Now()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		stats := s.gpuIndexer.GetStats()
		uptime := time.Since(s.startTime)

		fmt.Printf("\n Performance Stats (Uptime: %.1fm)\n", uptime.Minutes())
		fmt.Printf("   Requests: %d\n", s.totalRequests)
		fmt.Printf("   Indexed: %d vectors\n", s.totalIndexed)
		fmt.Printf("   GPU Memory: %.1f MB\n", stats.GPUMemoryMB)
		fmt.Printf("   Index Ready: %v\n", stats.IndexBuilt)

		if uptime.Seconds() > 0 {
			rps := float64(s.totalRequests) / uptime.Seconds()
			fmt.Printf("   Rate: %.2f req/sec\n", rps)
		}
	}
}

// runDemo runs a demonstration with sample data
func (s *SimpleGPUServer) runDemo(numVectors int) error {
	fmt.Printf(" Demo: Indexing %d vectors and testing search\n", numVectors)

	// Generate sample texts
	sampleTexts := generateSampleTexts(numVectors)
	
	// Index in batches
	fmt.Printf("📚 Indexing %d documents...\n", len(sampleTexts))
	start := time.Now()

	var allEmbeddings [][]int8
	for i, text := range sampleTexts {
		embedding, err := s.model.EmbedInt8(text)
		if err != nil {
			continue
		}
		allEmbeddings = append(allEmbeddings, embedding.Vector)

		if (i+1)%1000 == 0 {
			fmt.Printf("   Progress: %d/%d\n", i+1, len(sampleTexts))
		}
	}

	// Train and add vectors to GPU index
	if len(allEmbeddings) > 1000 {
		trainVectors := allEmbeddings[:1000] // Use first 1000 for training
		if err := s.gpuIndexer.TrainIndex(trainVectors); err != nil {
			return fmt.Errorf("training failed: %w", err)
		}
	}

	if err := s.gpuIndexer.AddVectors(allEmbeddings); err != nil {
		return fmt.Errorf("indexing failed: %w", err)
	}

	indexTime := time.Since(start)
	fmt.Printf(" Indexed %d vectors in %v\n", len(allEmbeddings), indexTime)

	// Test searches
	fmt.Printf("\n Testing GPU searches...\n")
	queries := []string{
		"machine learning algorithms",
		"neural network training",
		"GPU acceleration",
		"vector similarity",
		"data processing",
	}

	searchStart := time.Now()
	for i, query := range queries {
		embedding, err := s.model.EmbedInt8(query)
		if err != nil {
			continue
		}

		results, err := s.gpuIndexer.Search(embedding.Vector, 5)
		if err != nil {
			continue
		}

		fmt.Printf("   Query %d: \"%s\" -> %d results\n", i+1, query, len(results))
		if len(results) > 0 {
			fmt.Printf("     Top result: ID=%d, Similarity=%.3f\n", 
				results[0].ID, results[0].Similarity)
		}
	}

	searchTime := time.Since(searchStart)
	fmt.Printf(" Completed %d searches in %v\n", len(queries), searchTime)

	s.totalIndexed = uint64(len(allEmbeddings))
	return nil
}

// generateSampleTexts creates sample texts for demonstration
func generateSampleTexts(count int) []string {
	templates := []string{
		"Machine learning algorithms enable automated data analysis",
		"Deep neural networks process complex information patterns",
		"GPU acceleration dramatically improves computational performance",
		"Vector databases provide efficient similarity search capabilities",
		"Embedding models capture semantic relationships in text",
		"CUDA programming enables parallel GPU computation",
		"Information retrieval systems use advanced indexing techniques",
		"Natural language processing transforms text understanding",
		"Artificial intelligence drives innovation across industries",
		"High-performance computing leverages parallel architectures",
	}

	variations := []string{
		"with remarkable efficiency",
		"using cutting-edge technology",
		"through advanced optimization",
		"via innovative approaches",
		"by leveraging modern hardware",
		"with unprecedented speed",
		"using state-of-the-art methods",
		"through parallel processing",
		"with optimal resource utilization",
		"using scalable architectures",
	}

	texts := make([]string, count)
	for i := 0; i < count; i++ {
		template := templates[i%len(templates)]
		variation := variations[i%len(variations)]
		texts[i] = fmt.Sprintf("%s %s %d", template, variation, i)
	}

	return texts
}
