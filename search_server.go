package gobed

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// SearchServer provides a high-performance HTTP search server with shared memory
type SearchServer struct {
	model       *EmbeddingModel
	sharedIndex *SharedMemoryIndex

	// Server state
	server *http.Server
	port   int

	// Performance metrics
	totalRequests uint64
	totalLatency  uint64
	p95Latency    uint64
	p99Latency    uint64

	// Request pooling
	requestPool  sync.Pool
	responsePool sync.Pool

	// Configuration
	config ServerConfig

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// ServerConfig configures the search server
type ServerConfig struct {
	Port              int
	SharedIndexPath   string
	MaxVectors        int
	MaxConcurrency    int
	EnableProfiling   bool
	EnableMetrics     bool
	ReadOnly          bool
	PreloadEmbeddings bool
	WorkerThreads     int
}

// DefaultServerConfig returns optimized server configuration
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Port:              8080,
		SharedIndexPath:   "/tmp/gobed_shared_index",
		MaxVectors:        1000000,
		MaxConcurrency:    1000,
		EnableProfiling:   false,
		EnableMetrics:     true,
		ReadOnly:          false,
		PreloadEmbeddings: true,
		WorkerThreads:     runtime.NumCPU(),
	}
}

// SearchRequest represents a search API request
type SearchRequest struct {
	Query     string   `json:"query"`
	Queries   []string `json:"queries,omitempty"` // Batch search
	K         int      `json:"k"`
	Timeout   int      `json:"timeout_ms,omitempty"`
	RequestID string   `json:"request_id,omitempty"`
}

// SearchResponse represents a search API response
type SearchResponse struct {
	Results   []SearchResult   `json:"results,omitempty"`
	Batch     [][]SearchResult `json:"batch,omitempty"`
	Latency   int64            `json:"latency_us"`
	RequestID string           `json:"request_id,omitempty"`
	Error     string           `json:"error,omitempty"`
}

// ServerIndexRequest represents an indexing API request
type ServerIndexRequest struct {
	Documents []ServerDocument `json:"documents"`
	Async     bool             `json:"async,omitempty"`
	RequestID string           `json:"request_id,omitempty"`
}

// ServerDocument represents a document to index
type ServerDocument struct {
	ID   int    `json:"id"`
	Text string `json:"text"`
}

// ServerIndexResponse represents an indexing API response
type ServerIndexResponse struct {
	Indexed   int    `json:"indexed"`
	Latency   int64  `json:"latency_us"`
	RequestID string `json:"request_id,omitempty"`
	Error     string `json:"error,omitempty"`
}

// NewSearchServer creates a new high-performance search server
func NewSearchServer(model *EmbeddingModel, config ServerConfig) (*SearchServer, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Initialize shared memory index
	sharedConfig := SharedMemoryConfig{
		BasePath:    config.SharedIndexPath,
		MaxVectors:  config.MaxVectors,
		ReadOnly:    config.ReadOnly,
		CreateIfNew: !config.ReadOnly,
		CacheSize:   10000, // Cache hot vectors
		UseLockFree: true,
	}

	sharedIndex, err := NewSharedMemoryIndex(sharedConfig)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create shared index: %w", err)
	}

	server := &SearchServer{
		model:       model,
		sharedIndex: sharedIndex,
		port:        config.Port,
		config:      config,
		ctx:         ctx,
		cancel:      cancel,
		requestPool: sync.Pool{
			New: func() interface{} {
				return &SearchRequest{}
			},
		},
		responsePool: sync.Pool{
			New: func() interface{} {
				return &SearchResponse{}
			},
		},
	}

	// Setup HTTP routes
	mux := http.NewServeMux()

	// Search endpoints
	mux.HandleFunc("/search", server.handleSearch)
	mux.HandleFunc("/batch_search", server.handleBatchSearch)

	// Index endpoints (if not read-only)
	if !config.ReadOnly {
		mux.HandleFunc("/index", server.handleIndex)
		mux.HandleFunc("/batch_index", server.handleBatchIndex)
	}

	// Health and metrics
	mux.HandleFunc("/health", server.handleHealth)
	mux.HandleFunc("/metrics", server.handleMetrics)

	// Profiling endpoints (if enabled)
	if config.EnableProfiling {
		mux.HandleFunc("/debug/pprof/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.DefaultServeMux.ServeHTTP(w, r)
		}))
	}

	server.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", config.Port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	return server, nil
}

// Start starts the search server
func (s *SearchServer) Start() error {
	log.Printf("Starting search server on port %d", s.port)
	log.Printf("Shared index: %s (readonly=%v)", s.config.SharedIndexPath, s.config.ReadOnly)

	// Set GOMAXPROCS for optimal performance
	if s.config.WorkerThreads > 0 {
		runtime.GOMAXPROCS(s.config.WorkerThreads)
	}

	// Start server
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		if err := s.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("Server error: %v", err)
		}
	}()

	// Start metrics collector if enabled
	if s.config.EnableMetrics {
		s.wg.Add(1)
		go s.metricsCollector()
	}

	log.Printf("Search server started successfully")
	return nil
}

// handleSearch handles single search requests
func (s *SearchServer) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Track request
	atomic.AddUint64(&s.totalRequests, 1)
	start := time.Now()

	// Get request from pool
	req := s.requestPool.Get().(*SearchRequest)
	defer s.requestPool.Put(req)

	// Parse request
	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// Validate
	if req.Query == "" {
		http.Error(w, "Query is required", http.StatusBadRequest)
		return
	}
	if req.K <= 0 {
		req.K = 10 // Default
	}

	// Generate embedding
	embedding, err := s.model.EmbedInt8(req.Query)
	if err != nil {
		http.Error(w, fmt.Sprintf("Embedding error: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to SIMD vector
	var vec simd.Vec512
	copy(vec[:], embedding.Vector)

	// Perform zero-copy search on shared memory
	results := s.sharedIndex.SearchTopK(&vec, req.K)

	// Track latency
	latency := time.Since(start)
	atomic.AddUint64(&s.totalLatency, uint64(latency.Nanoseconds()))

	// Build response
	resp := s.responsePool.Get().(*SearchResponse)
	defer s.responsePool.Put(resp)

	resp.Results = results
	resp.Latency = latency.Microseconds()
	resp.RequestID = req.RequestID

	// Send response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleBatchSearch handles batch search requests
func (s *SearchServer) handleBatchSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	atomic.AddUint64(&s.totalRequests, 1)
	start := time.Now()

	req := &SearchRequest{}
	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
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

	// Generate embeddings in parallel
	vectors := make([]*simd.Vec512, len(req.Queries))
	var wg sync.WaitGroup

	for i, query := range req.Queries {
		wg.Add(1)
		go func(idx int, q string) {
			defer wg.Done()

			embedding, err := s.model.EmbedInt8(q)
			if err != nil {
				log.Printf("Embedding error for query %d: %v", idx, err)
				return
			}

			vec := &simd.Vec512{}
			copy(vec[:], embedding.Vector)
			vectors[idx] = vec
		}(i, query)
	}
	wg.Wait()

	// Batch search on shared memory
	results := s.sharedIndex.BatchSearch(vectors, req.K)

	latency := time.Since(start)

	resp := &SearchResponse{
		Batch:     results,
		Latency:   latency.Microseconds(),
		RequestID: req.RequestID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleIndex handles document indexing
func (s *SearchServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if s.config.ReadOnly {
		http.Error(w, "Server is in read-only mode", http.StatusForbidden)
		return
	}

	start := time.Now()

	req := &ServerIndexRequest{}
	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	indexed := 0
	for _, doc := range req.Documents {
		// Generate embedding
		embedding, err := s.model.EmbedInt8(doc.Text)
		if err != nil {
			log.Printf("Failed to embed document %d: %v", doc.ID, err)
			continue
		}

		// Convert to SIMD vector
		var vec simd.Vec512
		copy(vec[:], embedding.Vector)

		// Add to shared index (zero-copy)
		if err := s.sharedIndex.AddVector(&vec, embedding.Scale, doc.ID); err != nil {
			log.Printf("Failed to index document %d: %v", doc.ID, err)
			continue
		}

		indexed++
	}

	// Sync to ensure visibility
	s.sharedIndex.Sync()

	latency := time.Since(start)

	resp := &ServerIndexResponse{
		Indexed:   indexed,
		Latency:   latency.Microseconds(),
		RequestID: req.RequestID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleBatchIndex handles batch indexing with parallel processing
func (s *SearchServer) handleBatchIndex(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if s.config.ReadOnly {
		http.Error(w, "Server is in read-only mode", http.StatusForbidden)
		return
	}

	start := time.Now()

	req := &ServerIndexRequest{}
	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// Parallel embedding generation
	type embeddingResult struct {
		doc   ServerDocument
		vec   simd.Vec512
		scale float32
		err   error
	}

	results := make(chan embeddingResult, len(req.Documents))

	// Process in parallel
	var wg sync.WaitGroup
	for _, doc := range req.Documents {
		wg.Add(1)
		go func(d ServerDocument) {
			defer wg.Done()

			embedding, err := s.model.EmbedInt8(d.Text)
			if err != nil {
				results <- embeddingResult{doc: d, err: err}
				return
			}

			var vec simd.Vec512
			copy(vec[:], embedding.Vector)

			results <- embeddingResult{
				doc:   d,
				vec:   vec,
				scale: embedding.Scale,
			}
		}(doc)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Index results as they come in
	indexed := 0
	for result := range results {
		if result.err != nil {
			log.Printf("Failed to embed document %d: %v", result.doc.ID, result.err)
			continue
		}

		if err := s.sharedIndex.AddVector(&result.vec, result.scale, result.doc.ID); err != nil {
			log.Printf("Failed to index document %d: %v", result.doc.ID, err)
			continue
		}

		indexed++
	}

	// Sync to ensure visibility
	s.sharedIndex.Sync()

	latency := time.Since(start)

	resp := &ServerIndexResponse{
		Indexed:   indexed,
		Latency:   latency.Microseconds(),
		RequestID: req.RequestID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleHealth handles health check requests
func (s *SearchServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	stats := s.sharedIndex.Stats()

	health := map[string]interface{}{
		"status":      "healthy",
		"vectors":     stats.NumVectors,
		"max_vectors": stats.MaxVectors,
		"memory_mb":   stats.MemoryUsageMB,
		"uptime_sec":  0, // TODO: Track actual server start time
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// handleMetrics handles metrics requests
func (s *SearchServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	stats := s.sharedIndex.Stats()

	totalReqs := atomic.LoadUint64(&s.totalRequests)
	totalLat := atomic.LoadUint64(&s.totalLatency)
	avgLatency := uint64(0)
	if totalReqs > 0 {
		avgLatency = totalLat / totalReqs / 1000 // Convert to microseconds
	}

	metrics := map[string]interface{}{
		"total_requests":  totalReqs,
		"avg_latency_us":  avgLatency,
		"p95_latency_us":  atomic.LoadUint64(&s.p95Latency),
		"p99_latency_us":  atomic.LoadUint64(&s.p99Latency),
		"total_searches":  stats.TotalSearches,
		"total_reads":     stats.TotalReads,
		"vectors_indexed": stats.NumVectors,
		"memory_usage_mb": stats.MemoryUsageMB,
		"cache_size":      stats.CacheSize,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// metricsCollector collects performance metrics
func (s *SearchServer) metricsCollector() {
	defer s.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Collect and log metrics
			stats := s.sharedIndex.Stats()
			totalReqs := atomic.LoadUint64(&s.totalRequests)

			if totalReqs > 0 {
				avgLatency := atomic.LoadUint64(&s.totalLatency) / totalReqs / 1000
				log.Printf("Metrics: requests=%d, avg_latency=%dµs, vectors=%d, memory=%.2fMB",
					totalReqs, avgLatency, stats.NumVectors, stats.MemoryUsageMB)
			}

		case <-s.ctx.Done():
			return
		}
	}
}

// Stop gracefully stops the server
func (s *SearchServer) Stop() error {
	log.Println("Stopping search server...")

	// Cancel context
	s.cancel()

	// Shutdown HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := s.server.Shutdown(ctx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	// Wait for goroutines
	s.wg.Wait()

	// Close shared index
	if err := s.sharedIndex.Close(); err != nil {
		return fmt.Errorf("failed to close shared index: %w", err)
	}

	log.Println("Search server stopped")
	return nil
}
