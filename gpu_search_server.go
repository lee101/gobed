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

	"github.com/lee101/gobed/ann/simd"
)

// GPUSearchServer provides a high-performance CUDA-accelerated HTTP search server
type GPUSearchServer struct {
	model       *EmbeddingModel
	gpuIndexer  *GPUIndexer
	cpuIndexer  *SharedMemoryIndex // Fallback indexer

	// Server state
	server *http.Server
	port   int

	// Performance metrics
	totalRequests    uint64
	totalLatency     uint64
	totalGPURequests uint64
	totalCPURequests uint64
	p95Latency       uint64
	p99Latency       uint64

	// Request pooling for zero-copy operations
	requestPool  sync.Pool
	responsePool sync.Pool

	// Configuration
	config GPUServerConfig

	// Lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// GPU state management
	gpuAvailable bool
	useGPU       bool
}

// GPUServerConfig configures the GPU-accelerated search server
type GPUServerConfig struct {
	Port                int
	SharedIndexPath     string
	MaxVectors          int
	MaxConcurrency      int
	EnableProfiling     bool
	EnableMetrics       bool
	ReadOnly            bool
	PreloadEmbeddings   bool
	WorkerThreads       int
	
	// GPU-specific configuration
	GPUDeviceID         int
	GPUBatchSize        int
	EnableGPUFallback   bool
	GPUMemoryLimitMB    int
	IndexingBatchSize   int
}

// DefaultGPUServerConfig returns optimized GPU server configuration
func DefaultGPUServerConfig() GPUServerConfig {
	return GPUServerConfig{
		Port:                8080,
		SharedIndexPath:     "/tmp/gobed_gpu_index",
		MaxVectors:          10000000, // 10M vectors for GPU
		MaxConcurrency:      2000,     // Higher concurrency with GPU
		EnableProfiling:     false,
		EnableMetrics:       true,
		ReadOnly:            false,
		PreloadEmbeddings:   true,
		WorkerThreads:       runtime.NumCPU(),
		
		// GPU settings
		GPUDeviceID:         0,
		GPUBatchSize:        1024,
		EnableGPUFallback:   true,
		GPUMemoryLimitMB:    12000, // 12GB for RTX 3080
		IndexingBatchSize:   2048,
	}
}

// NewGPUSearchServer creates a new CUDA-accelerated search server
func NewGPUSearchServer(model *EmbeddingModel, config GPUServerConfig) (*GPUSearchServer, error) {
	ctx, cancel := context.WithCancel(context.Background())

	server := &GPUSearchServer{
		model:        model,
		port:         config.Port,
		config:       config,
		ctx:          ctx,
		cancel:       cancel,
		gpuAvailable: IsCUDAAvailable(),
		useGPU:       IsCUDAAvailable(),
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

	// Initialize GPU indexer if available
	if server.gpuAvailable {
		// Use adapter to create appropriate config based on build tags
		gpuConfig := createGPUIndexConfig(config.GPUDeviceID)

		var err error
		server.gpuIndexer, err = NewGPUIndexer(gpuConfig)
		if err != nil {
			log.Printf("⚠️  Failed to create GPU indexer: %v", err)
			server.gpuAvailable = false
			server.useGPU = false
		} else {
			log.Printf("🚀 GPU indexer initialized on device %d", config.GPUDeviceID)
		}
	}

	// Initialize CPU fallback indexer if needed
	if config.EnableGPUFallback || !server.gpuAvailable {
		sharedConfig := SharedMemoryConfig{
			BasePath:    config.SharedIndexPath + "_cpu",
			MaxVectors:  config.MaxVectors,
			ReadOnly:    config.ReadOnly,
			CreateIfNew: !config.ReadOnly,
			CacheSize:   100000, // Large cache for CPU fallback
			UseLockFree: true,
		}

		var err error
		server.cpuIndexer, err = NewSharedMemoryIndex(sharedConfig)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to create CPU fallback index: %w", err)
		}
		log.Printf("💾 CPU fallback indexer initialized")
	}

	// Setup HTTP routes with GPU-optimized handlers
	server.setupRoutes()

	return server, nil
}

// setupRoutes configures HTTP routes with GPU acceleration
func (s *GPUSearchServer) setupRoutes() {
	mux := http.NewServeMux()

	// High-performance search endpoints
	mux.HandleFunc("/search", s.handleGPUSearch)
	mux.HandleFunc("/batch_search", s.handleGPUBatchSearch)
	mux.HandleFunc("/gpu_search", s.handleGPUSearch) // Force GPU

	// GPU-accelerated indexing endpoints
	if !s.config.ReadOnly {
		mux.HandleFunc("/index", s.handleGPUIndex)
		mux.HandleFunc("/batch_index", s.handleGPUIndex)
	}

	// System and monitoring
	mux.HandleFunc("/health", s.handleGPUHealth)
	mux.HandleFunc("/metrics", s.handleGPUMetrics)
	mux.HandleFunc("/gpu_stats", s.handleGPUStats)

	// Profiling if enabled
	if s.config.EnableProfiling {
		mux.HandleFunc("/debug/pprof/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.DefaultServeMux.ServeHTTP(w, r)
		}))
	}

	s.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.config.Port),
		Handler:      mux,
		ReadTimeout:  60 * time.Second, // Longer for GPU processing
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
}

// Start starts the GPU-accelerated search server
func (s *GPUSearchServer) Start() error {
	log.Printf("🚀 Starting GPU-accelerated search server on port %d", s.port)
	log.Printf("   GPU Available: %v (Device: %d)", s.gpuAvailable, s.config.GPUDeviceID)
	log.Printf("   Fallback Enabled: %v", s.config.EnableGPUFallback)
	
	if s.gpuAvailable {
		log.Printf("   CUDA Available: Yes")
		log.Printf("   Using Pure CUDA Implementation")
	}

	// Optimize for GPU workloads
	if s.config.WorkerThreads > 0 {
		runtime.GOMAXPROCS(s.config.WorkerThreads)
	}

	// Start server
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		if err := s.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("❌ Server error: %v", err)
		}
	}()

	// Start GPU metrics collector
	if s.config.EnableMetrics {
		s.wg.Add(1)
		go s.gpuMetricsCollector()
	}

	log.Printf("✅ GPU-accelerated search server started successfully")
	return nil
}

// handleGPUSearch handles GPU-accelerated single search requests
func (s *GPUSearchServer) handleGPUSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	atomic.AddUint64(&s.totalRequests, 1)
	start := time.Now()

	req := s.requestPool.Get().(*SearchRequest)
	defer s.requestPool.Put(req)

	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
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

	var results []SearchResult
	var searchErr error
	usedGPU := false

	// Try GPU search first with optimized memory usage
	if s.useGPU && s.gpuIndexer != nil {
		indices, scores, err := OptimizedGPUSearch(s.gpuIndexer, embedding, req.K)
		if err == nil {
			// Convert to SearchResult format
			results = make([]SearchResult, len(indices))
			for i := range indices {
				results[i] = SearchResult{
					ID:         int(indices[i]),
					Similarity: scores[i],
				}
			}
			atomic.AddUint64(&s.totalGPURequests, 1)
			usedGPU = true
		} else {
			log.Printf("⚠️  GPU search failed: %v", err)
			searchErr = err
		}
	}

	// Fallback to CPU if GPU failed or unavailable
	if !usedGPU && s.config.EnableGPUFallback && s.cpuIndexer != nil {
		var vec simd.Vec512
		copy(vec[:], embedding.Vector)
		cpuResults := s.cpuIndexer.SearchTopK(&vec, req.K)
		
		// Convert CPU results to SearchResult format
		results = make([]SearchResult, len(cpuResults))
		for i, result := range cpuResults {
			results[i] = SearchResult{
				ID:         result.ID,
				Similarity: result.Similarity,
			}
		}
		atomic.AddUint64(&s.totalCPURequests, 1)
		searchErr = nil
	}

	if searchErr != nil {
		http.Error(w, fmt.Sprintf("Search error: %v", searchErr), http.StatusInternalServerError)
		return
	}

	latency := time.Since(start)
	atomic.AddUint64(&s.totalLatency, uint64(latency.Nanoseconds()))

	resp := s.responsePool.Get().(*SearchResponse)
	defer s.responsePool.Put(resp)

	resp.Results = results
	resp.Latency = latency.Microseconds()
	resp.RequestID = req.RequestID

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-GPU-Accelerated", fmt.Sprintf("%v", usedGPU))
	json.NewEncoder(w).Encode(resp)
}

// handleGPUBatchSearch handles GPU-accelerated batch search
func (s *GPUSearchServer) handleGPUBatchSearch(w http.ResponseWriter, r *http.Request) {
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

	// Generate embeddings in parallel batches for GPU efficiency
	embeddings := make([][]int8, len(req.Queries))
	var wg sync.WaitGroup
	
	// Process embeddings in GPU-optimized batches
	batchSize := s.config.GPUBatchSize
	for i := 0; i < len(req.Queries); i += batchSize {
		wg.Add(1)
		go func(startIdx int) {
			defer wg.Done()
			endIdx := startIdx + batchSize
			if endIdx > len(req.Queries) {
				endIdx = len(req.Queries)
			}

			for j := startIdx; j < endIdx; j++ {
				embedding, err := s.model.EmbedInt8(req.Queries[j])
				if err != nil {
					log.Printf("⚠️  Embedding error for query %d: %v", j, err)
					continue
				}
				embeddings[j] = embedding.Vector
			}
		}(i)
	}
	wg.Wait()

	var results [][]SearchResult
	var searchErr error
	usedGPU := false

	// GPU batch search not yet implemented in pure CUDA version
	// TODO: Implement batch search in pure CUDA
	// if s.useGPU && s.gpuIndexer != nil {
	//     results, searchErr = s.gpuIndexer.BatchSearch(embeddings, req.K)
	//     if searchErr == nil {
	//         atomic.AddUint64(&s.totalGPURequests, 1)
	//         usedGPU = true
	//     } else {
	//         log.Printf("⚠️  GPU batch search failed: %v", searchErr)
	//     }
	// }

	// CPU fallback for batch search
	if !usedGPU && s.config.EnableGPUFallback && s.cpuIndexer != nil {
		results = make([][]SearchResult, len(embeddings))
		vectors := make([]*simd.Vec512, len(embeddings))
		
		for i, emb := range embeddings {
			if emb != nil {
				vec := &simd.Vec512{}
				copy(vec[:], emb)
				vectors[i] = vec
			}
		}

		cpuResults := s.cpuIndexer.BatchSearch(vectors, req.K)
		for i, batch := range cpuResults {
			results[i] = make([]SearchResult, len(batch))
			for j, result := range batch {
				results[i][j] = SearchResult{
					ID:         result.ID,
					Similarity: result.Similarity,
				}
			}
		}
		atomic.AddUint64(&s.totalCPURequests, 1)
		searchErr = nil
	}

	if searchErr != nil {
		http.Error(w, fmt.Sprintf("Batch search error: %v", searchErr), http.StatusInternalServerError)
		return
	}

	latency := time.Since(start)

	resp := &SearchResponse{
		Batch:     results,
		Latency:   latency.Microseconds(),
		RequestID: req.RequestID,
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-GPU-Accelerated", fmt.Sprintf("%v", usedGPU))
	w.Header().Set("X-Batch-Size", fmt.Sprintf("%d", len(req.Queries)))
	json.NewEncoder(w).Encode(resp)
}

// handleGPUIndex handles GPU-accelerated document indexing
func (s *GPUSearchServer) handleGPUIndex(w http.ResponseWriter, r *http.Request) {
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

	// Process documents in GPU-optimized batches
	indexed := 0
	batchSize := s.config.IndexingBatchSize
	
	for i := 0; i < len(req.Documents); i += batchSize {
		end := i + batchSize
		if end > len(req.Documents) {
			end = len(req.Documents)
		}

		batch := req.Documents[i:end]
		batchIndexed, err := s.indexDocumentBatch(batch)
		if err != nil {
			log.Printf("⚠️  Failed to index batch %d-%d: %v", i, end, err)
			continue
		}
		indexed += batchIndexed
	}

	latency := time.Since(start)

	resp := &ServerIndexResponse{
		Indexed:   indexed,
		Latency:   latency.Microseconds(),
		RequestID: req.RequestID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// indexDocumentBatch indexes a batch of documents using GPU acceleration
func (s *GPUSearchServer) indexDocumentBatch(documents []ServerDocument) (int, error) {
	// Generate embeddings in parallel
	type embedResult struct {
		vector []int8
		scale  float32
	}
	embeddings := make([]embedResult, len(documents))
	var wg sync.WaitGroup
	var mu sync.Mutex
	errors := make([]error, len(documents))

	for i, doc := range documents {
		wg.Add(1)
		go func(idx int, d ServerDocument) {
			defer wg.Done()
			
			embedding, err := s.model.EmbedInt8(d.Text)
			mu.Lock()
			if err != nil {
				errors[idx] = err
			} else {
				embeddings[idx] = embedResult{
					vector: embedding.Vector,
					scale:  embedding.Scale,
				}
			}
			mu.Unlock()
		}(i, doc)
	}
	wg.Wait()

	// Filter successful embeddings
	var validEmbeddings [][]int8
	var validScales []float32
	indexed := 0

	for i, emb := range embeddings {
		if errors[i] != nil {
			log.Printf("⚠️  Failed to embed document %d: %v", documents[i].ID, errors[i])
			continue
		}

		validEmbeddings = append(validEmbeddings, emb.vector)
		validScales = append(validScales, emb.scale)
		indexed++
	}

	if len(validEmbeddings) == 0 {
		return 0, fmt.Errorf("no valid embeddings generated")
	}

	// Add to GPU index with optimized memory usage
	if s.useGPU && s.gpuIndexer != nil {
		// Convert to EmbedInt8Result format for optimized indexing
		embedResults := make([]*EmbedInt8Result, len(validEmbeddings))
		for i, emb := range validEmbeddings {
			embedResults[i] = &EmbedInt8Result{
				Vector: emb,
				Scale:  validScales[i],
			}
		}
		
		if err := OptimizedBatchIndex(s.gpuIndexer, embedResults); err != nil {
			log.Printf("⚠️  GPU indexing failed: %v", err)
			// Try CPU fallback
			if s.config.EnableGPUFallback && s.cpuIndexer != nil {
				return s.indexBatchCPU(validEmbeddings)
			}
			return 0, err
		}
		return indexed, nil
	}

	// CPU fallback
	if s.config.EnableGPUFallback && s.cpuIndexer != nil {
		return s.indexBatchCPU(validEmbeddings)
	}

	return 0, fmt.Errorf("no indexer available")
}

// indexBatchCPU indexes batch using CPU fallback
func (s *GPUSearchServer) indexBatchCPU(embeddings [][]int8) (int, error) {
	indexed := 0
	for _, emb := range embeddings {
		var vec simd.Vec512
		copy(vec[:], emb)
		
		if err := s.cpuIndexer.AddVector(&vec, 1.0, indexed); err != nil {
			log.Printf("⚠️  CPU indexing failed: %v", err)
			continue
		}
		indexed++
	}
	
	s.cpuIndexer.Sync()
	return indexed, nil
}

// handleGPUStats returns detailed GPU statistics
func (s *GPUSearchServer) handleGPUStats(w http.ResponseWriter, r *http.Request) {
	stats := map[string]interface{}{
		"gpu_available": s.gpuAvailable,
		"using_gpu":     s.useGPU,
		// "cuda_version":  GetCUDAVersion(),
		// "gpu_devices":   GetCUDADeviceCount(),
	}

	if s.gpuIndexer != nil {
		// GetStats not yet implemented in pure CUDA version
		// gpuStats := s.gpuIndexer.GetStats()
		// stats["gpu_stats"] = gpuStats
		stats["gpu_memory_usage"] = s.gpuIndexer.GetMemoryUsage()
	}

	if s.cpuIndexer != nil {
		cpuStats := s.cpuIndexer.Stats()
		stats["cpu_stats"] = cpuStats
	}

	stats["requests"] = map[string]interface{}{
		"total":     atomic.LoadUint64(&s.totalRequests),
		"gpu":       atomic.LoadUint64(&s.totalGPURequests),
		"cpu":       atomic.LoadUint64(&s.totalCPURequests),
		"gpu_ratio": float64(atomic.LoadUint64(&s.totalGPURequests)) / float64(atomic.LoadUint64(&s.totalRequests)),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// handleGPUHealth provides health check with GPU status
func (s *GPUSearchServer) handleGPUHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":        "healthy",
		"gpu_available": s.gpuAvailable,
		"using_gpu":     s.useGPU,
	}

	if s.gpuIndexer != nil {
		// GetStats not yet implemented in pure CUDA version
		// stats := s.gpuIndexer.GetStats()
		// health["gpu_vectors"] = stats.NumVectors
		// health["gpu_memory_mb"] = stats.GPUMemoryMB
		health["gpu_memory_bytes"] = s.gpuIndexer.GetMemoryUsage()
	}

	if s.cpuIndexer != nil {
		stats := s.cpuIndexer.Stats()
		health["cpu_vectors"] = stats.NumVectors
		health["cpu_memory_mb"] = stats.MemoryUsageMB
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// handleGPUMetrics provides detailed performance metrics
func (s *GPUSearchServer) handleGPUMetrics(w http.ResponseWriter, r *http.Request) {
	totalReqs := atomic.LoadUint64(&s.totalRequests)
	gpuReqs := atomic.LoadUint64(&s.totalGPURequests)
	cpuReqs := atomic.LoadUint64(&s.totalCPURequests)
	totalLat := atomic.LoadUint64(&s.totalLatency)
	
	avgLatency := uint64(0)
	if totalReqs > 0 {
		avgLatency = totalLat / totalReqs / 1000
	}

	metrics := map[string]interface{}{
		"total_requests":    totalReqs,
		"gpu_requests":      gpuReqs,
		"cpu_requests":      cpuReqs,
		"gpu_usage_ratio":   float64(gpuReqs) / float64(totalReqs),
		"avg_latency_us":    avgLatency,
		"p95_latency_us":    atomic.LoadUint64(&s.p95Latency),
		"p99_latency_us":    atomic.LoadUint64(&s.p99Latency),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// gpuMetricsCollector collects GPU-specific performance metrics
func (s *GPUSearchServer) gpuMetricsCollector() {
	defer s.wg.Done()

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			totalReqs := atomic.LoadUint64(&s.totalRequests)
			gpuReqs := atomic.LoadUint64(&s.totalGPURequests)
			cpuReqs := atomic.LoadUint64(&s.totalCPURequests)

			if totalReqs > 0 {
				avgLatency := atomic.LoadUint64(&s.totalLatency) / totalReqs / 1000
				gpuRatio := float64(gpuReqs) / float64(totalReqs) * 100

				log.Printf("🚀 GPU Metrics: total=%d, gpu=%d(%.1f%%), cpu=%d, avg_latency=%dµs",
					totalReqs, gpuReqs, gpuRatio, cpuReqs, avgLatency)

				if s.gpuIndexer != nil {
					// GetStats not yet implemented in pure CUDA version
					memUsage := s.gpuIndexer.GetMemoryUsage()
					log.Printf("   GPU Index: memory=%d bytes", memUsage)
				}
			}

		case <-s.ctx.Done():
			return
		}
	}
}

// Stop gracefully stops the GPU search server
func (s *GPUSearchServer) Stop() error {
	log.Println("🛑 Stopping GPU search server...")

	s.cancel()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := s.server.Shutdown(ctx); err != nil {
		log.Printf("❌ Server shutdown error: %v", err)
	}

	s.wg.Wait()

	// Close indexers
	if s.gpuIndexer != nil {
		s.gpuIndexer.Close()
		log.Printf("✅ GPU indexer closed")
	}

	if s.cpuIndexer != nil {
		if err := s.cpuIndexer.Close(); err != nil {
			log.Printf("⚠️  CPU indexer close error: %v", err)
		}
	}

	log.Println("✅ GPU search server stopped")
	return nil
}