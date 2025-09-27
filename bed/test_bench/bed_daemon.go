package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

// BedDaemon runs a persistent search service with real-time indexing
type BedDaemon struct {
	index      *PersistentIndex
	config     *DaemonConfig
	httpServer *http.Server
	wg         sync.WaitGroup
	shutdown   chan struct{}
	startTime  time.Time
}

// DaemonConfig holds configuration for the daemon
type DaemonConfig struct {
	// Index settings
	MaxDocuments  int      `json:"max_documents"`
	IndexDirs     []string `json:"index_dirs"`
	FileExtensions []string `json:"file_extensions"`
	ChunkSize     int      `json:"chunk_size"`
	MaxLines      int      `json:"max_lines_per_chunk"`

	// Performance settings
	FileWorkers   int `json:"file_workers"`
	IndexWorkers  int `json:"index_workers"`
	SearchWorkers int `json:"search_workers"`

	// Server settings
	HTTPPort      int    `json:"http_port"`
	EnableMetrics bool   `json:"enable_metrics"`
	EnableProfile bool   `json:"enable_profile"`

	// Auto-indexing
	WatchInterval    time.Duration `json:"watch_interval"`
	AutoReindex      bool          `json:"auto_reindex"`
	ReindexThreshold float64       `json:"reindex_threshold"` // CPU load threshold
}

// DefaultDaemonConfig returns sensible defaults
func DefaultDaemonConfig() *DaemonConfig {
	return &DaemonConfig{
		MaxDocuments:     1000000, // 1M documents
		FileExtensions:   []string{".go", ".py", ".js", ".ts", ".txt", ".md", ".c", ".cpp", ".h", ".rs", ".java"},
		ChunkSize:        500,
		MaxLines:         10,
		FileWorkers:      runtime.NumCPU(),
		IndexWorkers:     4,
		SearchWorkers:    2,
		HTTPPort:         8080,
		EnableMetrics:    true,
		EnableProfile:    false,
		WatchInterval:    5 * time.Second,
		AutoReindex:      true,
		ReindexThreshold: 0.5, // 50% CPU threshold
	}
}

// NewBedDaemon creates a new daemon instance
func NewBedDaemon(config *DaemonConfig) (*BedDaemon, error) {
	// Load model
	model, err := LoadFastModel("../../model/modelint8_512dim.safetensors", "../../model/tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	// Create persistent index
	index := NewPersistentIndex(config.MaxDocuments, model)

	daemon := &BedDaemon{
		index:     index,
		config:    config,
		shutdown:  make(chan struct{}),
		startTime: time.Now(),
	}

	// Setup HTTP server
	daemon.setupHTTPServer()

	return daemon, nil
}

// setupHTTPServer configures HTTP endpoints
func (d *BedDaemon) setupHTTPServer() {
	mux := http.NewServeMux()

	// Search endpoint
	mux.HandleFunc("/search", d.handleSearch)

	// Index management endpoints
	mux.HandleFunc("/index/add", d.handleIndexAdd)
	mux.HandleFunc("/index/remove", d.handleIndexRemove)
	mux.HandleFunc("/index/refresh", d.handleIndexRefresh)
	mux.HandleFunc("/index/stats", d.handleIndexStats)

	// Health and metrics
	mux.HandleFunc("/health", d.handleHealth)
	if d.config.EnableMetrics {
		mux.HandleFunc("/metrics", d.handleMetrics)
	}

	// Profiling endpoints (handled by import _ "net/http/pprof")
	if d.config.EnableProfile {
		// pprof handlers are automatically registered
	}

	d.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", d.config.HTTPPort),
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}
}

// Start begins daemon operation
func (d *BedDaemon) Start() error {
	log.Printf("🚀 Starting Bed Daemon on port %d...", d.config.HTTPPort)

	// Initial indexing of configured directories
	if len(d.config.IndexDirs) > 0 {
		d.wg.Add(1)
		go d.initialIndexing()
	}

	// Start file watcher
	if d.config.AutoReindex {
		d.wg.Add(1)
		go d.fileWatcher()
	}

	// Start HTTP server
	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		if err := d.httpServer.ListenAndServe(); err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()

	// Start stats reporter
	d.wg.Add(1)
	go d.statsReporter()

	log.Println("✅ Bed Daemon started successfully")

	// Wait for shutdown signal
	d.waitForShutdown()

	return nil
}

// initialIndexing performs the initial directory indexing
func (d *BedDaemon) initialIndexing() {
	defer d.wg.Done()

	log.Println("📂 Starting initial indexing...")
	startTime := time.Now()

	for _, dir := range d.config.IndexDirs {
		log.Printf("  Indexing directory: %s", dir)
		err := d.index.IndexDirectory(dir, d.config.FileExtensions)
		if err != nil {
			log.Printf("  ⚠️  Error indexing %s: %v", dir, err)
		}
	}

	// Wait for indexing to complete
	time.Sleep(2 * time.Second)

	stats := d.index.GetStats()
	duration := time.Since(startTime)

	log.Printf("✅ Initial indexing complete:")
	log.Printf("  Files: %d", stats["files_processed"])
	log.Printf("  Documents: %d", stats["docs_indexed"])
	log.Printf("  Time: %.2fs", duration.Seconds())
	log.Printf("  Rate: %.0f docs/sec", float64(stats["docs_indexed"].(uint64))/duration.Seconds())
}

// fileWatcher monitors directories for changes
func (d *BedDaemon) fileWatcher() {
	defer d.wg.Done()

	ticker := time.NewTicker(d.config.WatchInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Check CPU load before triggering reindex
			if d.shouldReindex() {
				d.checkForChanges()
			}
		case <-d.shutdown:
			return
		}
	}
}

// shouldReindex checks if we should perform reindexing
func (d *BedDaemon) shouldReindex() bool {
	// Simple CPU load check (can be enhanced)
	var stat syscall.Sysinfo_t
	if err := syscall.Sysinfo(&stat); err == nil {
		load1 := float64(stat.Loads[0]) / 65536.0
		cpus := float64(runtime.NumCPU())
		cpuUsage := load1 / cpus

		if cpuUsage > d.config.ReindexThreshold {
			return false // Too busy
		}
	}
	return true
}

// checkForChanges scans for file changes
func (d *BedDaemon) checkForChanges() {
	for _, dir := range d.config.IndexDirs {
		filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
			if err != nil || entry.IsDir() {
				return nil
			}

			ext := filepath.Ext(path)
			valid := false
			for _, allowedExt := range d.config.FileExtensions {
				if ext == allowedExt {
					valid = true
					break
				}
			}

			if valid {
				info, _ := entry.Info()
				// The persistent index will check if the file needs reindexing
				d.index.IndexFile(path, false)
			}

			return nil
		})
	}
}

// statsReporter periodically reports statistics
func (d *BedDaemon) statsReporter() {
	defer d.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			stats := d.index.GetStats()
			log.Printf("📊 Stats: %d docs, %d files, %d searches, %.1f MB GPU",
				stats["active_docs"],
				stats["files_tracked"],
				stats["searches_handled"],
				stats["gpu_memory_mb"])
		case <-d.shutdown:
			return
		}
	}
}

// HTTP Handlers

func (d *BedDaemon) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.TopK == 0 {
		req.TopK = 10
	}

	// Perform search
	resp, err := d.index.Search(req.Query, req.TopK)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Format response
	result := map[string]interface{}{
		"query":   req.Query,
		"results": resp.Results,
		"time_ms": resp.Time.Milliseconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (d *BedDaemon) handleIndexAdd(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Path  string `json:"path"`
		Force bool   `json:"force"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := d.index.IndexFile(req.Path, req.Force); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]string{
		"status": "indexing",
		"path":   req.Path,
	})
}

func (d *BedDaemon) handleIndexRemove(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Path string `json:"path"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	d.index.RemoveFile(req.Path)

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status": "removed",
		"path":   req.Path,
	})
}

func (d *BedDaemon) handleIndexRefresh(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Trigger reindexing of all directories
	go d.checkForChanges()

	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]string{
		"status": "refreshing",
	})
}

func (d *BedDaemon) handleIndexStats(w http.ResponseWriter, r *http.Request) {
	stats := d.index.GetStats()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (d *BedDaemon) handleHealth(w http.ResponseWriter, r *http.Request) {
	stats := d.index.GetStats()

	health := map[string]interface{}{
		"status": "healthy",
		"uptime": time.Since(d.startTime).Seconds(),
		"index": map[string]interface{}{
			"documents": stats["active_docs"],
			"capacity":  stats["capacity"],
			"files":     stats["files_tracked"],
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (d *BedDaemon) handleMetrics(w http.ResponseWriter, r *http.Request) {
	stats := d.index.GetStats()

	// Prometheus-style metrics
	fmt.Fprintf(w, "# TYPE bed_documents_total gauge\n")
	fmt.Fprintf(w, "bed_documents_total %d\n", stats["active_docs"])

	fmt.Fprintf(w, "# TYPE bed_files_total gauge\n")
	fmt.Fprintf(w, "bed_files_total %d\n", stats["files_tracked"])

	fmt.Fprintf(w, "# TYPE bed_searches_total counter\n")
	fmt.Fprintf(w, "bed_searches_total %d\n", stats["searches_handled"])

	fmt.Fprintf(w, "# TYPE bed_gpu_memory_mb gauge\n")
	fmt.Fprintf(w, "bed_gpu_memory_mb %.2f\n", stats["gpu_memory_mb"])
}

// waitForShutdown handles graceful shutdown
func (d *BedDaemon) waitForShutdown() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	sig := <-sigChan
	log.Printf("⚠️  Received signal %v, shutting down...", sig)

	// Initiate shutdown
	close(d.shutdown)

	// Shutdown HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	d.httpServer.Shutdown(ctx)

	// Shutdown index
	d.index.Shutdown()

	// Wait for all goroutines
	d.wg.Wait()

	log.Println("✅ Bed Daemon shutdown complete")
}

var (
	configFile = flag.String("config", "", "Configuration file")
	daemonMode = flag.Bool("daemon", false, "Run as daemon")
	indexDirs  = flag.String("dirs", ".", "Comma-separated directories to index")
	port       = flag.Int("port", 8080, "HTTP port")
	maxDocs    = flag.Int("max-docs", 1000000, "Maximum documents")
)

func main() {
	flag.Parse()

	// Load or create config
	config := DefaultDaemonConfig()

	if *configFile != "" {
		data, err := os.ReadFile(*configFile)
		if err != nil {
			log.Fatalf("Failed to read config: %v", err)
		}
		if err := json.Unmarshal(data, config); err != nil {
			log.Fatalf("Failed to parse config: %v", err)
		}
	} else {
		// Use command-line flags
		config.HTTPPort = *port
		config.MaxDocuments = *maxDocs
		config.IndexDirs = strings.Split(*indexDirs, ",")
	}

	// Create and start daemon
	daemon, err := NewBedDaemon(config)
	if err != nil {
		log.Fatalf("Failed to create daemon: %v", err)
	}

	if err := daemon.Start(); err != nil {
		log.Fatalf("Failed to start daemon: %v", err)
	}
}

// Client example for testing
func searchClient(query string, topK int) {
	reqBody, _ := json.Marshal(map[string]interface{}{
		"query": query,
		"top_k": topK,
	})

	resp, err := http.Post("http://localhost:8080/search", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		log.Printf("Search error: %v", err)
		return
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	fmt.Printf("Search results for '%s':\n", query)
	fmt.Printf("Time: %v ms\n", result["time_ms"])

	if results, ok := result["results"].([]interface{}); ok {
		for i, r := range results {
			if res, ok := r.(map[string]interface{}); ok {
				fmt.Printf("%d. %s (%.3f)\n", i+1, res["FilePath"], res["Score"])
			}
		}
	}
}