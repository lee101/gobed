package src

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
)

type DaemonConfig struct {
	WatchPaths   []string
	SocketPath   string
	HTTPPort     int
	Verbose      bool
	BatchDelay   time.Duration
	MaxFileSize  int64
}

type BedDaemon struct {
	config       DaemonConfig
	searcher     *FastBedSearcher
	watcher      *fsnotify.Watcher
	ignoreFilter *EnhancedIgnoreFilter

	pendingUpdates map[string]time.Time
	updateMu       sync.Mutex

	ctx    context.Context
	cancel context.CancelFunc
}

type SearchRequest struct {
	Query     string  `json:"query"`
	Limit     int     `json:"limit"`
	Threshold float32 `json:"threshold"`
}

type SearchResponse struct {
	Results []SearchMatch `json:"results"`
	Latency float64       `json:"latency_ms"`
	Count   int           `json:"count"`
}

type StatusResponse struct {
	Indexed    int     `json:"indexed"`
	WatchPaths int     `json:"watch_paths"`
	Uptime     float64 `json:"uptime_seconds"`
}

func NewBedDaemon(config DaemonConfig) (*BedDaemon, error) {
	searcher, err := NewFastBedSearcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create searcher: %w", err)
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create watcher: %w", err)
	}

	if config.BatchDelay == 0 {
		config.BatchDelay = 100 * time.Millisecond
	}
	if config.MaxFileSize == 0 {
		config.MaxFileSize = 10 * 1024 * 1024
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &BedDaemon{
		config:         config,
		searcher:       searcher,
		watcher:        watcher,
		pendingUpdates: make(map[string]time.Time),
		ctx:            ctx,
		cancel:         cancel,
	}, nil
}

func (d *BedDaemon) Run() error {
	startTime := time.Now()

	// Initial index build
	fmt.Printf("Building initial index...\n")
	for _, path := range d.config.WatchPaths {
		if err := d.searcher.IndexDirectory(path, BedSearchOptions{Verbose: d.config.Verbose}); err != nil {
			return fmt.Errorf("failed to index %s: %w", path, err)
		}
	}
	fmt.Printf("Initial index: %d documents in %.2fs\n",
		len(d.searcher.documents), time.Since(startTime).Seconds())

	// Setup file watching
	for _, path := range d.config.WatchPaths {
		if err := d.addWatchRecursive(path); err != nil {
			return fmt.Errorf("failed to watch %s: %w", path, err)
		}
	}
	fmt.Printf("Watching %d paths for changes\n", len(d.config.WatchPaths))

	// Start watcher goroutine
	go d.watchLoop()

	// Start batch processor
	go d.batchProcessor()

	// Start servers
	errChan := make(chan error, 2)

	if d.config.SocketPath != "" {
		go func() { errChan <- d.runUnixSocket() }()
	}

	if d.config.HTTPPort > 0 {
		go func() { errChan <- d.runHTTPServer() }()
	}

	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-sigChan:
		fmt.Printf("\nReceived %v, shutting down...\n", sig)
	case err := <-errChan:
		return err
	case <-d.ctx.Done():
	}

	return d.Close()
}

func (d *BedDaemon) addWatchRecursive(root string) error {
	return filepath.WalkDir(root, func(path string, info os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			// Skip hidden and common ignore dirs
			name := filepath.Base(path)
			if len(name) > 0 && name[0] == '.' {
				return filepath.SkipDir
			}
			if name == "node_modules" || name == "vendor" || name == "__pycache__" {
				return filepath.SkipDir
			}
			return d.watcher.Add(path)
		}
		return nil
	})
}

func (d *BedDaemon) watchLoop() {
	for {
		select {
		case event, ok := <-d.watcher.Events:
			if !ok {
				return
			}
			if event.Op&(fsnotify.Write|fsnotify.Create) != 0 {
				d.updateMu.Lock()
				d.pendingUpdates[event.Name] = time.Now()
				d.updateMu.Unlock()
			}
		case err, ok := <-d.watcher.Errors:
			if !ok {
				return
			}
			if d.config.Verbose {
				fmt.Printf("Watch error: %v\n", err)
			}
		case <-d.ctx.Done():
			return
		}
	}
}

func (d *BedDaemon) batchProcessor() {
	ticker := time.NewTicker(d.config.BatchDelay)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			d.processPendingUpdates()
		case <-d.ctx.Done():
			return
		}
	}
}

func (d *BedDaemon) processPendingUpdates() {
	d.updateMu.Lock()
	if len(d.pendingUpdates) == 0 {
		d.updateMu.Unlock()
		return
	}

	// Collect files older than batch delay
	cutoff := time.Now().Add(-d.config.BatchDelay)
	var toProcess []string
	for path, t := range d.pendingUpdates {
		if t.Before(cutoff) {
			toProcess = append(toProcess, path)
			delete(d.pendingUpdates, path)
		}
	}
	d.updateMu.Unlock()

	if len(toProcess) == 0 {
		return
	}

	// Update embeddings for changed files
	for _, path := range toProcess {
		d.updateFile(path)
	}

	if d.config.Verbose {
		fmt.Printf("Updated %d files\n", len(toProcess))
	}
}

func (d *BedDaemon) updateFile(path string) {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() || info.Size() > d.config.MaxFileSize {
		return
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return
	}

	// Check if binary
	for i, b := range content {
		if b == 0 {
			return
		}
		if i > 8192 {
			break
		}
	}

	// Re-embed the file (for now just log - full incremental update needs more work)
	if d.config.Verbose {
		fmt.Printf("File changed: %s\n", path)
	}
}

func (d *BedDaemon) runUnixSocket() error {
	os.Remove(d.config.SocketPath)
	listener, err := net.Listen("unix", d.config.SocketPath)
	if err != nil {
		return err
	}
	defer listener.Close()
	defer os.Remove(d.config.SocketPath)

	fmt.Printf("Unix socket: %s\n", d.config.SocketPath)

	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-d.ctx.Done():
				return nil
			default:
				continue
			}
		}
		go d.handleConnection(conn)
	}
}

func (d *BedDaemon) handleConnection(conn net.Conn) {
	defer conn.Close()

	var req SearchRequest
	if err := json.NewDecoder(conn).Decode(&req); err != nil {
		return
	}

	if req.Limit == 0 {
		req.Limit = 10
	}
	if req.Threshold == 0 {
		req.Threshold = 0.5
	}

	start := time.Now()
	results, _ := d.searcher.SearchMatches(BedSearchOptions{
		Query:     req.Query,
		Limit:     req.Limit,
		Threshold: req.Threshold,
		NoIndex:   true,
	})

	resp := SearchResponse{
		Results: results,
		Latency: float64(time.Since(start).Microseconds()) / 1000.0,
		Count:   len(results),
	}

	json.NewEncoder(conn).Encode(resp)
}

func (d *BedDaemon) runHTTPServer() error {
	mux := http.NewServeMux()
	startTime := time.Now()

	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var req SearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if req.Limit == 0 {
			req.Limit = 10
		}
		if req.Threshold == 0 {
			req.Threshold = 0.5
		}

		start := time.Now()
		results, err := d.searcher.SearchMatches(BedSearchOptions{
			Query:     req.Query,
			Limit:     req.Limit,
			Threshold: req.Threshold,
			NoIndex:   true,
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		resp := SearchResponse{
			Results: results,
			Latency: float64(time.Since(start).Microseconds()) / 1000.0,
			Count:   len(results),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	mux.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		d.searcher.mu.RLock()
		indexed := len(d.searcher.documents)
		d.searcher.mu.RUnlock()

		resp := StatusResponse{
			Indexed:    indexed,
			WatchPaths: len(d.config.WatchPaths),
			Uptime:     time.Since(startTime).Seconds(),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	addr := fmt.Sprintf(":%d", d.config.HTTPPort)
	fmt.Printf("HTTP server: http://localhost%s\n", addr)

	server := &http.Server{Addr: addr, Handler: mux}
	go func() {
		<-d.ctx.Done()
		server.Close()
	}()

	return server.ListenAndServe()
}

func (d *BedDaemon) Close() error {
	d.cancel()
	if d.watcher != nil {
		d.watcher.Close()
	}
	if d.searcher != nil {
		d.searcher.Close()
	}
	return nil
}
