package src

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/fsnotify/fsnotify"
)

// FileInfo represents a file discovered during crawling
type FileInfo struct {
	Path         string
	Size         int64
	ModTime      time.Time
	IsText       bool
	LineCount    int
	Content      string
	RelativePath string
}

// CrawlStats tracks crawling statistics
type CrawlStats struct {
	FilesFound     int64
	FilesProcessed int64
	BytesProcessed int64
	LinesProcessed int64
	Errors         int64
	StartTime      time.Time
	EndTime        time.Time
	ElapsedTime    time.Duration
}

// Crawler handles async filesystem crawling with ignore pattern filtering
type Crawler struct {
	basePath     string
	ignoreFilter *IgnoreFilter
	config       *Config
	
	// Async processing
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
	
	// Channels for pipeline
	filePaths    chan string
	fileInfos    chan *FileInfo
	results      chan *FileInfo
	errors       chan error
	
	// Stats
	stats        *CrawlStats
	statsLock    sync.RWMutex
	
	// File watcher for incremental updates
	watcher      *fsnotify.Watcher
	watchEnabled bool
}

// CrawlOptions configures the crawling behavior
type CrawlOptions struct {
	MaxWorkers     int
	BufferSize     int
	MaxFileSize    int64
	IncludeContent bool
	WatchMode      bool
	Recursive      bool
	FollowSymlinks bool
}

// DefaultCrawlOptions returns sensible defaults
func DefaultCrawlOptions() CrawlOptions {
	return CrawlOptions{
		MaxWorkers:     8,
		BufferSize:     1000,
		MaxFileSize:    10 * 1024 * 1024, // 10MB
		IncludeContent: true,
		WatchMode:      false,
		Recursive:      true,
		FollowSymlinks: false,
	}
}

// NewCrawler creates a new filesystem crawler
func NewCrawler(basePath string, config *Config) (*Crawler, error) {
	// Normalize path
	absPath, err := filepath.Abs(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path: %w", err)
	}

	// Create ignore filter
	ignoreFilter, err := NewIgnoreFilter(absPath, config.RespectGitignore)
	if err != nil {
		return nil, fmt.Errorf("failed to create ignore filter: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	crawler := &Crawler{
		basePath:     absPath,
		ignoreFilter: ignoreFilter,
		config:       config,
		ctx:          ctx,
		cancel:       cancel,
		stats: &CrawlStats{
			StartTime: time.Now(),
		},
	}

	return crawler, nil
}

// Crawl starts the async crawling process
func (c *Crawler) Crawl(options CrawlOptions) (<-chan *FileInfo, <-chan error, error) {
	// Initialize channels
	c.filePaths = make(chan string, options.BufferSize)
	c.fileInfos = make(chan *FileInfo, options.BufferSize)
	c.results = make(chan *FileInfo, options.BufferSize)
	c.errors = make(chan error, options.BufferSize/4)

	// Start file discovery goroutine
	c.wg.Add(1)
	go c.discoverFiles(options)

	// Start file processing workers
	for i := 0; i < options.MaxWorkers; i++ {
		c.wg.Add(1)
		go c.processFiles(options)
	}

	// Start result aggregation
	c.wg.Add(1)
	go c.aggregateResults()

	// Setup file watcher if requested
	if options.WatchMode {
		if err := c.setupWatcher(); err != nil {
			c.Close()
			return nil, nil, fmt.Errorf("failed to setup file watcher: %w", err)
		}
	}

	return c.results, c.errors, nil
}

// discoverFiles recursively discovers files in the filesystem
func (c *Crawler) discoverFiles(options CrawlOptions) {
	defer close(c.filePaths)
	defer c.wg.Done()

	err := filepath.WalkDir(c.basePath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			select {
			case c.errors <- fmt.Errorf("walk error %s: %w", path, err):
			case <-c.ctx.Done():
				return c.ctx.Err()
			}
			return nil
		}

		// Check if context is cancelled
		select {
		case <-c.ctx.Done():
			return c.ctx.Err()
		default:
		}

		// Skip if should be ignored
		if c.ignoreFilter.ShouldIgnore(path) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Only process regular files
		if !d.Type().IsRegular() {
			return nil
		}

		// Check if it's a text file
		if !c.ignoreFilter.IsTextFile(path) {
			return nil
		}

		// Check file size
		info, err := d.Info()
		if err != nil {
			select {
			case c.errors <- fmt.Errorf("stat error %s: %w", path, err):
			case <-c.ctx.Done():
				return c.ctx.Err()
			}
			return nil
		}

		if info.Size() > options.MaxFileSize {
			return nil // Skip large files
		}

		// Add to processing queue
		atomic.AddInt64(&c.stats.FilesFound, 1)
		select {
		case c.filePaths <- path:
		case <-c.ctx.Done():
			return c.ctx.Err()
		}

		return nil
	})

	if err != nil && err != c.ctx.Err() {
		select {
		case c.errors <- fmt.Errorf("filesystem walk failed: %w", err):
		case <-c.ctx.Done():
		}
	}
}

// processFiles processes discovered files
func (c *Crawler) processFiles(options CrawlOptions) {
	defer c.wg.Done()

	for {
		select {
		case path, ok := <-c.filePaths:
			if !ok {
				return // Channel closed
			}

			if fileInfo, err := c.processFile(path, options); err != nil {
				atomic.AddInt64(&c.stats.Errors, 1)
				select {
				case c.errors <- err:
				case <-c.ctx.Done():
					return
				}
			} else if fileInfo != nil {
				select {
				case c.fileInfos <- fileInfo:
				case <-c.ctx.Done():
					return
				}
			}

		case <-c.ctx.Done():
			return
		}
	}
}

// processFile processes a single file
func (c *Crawler) processFile(path string, options CrawlOptions) (*FileInfo, error) {
	// Get file info
	stat, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to stat %s: %w", path, err)
	}

	relPath, _ := filepath.Rel(c.basePath, path)

	fileInfo := &FileInfo{
		Path:         path,
		RelativePath: relPath,
		Size:         stat.Size(),
		ModTime:      stat.ModTime(),
		IsText:       true,
	}

	// Read file content if requested
	if options.IncludeContent {
		content, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("failed to read %s: %w", path, err)
		}

		// Check if content is actually text
		if !isTextContent(content) {
			return nil, nil // Skip binary files
		}

		fileInfo.Content = string(content)
		fileInfo.LineCount = countLines(fileInfo.Content)

		// Update stats
		atomic.AddInt64(&c.stats.BytesProcessed, stat.Size())
		atomic.AddInt64(&c.stats.LinesProcessed, int64(fileInfo.LineCount))
	}

	atomic.AddInt64(&c.stats.FilesProcessed, 1)
	return fileInfo, nil
}

// aggregateResults handles the final result aggregation
func (c *Crawler) aggregateResults() {
	defer close(c.results)
	defer close(c.errors)
	defer c.wg.Done()

	for {
		select {
		case fileInfo, ok := <-c.fileInfos:
			if !ok {
				return // Channel closed
			}

			select {
			case c.results <- fileInfo:
			case <-c.ctx.Done():
				return
			}

		case <-c.ctx.Done():
			return
		}
	}
}

// setupWatcher sets up filesystem watching for incremental updates
func (c *Crawler) setupWatcher() error {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to create file watcher: %w", err)
	}

	c.watcher = watcher
	c.watchEnabled = true

	// Add base path to watcher
	if err := watcher.Add(c.basePath); err != nil {
		watcher.Close()
		return fmt.Errorf("failed to watch directory: %w", err)
	}

	// Start watching goroutine
	c.wg.Add(1)
	go c.watchFiles()

	return nil
}

// watchFiles monitors filesystem changes
func (c *Crawler) watchFiles() {
	defer c.wg.Done()
	defer c.watcher.Close()

	for {
		select {
		case event, ok := <-c.watcher.Events:
			if !ok {
				return
			}

			// Only care about write and create events
			if event.Op&fsnotify.Write == fsnotify.Write || 
			   event.Op&fsnotify.Create == fsnotify.Create {
				
				// Check if we should ignore this file
				if c.ignoreFilter.ShouldIgnore(event.Name) {
					continue
				}

				// Process the changed file
				if fileInfo, err := c.processFile(event.Name, CrawlOptions{
					IncludeContent: true,
					MaxFileSize:    c.config.MaxFileSize,
				}); err == nil && fileInfo != nil {
					select {
					case c.results <- fileInfo:
					case <-c.ctx.Done():
						return
					}
				}
			}

		case err, ok := <-c.watcher.Errors:
			if !ok {
				return
			}
			
			select {
			case c.errors <- fmt.Errorf("watcher error: %w", err):
			case <-c.ctx.Done():
				return
			}

		case <-c.ctx.Done():
			return
		}
	}
}

// Stats returns current crawling statistics
func (c *Crawler) Stats() CrawlStats {
	c.statsLock.RLock()
	defer c.statsLock.RUnlock()
	
	stats := *c.stats
	if stats.EndTime.IsZero() {
		stats.ElapsedTime = time.Since(stats.StartTime)
	} else {
		stats.ElapsedTime = stats.EndTime.Sub(stats.StartTime)
	}
	
	return stats
}

// Close shuts down the crawler gracefully
func (c *Crawler) Close() error {
	c.cancel()
	c.wg.Wait()
	
	c.statsLock.Lock()
	if c.stats.EndTime.IsZero() {
		c.stats.EndTime = time.Now()
	}
	c.statsLock.Unlock()
	
	if c.watcher != nil {
		return c.watcher.Close()
	}
	
	return nil
}

// isTextContent checks if content appears to be text
func isTextContent(data []byte) bool {
	if len(data) == 0 {
		return true
	}
	
	// Check for null bytes (common in binary files)
	for i, b := range data {
		if b == 0 {
			return false
		}
		
		// Only check first 8KB for performance
		if i > 8192 {
			break
		}
	}
	
	// Check for valid UTF-8 (simple check - strings.ValidString is not available in all Go versions)
	return utf8.Valid(data)
}

// countLines counts the number of lines in content
func countLines(content string) int {
	if content == "" {
		return 0
	}
	
	lines := 1
	for _, c := range content {
		if c == '\n' {
			lines++
		}
	}
	
	return lines
}