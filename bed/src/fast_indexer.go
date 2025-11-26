package src

import (
	"bufio"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// FastIndexer provides high-performance parallel filesystem indexing
type FastIndexer struct {
	engine       *Int8SearchEngine
	ignoreFilter *EnhancedIgnoreFilter

	// Stats
	filesProcessed int64
	linesIndexed   int64
	bytesRead      int64
	errors         int64
	startTime      time.Time

	// Config
	maxFileSize  int64
	maxLineLen   int
	batchSize    int
	numWorkers   int
}

// FastIndexerConfig configures the fast indexer
type FastIndexerConfig struct {
	MaxFileSize int64
	MaxLineLen  int
	BatchSize   int
	NumWorkers  int
}

// DefaultFastIndexerConfig returns sensible defaults
func DefaultFastIndexerConfig() FastIndexerConfig {
	return FastIndexerConfig{
		MaxFileSize: 10 << 20, // 10MB
		MaxLineLen:  1000,
		BatchSize:   500,
		NumWorkers:  runtime.NumCPU(),
	}
}

// NewFastIndexer creates a new fast indexer
func NewFastIndexer(config FastIndexerConfig) (*FastIndexer, error) {
	engine, err := NewInt8SearchEngine()
	if err != nil {
		return nil, err
	}

	ignoreFilter, err := NewEnhancedIgnoreFilter(".",
		WithMaxFileSize(config.MaxFileSize),
		WithBinarySearch(false),
	)
	if err != nil {
		return nil, err
	}

	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}

	return &FastIndexer{
		engine:       engine,
		ignoreFilter: ignoreFilter,
		maxFileSize:  config.MaxFileSize,
		maxLineLen:   config.MaxLineLen,
		batchSize:    config.BatchSize,
		numWorkers:   config.NumWorkers,
	}, nil
}

// fileWork represents a file to be processed
type fileWork struct {
	path string
	info os.FileInfo
}

// lineWork represents lines to be indexed
type lineWork struct {
	path       string
	lineNumber int
	content    string
}

// IndexDirectory indexes all files in a directory with parallel processing
func (fi *FastIndexer) IndexDirectory(dir string, verbose bool) error {
	fi.startTime = time.Now()
	atomic.StoreInt64(&fi.filesProcessed, 0)
	atomic.StoreInt64(&fi.linesIndexed, 0)
	atomic.StoreInt64(&fi.bytesRead, 0)
	atomic.StoreInt64(&fi.errors, 0)

	// Stage 1: Parallel file discovery
	fileChan := make(chan fileWork, 1000)
	lineChan := make(chan []lineWork, fi.batchSize)

	var discoveryWg sync.WaitGroup
	discoveryWg.Add(1)
	go func() {
		defer discoveryWg.Done()
		defer close(fileChan)
		fi.discoverFiles(dir, fileChan)
	}()

	// Stage 2: Parallel file reading
	var readWg sync.WaitGroup
	for i := 0; i < fi.numWorkers; i++ {
		readWg.Add(1)
		go func() {
			defer readWg.Done()
			fi.readFiles(fileChan, lineChan)
		}()
	}

	// Close line channel when reading is done
	go func() {
		readWg.Wait()
		close(lineChan)
	}()

	// Stage 3: Batch embedding and indexing
	var indexWg sync.WaitGroup
	indexWg.Add(1)
	go func() {
		defer indexWg.Done()
		fi.indexLines(lineChan)
	}()

	// Wait for all stages
	discoveryWg.Wait()
	indexWg.Wait()

	if verbose {
		fi.printStats()
	}

	return nil
}

// discoverFiles walks the directory tree and sends files to process
func (fi *FastIndexer) discoverFiles(dir string, fileChan chan<- fileWork) {
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			atomic.AddInt64(&fi.errors, 1)
			return nil
		}

		if info.IsDir() {
			// Skip hidden directories and common ignore patterns
			name := info.Name()
			if name != "." && strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			if name == "node_modules" || name == "vendor" || name == "__pycache__" {
				return filepath.SkipDir
			}
			return nil
		}

		// Check file size
		if info.Size() > fi.maxFileSize {
			return nil
		}

		// Check if we should process this file
		shouldProcess, fileType := fi.ignoreFilter.ShouldProcess(path)
		if !shouldProcess || fileType == FileTypeBinary {
			return nil
		}

		fileChan <- fileWork{path: path, info: info}
		return nil
	})
}

// readFiles reads files and extracts lines
func (fi *FastIndexer) readFiles(fileChan <-chan fileWork, lineChan chan<- []lineWork) {
	batch := make([]lineWork, 0, fi.batchSize)

	for fw := range fileChan {
		lines, bytesRead := fi.readFileLines(fw.path)
		if lines == nil {
			atomic.AddInt64(&fi.errors, 1)
			continue
		}

		atomic.AddInt64(&fi.filesProcessed, 1)
		atomic.AddInt64(&fi.bytesRead, bytesRead)

		for _, lw := range lines {
			batch = append(batch, lw)
			if len(batch) >= fi.batchSize {
				// Send batch
				batchCopy := make([]lineWork, len(batch))
				copy(batchCopy, batch)
				lineChan <- batchCopy
				batch = batch[:0]
			}
		}
	}

	// Send remaining
	if len(batch) > 0 {
		lineChan <- batch
	}
}

// readFileLines reads a file and returns line work items
func (fi *FastIndexer) readFileLines(path string) ([]lineWork, int64) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0
	}
	defer file.Close()

	var lines []lineWork
	var bytesRead int64

	scanner := bufio.NewScanner(file)
	// Increase buffer size for long lines
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		bytesRead += int64(len(line) + 1)

		// Skip empty lines
		trimmed := strings.TrimSpace(line)
		if len(trimmed) == 0 {
			continue
		}

		// Truncate very long lines
		if len(trimmed) > fi.maxLineLen {
			trimmed = trimmed[:fi.maxLineLen]
		}

		lines = append(lines, lineWork{
			path:       path,
			lineNumber: lineNum,
			content:    trimmed,
		})
	}

	return lines, bytesRead
}

// indexLines indexes batches of lines
func (fi *FastIndexer) indexLines(lineChan <-chan []lineWork) {
	for batch := range lineChan {
		// Convert to documents
		docs := make([]Document, len(batch))
		for i, lw := range batch {
			docs[i] = Document{
				Path:       lw.path,
				LineNumber: lw.lineNumber,
				Content:    lw.content,
			}
		}

		// Batch add with parallel embedding
		err := fi.engine.BatchAddDocuments(docs, fi.numWorkers)
		if err != nil {
			atomic.AddInt64(&fi.errors, int64(len(batch)))
		} else {
			atomic.AddInt64(&fi.linesIndexed, int64(len(batch)))
		}
	}
}

// Search performs a search on the indexed content
func (fi *FastIndexer) Search(query string, topK int, threshold float32) ([]SearchMatch, error) {
	return fi.engine.Search(query, topK, threshold)
}

// Stats returns indexing statistics
func (fi *FastIndexer) Stats() map[string]interface{} {
	elapsed := time.Since(fi.startTime)
	files := atomic.LoadInt64(&fi.filesProcessed)
	lines := atomic.LoadInt64(&fi.linesIndexed)
	bytes := atomic.LoadInt64(&fi.bytesRead)
	errors := atomic.LoadInt64(&fi.errors)

	var filesPerSec, linesPerSec, mbPerSec float64
	if elapsed.Seconds() > 0 {
		filesPerSec = float64(files) / elapsed.Seconds()
		linesPerSec = float64(lines) / elapsed.Seconds()
		mbPerSec = float64(bytes) / (1024 * 1024) / elapsed.Seconds()
	}

	return map[string]interface{}{
		"files_processed": files,
		"lines_indexed":   lines,
		"bytes_read":      bytes,
		"errors":          errors,
		"elapsed":         elapsed.String(),
		"files_per_sec":   filesPerSec,
		"lines_per_sec":   linesPerSec,
		"mb_per_sec":      mbPerSec,
		"docs_in_index":   fi.engine.NumDocuments(),
	}
}

// printStats prints statistics to stdout
func (fi *FastIndexer) printStats() {
	stats := fi.Stats()
	println("Indexing complete:")
	println("  Files:", stats["files_processed"].(int64))
	println("  Lines:", stats["lines_indexed"].(int64))
	println("  Errors:", stats["errors"].(int64))
	println("  Time:", stats["elapsed"].(string))
	println("  Rate:", int(stats["lines_per_sec"].(float64)), "lines/sec")
}

// GetEngine returns the underlying search engine
func (fi *FastIndexer) GetEngine() *Int8SearchEngine {
	return fi.engine
}

// Clear clears the index
func (fi *FastIndexer) Clear() {
	fi.engine.Clear()
}
