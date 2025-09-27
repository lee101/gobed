package src

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/schollz/progressbar/v3"
)

// Indexer handles building and maintaining the embedding index
type Indexer struct {
	config    *Config
	embedder  *Embedder
	index     *EmbeddingIndex
	crawler   *Crawler
	
	// Async processing
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	
	// Progress tracking
	totalFiles     int64
	processedFiles int64
	errorCount     int64
	progressBar    *progressbar.ProgressBar
	
	// Performance monitoring
	startTime      time.Time
	indexingTime   time.Duration
	embeddingTime  time.Duration
	
	// Batch processing
	batchSize      int
	workerCount    int
}

// IndexOptions configures the indexing process
type IndexOptions struct {
	Path             string
	Force            bool
	BatchSize        int
	UseGPU           bool
	Verbose          bool
	ShowProgress     bool
	MaxWorkers       int
	IncludeContent   bool
	LineLevel        bool
	FileLevel        bool
	WatchMode        bool
}

// DefaultIndexOptions returns sensible defaults
func DefaultIndexOptions() IndexOptions {
	return IndexOptions{
		Path:           ".",
		Force:          false,
		BatchSize:      1000,
		UseGPU:         false,
		Verbose:        false,
		ShowProgress:   true,
		MaxWorkers:     8,
		IncludeContent: true,
		LineLevel:      true,
		FileLevel:      true,
		WatchMode:      false,
	}
}

// NewIndexer creates a new indexer instance
func NewIndexer() (*Indexer, error) {
	// Load configuration
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}
	
	// Create embedder
	embedder, err := NewEmbedder(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedder: %w", err)
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	indexer := &Indexer{
		config:      config,
		embedder:    embedder,
		ctx:         ctx,
		cancel:      cancel,
		batchSize:   config.BatchSize,
		workerCount: config.WorkerThreads,
		startTime:   time.Now(),
	}
	
	return indexer, nil
}

// Index builds or updates the semantic index
func (idx *Indexer) Index(options IndexOptions) error {
	if options.Verbose {
		fmt.Printf("Starting indexing process for: %s\n", options.Path)
	}
	
	// Create index configuration
	indexConfig := IndexConfig{
		MaxFileSize:       idx.config.MaxFileSize,
		IncludeExtensions: idx.config.IncludeExtensions,
		ExcludeExtensions: idx.config.ExcludeExtensions,
		EmbeddingModel:    idx.config.EmbeddingModel,
		UseGPU:            options.UseGPU,
		CompressionLevel:  idx.config.CompressionLevel,
		LineBasedIndex:    options.LineLevel,
		ContextWindow:     2, // Default context window
	}
	
	// Create or load existing index
	if options.Force {
		idx.index = NewEmbeddingIndex(options.Path, idx.config.IndexPath, indexConfig)
	} else {
		// Try to load existing index
		if existing, err := LoadEmbeddingIndex(idx.config.IndexPath); err == nil {
			idx.index = existing
			if options.Verbose {
				fmt.Println("Loaded existing index, performing incremental update")
			}
		} else {
			idx.index = NewEmbeddingIndex(options.Path, idx.config.IndexPath, indexConfig)
			if options.Verbose {
				fmt.Println("Creating new index")
			}
		}
	}
	
	// Create crawler
	crawler, err := NewCrawler(options.Path, idx.config)
	if err != nil {
		return fmt.Errorf("failed to create crawler: %w", err)
	}
	idx.crawler = crawler
	
	// Configure crawling options
	crawlOptions := CrawlOptions{
		MaxWorkers:     options.MaxWorkers,
		BufferSize:     options.BatchSize,
		MaxFileSize:    idx.config.MaxFileSize,
		IncludeContent: options.IncludeContent,
		WatchMode:      options.WatchMode,
		Recursive:      true,
		FollowSymlinks: false,
	}
	
	// Start crawling
	files, errors, err := crawler.Crawl(crawlOptions)
	if err != nil {
		return fmt.Errorf("failed to start crawling: %w", err)
	}
	
	// Configure embedding options
	embeddingOptions := EmbeddingOptions{
		UseGPU:       options.UseGPU,
		BatchSize:    options.BatchSize,
		MaxWorkers:   options.MaxWorkers,
		ShowProgress: false, // We'll handle progress ourselves
		LineLevel:    options.LineLevel,
		FileLevel:    options.FileLevel,
		ContextSize:  indexConfig.ContextWindow,
	}
	
	// Start embedding processing
	embeddingResults, err := idx.embedder.ProcessFiles(files, embeddingOptions)
	if err != nil {
		return fmt.Errorf("failed to start embedding processing: %w", err)
	}
	
	// Setup progress tracking if requested
	if options.ShowProgress {
		idx.progressBar = progressbar.NewOptions(-1,
			progressbar.OptionSetDescription("Indexing files"),
			progressbar.OptionSetRenderBlankState(true),
			progressbar.OptionShowCount(),
			progressbar.OptionShowIts(),
			progressbar.OptionSetItsString("files"),
			progressbar.OptionThrottle(100*time.Millisecond),
		)
	}
	
	// Process results asynchronously
	idx.wg.Add(3)
	go idx.processEmbeddingResults(embeddingResults, options)
	go idx.handleErrors(errors, options)
	go idx.monitorProgress(options)
	
	// Wait for processing to complete
	idx.wg.Wait()
	
	// Save the index
	if err := idx.index.Save(); err != nil {
		return fmt.Errorf("failed to save index: %w", err)
	}
	
	// Display final statistics
	if options.Verbose || options.ShowProgress {
		idx.displayStatistics()
	}
	
	return nil
}

// processEmbeddingResults processes embedding results and updates the index
func (idx *Indexer) processEmbeddingResults(results <-chan *EmbeddingResult, options IndexOptions) {
	defer idx.wg.Done()
	
	batch := make([]*EmbeddingResult, 0, idx.batchSize)
	
	for result := range results {
		if result.Error != nil {
			atomic.AddInt64(&idx.errorCount, 1)
			if options.Verbose {
				fmt.Printf("Error processing %s: %v\n", result.FileInfo.RelativePath, result.Error)
			}
			continue
		}
		
		batch = append(batch, result)
		
		// Process batch when full
		if len(batch) >= idx.batchSize {
			idx.processBatch(batch, options)
			batch = batch[:0] // Reset batch
		}
		
		atomic.AddInt64(&idx.processedFiles, 1)
	}
	
	// Process remaining results
	if len(batch) > 0 {
		idx.processBatch(batch, options)
	}
}

// processBatch processes a batch of embedding results
func (idx *Indexer) processBatch(batch []*EmbeddingResult, options IndexOptions) {
	batchStart := time.Now()
	
	for _, result := range batch {
		// Prepare embeddings for indexing
		var embeddings [][]float32
		
		if options.LineLevel {
			embeddings = result.LineEmbeddings
		} else if options.FileLevel && result.FileEmbedding != nil {
			embeddings = [][]float32{result.FileEmbedding}
		}
		
		// Add to index
		if len(embeddings) > 0 {
			if err := idx.index.AddFile(result.FileInfo, embeddings); err != nil {
				atomic.AddInt64(&idx.errorCount, 1)
				if options.Verbose {
					fmt.Printf("Error adding %s to index: %v\n", result.FileInfo.RelativePath, err)
				}
			}
		}
	}
	
	idx.indexingTime += time.Since(batchStart)
	
	// Update progress
	if idx.progressBar != nil {
		idx.progressBar.Add(len(batch))
	}
}

// handleErrors processes crawling and processing errors
func (idx *Indexer) handleErrors(errors <-chan error, options IndexOptions) {
	defer idx.wg.Done()
	
	for err := range errors {
		atomic.AddInt64(&idx.errorCount, 1)
		if options.Verbose {
			fmt.Printf("Error: %v\n", err)
		}
	}
}

// monitorProgress monitors the indexing progress
func (idx *Indexer) monitorProgress(options IndexOptions) {
	defer idx.wg.Done()
	
	if !options.ShowProgress && !options.Verbose {
		return
	}
	
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			if options.Verbose {
				processed := atomic.LoadInt64(&idx.processedFiles)
				errors := atomic.LoadInt64(&idx.errorCount)
				elapsed := time.Since(idx.startTime)
				
				rate := float64(processed) / elapsed.Seconds()
				fmt.Printf("Progress: %d files processed, %d errors, %.1f files/sec\n", 
					processed, errors, rate)
			}
			
		case <-idx.ctx.Done():
			return
		}
	}
}

// displayStatistics shows final indexing statistics
func (idx *Indexer) displayStatistics() {
	if idx.progressBar != nil {
		idx.progressBar.Finish()
	}
	
	processed := atomic.LoadInt64(&idx.processedFiles)
	errors := atomic.LoadInt64(&idx.errorCount)
	elapsed := time.Since(idx.startTime)
	
	fmt.Printf("\nIndexing completed:\n")
	fmt.Printf("  Files processed: %d\n", processed)
	fmt.Printf("  Errors:          %d\n", errors)
	fmt.Printf("  Total time:      %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  Processing rate: %.1f files/sec\n", float64(processed)/elapsed.Seconds())
	
	if idx.index != nil {
		stats := idx.index.Stats()
		if totalLines, ok := stats["total_lines"].(int); ok {
			fmt.Printf("  Lines indexed:   %d\n", totalLines)
		}
		if memUsage, ok := stats["memory_usage"].(int64); ok {
			fmt.Printf("  Memory usage:    %.2f MB\n", float64(memUsage)/1024/1024)
		}
	}
	
	// Display GPU statistics if available
	if idx.embedder.IsGPUAvailable() {
		gpuStats := idx.embedder.Stats()
		fmt.Printf("  GPU accelerated: Yes \n")
		if gpuRate, ok := gpuStats["processing_rate"]; ok {
			fmt.Printf("  GPU processing:  %s\n", gpuRate)
		}
	}
}

// UpdateIndex performs incremental index updates
func (idx *Indexer) UpdateIndex(changedFiles []string, options IndexOptions) error {
	if idx.index == nil {
		return fmt.Errorf("no index loaded")
	}
	
	if options.Verbose {
		fmt.Printf("Updating index with %d changed files\n", len(changedFiles))
	}
	
	// Create a focused crawler for changed files only
	fileInfos := make([]*FileInfo, 0, len(changedFiles))
	
	for _, path := range changedFiles {
		crawler, err := NewCrawler(path, idx.config)
		if err != nil {
			continue
		}
		
		// Process single file
		crawlOptions := DefaultCrawlOptions()
		crawlOptions.WatchMode = false
		
		files, _, err := crawler.Crawl(crawlOptions)
		if err != nil {
			crawler.Close()
			continue
		}
		
		// Collect file info
		for file := range files {
			fileInfos = append(fileInfos, file)
		}
		
		crawler.Close()
	}
	
	if len(fileInfos) == 0 {
		if options.Verbose {
			fmt.Println("No valid files to update")
		}
		return nil
	}
	
	// Process files through embedder
	fileChan := make(chan *FileInfo, len(fileInfos))
	for _, fileInfo := range fileInfos {
		fileChan <- fileInfo
	}
	close(fileChan)
	
	embeddingOptions := EmbeddingOptions{
		UseGPU:       options.UseGPU,
		BatchSize:    options.BatchSize,
		MaxWorkers:   options.MaxWorkers,
		ShowProgress: options.ShowProgress,
		LineLevel:    options.LineLevel,
		FileLevel:    options.FileLevel,
	}
	
	results, err := idx.embedder.ProcessFiles(fileChan, embeddingOptions)
	if err != nil {
		return fmt.Errorf("failed to process changed files: %w", err)
	}
	
	// Update index with new embeddings
	for result := range results {
		if result.Error != nil {
			if options.Verbose {
				fmt.Printf("Error processing %s: %v\n", result.FileInfo.RelativePath, result.Error)
			}
			continue
		}
		
		var embeddings [][]float32
		if options.LineLevel {
			embeddings = result.LineEmbeddings
		} else if options.FileLevel && result.FileEmbedding != nil {
			embeddings = [][]float32{result.FileEmbedding}
		}
		
		if len(embeddings) > 0 {
			if err := idx.index.AddFile(result.FileInfo, embeddings); err != nil {
				if options.Verbose {
					fmt.Printf("Error updating %s in index: %v\n", result.FileInfo.RelativePath, err)
				}
			}
		}
	}
	
	// Save updated index
	return idx.index.Save()
}

// Stats returns current indexing statistics
func (idx *Indexer) Stats() map[string]interface{} {
	stats := map[string]interface{}{
		"total_files":      atomic.LoadInt64(&idx.totalFiles),
		"processed_files":  atomic.LoadInt64(&idx.processedFiles),
		"error_count":      atomic.LoadInt64(&idx.errorCount),
		"indexing_time":    idx.indexingTime.String(),
		"embedding_time":   idx.embeddingTime.String(),
		"elapsed_time":     time.Since(idx.startTime).String(),
		"batch_size":       idx.batchSize,
		"worker_count":     idx.workerCount,
	}
	
	// Add embedder stats
	if idx.embedder != nil {
		embedderStats := idx.embedder.Stats()
		for k, v := range embedderStats {
			stats["embedder_"+k] = v
		}
	}
	
	// Add crawler stats
	if idx.crawler != nil {
		crawlerStats := idx.crawler.Stats()
		stats["files_found"] = crawlerStats.FilesFound
		stats["bytes_processed"] = crawlerStats.BytesProcessed
		stats["lines_processed"] = crawlerStats.LinesProcessed
	}
	
	return stats
}

// Close shuts down the indexer and releases resources
func (idx *Indexer) Close() error {
	idx.cancel()
	idx.wg.Wait()
	
	var errs []error
	
	if idx.embedder != nil {
		if err := idx.embedder.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	
	if idx.crawler != nil {
		if err := idx.crawler.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	
	if len(errs) > 0 {
		return fmt.Errorf("errors during close: %v", errs)
	}
	
	return nil
}

// EstimateIndexSize estimates the disk space required for indexing
func (idx *Indexer) EstimateIndexSize(path string, options IndexOptions) (int64, error) {
	// Create temporary crawler to estimate file count and size
	crawler, err := NewCrawler(path, idx.config)
	if err != nil {
		return 0, err
	}
	defer crawler.Close()
	
	crawlOptions := CrawlOptions{
		MaxWorkers:     4,
		BufferSize:     1000,
		MaxFileSize:    idx.config.MaxFileSize,
		IncludeContent: false, // Just need file counts
		Recursive:      true,
	}
	
	files, _, err := crawler.Crawl(crawlOptions)
	if err != nil {
		return 0, err
	}
	
	var totalFiles, totalLines int64
	
	for file := range files {
		totalFiles++
		totalLines += int64(file.LineCount)
	}
	
	// Estimate index size based on embeddings
	// Assume 1024 dimensions * 4 bytes per float32 = 4KB per embedding
	embeddingSize := int64(4096) // 4KB per embedding
	
	var estimatedSize int64
	if options.LineLevel {
		estimatedSize += totalLines * embeddingSize
	}
	if options.FileLevel {
		estimatedSize += totalFiles * embeddingSize
	}
	
	// Add overhead for metadata, file entries, etc. (approximately 20%)
	estimatedSize = int64(float64(estimatedSize) * 1.2)
	
	return estimatedSize, nil
}