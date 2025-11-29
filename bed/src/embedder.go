package src

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lee101/gobed"
	"github.com/schollz/progressbar/v3"
)

// Embedder handles GPU-accelerated embedding generation
type Embedder struct {
	model       *gobed.EmbeddingModel
	gpuIndexer  *gobed.GPUIndexer
	config      *Config
	useGPU      bool
	
	// Stats
	totalProcessed int64
	totalErrors    int64
	startTime      time.Time
	
	// Progress tracking
	progressBar *progressbar.ProgressBar
	showProgress bool
}

// EmbeddingOptions configures embedding generation
type EmbeddingOptions struct {
	UseGPU       bool
	BatchSize    int
	MaxWorkers   int
	ShowProgress bool
	LineLevel    bool
	FileLevel    bool
	ContextSize  int
}

// EmbeddingResult contains embeddings for a file
type EmbeddingResult struct {
	FileInfo        *FileInfo
	FileEmbedding   []float32
	LineEmbeddings  [][]float32
	ProcessingTime  time.Duration
	Error           error
}

// NewEmbedder creates a new embedder with GPU support
func NewEmbedder(config *Config) (*Embedder, error) {
	// Load the embedding model
	model, err := gobed.LoadModel()
	if err != nil {
		return nil, fmt.Errorf("failed to load embedding model: %w", err)
	}
	
	embedder := &Embedder{
		model:  model,
		config: config,
		useGPU: config.UseGPU,
		startTime: time.Now(),
	}
	
	// GPU support is available but simplified for now
	// The existing gobed codebase has GPU infrastructure that can be integrated
	if config.UseGPU {
		// For now, we'll use CPU with a warning
		// TODO: Integrate with existing GPU infrastructure in parent project
		fmt.Printf("Note: GPU acceleration will be integrated with existing gobed GPU infrastructure\n")
		embedder.useGPU = false
	}
	
	return embedder, nil
}

// ProcessFiles processes files and generates embeddings
func (e *Embedder) ProcessFiles(files <-chan *FileInfo, options EmbeddingOptions) (<-chan *EmbeddingResult, error) {
	results := make(chan *EmbeddingResult, options.BatchSize)
	
	// Setup progress bar if requested
	if options.ShowProgress {
		e.progressBar = progressbar.NewOptions(-1,
			progressbar.OptionSetDescription("Generating embeddings"),
			progressbar.OptionSetRenderBlankState(true),
			progressbar.OptionShowCount(),
			progressbar.OptionShowIts(),
			progressbar.OptionSetItsString("files"),
			progressbar.OptionThrottle(100*time.Millisecond),
		)
		e.showProgress = true
	}
	
	// Use GPU batch processing if available
	if e.useGPU && e.gpuIndexer != nil {
		go e.processFilesGPU(files, results, options)
	} else {
		go e.processFilesCPU(files, results, options)
	}
	
	return results, nil
}

// processFilesGPU processes files using GPU acceleration (fallback to CPU for now)
func (e *Embedder) processFilesGPU(files <-chan *FileInfo, results chan<- *EmbeddingResult, options EmbeddingOptions) {
	// For now, fallback to CPU processing
	// TODO: Integrate with existing gobed GPU infrastructure
	e.processFilesCPU(files, results, options)
}

// processBatchGPU is no longer used - kept for future GPU integration
func (e *Embedder) processBatchGPU(batch []*FileInfo, results chan<- *EmbeddingResult, options EmbeddingOptions) {
	// Placeholder for future GPU batch processing
	// TODO: Integrate with existing gobed GPU infrastructure
}

// processFilesCPU processes files using CPU (fallback)
func (e *Embedder) processFilesCPU(files <-chan *FileInfo, results chan<- *EmbeddingResult, options EmbeddingOptions) {
	defer close(results)
	
	// Create worker pool for CPU processing
	var wg sync.WaitGroup
	workers := options.MaxWorkers
	if workers <= 0 {
		workers = e.config.WorkerThreads
	}
	
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for file := range files {
				result := e.processFileCPU(file, options)
				
				if e.showProgress {
					e.progressBar.Add(1)
				}
				
				results <- result
			}
		}()
	}
	
	wg.Wait()
	
	if e.showProgress {
		e.progressBar.Finish()
	}
}

// processFileCPU processes a single file using CPU
func (e *Embedder) processFileCPU(file *FileInfo, options EmbeddingOptions) *EmbeddingResult {
	startTime := time.Now()
	
	result := &EmbeddingResult{
		FileInfo:       file,
		ProcessingTime: time.Since(startTime),
	}
	
	// Generate file-level embedding if requested
	if options.FileLevel {
		if embedding, err := e.model.Encode(file.Content); err != nil {
			result.Error = fmt.Errorf("failed to generate file embedding: %w", err)
			atomic.AddInt64(&e.totalErrors, 1)
			return result
		} else {
			result.FileEmbedding = embedding
		}
	}
	
	// Generate line-level embeddings if requested
	if options.LineLevel {
		lines := splitIntoLines(file.Content)
		lineEmbeddings := make([][]float32, 0, len(lines))
		
		for _, line := range lines {
			if len(line) == 0 {
				continue // Skip empty lines
			}
			
			if embedding, err := e.model.Encode(line); err != nil {
				result.Error = fmt.Errorf("failed to generate line embedding: %w", err)
				atomic.AddInt64(&e.totalErrors, 1)
				return result
			} else {
				lineEmbeddings = append(lineEmbeddings, embedding)
			}
		}
		
		result.LineEmbeddings = lineEmbeddings
	}
	
	atomic.AddInt64(&e.totalProcessed, 1)
	result.ProcessingTime = time.Since(startTime)
	
	return result
}

// ProcessBatch processes a batch of texts and returns embeddings
func (e *Embedder) ProcessBatch(texts []string) ([][]float32, error) {
	// CPU processing for now
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		embedding, err := e.model.Encode(text)
		if err != nil {
			return nil, fmt.Errorf("failed to encode text %d: %w", i, err)
		}
		embeddings[i] = embedding
	}
	
	return embeddings, nil
}

// Stats returns processing statistics
func (e *Embedder) Stats() map[string]interface{} {
	elapsed := time.Since(e.startTime)
	processed := atomic.LoadInt64(&e.totalProcessed)
	errors := atomic.LoadInt64(&e.totalErrors)
	
	var rate float64
	if elapsed.Seconds() > 0 {
		rate = float64(processed) / elapsed.Seconds()
	}
	
	stats := map[string]interface{}{
		"total_processed":    processed,
		"total_errors":       errors,
		"processing_time":    elapsed.String(),
		"processing_rate":    fmt.Sprintf("%.1f files/sec", rate),
		"using_gpu":          e.useGPU,
		"model_type":         e.config.EmbeddingModel,
	}
	
	// GPU stats placeholder
	if e.useGPU {
		stats["gpu_device_id"] = e.config.GPUDeviceID
		stats["gpu_batch_size"] = e.config.BatchSize
		// TODO: Add real GPU stats when integrated
	}
	
	return stats
}

// Close releases resources
func (e *Embedder) Close() error {
	// No resources to close for CPU-only version
	return nil
}

// Enhanced line processing with context
func (e *Embedder) processLinesWithContext(content string, contextSize int) ([]string, error) {
	lines := splitIntoLines(content)
	processedLines := make([]string, 0, len(lines))
	
	for i, line := range lines {
		if len(line) == 0 {
			continue
		}
		
		// Add context if configured
		if contextSize > 0 {
			contextBefore := getContextBefore(lines, i, contextSize)
			contextAfter := getContextAfter(lines, i, contextSize)
			
			// Combine line with context
			fullContext := ""
			if contextBefore != "" {
				fullContext += contextBefore + "\n"
			}
			fullContext += line
			if contextAfter != "" {
				fullContext += "\n" + contextAfter
			}
			
			processedLines = append(processedLines, fullContext)
		} else {
			processedLines = append(processedLines, line)
		}
	}
	
	return processedLines, nil
}

// IsGPUAvailable checks if GPU acceleration is available
func (e *Embedder) IsGPUAvailable() bool {
	return false // GPU integration pending
}

// GetModel returns the underlying embedding model
func (e *Embedder) GetModel() *gobed.EmbeddingModel {
	return e.model
}