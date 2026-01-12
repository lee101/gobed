package src

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lee101/gobed"
	"github.com/schollz/progressbar/v3"
)

// FastEmbedder uses the optimized int8 512-dim model with zero-alloc hot path
type FastEmbedder struct {
	model        *gobed.SimpleInt8Model512
	config       *Config
	totalProcessed int64
	totalErrors    int64
	startTime      time.Time
	progressBar    *progressbar.ProgressBar
	showProgress   bool
}

// NewFastEmbedder creates a new embedder using the optimized int8 model
func NewFastEmbedder(config *Config) (*FastEmbedder, error) {
	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		return nil, fmt.Errorf("failed to load int8 model: %w", err)
	}

	return &FastEmbedder{
		model:     model,
		config:    config,
		startTime: time.Now(),
	}, nil
}

// ProcessFiles processes files using the fast int8 model
func (e *FastEmbedder) ProcessFiles(files <-chan *FileInfo, options EmbeddingOptions) (<-chan *EmbeddingResult, error) {
	results := make(chan *EmbeddingResult, options.BatchSize)

	if options.ShowProgress {
		e.progressBar = progressbar.NewOptions(-1,
			progressbar.OptionSetDescription("Embedding (int8 fast)"),
			progressbar.OptionSetRenderBlankState(true),
			progressbar.OptionShowCount(),
			progressbar.OptionShowIts(),
			progressbar.OptionSetItsString("files"),
			progressbar.OptionThrottle(100*time.Millisecond),
		)
		e.showProgress = true
	}

	go e.processFilesCPU(files, results, options)
	return results, nil
}

func (e *FastEmbedder) processFilesCPU(files <-chan *FileInfo, results chan<- *EmbeddingResult, options EmbeddingOptions) {
	defer close(results)

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
				result := e.processFile(file, options)
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

func (e *FastEmbedder) processFile(file *FileInfo, options EmbeddingOptions) *EmbeddingResult {
	startTime := time.Now()
	result := &EmbeddingResult{
		FileInfo: file,
	}

	// File-level embedding using zero-alloc path
	if options.FileLevel {
		emb, release := e.model.EmbedFast(file.Content)
		// Copy to persistent storage
		result.FileEmbedding = make([]float32, len(emb))
		copy(result.FileEmbedding, emb)
		release()
	}

	// Line-level embeddings
	if options.LineLevel {
		lines := splitIntoLinesOptimized(file.Content)
		lineEmbeddings := make([][]float32, 0, len(lines))

		for _, line := range lines {
			if len(line) == 0 {
				continue
			}
			emb, release := e.model.EmbedFast(line)
			// Copy to persistent storage
			embCopy := make([]float32, len(emb))
			copy(embCopy, emb)
			release()
			lineEmbeddings = append(lineEmbeddings, embCopy)
		}
		result.LineEmbeddings = lineEmbeddings
	}

	atomic.AddInt64(&e.totalProcessed, 1)
	result.ProcessingTime = time.Since(startTime)
	return result
}

// ProcessBatch processes multiple texts efficiently
func (e *FastEmbedder) ProcessBatch(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		emb, release := e.model.EmbedFast(text)
		embeddings[i] = make([]float32, len(emb))
		copy(embeddings[i], emb)
		release()
	}
	return embeddings, nil
}

// EmbedQuery embeds a single query with zero-copy for search
func (e *FastEmbedder) EmbedQuery(query string) ([]float32, func()) {
	return e.model.EmbedFast(query)
}

// EmbedQueryCopy embeds and returns a copy (for storage)
func (e *FastEmbedder) EmbedQueryCopy(query string) ([]float32, error) {
	emb, release := e.model.EmbedFast(query)
	defer release()
	result := make([]float32, len(emb))
	copy(result, emb)
	return result, nil
}

func (e *FastEmbedder) Stats() map[string]interface{} {
	elapsed := time.Since(e.startTime)
	processed := atomic.LoadInt64(&e.totalProcessed)
	var rate float64
	if elapsed.Seconds() > 0 {
		rate = float64(processed) / elapsed.Seconds()
	}

	return map[string]interface{}{
		"total_processed": processed,
		"total_errors":    atomic.LoadInt64(&e.totalErrors),
		"processing_time": elapsed.String(),
		"processing_rate": fmt.Sprintf("%.1f files/sec", rate),
		"model_type":      "int8_512dim",
		"using_gpu":       false,
	}
}

func (e *FastEmbedder) Close() error {
	return e.model.Close()
}

func (e *FastEmbedder) IsGPUAvailable() bool {
	return false // CPU-optimized path
}

// splitIntoLinesOptimized splits content into lines without allocating slices for empty lines
func splitIntoLinesOptimized(content string) []string {
	if content == "" {
		return nil
	}

	// Count lines first
	lineCount := 1
	for i := 0; i < len(content); i++ {
		if content[i] == '\n' {
			lineCount++
		}
	}

	lines := make([]string, 0, lineCount)
	start := 0

	for i := 0; i < len(content); i++ {
		if content[i] == '\n' {
			if i > start {
				lines = append(lines, content[start:i])
			}
			start = i + 1
		}
	}

	if start < len(content) {
		lines = append(lines, content[start:])
	}

	return lines
}
