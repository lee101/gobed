// optimized_indexing.go - Drop-in optimization for gobed GPU indexing
package main // Keep this as main package

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// IndexTextsParallel processes texts in parallel with optimal GPU utilization
func IndexTextsParallel(pipeline interface{}, texts []string, chunkSize int) error {
	if len(texts) == 0 {
		return nil
	}

	log.Printf("🚀 Starting parallel GPU indexing of %d texts", len(texts))
	log.Printf("📦 Chunk size: %d (optimized for GPU)", chunkSize)

	start := time.Now()

	// Create chunks for optimal GPU batching
	chunks := make([][]string, 0)
	for i := 0; i < len(texts); i += chunkSize {
		end := i + chunkSize
		if end > len(texts) {
			end = len(texts)
		}
		chunks = append(chunks, texts[i:end])
	}

	log.Printf("📊 Created %d chunks (avg: %d texts/chunk)", len(chunks), len(texts)/len(chunks))

	// Parallel processing with controlled concurrency
	const maxConcurrent = 8 // Adjust based on GPU memory
	semaphore := make(chan struct{}, maxConcurrent)

	var wg sync.WaitGroup
	errors := make(chan error, len(chunks))
	progress := make(chan int, len(chunks))

	// Progress monitoring goroutine
	go func() {
		completed := 0
		for range progress {
			completed++
			if completed%10 == 0 || completed == len(chunks) {
				percent := float64(completed) / float64(len(chunks)) * 100
				elapsed := time.Since(start)
				rate := float64(completed*chunkSize) / elapsed.Seconds()

				log.Printf("📈 Progress: %.1f%% (%d/%d chunks, %.0f texts/sec)",
					percent, completed, len(chunks), rate)
			}
		}
	}()

	// Process chunks in parallel
	for i, chunk := range chunks {
		wg.Add(1)
		go func(chunkNum int, chunkTexts []string) {
			defer wg.Done()

			// Acquire semaphore (limit concurrent GPU operations)
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// Type assertion for pipeline interface
			// Replace with your actual pipeline interface
			type Indexer interface {
				IndexTexts([]string) error
			}

			indexer, ok := pipeline.(Indexer)
			if !ok {
				errors <- fmt.Errorf("pipeline does not implement IndexTexts")
				return
			}

			// Index the chunk
			chunkStart := time.Now()
			if err := indexer.IndexTexts(chunkTexts); err != nil {
				errors <- fmt.Errorf("chunk %d failed: %w", chunkNum, err)
				return
			}

			chunkTime := time.Since(chunkStart)
			chunkRate := float64(len(chunkTexts)) / chunkTime.Seconds()

			// Report chunk completion (optional detailed logging)
			if len(chunks) <= 20 { // Only log details for smaller jobs
				log.Printf("✅ Chunk %d: %d texts in %v (%.0f texts/sec)",
					chunkNum+1, len(chunkTexts), chunkTime, chunkRate)
			}

			progress <- 1
		}(i, chunk)
	}

	// Wait for all chunks to complete
	wg.Wait()
	close(errors)
	close(progress)

	// Check for errors
	var firstError error
	errorCount := 0
	for err := range errors {
		if firstError == nil {
			firstError = err
		}
		errorCount++
	}

	if firstError != nil {
		return fmt.Errorf("indexing failed (%d/%d chunks failed): %w", errorCount, len(chunks), firstError)
	}

	// Success metrics
	totalTime := time.Since(start)
	totalThroughput := float64(len(texts)) / totalTime.Seconds()

	log.Printf("✅ Parallel indexing complete!")
	log.Printf("   Total texts: %d", len(texts))
	log.Printf("   Total time: %v", totalTime)
	log.Printf("   Throughput: %.0f texts/sec", totalThroughput)
	log.Printf("   Chunks: %d", len(chunks))
	log.Printf("   Concurrency: %d", maxConcurrent)

	// Performance analysis
	if totalThroughput > 3000 {
		log.Printf("🚀 Excellent performance! GPU well utilized.")
	} else if totalThroughput > 1500 {
		log.Printf("✅ Good performance. Consider larger batches for even better GPU utilization.")
	} else {
		log.Printf("⚠️  Performance below expectations. Check GPU utilization and batch sizes.")
	}

	return nil
}

// OptimizedConfig provides optimized settings for maximum throughput
type OptimizedConfig_disabled struct { // Renamed to avoid conflict
	BatchSize      int  `json:"batch_size"`     // GPU batch size
	ChunkSize      int  `json:"chunk_size"`     // Parallel chunk size
	MaxConcurrent  int  `json:"max_concurrent"` // Parallel workers
	UseGPUIndexing bool `json:"use_gpu_indexing"`
	PreloadGPU     bool `json:"preload_gpu"`
	MaxVectors     int  `json:"max_vectors"`
	GPUOnlyMode    bool `json:"gpu_only_mode"`
}

// GetOptimizedConfig returns configuration optimized for maximum GPU throughput
func GetOptimizedConfig(gpuMemoryGB float64) OptimizedConfig {
	// Scale settings based on GPU memory
	var batchSize, chunkSize, maxConcurrent int

	if gpuMemoryGB >= 16 {
		// High-end GPU (RTX 3080, 4080, etc.)
		batchSize = 4096
		chunkSize = 8192
		maxConcurrent = 8
	} else if gpuMemoryGB >= 8 {
		// Mid-range GPU
		batchSize = 2048
		chunkSize = 4096
		maxConcurrent = 6
	} else {
		// Lower-end GPU
		batchSize = 1024
		chunkSize = 2048
		maxConcurrent = 4
	}

	return OptimizedConfig{
		BatchSize:      batchSize,
		ChunkSize:      chunkSize,
		MaxConcurrent:  maxConcurrent,
		UseGPUIndexing: true,
		PreloadGPU:     true,
		MaxVectors:     1000000,
		GPUOnlyMode:    true,
	}
}

// Example usage for your main.go:
/*
func main() {
	// ... existing flag parsing ...

	// Get optimized configuration
	optConfig := GetOptimizedConfig(16.0) // Your GPU memory in GB

	config := gpu.Config{
		ModelPath:      *modelPath,
		GPUServerURL:   *gpuServer,
		BatchSize:      optConfig.BatchSize,     // Much larger!
		UseGPUIndexing: optConfig.UseGPUIndexing,
		PreloadGPU:     optConfig.PreloadGPU,
		MaxVectors:     optConfig.MaxVectors,
		GPUOnlyMode:    optConfig.GPUOnlyMode,
	}

	pipeline, err := gpu.NewPipeline(config)
	if err != nil {
		log.Fatalf("Failed to create GPU pipeline: %v", err)
	}

	log.Printf("🚀 Optimized GPU Pipeline initialized")
	log.Printf("   Batch size: %d", optConfig.BatchSize)
	log.Printf("   Chunk size: %d", optConfig.ChunkSize)
	log.Printf("   Max concurrent: %d", optConfig.MaxConcurrent)

	// Load texts
	texts, err := loadTexts(*dataFile)
	if err != nil {
		// ... handle error ...
	}

	// OPTIMIZED INDEXING - Replace your IndexTexts call with this:
	log.Println("🚀 Starting optimized indexing...")
	start := time.Now()

	if err := IndexTextsParallel(pipeline, texts, optConfig.ChunkSize); err != nil {
		log.Fatalf("Failed to index texts: %v", err)
	}

	indexTime := time.Since(start)
	throughput := float64(len(texts)) / indexTime.Seconds()

	log.Printf("✅ Indexing complete: %.0f texts/sec (%.1fx improvement)",
		throughput, throughput/700) // Compare to current ~700 texts/sec

	// ... rest of your code ...
}
*/
