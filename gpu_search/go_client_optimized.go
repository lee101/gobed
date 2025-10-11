//go:build legacy

// go_client_optimized.go - High-performance Go client for GPU server
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
)

// OptimizedConfig for maximum throughput
type OptimizedConfig struct {
	ServerURL       string
	MaxBatchSize    int // Much larger batches
	MaxConcurrent   int // Parallel requests
	RequestTimeout  time.Duration
	PreallocBuffers bool
}

// EmbedRequest with optimizations
type EmbedRequest struct {
	Texts     []string `json:"texts"`
	RequestID string   `json:"request_id"`
}

// EmbedResponse with statistics
type EmbedResponse struct {
	Embeddings  [][]float64 `json:"embeddings"`
	Stats       Stats       `json:"stats"`
	ServerStats ServerStats `json:"server_stats"`
}

type Stats struct {
	Texts       int     `json:"texts"`
	TimeSeconds float64 `json:"time_seconds"`
	Throughput  float64 `json:"throughput"`
	Chunks      int     `json:"chunks"`
}

type ServerStats struct {
	TotalEmbedded  int     `json:"total_embedded"`
	TotalIndexed   int     `json:"total_indexed"`
	AvgBatchSize   int     `json:"avg_batch_size"`
	PeakThroughput float64 `json:"peak_throughput"`
}

// OptimizedGPUClient for maximum performance
type OptimizedGPUClient struct {
	config     OptimizedConfig
	httpClient *http.Client
	bufferPool sync.Pool
}

// NewOptimizedGPUClient creates a high-performance client
func NewOptimizedGPUClient(config OptimizedConfig) *OptimizedGPUClient {
	client := &OptimizedGPUClient{
		config: config,
		httpClient: &http.Client{
			Timeout: config.RequestTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}

	// Pre-allocate buffers if enabled
	if config.PreallocBuffers {
		client.bufferPool = sync.Pool{
			New: func() interface{} {
				return bytes.NewBuffer(make([]byte, 0, 1024*1024)) // 1MB buffers
			},
		}
	}

	return client
}

// EmbedTextsBatch processes texts in optimal batches
func (c *OptimizedGPUClient) EmbedTextsBatch(texts []string) (*EmbedResponse, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	// Create optimized request
	requestID := fmt.Sprintf("go_batch_%d", time.Now().UnixNano())
	request := EmbedRequest{
		Texts:     texts,
		RequestID: requestID,
	}

	// Serialize request
	var reqBuffer *bytes.Buffer
	if c.config.PreallocBuffers {
		reqBuffer = c.bufferPool.Get().(*bytes.Buffer)
		defer c.bufferPool.Put(reqBuffer)
		reqBuffer.Reset()
	} else {
		reqBuffer = &bytes.Buffer{}
	}

	if err := json.NewEncoder(reqBuffer).Encode(request); err != nil {
		return nil, fmt.Errorf("failed to encode request: %w", err)
	}

	// Send request
	resp, err := c.httpClient.Post(
		c.config.ServerURL+"/embed_optimized",
		"application/json",
		reqBuffer,
	)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server error %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var response EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// IndexTextsParallel indexes texts with maximum parallelism
func (c *OptimizedGPUClient) IndexTextsParallel(texts []string) error {
	if len(texts) == 0 {
		return nil
	}

	log.Printf(" Starting parallel indexing of %d texts", len(texts))
	start := time.Now()

	// Create large batches for optimal GPU utilization
	batchSize := c.config.MaxBatchSize
	if batchSize > len(texts) {
		batchSize = len(texts)
	}

	batches := make([][]string, 0)
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batches = append(batches, texts[i:end])
	}

	log.Printf(" Created %d batches (avg size: %d)", len(batches), len(texts)/len(batches))

	// Process batches in parallel
	var wg sync.WaitGroup
	errors := make(chan error, len(batches))
	results := make(chan Stats, len(batches))

	// Limit concurrent requests
	semaphore := make(chan struct{}, c.config.MaxConcurrent)

	for i, batch := range batches {
		wg.Add(1)
		go func(batchNum int, batchTexts []string) {
			defer wg.Done()

			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// Send index request
			if err := c.indexBatch(batchTexts, batchNum); err != nil {
				errors <- fmt.Errorf("batch %d failed: %w", batchNum, err)
				return
			}

			// Report progress
			if batchNum%10 == 0 || batchNum == len(batches)-1 {
				progress := float64(batchNum+1) / float64(len(batches)) * 100
				log.Printf(" Progress: %.1f%% (%d/%d batches)", progress, batchNum+1, len(batches))
			}

		}(i, batch)
	}

	// Wait for completion
	wg.Wait()
	close(errors)
	close(results)

	// Check for errors
	if len(errors) > 0 {
		return <-errors
	}

	totalTime := time.Since(start)
	throughput := float64(len(texts)) / totalTime.Seconds()

	log.Printf(" Parallel indexing complete!")
	log.Printf("   Total time: %v", totalTime)
	log.Printf("   Throughput: %.0f texts/sec", throughput)
	log.Printf("   Batches: %d", len(batches))

	return nil
}

// indexBatch sends a single batch for indexing
func (c *OptimizedGPUClient) indexBatch(texts []string, batchNum int) error {
	request := map[string]interface{}{
		"texts": texts,
	}

	var reqBuffer *bytes.Buffer
	if c.config.PreallocBuffers {
		reqBuffer = c.bufferPool.Get().(*bytes.Buffer)
		defer c.bufferPool.Put(reqBuffer)
		reqBuffer.Reset()
	} else {
		reqBuffer = &bytes.Buffer{}
	}

	if err := json.NewEncoder(reqBuffer).Encode(request); err != nil {
		return fmt.Errorf("encode failed: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.config.ServerURL+"/index_streaming",
		"application/json",
		reqBuffer,
	)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server error %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// GetServerStats retrieves performance statistics
func (c *OptimizedGPUClient) GetServerStats() (*ServerStats, error) {
	resp, err := c.httpClient.Get(c.config.ServerURL + "/stats")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var stats struct {
		ServerStats ServerStats `json:"server_stats"`
		GPUMemory   float64     `json:"gpu_memory"`
		IndexSize   int         `json:"index_size"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return nil, err
	}

	log.Printf(" Server Stats:")
	log.Printf("   Total embedded: %d", stats.ServerStats.TotalEmbedded)
	log.Printf("   Total indexed: %d", stats.ServerStats.TotalIndexed)
	log.Printf("   Peak throughput: %.0f texts/sec", stats.ServerStats.PeakThroughput)
	log.Printf("   GPU memory: %.1f GB", stats.GPUMemory)
	log.Printf("   Index size: %d", stats.IndexSize)

	return &stats.ServerStats, nil
}

// Benchmark tests the optimized client
func (c *OptimizedGPUClient) Benchmark(numTexts int) {
	log.Printf("🧪 Benchmarking with %d texts", numTexts)

	// Generate test data
	texts := make([]string, numTexts)
	for i := 0; i < numTexts; i++ {
		texts[i] = fmt.Sprintf("Sample text %d with some content for embedding", i)
	}

	// Test large batch embedding
	log.Printf(" Testing large batch embedding...")
	start := time.Now()

	response, err := c.EmbedTextsBatch(texts)
	if err != nil {
		log.Fatalf("Embedding failed: %v", err)
	}

	embedTime := time.Since(start)
	throughput := float64(numTexts) / embedTime.Seconds()

	log.Printf(" Embedding Results:")
	log.Printf("   Texts: %d", numTexts)
	log.Printf("   Time: %v", embedTime)
	log.Printf("   Throughput: %.0f texts/sec", throughput)
	log.Printf("   Server throughput: %.0f texts/sec", response.Stats.Throughput)
	log.Printf("   Chunks processed: %d", response.Stats.Chunks)

	// Test parallel indexing
	log.Printf("\n📚 Testing parallel indexing...")
	if err := c.IndexTextsParallel(texts); err != nil {
		log.Fatalf("Indexing failed: %v", err)
	}

	// Get final stats
	log.Printf("\n Final server statistics:")
	c.GetServerStats()
}

func main() {
	// Optimized configuration for maximum throughput
	config := OptimizedConfig{
		ServerURL:       "http://localhost:5000",
		MaxBatchSize:    4096, // 16x larger than current 256
		MaxConcurrent:   8,    // Parallel requests
		RequestTimeout:  60 * time.Second,
		PreallocBuffers: true, // Memory optimization
	}

	client := NewOptimizedGPUClient(config)

	log.Printf(" Optimized GPU Client initialized")
	log.Printf("   Server: %s", config.ServerURL)
	log.Printf("   Max batch size: %d", config.MaxBatchSize)
	log.Printf("   Max concurrent: %d", config.MaxConcurrent)
	log.Printf("   Timeout: %v", config.RequestTimeout)

	// Run benchmark
	client.Benchmark(10000) // Test with 10K texts
}
