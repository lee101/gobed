package main

import (
	"fmt"
	"log"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🚀 GPU-Optimized Batch Embedding Benchmark")
	fmt.Println("==========================================")
	
	// Load model
	fmt.Print("Loading model... ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("✅ Done (%v)\n\n", time.Since(start))
	
	// Generate test data
	texts := generateTestTexts(10000)
	fmt.Printf("Generated %d test texts\n\n", len(texts))
	
	// Test different batch configurations
	batchConfigs := []struct {
		name       string
		batchSize  int
		numBatches int
		workers    int
	}{
		{"Small Batch", 32, 10, 4},
		{"Medium Batch", 64, 20, 8},
		{"Large Batch", 128, 25, 8},
		{"Mega Batch", 256, 10, 12},
		{"GPU-Style Batch", 512, 5, 16},
	}
	
	fmt.Printf("%-15s %-10s %-10s %-12s %-15s %-10s\n", 
		"Config", "Batch", "Workers", "Total", "Items/sec", "ms/item")
	fmt.Println(strings.Repeat("-", 75))
	
	for _, config := range batchConfigs {
		totalItems := config.batchSize * config.numBatches
		if totalItems > len(texts) {
			totalItems = len(texts)
		}
		
		result := benchmarkBatchProcessing(model, texts[:totalItems], config.batchSize, config.workers)
		
		fmt.Printf("%-15s %-10d %-10d %-12d %-15.0f %-10.3f\n",
			config.name,
			config.batchSize,
			config.workers,
			totalItems,
			result.itemsPerSec,
			result.msPerItem)
	}
	
	// Test GPU-style parallel batching
	fmt.Printf("\n🔥 GPU-Style Parallel Batch Processing\n")
	fmt.Println(strings.Repeat("=", 50))
	
	gpuResults := testGPUStyleBatching(model, texts[:5000])
	for size, result := range gpuResults {
		fmt.Printf("Batch size %d: %.0f items/sec (%.3f ms/item)\n",
			size, result.itemsPerSec, result.msPerItem)
	}
	
	// Estimate large scale performance
	fmt.Printf("\n📊 Large Scale Performance Estimates\n")
	fmt.Println(strings.Repeat("=", 50))
	
	bestPerf := float64(0)
	for _, result := range gpuResults {
		if result.itemsPerSec > bestPerf {
			bestPerf = result.itemsPerSec
		}
	}
	
	scales := []int{50000, 500000, 5000000}
	for _, scale := range scales {
		timeSeconds := float64(scale) / bestPerf
		fmt.Printf("%7d documents: ~%.1f seconds (~%.1f minutes)\n", 
			scale, timeSeconds, timeSeconds/60)
	}
}

type BenchResult struct {
	itemsPerSec float64
	msPerItem   float64
	duration    time.Duration
}

func benchmarkBatchProcessing(model *gobed.EmbeddingModel, texts []string, batchSize, workers int) BenchResult {
	runtime.GC()
	start := time.Now()
	
	// Create work batches
	batches := make([][]string, 0)
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batches = append(batches, texts[i:end])
	}
	
	// Process batches in parallel
	batchChan := make(chan []string, len(batches))
	var wg sync.WaitGroup
	
	// Start workers
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range batchChan {
				processBatch(model, batch)
			}
		}()
	}
	
	// Send work
	for _, batch := range batches {
		batchChan <- batch
	}
	close(batchChan)
	wg.Wait()
	
	duration := time.Since(start)
	itemsPerSec := float64(len(texts)) / duration.Seconds()
	msPerItem := float64(duration.Nanoseconds()) / float64(len(texts)) / 1e6
	
	return BenchResult{
		itemsPerSec: itemsPerSec,
		msPerItem:   msPerItem,
		duration:    duration,
	}
}

func processBatch(model *gobed.EmbeddingModel, batch []string) {
	// Process each text in the batch
	for _, text := range batch {
		_, err := model.Encode(text)
		if err != nil {
			// Skip errors for benchmark
			continue
		}
	}
}

func testGPUStyleBatching(model *gobed.EmbeddingModel, texts []string) map[int]BenchResult {
	batchSizes := []int{64, 128, 256, 512, 1024}
	results := make(map[int]BenchResult)
	
	for _, batchSize := range batchSizes {
		// Use optimal workers for each batch size
		workers := calculateOptimalWorkers(batchSize)
		
		// Test with subset to keep benchmark reasonable
		testSize := min(len(texts), batchSize*10)
		result := benchmarkBatchProcessing(model, texts[:testSize], batchSize, workers)
		results[batchSize] = result
	}
	
	return results
}

func calculateOptimalWorkers(batchSize int) int {
	numCPU := runtime.NumCPU()
	
	// For smaller batches, use more workers
	if batchSize <= 64 {
		return min(numCPU, 16)
	} else if batchSize <= 256 {
		return min(numCPU, 12)
	} else {
		return min(numCPU, 8)
	}
}

func generateTestTexts(count int) []string {
	templates := []string{
		"Analyzing performance metrics for %s optimization in distributed systems.",
		"Research findings on %s implementation strategies and best practices.",
		"Deep dive into %s architecture patterns for scalable applications.",
		"Comprehensive guide to %s deployment in cloud environments.",
		"Performance benchmarking of %s algorithms across different platforms.",
		"Security considerations for %s integration in enterprise systems.",
		"Comparative analysis of %s frameworks and their efficiency metrics.",
		"Machine learning applications in %s for improved system performance.",
		"Real-time processing capabilities of %s in high-throughput scenarios.",
		"Cost optimization strategies for %s infrastructure at scale.",
	}
	
	topics := []string{
		"neural networks", "distributed computing", "container orchestration",
		"microservices", "data processing", "machine learning pipelines",
		"real-time analytics", "cloud infrastructure", "edge computing",
		"artificial intelligence", "big data", "stream processing",
		"graph databases", "time series analysis", "natural language processing",
		"computer vision", "recommendation systems", "fraud detection",
		"predictive analytics", "deep learning", "reinforcement learning",
		"federated learning", "transfer learning", "model optimization",
	}
	
	texts := make([]string, count)
	for i := 0; i < count; i++ {
		template := templates[i%len(templates)]
		topic := topics[i%len(topics)]
		texts[i] = fmt.Sprintf(template, topic)
	}
	
	return texts
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

