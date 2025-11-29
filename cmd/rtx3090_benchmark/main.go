//go:build legacy

package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println(" RTX 3090 ULTIMATE GPU INDEXING BENCHMARK")
	fmt.Println("================================================================================")
	fmt.Printf("GPU: NVIDIA RTX 3090 (24GB VRAM)\n")
	fmt.Printf("CPU: %d cores\n", runtime.NumCPU())
	fmt.Printf("Go Version: %s\n\n", runtime.Version())

	// Load model
	fmt.Println("Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(" Model loaded successfully\n")

	// Test configurations optimized for RTX 3090
	configs := []struct {
		name      string
		batchSize int
		docCount  int
		workers   int
	}{
		// Baseline tests
		{"Baseline_Small", 100, 1000, 4},
		{"Baseline_Medium", 500, 5000, 8},
		{"Baseline_Large", 1000, 10000, 16},

		// Aggressive batch sizes for 24GB VRAM
		{"RTX3090_Optimized_5K", 5000, 50000, 24},
		{"RTX3090_Optimized_10K", 10000, 100000, 32},
		{"RTX3090_Optimized_20K", 20000, 200000, 32},
		{"RTX3090_Extreme_50K", 50000, 500000, 48},
		
		// Memory stress test
		{"RTX3090_MaxMemory_100K", 100000, 1000000, 64},
	}

	bestThroughput := 0.0
	bestConfig := ""

	for _, cfg := range configs {
		fmt.Printf("\n============================================================\n")
		fmt.Printf("Testing: %s\n", cfg.name)
		fmt.Printf("  Batch Size: %d | Documents: %d | Workers: %d\n", 
			cfg.batchSize, cfg.docCount, cfg.workers)
		fmt.Println("------------------------------------------------------------")

		// Generate test documents
		docs := generateMassiveCorpus(cfg.docCount)
		
		// Create search engine
		engine := gobed.NewSearchEngine(model)

		// Warmup
		if cfg.docCount > 100 {
			warmupDocs := docs[:min(100, len(docs))]
			engine.IndexBatch(warmupDocs)
		}

		// Measure indexing performance
		start := time.Now()
		totalIndexed := 0
		errors := 0

		// Process in parallel batches
		batchCount := (cfg.docCount + cfg.batchSize - 1) / cfg.batchSize
		var wg sync.WaitGroup
		var indexedCount int64
		var errorCount int64

		for i := 0; i < batchCount; i++ {
			wg.Add(1)
			go func(batchIdx int) {
				defer wg.Done()
				
				startIdx := batchIdx * cfg.batchSize
				endIdx := min(startIdx+cfg.batchSize, cfg.docCount)
				
				if startIdx >= cfg.docCount {
					return
				}
				
				batch := docs[startIdx:endIdx]
				_, err := engine.IndexBatch(batch)
				if err != nil {
					atomic.AddInt64(&errorCount, 1)
					fmt.Printf("    Batch %d error: %v\n", batchIdx, err)
				} else {
					atomic.AddInt64(&indexedCount, int64(len(batch)))
				}
			}(i)
		}

		wg.Wait()
		elapsed := time.Since(start)
		totalIndexed = int(indexedCount)
		errors = int(errorCount)

		// Calculate metrics
		docsPerSec := float64(totalIndexed) / elapsed.Seconds()
		latencyMs := elapsed.Seconds() * 1000 / float64(batchCount)
		
		// Get engine stats
		stats := engine.Stats()

		// Display results
		fmt.Printf("\n Results:\n")
		fmt.Printf("  ✓ Indexed: %d documents\n", totalIndexed)
		fmt.Printf("  ✓ Time: %v\n", elapsed)
		fmt.Printf("  ✓ Throughput: %.0f docs/sec\n", docsPerSec)
		fmt.Printf("  ✓ Batch Latency: %.2f ms\n", latencyMs)
		fmt.Printf("  ✓ Memory Usage: %.2f MB\n", stats.MemoryUsageMB)
		fmt.Printf("  ✓ Index Type: %s\n", stats.IndexType)
		if errors > 0 {
			fmt.Printf("    Errors: %d\n", errors)
		}

		if docsPerSec > bestThroughput {
			bestThroughput = docsPerSec
			bestConfig = cfg.name
		}

		// Run search benchmark
		fmt.Printf("\n Search Performance:\n")
		queries := []string{
			"machine learning algorithms",
			"deep neural networks",
			"transformer architecture",
			"gpu acceleration techniques",
			"vector similarity search",
		}

		searchStart := time.Now()
		searchCount := 100
		searchErrors := 0
		for i := 0; i < searchCount; i++ {
			query := queries[i%len(queries)]
			_, err := engine.Search(query, 10)
			if err != nil {
				searchErrors++
			}
		}
		searchElapsed := time.Since(searchStart)
		searchQPS := float64(searchCount) / searchElapsed.Seconds()
		avgSearchLatency := searchElapsed / time.Duration(searchCount)

		fmt.Printf("  ✓ Search QPS: %.0f\n", searchQPS)
		fmt.Printf("  ✓ Avg Latency: %v\n", avgSearchLatency)
		if searchErrors > 0 {
			fmt.Printf("    Search Errors: %d\n", searchErrors)
		}
	}

	// Final summary
	fmt.Printf("\n================================================================================\n")
	fmt.Printf(" BEST CONFIGURATION: %s\n", bestConfig)
	fmt.Printf(" BEST THROUGHPUT: %.0f docs/sec\n", bestThroughput)
	fmt.Printf("================================================================================\n")
}

func generateMassiveCorpus(size int) []string {
	templates := []string{
		"Advanced %s techniques for modern applications",
		"Implementing %s at scale with cloud infrastructure",
		"Optimizing %s performance using GPU acceleration",
		"Building robust %s systems with high availability",
		"Real-time %s processing for enterprise solutions",
		"Distributed %s architecture patterns and best practices",
		"Security considerations for %s deployment",
		"Cost optimization strategies for %s workloads",
		"Monitoring and debugging %s in production",
		"Future trends in %s technology and innovation",
	}

	topics := []string{
		"machine learning", "deep learning", "neural networks",
		"computer vision", "natural language processing", "reinforcement learning",
		"data pipelines", "stream processing", "batch processing",
		"microservices", "serverless computing", "container orchestration",
		"blockchain", "quantum computing", "edge computing",
		"cybersecurity", "cryptography", "zero trust architecture",
		"data lakes", "data warehouses", "data mesh",
		"DevOps", "GitOps", "MLOps", "DataOps", "AIOps",
	}

	corpus := make([]string, size)
	for i := 0; i < size; i++ {
		template := templates[i%len(templates)]
		topic := topics[(i/len(templates))%len(topics)]
		// Add variation to prevent caching
		corpus[i] = fmt.Sprintf("%s [ID:%d Time:%d]", 
			fmt.Sprintf(template, topic), i, time.Now().UnixNano())
	}

	return corpus
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
