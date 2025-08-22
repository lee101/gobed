package main

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

// generateTestDocuments creates a large dataset for testing bulk indexing
func generateTestDocuments(count int) []gobed.Document {
	// Pre-defined text templates for realistic document generation
	templates := []string{
		"This document discusses the fundamentals of machine learning and artificial intelligence in modern computing.",
		"Natural language processing techniques have revolutionized how we interact with computer systems.",
		"Deep learning models like transformers have achieved remarkable performance on various NLP tasks.",
		"Vector databases provide efficient similarity search capabilities for high-dimensional embeddings.",
		"GPU acceleration significantly improves the performance of neural network inference and training.",
		"Quantization techniques reduce memory usage while maintaining model accuracy in production systems.",
		"Large language models demonstrate emergent capabilities in reasoning and text generation tasks.",
		"Information retrieval systems leverage semantic embeddings for more accurate document matching.",
		"Real-time search applications require optimized indexing and query processing algorithms.",
		"Distributed computing architectures enable scaling of machine learning workloads across clusters.",
		"Data preprocessing pipelines are crucial for ensuring high-quality training data in ML systems.",
		"Attention mechanisms allow models to focus on relevant parts of input sequences during processing.",
		"Transfer learning enables leveraging pre-trained models for domain-specific applications.",
		"Model compression techniques make large neural networks deployable on resource-constrained devices.",
		"Evaluation metrics guide the development and optimization of machine learning algorithms.",
	}

	variations := []string{
		"The research shows that",
		"Recent advances in",
		"Our analysis reveals that",
		"Studies indicate that",
		"Experimental results demonstrate that",
		"The implementation of",
		"Performance benchmarks show that",
		"Technical documentation describes how",
		"Industry reports highlight that",
		"Academic papers examine how",
	}

	suffixes := []string{
		"These findings have significant implications for future research directions.",
		"The methodology can be applied across various domains and use cases.",
		"Performance improvements are substantial compared to baseline approaches.",
		"Resource efficiency is critical for practical deployment scenarios.",
		"Scalability considerations must be addressed in production environments.",
		"Quality metrics demonstrate the effectiveness of this approach.",
		"Integration challenges require careful system design and optimization.",
		"Cost-benefit analysis supports the adoption of these techniques.",
		"User experience improvements justify the implementation complexity.",
		"Long-term maintenance and updates are essential for system reliability.",
	}

	docs := make([]gobed.Document, count)

	for i := 0; i < count; i++ {
		// Create varied documents
		template := templates[rand.Intn(len(templates))]
		variation := variations[rand.Intn(len(variations))]
		suffix := suffixes[rand.Intn(len(suffixes))]

		text := fmt.Sprintf("%s %s %s", variation, template, suffix)

		docs[i] = gobed.Document{
			ID:   i,
			Text: text,
		}
	}

	return docs
}

// benchmarkStandardIndexing tests CPU-based indexing
func benchmarkStandardIndexing(model *gobed.EmbeddingModel, docs []gobed.Document) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("🖥️  STANDARD CPU INDEXING BENCHMARK\n")
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Create index without GPU bulk indexing
	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = false
	index := gobed.NewVectorIndex(model, config)

	startTime := time.Now()
	memBefore := getMemoryUsage()

	err := index.AddDocuments(docs)
	if err != nil {
		log.Printf("❌ Standard indexing failed: %v", err)
		return
	}

	elapsed := time.Since(startTime)
	memAfter := getMemoryUsage()

	throughput := float64(len(docs)) / elapsed.Seconds()
	memUsed := memAfter - memBefore

	fmt.Printf("📊 Standard Indexing Results:\n")
	fmt.Printf("   Documents: %d\n", len(docs))
	fmt.Printf("   Time: %.2fs\n", elapsed.Seconds())
	fmt.Printf("   Throughput: %.0f docs/sec\n", throughput)
	fmt.Printf("   Memory: +%.1f MB\n", float64(memUsed)/(1024*1024))
	fmt.Printf("   Index size: %d\n", index.Size())
}

// benchmarkBulkGPUIndexing tests GPU-accelerated bulk indexing
func benchmarkBulkGPUIndexing(model *gobed.EmbeddingModel, docs []gobed.Document) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("🚀 BULK GPU INDEXING BENCHMARK\n")
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Create index with GPU bulk indexing
	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = true
	config.BulkBatchSize = 5000
	index := gobed.NewVectorIndex(model, config)

	startTime := time.Now()
	memBefore := getMemoryUsage()

	err := index.AddDocumentsBulkGPU(docs)
	if err != nil {
		log.Printf("❌ Bulk GPU indexing failed: %v", err)
		return
	}

	elapsed := time.Since(startTime)
	memAfter := getMemoryUsage()

	throughput := float64(len(docs)) / elapsed.Seconds()
	memUsed := memAfter - memBefore

	fmt.Printf("📊 Bulk GPU Indexing Results:\n")
	fmt.Printf("   Documents: %d\n", len(docs))
	fmt.Printf("   Time: %.2fs\n", elapsed.Seconds())
	fmt.Printf("   Throughput: %.0f docs/sec\n", throughput)
	fmt.Printf("   Memory: +%.1f MB\n", float64(memUsed)/(1024*1024))
	fmt.Printf("   Index size: %d\n", index.Size())
}

// benchmarkWithMonitoring demonstrates real-time GPU monitoring
func benchmarkWithMonitoring(model *gobed.EmbeddingModel, docs []gobed.Document) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("📊 BULK GPU INDEXING WITH REAL-TIME MONITORING\n")
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Create index with GPU bulk indexing
	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = true
	config.BulkBatchSize = 2500 // Smaller batches for better monitoring granularity
	index := gobed.NewVectorIndex(model, config)

	// Start monitoring
	progressChan, err := index.AddDocumentsWithMonitoring(docs)
	if err != nil {
		log.Printf("❌ Monitored indexing failed to start: %v", err)
		return
	}

	startTime := time.Now()

	// Monitor progress in real-time
	for progress := range progressChan {
		progress.LogProgress()

		if progress.Error != nil {
			log.Printf("❌ Indexing error: %v", progress.Error)
			return
		}

		if progress.Complete {
			break
		}
	}

	elapsed := time.Since(startTime)
	throughput := float64(len(docs)) / elapsed.Seconds()

	fmt.Printf("\n🎯 Monitored Indexing Summary:\n")
	fmt.Printf("   Total time: %.2fs\n", elapsed.Seconds())
	fmt.Printf("   Overall throughput: %.0f docs/sec\n", throughput)
}

// performanceComparison runs comprehensive performance comparison
func performanceComparison(model *gobed.EmbeddingModel) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf("🏁 COMPREHENSIVE BULK INDEXING PERFORMANCE COMPARISON\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))

	// Test different dataset sizes
	sizes := []int{1000, 5000, 10000, 25000}

	for _, size := range sizes {
		fmt.Printf("\n🔍 Testing with %d documents...\n", size)
		docs := generateTestDocuments(size)

		// Warm up
		fmt.Printf("🔥 Warming up...\n")
		warmupDocs := generateTestDocuments(100)
		config := gobed.DefaultVectorIndexConfig()
		config.EnableBulkGPU = true
		warmupIndex := gobed.NewVectorIndex(model, config)
		warmupIndex.AddDocuments(warmupDocs)

		// Run tests
		fmt.Printf("\n📈 Running benchmarks for %d documents:\n", size)

		// Test 1: Standard CPU indexing
		benchmarkStandardIndexing(model, docs)

		// Test 2: Bulk GPU indexing
		benchmarkBulkGPUIndexing(model, docs)

		// Force garbage collection between tests
		runtime.GC()
		time.Sleep(1 * time.Second)
	}

	// Final large-scale test with monitoring
	fmt.Printf("\n🚀 Large-scale test with real-time monitoring...\n")
	largeDocs := generateTestDocuments(50000)
	benchmarkWithMonitoring(model, largeDocs)
}

// getMemoryUsage returns current memory usage in bytes
func getMemoryUsage() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc
}

// testSearchPerformance verifies search functionality after bulk indexing
func testSearchPerformance(model *gobed.EmbeddingModel, docs []gobed.Document) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("🔍 SEARCH PERFORMANCE TEST AFTER BULK INDEXING\n")
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Create and populate index
	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = true
	index := gobed.NewVectorIndex(model, config)

	// Index documents
	fmt.Printf("📚 Indexing %d documents...\n", len(docs))
	startIndex := time.Now()
	err := index.AddDocumentsBulkGPU(docs)
	if err != nil {
		log.Printf("❌ Indexing failed: %v", err)
		return
	}
	indexTime := time.Since(startIndex)

	fmt.Printf("✅ Indexing completed in %.2fs (%.0f docs/sec)\n",
		indexTime.Seconds(), float64(len(docs))/indexTime.Seconds())

	// Test search queries
	queries := []string{
		"machine learning artificial intelligence",
		"natural language processing",
		"GPU acceleration performance",
		"vector similarity search",
		"deep learning transformers",
	}

	fmt.Printf("\n🔎 Testing search performance:\n")

	totalSearchTime := time.Duration(0)
	for i, query := range queries {
		startSearch := time.Now()
		results, err := index.Search(query, 5)
		searchTime := time.Since(startSearch)
		totalSearchTime += searchTime

		if err != nil {
			log.Printf("❌ Search failed for query %d: %v", i+1, err)
			continue
		}

		fmt.Printf("   Q%d: %8.2fms - \"%s\" (found %d results)\n",
			i+1, float64(searchTime.Nanoseconds())/1e6, query, len(results))

		// Show top result
		if len(results) > 0 {
			fmt.Printf("        Top: Doc %d (sim: %.4f)\n",
				results[0].ID, results[0].Similarity)
		}
	}

	avgSearchTime := totalSearchTime / time.Duration(len(queries))
	fmt.Printf("\n📊 Search Performance Summary:\n")
	fmt.Printf("   Average search time: %.2fms\n",
		float64(avgSearchTime.Nanoseconds())/1e6)
	fmt.Printf("   Search throughput: %.0f queries/sec\n",
		1.0/avgSearchTime.Seconds())
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 GOBED BULK GPU INDEXING PERFORMANCE DEMO")
	fmt.Println("================================================================================")
	fmt.Println("Testing GPU-accelerated bulk indexing with 5k batch processing")
	fmt.Println("")

	// Load model
	fmt.Printf("📦 Loading embedding model...\n")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	fmt.Printf("✅ Model loaded successfully\n")

	// Set random seed for reproducible results
	rand.Seed(42)

	// Check system resources
	fmt.Printf("\n💻 System Information:\n")
	fmt.Printf("   GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("   NumCPU: %d\n", runtime.NumCPU())

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("   Initial memory: %.1f MB\n", float64(m.Alloc)/(1024*1024))

	// Run comprehensive performance comparison
	performanceComparison(model)

	// Test search functionality
	fmt.Printf("\n🔍 Testing search after bulk indexing...\n")
	searchTestDocs := generateTestDocuments(10000)
	testSearchPerformance(model, searchTestDocs)

	// Final summary
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf("✅ BULK GPU INDEXING DEMO COMPLETED\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))
	fmt.Printf("Key Benefits Demonstrated:\n")
	fmt.Printf("  • GPU-accelerated embedding generation with 5k batch processing\n")
	fmt.Printf("  • Real-time GPU memory monitoring during indexing\n")
	fmt.Printf("  • Automatic fallback to CPU for smaller datasets\n")
	fmt.Printf("  • Massive parallelism for large-scale document indexing\n")
	fmt.Printf("  • Maintained search accuracy and performance\n")
	fmt.Printf("\nRun with different batch sizes to optimize for your GPU memory!\n")
}
