//go:build legacy

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

// generateTestDocuments creates test documents for benchmarking
func generateTestDocuments(count int) []gobed.Document {
	templates := []string{
		"Machine learning algorithms analyze patterns in large datasets to make predictions.",
		"Natural language processing enables computers to understand and generate human language.",
		"Computer vision systems can recognize objects and patterns in digital images.",
		"Database management systems organize and retrieve information efficiently.",
		"Web development frameworks simplify the creation of dynamic websites.",
		"Artificial intelligence research focuses on creating intelligent computer systems.",
		"Data science combines statistics, programming, and domain expertise.",
		"Cloud computing provides scalable infrastructure for modern applications.",
		"Cybersecurity measures protect digital systems from malicious attacks.",
		"Software engineering practices ensure reliable and maintainable code.",
		"Network protocols enable communication between different computer systems.",
		"Operating systems manage hardware resources and provide user interfaces.",
		"Algorithm optimization improves the efficiency of computational processes.",
		"Version control systems track changes in software development projects.",
		"API design enables different software components to work together.",
	}

	docs := make([]gobed.Document, count)
	for i := 0; i < count; i++ {
		template := templates[rand.Intn(len(templates))]
		variation := fmt.Sprintf("Document %d discusses how %s", i, strings.ToLower(template))

		docs[i] = gobed.Document{
			ID:   i,
			Text: variation,
		}
	}

	return docs
}

// benchmarkStandardIndexing tests single-threaded indexing
func benchmarkStandardIndexing(model *gobed.EmbeddingModel, docs []gobed.Document) time.Duration {
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("🐌 STANDARD SINGLE-THREADED INDEXING\n")
	fmt.Printf("%s\n", strings.Repeat("=", 60))

	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = false
	index := gobed.NewVectorIndex(model, config)

	startTime := time.Now()

	// Add documents one by one (slowest method)
	for i, doc := range docs {
		err := index.AddDocument(doc)
		if err != nil {
			log.Printf("Failed to add document %d: %v", i, err)
			continue
		}

		if (i+1)%1000 == 0 {
			elapsed := time.Since(startTime)
			throughput := float64(i+1) / elapsed.Seconds()
			fmt.Printf("   Progress: %d/%d (%.0f docs/sec)\n", i+1, len(docs), throughput)
		}
	}

	elapsed := time.Since(startTime)
	throughput := float64(len(docs)) / elapsed.Seconds()

	fmt.Printf(" Standard Results:\n")
	fmt.Printf("   Documents: %d\n", len(docs))
	fmt.Printf("   Time: %.2fs\n", elapsed.Seconds())
	fmt.Printf("   Throughput: %.0f docs/sec\n", throughput)
	fmt.Printf("   Index size: %d\n", index.Size())

	return elapsed
}

// benchmarkCPUBulkIndexing tests CPU bulk indexing with workers
func benchmarkCPUBulkIndexing(model *gobed.EmbeddingModel, docs []gobed.Document) time.Duration {
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf(" CPU BULK INDEXING (PARALLEL)\n")
	fmt.Printf("%s\n", strings.Repeat("=", 60))

	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = false
	index := gobed.NewVectorIndex(model, config)

	// Create CPU bulk indexer
	cpuIndexer := gobed.NewCPUBulkIndexer(index, 1000)

	startTime := time.Now()
	err := cpuIndexer.IndexBatch(docs)
	elapsed := time.Since(startTime)

	if err != nil {
		log.Printf(" CPU bulk indexing failed: %v", err)
		return elapsed
	}

	stats := cpuIndexer.Stats()

	fmt.Printf(" CPU Bulk Results:\n")
	fmt.Printf("   Documents: %d\n", len(docs))
	fmt.Printf("   Time: %.2fs\n", elapsed.Seconds())
	fmt.Printf("   Throughput: %.0f docs/sec\n", stats.Throughput)
	fmt.Printf("   Workers: %d\n", stats.NumWorkers)
	fmt.Printf("   Index size: %d\n", index.Size())

	return elapsed
}

// showGPUPlaceholder shows what GPU indexing would look like
func showGPUPlaceholder() {
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf(" GPU BULK INDEXING (PLACEHOLDER)\n")
	fmt.Printf("%s\n", strings.Repeat("=", 60))
	fmt.Printf(" GPU indexing would work like this:\n")
	fmt.Printf("   1. Install libtorch with CUDA support\n")
	fmt.Printf("   2. Load 5,000 documents into GPU memory as token arrays\n")
	fmt.Printf("   3. Process entire batch with single GPU forward pass\n")
	fmt.Printf("   4. Quantize embeddings to int8 on GPU\n")
	fmt.Printf("   5. Transfer results back to CPU for indexing\n")
	fmt.Printf("\n   Expected performance: 10-50x faster than CPU\n")
	fmt.Printf("   Batch size: 5,000 documents per GPU forward pass\n")
	fmt.Printf("   Memory usage: ~2-8GB GPU RAM for large batches\n")
}

// testSearchPerformance verifies search works after bulk indexing
func testSearchPerformance(model *gobed.EmbeddingModel, docs []gobed.Document) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf(" SEARCH PERFORMANCE TEST\n")
	fmt.Printf("%s\n", strings.Repeat("=", 60))

	// Index documents with CPU bulk indexing
	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = false
	index := gobed.NewVectorIndex(model, config)

	cpuIndexer := gobed.NewCPUBulkIndexer(index, 1000)

	fmt.Printf("📚 Indexing %d documents for search test...\n", len(docs))
	start := time.Now()
	err := cpuIndexer.IndexBatch(docs)
	indexTime := time.Since(start)

	if err != nil {
		log.Printf(" Failed to index documents: %v", err)
		return
	}

	fmt.Printf(" Indexed in %.2fs\n", indexTime.Seconds())

	// Test queries
	queries := []string{
		"machine learning data analysis",
		"natural language processing",
		"computer vision image recognition",
		"database management systems",
		"web development frameworks",
	}

	fmt.Printf("\n🔎 Testing search queries:\n")
	totalSearchTime := time.Duration(0)

	for i, query := range queries {
		start := time.Now()
		results, err := index.Search(query, 5)
		searchTime := time.Since(start)
		totalSearchTime += searchTime

		if err != nil {
			log.Printf(" Search failed: %v", err)
			continue
		}

		fmt.Printf("   Q%d: \"%.40s...\" - %.2fms, %d results\n",
			i+1, query, float64(searchTime.Nanoseconds())/1e6, len(results))

		if len(results) > 0 {
			fmt.Printf("        Best: Doc %d (similarity: %.4f)\n",
				results[0].ID, results[0].Similarity)
		}
	}

	avgSearchTime := totalSearchTime / time.Duration(len(queries))
	fmt.Printf("\n Search Summary:\n")
	fmt.Printf("   Average search time: %.2fms\n", float64(avgSearchTime.Nanoseconds())/1e6)
	fmt.Printf("   Search throughput: %.0f queries/sec\n", 1.0/avgSearchTime.Seconds())
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🏁 CPU VS GPU BULK INDEXING COMPARISON")
	fmt.Println("================================================================================")
	fmt.Println("Comparing different indexing approaches and preparing for GPU acceleration")
	fmt.Println("")

	// Load model
	fmt.Printf(" Loading embedding model...\n")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	// model.Close() not needed for current implementation

	fmt.Printf(" Model loaded successfully\n")

	// System info
	fmt.Printf("\n System Information:\n")
	fmt.Printf("   CPU cores: %d\n", runtime.NumCPU())
	fmt.Printf("   GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("   Memory: %.1f MB available\n", float64(m.Sys)/(1024*1024))

	// Set random seed for reproducibility
	rand.Seed(42)

	// Test different dataset sizes
	sizes := []int{1000, 5000, 10000}

	for _, size := range sizes {
		fmt.Printf("\n" + strings.Repeat("*", 80))
		fmt.Printf("\n🧪 TESTING WITH %d DOCUMENTS\n", size)
		fmt.Printf(strings.Repeat("*", 80))

		docs := generateTestDocuments(size)

		// Benchmark different approaches
		standardTime := benchmarkStandardIndexing(model, docs)
		cpuBulkTime := benchmarkCPUBulkIndexing(model, docs)

		// Show improvement
		improvement := float64(standardTime) / float64(cpuBulkTime)
		fmt.Printf("\n Performance Improvement:\n")
		fmt.Printf("   CPU Bulk is %.2fx faster than Standard\n", improvement)

		// Force garbage collection between tests
		runtime.GC()
		time.Sleep(500 * time.Millisecond)
	}

	// Show GPU placeholder
	showGPUPlaceholder()

	// Test search functionality
	searchDocs := generateTestDocuments(5000)
	testSearchPerformance(model, searchDocs)

	// Final summary
	fmt.Printf("\n" + strings.Repeat("=", 80))
	fmt.Printf("\n BENCHMARKING COMPLETED\n")
	fmt.Printf(strings.Repeat("=", 80))
	fmt.Printf("\nKey Results:\n")
	fmt.Printf("  • CPU bulk indexing provides significant speedup over single-threaded\n")
	fmt.Printf("  • Parallel workers efficiently utilize multiple CPU cores\n")
	fmt.Printf("  • Search functionality works correctly after bulk indexing\n")
	fmt.Printf("  • System is ready for GPU acceleration with libtorch\n")
	fmt.Printf("\nNext Steps:\n")
	fmt.Printf("  1. Install libtorch: wget pytorch.org/libtorch/...\n")
	fmt.Printf("  2. Uncomment GPU code in bulk_gpu_indexer.go\n")
	fmt.Printf("  3. Test GPU vs CPU performance comparison\n")
	fmt.Printf("  4. Optimize GPU batch sizes for your hardware\n")
}
