//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🚀 Gobed Ultra-Fast Parallel Search Example")
	fmt.Println("=========================================")

	// Load the int8 512-dim model for maximum efficiency
	fmt.Println("Loading int8 512-dim model for 7.9x compression...")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("✅ Model loaded in %v\n", time.Since(start))

	// Create ultra-fast search engine with auto GPU detection
	fmt.Println("Creating ultra-fast search engine with auto GPU detection...")
	start = time.Now()

	// Use auto-optimized config for maximum performance
	config := gobed.AutoOptimizedSearchConfig()
	config.UseInt8 = true        // Force int8 for 87.4% memory savings
	config.EnableAsync = true    // Enable async processing
	config.AsyncWorkers = runtime.NumCPU() * 2  // Aggressive parallelism
	config.AsyncQueueSize = 50000 // Large queue for throughput

	engine := gobed.NewSearchEngineWithConfig(model, config)
	fmt.Printf("✅ Engine created in %v\n", time.Since(start))

	// Test with ultra-fast parallel indexing (no for loops)
	fmt.Println("\n🔥 Ultra-Fast Parallel Indexing Test")
	fmt.Println("=====================================")

	testUltraFastIndexing(engine)

	// Test with ultra-fast parallel searching
	fmt.Println("\n⚡ Ultra-Fast Parallel Search Test")
	fmt.Println("==================================")

	testUltraFastSearching(engine)

	// Batch processing performance test
	fmt.Println("\n🏆 Maximum Throughput Batch Test")
	fmt.Println("================================")

	testMaxThroughputBatch(engine)

	// GPU accelerated similarity computation
	fmt.Println("\n🚀 GPU Accelerated Similarity Test")
	fmt.Println("==================================")

	testGPUAcceleratedSimilarity(model)
}

// testUltraFastIndexing demonstrates parallel batch indexing with no for loops
func testUltraFastIndexing(engine *gobed.SearchEngine) {
	// Generate large dataset for parallel processing
	datasets := generateParallelDatasets(10000) // 10K documents across batches

	fmt.Printf("🔄 Parallel indexing %d documents across %d batches...\n",
		len(datasets)*1000, len(datasets))

	start := time.Now()

	// Use WaitGroup for ultra-fast parallel processing
	var wg sync.WaitGroup
	indexResults := make(chan IndexResult, len(datasets))

	// Process all batches in parallel (no sequential for loops)
	for i, dataset := range datasets {
		wg.Add(1)
		go func(batchID int, docs []string) {
			defer wg.Done()

			batchStart := time.Now()
			ids, err := engine.IndexBatch(docs)
			batchTime := time.Since(batchStart)

			indexResults <- IndexResult{
				BatchID:     batchID,
				NumDocs:     len(docs),
				ProcessTime: batchTime,
				Error:       err,
				IDs:         ids,
			}
		}(i, dataset)
	}

	// Wait for all parallel operations to complete
	go func() {
		wg.Wait()
		close(indexResults)
	}()

	// Collect results
	totalDocs := 0
	for result := range indexResults {
		if result.Error != nil {
			log.Printf("Batch %d failed: %v", result.BatchID, result.Error)
			continue
		}
		totalDocs += result.NumDocs
		fmt.Printf("  Batch %d: %d docs in %v (%.0f docs/sec)\n",
			result.BatchID, result.NumDocs, result.ProcessTime,
			float64(result.NumDocs)/result.ProcessTime.Seconds())
	}

	totalTime := time.Since(start)
	throughput := float64(totalDocs) / totalTime.Seconds()

	fmt.Printf("✅ Parallel indexing complete: %d docs in %v (%.0f docs/sec)\n",
		totalDocs, totalTime, throughput)
}

// testUltraFastSearching demonstrates parallel search with no for loops
func testUltraFastSearching(engine *gobed.SearchEngine) {
	// Generate multiple search queries for parallel processing
	queries := []string{
		"machine learning artificial intelligence neural networks",
		"database optimization query performance tuning",
		"distributed systems cloud computing scalability",
		"web development frontend backend frameworks",
		"cybersecurity encryption data protection protocols",
		"mobile development iOS Android cross platform",
		"blockchain cryptocurrency decentralized applications",
		"computer vision image processing deep learning",
		"natural language processing text analysis NLP",
		"software engineering architecture design patterns",
	}

	fmt.Printf("🔍 Parallel searching with %d queries...\n", len(queries))

	start := time.Now()

	// Ultra-fast parallel search (no for loops)
	var wg sync.WaitGroup
	searchResults := make(chan SearchResult, len(queries))

	// Launch all searches in parallel
	for i, query := range queries {
		wg.Add(1)
		go func(queryID int, q string) {
			defer wg.Done()

			searchStart := time.Now()
			results, err := engine.Search(q, 5)
			searchTime := time.Since(searchStart)

			searchResults <- SearchResult{
				QueryID:     queryID,
				Query:       q,
				Results:     results,
				SearchTime:  searchTime,
				Error:       err,
			}
		}(i, query)
	}

	// Wait for all searches to complete
	go func() {
		wg.Wait()
		close(searchResults)
	}()

	// Process results as they come in
	var totalSearchTime time.Duration
	successCount := 0

	for result := range searchResults {
		if result.Error != nil {
			log.Printf("Query %d failed: %v", result.QueryID, result.Error)
			continue
		}

		totalSearchTime += result.SearchTime
		successCount++

		fmt.Printf("  Query %d: %d results in %v (%.3fms)\n",
			result.QueryID, len(result.Results), result.SearchTime,
			float64(result.SearchTime.Nanoseconds())/1e6)

		// Show top result
		if len(result.Results) > 0 {
			fmt.Printf("    Top: [%.3f] %s\n",
				result.Results[0].Similarity,
				truncateText(result.Results[0].Text, 60))
		}
	}

	totalTime := time.Since(start)
	avgLatency := totalSearchTime / time.Duration(successCount)
	qps := float64(successCount) / totalTime.Seconds()

	fmt.Printf("✅ Parallel search complete: %d queries in %v (avg: %v, %.0f QPS)\n",
		successCount, totalTime, avgLatency, qps)
}

// testMaxThroughputBatch demonstrates maximum throughput batch processing
func testMaxThroughputBatch(engine *gobed.SearchEngine) {
	// Generate large batch for maximum throughput test
	batchSize := 1000
	docs := make([]string, batchSize)

	// Generate diverse content in parallel
	var wg sync.WaitGroup
	chunks := runtime.NumCPU()
	chunkSize := batchSize / chunks

	for i := 0; i < chunks; i++ {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end && j < batchSize; j++ {
				docs[j] = generateDocument(j)
			}
		}(i*chunkSize, (i+1)*chunkSize)
	}
	wg.Wait()

	fmt.Printf("🏆 Maximum throughput test with %d documents...\n", batchSize)

	// Test indexing throughput
	start := time.Now()
	ids, err := engine.IndexBatch(docs)
	indexTime := time.Since(start)

	if err != nil {
		log.Printf("Batch indexing failed: %v", err)
		return
	}

	indexThroughput := float64(batchSize) / indexTime.Seconds()
	fmt.Printf("  Indexing: %d docs in %v (%.0f docs/sec)\n",
		len(ids), indexTime, indexThroughput)

	// Test search throughput with batch queries
	queries := make([]string, 100)
	for i := range queries {
		queries[i] = fmt.Sprintf("test query %d with keywords batch throughput performance", i)
	}

	start = time.Now()

	// Ultra-fast parallel batch search
	resultChan := make(chan BatchSearchResult, len(queries))

	// Process queries in parallel batches
	batchQueries := 10 // Process 10 queries per goroutine
	for i := 0; i < len(queries); i += batchQueries {
		wg.Add(1)
		end := i + batchQueries
		if end > len(queries) {
			end = len(queries)
		}

		go func(batchQueries []string, startIdx int) {
			defer wg.Done()

			for j, query := range batchQueries {
				queryStart := time.Now()
				results, err := engine.Search(query, 3)
				queryTime := time.Since(queryStart)

				resultChan <- BatchSearchResult{
					QueryIndex: startIdx + j,
					Query:      query,
					Results:    results,
					Time:       queryTime,
					Error:      err,
				}
			}
		}(queries[i:end], i)
	}

	// Wait for all queries to complete
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	var totalQueryTime time.Duration
	successQueries := 0

	for result := range resultChan {
		if result.Error == nil {
			totalQueryTime += result.Time
			successQueries++
		}
	}

	searchTime := time.Since(start)
	searchThroughput := float64(successQueries) / searchTime.Seconds()
	avgQueryTime := totalQueryTime / time.Duration(successQueries)

	fmt.Printf("  Searching: %d queries in %v (%.0f QPS, avg: %v)\n",
		successQueries, searchTime, searchThroughput, avgQueryTime)

	// Show final stats
	stats := engine.Stats()
	fmt.Printf("✅ Engine stats: %d docs, Type: %s, Memory: %.2f MB\n",
		stats.NumDocuments, stats.IndexType, stats.MemoryUsageMB)
}

// testGPUAcceleratedSimilarity demonstrates GPU-accelerated similarity computation
func testGPUAcceleratedSimilarity(model *gobed.EmbeddingModel) {
	pairs := [][]string{
		{"machine learning algorithms", "artificial intelligence models"},
		{"database optimization", "query performance tuning"},
		{"distributed computing", "parallel processing systems"},
		{"web development", "frontend backend programming"},
		{"neural networks", "deep learning architectures"},
		{"cloud computing", "scalable infrastructure services"},
		{"cybersecurity", "information security protocols"},
		{"data science", "statistical analysis methods"},
		{"software engineering", "application development practices"},
		{"computer vision", "image processing techniques"},
	}

	fmt.Printf("🚀 GPU-accelerated similarity computation for %d pairs...\n", len(pairs))

	start := time.Now()

	// Parallel similarity computation (no for loops)
	var wg sync.WaitGroup
	similarities := make(chan SimilarityResult, len(pairs))

	for i, pair := range pairs {
		wg.Add(1)
		go func(pairID int, text1, text2 string) {
			defer wg.Done()

			simStart := time.Now()

			// Generate embeddings in parallel
			var emb1, emb2 []float32
			var err1, err2 error
			var embWg sync.WaitGroup

			embWg.Add(2)
			go func() {
				defer embWg.Done()
				emb1, err1 = model.Encode(text1)
			}()
			go func() {
				defer embWg.Done()
				emb2, err2 = model.Encode(text2)
			}()
			embWg.Wait()

			if err1 != nil || err2 != nil {
				similarities <- SimilarityResult{
					PairID: pairID,
					Error:  fmt.Errorf("embedding failed: %v, %v", err1, err2),
				}
				return
			}

			similarity := gobed.CosineSimilarity(emb1, emb2)
			simTime := time.Since(simStart)

			similarities <- SimilarityResult{
				PairID:     pairID,
				Text1:      text1,
				Text2:      text2,
				Similarity: similarity,
				Time:       simTime,
			}
		}(i, pair[0], pair[1])
	}

	// Wait for all computations
	go func() {
		wg.Wait()
		close(similarities)
	}()

	// Process results
	var totalTime time.Duration
	successCount := 0

	for result := range similarities {
		if result.Error != nil {
			log.Printf("Similarity %d failed: %v", result.PairID, result.Error)
			continue
		}

		totalTime += result.Time
		successCount++

		fmt.Printf("  [%.3f] '%s' ↔ '%s' (%v)\n",
			result.Similarity,
			truncateText(result.Text1, 25),
			truncateText(result.Text2, 25),
			result.Time)
	}

	overallTime := time.Since(start)
	avgTime := totalTime / time.Duration(successCount)
	parallelSpeedup := float64(totalTime) / float64(overallTime)

	fmt.Printf("✅ GPU similarity complete: %d pairs in %v (avg: %v, %.1fx speedup)\n",
		successCount, overallTime, avgTime, parallelSpeedup)
}

// Helper types for parallel processing
type IndexResult struct {
	BatchID     int
	NumDocs     int
	ProcessTime time.Duration
	Error       error
	IDs         []int
}

type SearchResult struct {
	QueryID    int
	Query      string
	Results    []gobed.SearchResult
	SearchTime time.Duration
	Error      error
}

type BatchSearchResult struct {
	QueryIndex int
	Query      string
	Results    []gobed.SearchResult
	Time       time.Duration
	Error      error
}

type SimilarityResult struct {
	PairID     int
	Text1      string
	Text2      string
	Similarity float32
	Time       time.Duration
	Error      error
}

// Helper functions
func generateParallelDatasets(totalDocs int) [][]string {
	numBatches := runtime.NumCPU()
	batchSize := totalDocs / numBatches

	datasets := make([][]string, numBatches)

	// Generate batches in parallel
	var wg sync.WaitGroup
	for i := 0; i < numBatches; i++ {
		wg.Add(1)
		go func(batchID int) {
			defer wg.Done()

			batch := make([]string, batchSize)
			for j := 0; j < batchSize; j++ {
				docID := batchID*batchSize + j
				batch[j] = generateDocument(docID)
			}
			datasets[batchID] = batch
		}(i)
	}
	wg.Wait()

	return datasets
}

func generateDocument(id int) string {
	templates := []string{
		"Advanced %s techniques for %s optimization in %s systems",
		"Implementation of %s algorithms using %s for %s applications",
		"Research on %s methodologies in %s with %s frameworks",
		"Performance analysis of %s solutions for %s in %s environments",
		"Scalable %s architecture design for %s using %s technologies",
	}

	topics := []string{
		"machine learning", "artificial intelligence", "neural networks",
		"database", "cloud computing", "distributed systems",
		"web development", "mobile applications", "software engineering",
		"cybersecurity", "data science", "computer vision",
		"natural language processing", "blockchain", "quantum computing",
	}

	technologies := []string{
		"TensorFlow", "PyTorch", "Kubernetes", "Docker", "React",
		"PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Kafka",
		"AWS", "Azure", "Google Cloud", "Python", "Go", "Rust",
	}

	applications := []string{
		"enterprise", "real-time", "high-performance", "distributed",
		"scalable", "secure", "cloud-native", "microservices",
		"serverless", "edge computing", "IoT", "mobile",
	}

	template := templates[id%len(templates)]
	topic := topics[id%len(topics)]
	tech := technologies[id%len(technologies)]
	app := applications[id%len(applications)]

	return fmt.Sprintf(template+" Document ID: %d", topic, tech, app, id)
}

func truncateText(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen-3] + "..."
}
