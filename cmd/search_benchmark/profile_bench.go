package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

// ProfileConfig contains profiling configuration
type ProfileConfig struct {
	CPUProfile  bool
	MemProfile  bool
	HTTPPprof   bool
	PProfPort   string
	ProfileTime time.Duration
	OutputDir   string
}

// AsyncSearchEngine wraps the search engine with async capabilities
type AsyncSearchEngine struct {
	*gobed.SearchEngine
	indexQueue   chan IndexRequest
	indexWorkers int
	wg           sync.WaitGroup
	ctx          context.Context
	cancel       context.CancelFunc
}

// IndexRequest represents an async indexing request
type IndexRequest struct {
	Documents []string
	Response  chan IndexResponse
}

// IndexResponse contains the result of async indexing
type IndexResponse struct {
	IDs   []int
	Error error
}

// NewAsyncSearchEngine creates a new async search engine
func NewAsyncSearchEngine(model *gobed.EmbeddingModel, workers int) *AsyncSearchEngine {
	ctx, cancel := context.WithCancel(context.Background())

	ase := &AsyncSearchEngine{
		SearchEngine: gobed.NewSearchEngine(model),
		indexQueue:   make(chan IndexRequest, 100), // Buffer for 100 requests
		indexWorkers: workers,
		ctx:          ctx,
		cancel:       cancel,
	}

	// Start index workers
	for i := 0; i < workers; i++ {
		ase.wg.Add(1)
		go ase.indexWorker()
	}

	return ase
}

// indexWorker processes async indexing requests
func (ase *AsyncSearchEngine) indexWorker() {
	defer ase.wg.Done()

	for {
		select {
		case req := <-ase.indexQueue:
			ids, err := ase.SearchEngine.IndexBatch(req.Documents)
			req.Response <- IndexResponse{IDs: ids, Error: err}
		case <-ase.ctx.Done():
			return
		}
	}
}

// IndexBatchAsync performs async batch indexing
func (ase *AsyncSearchEngine) IndexBatchAsync(documents []string) <-chan IndexResponse {
	response := make(chan IndexResponse, 1)

	select {
	case ase.indexQueue <- IndexRequest{Documents: documents, Response: response}:
		return response
	case <-ase.ctx.Done():
		response <- IndexResponse{Error: fmt.Errorf("engine is shutting down")}
		return response
	}
}

// Close shuts down the async engine
func (ase *AsyncSearchEngine) Close() {
	ase.cancel()
	close(ase.indexQueue)
	ase.wg.Wait()
}

// ProfileRunner manages profiling execution
type ProfileRunner struct {
	config ProfileConfig
}

// NewProfileRunner creates a new profile runner
func NewProfileRunner(config ProfileConfig) *ProfileRunner {
	return &ProfileRunner{config: config}
}

// StartProfiling begins profiling with specified configuration
func (pr *ProfileRunner) StartProfiling() error {
	// Create output directory
	if err := os.MkdirAll(pr.config.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Start HTTP pprof server if enabled
	if pr.config.HTTPPprof {
		go func() {
			fmt.Printf("Starting pprof HTTP server on :%s\n", pr.config.PProfPort)
			fmt.Printf("Access profiles at: http://localhost:%s/debug/pprof/\n", pr.config.PProfPort)
			log.Println(http.ListenAndServe(":"+pr.config.PProfPort, nil))
		}()

		// Give server time to start
		time.Sleep(100 * time.Millisecond)
	}

	// Start CPU profiling if enabled
	if pr.config.CPUProfile {
		cpuFile, err := os.Create(fmt.Sprintf("%s/cpu_profile.pprof", pr.config.OutputDir))
		if err != nil {
			return fmt.Errorf("failed to create CPU profile file: %w", err)
		}

		if err := pprof.StartCPUProfile(cpuFile); err != nil {
			cpuFile.Close()
			return fmt.Errorf("failed to start CPU profiling: %w", err)
		}

		fmt.Printf("CPU profiling started, will run for %v\n", pr.config.ProfileTime)
	}

	return nil
}

// StopProfiling stops profiling and writes profiles
func (pr *ProfileRunner) StopProfiling() error {
	// Stop CPU profiling
	if pr.config.CPUProfile {
		pprof.StopCPUProfile()
		fmt.Println("CPU profiling stopped")
	}

	// Write memory profile if enabled
	if pr.config.MemProfile {
		memFile, err := os.Create(fmt.Sprintf("%s/mem_profile.pprof", pr.config.OutputDir))
		if err != nil {
			return fmt.Errorf("failed to create memory profile file: %w", err)
		}
		defer memFile.Close()

		runtime.GC() // Force GC before memory profiling
		if err := pprof.WriteHeapProfile(memFile); err != nil {
			return fmt.Errorf("failed to write memory profile: %w", err)
		}

		fmt.Println("Memory profile written")
	}

	return nil
}

// BenchmarkScenario represents a benchmark scenario
type BenchmarkScenario struct {
	Name            string
	DocumentCount   int
	QueryCount      int
	ConcurrentUsers int
	BatchSize       int
	UseAsync        bool
}

// runBenchmarkScenario executes a specific benchmark scenario
func runBenchmarkScenario(scenario BenchmarkScenario, model *gobed.EmbeddingModel) (*BenchmarkResults, error) {
	fmt.Printf("\n=== Running Scenario: %s ===\n", scenario.Name)

	results := &BenchmarkResults{
		Scenario: scenario,
		Metrics:  make(map[string]interface{}),
	}

	// Generate test data
	documents := generateLargeCorpus(scenario.DocumentCount)
	queries := generateTestQueries(scenario.QueryCount)

	var engine interface {
		IndexBatch([]string) ([]int, error)
		Search(string, int) ([]gobed.SearchResult, error)
		Stats() gobed.SearchEngineStats
	}

	var asyncEngine *AsyncSearchEngine

	if scenario.UseAsync {
		asyncEngine = NewAsyncSearchEngine(model, 4) // 4 workers
		engine = asyncEngine.SearchEngine
		defer asyncEngine.Close()
	} else {
		engine = gobed.NewSearchEngine(model)
	}

	// Measure indexing performance
	fmt.Printf("Indexing %d documents...\n", scenario.DocumentCount)
	indexStart := time.Now()

	if scenario.UseAsync {
		// Async indexing in batches
		var responses []<-chan IndexResponse

		for i := 0; i < scenario.DocumentCount; i += scenario.BatchSize {
			end := min(i+scenario.BatchSize, scenario.DocumentCount)
			batch := documents[i:end]

			response := asyncEngine.IndexBatchAsync(batch)
			responses = append(responses, response)
		}

		// Wait for all indexing to complete
		totalIndexed := 0
		for _, response := range responses {
			result := <-response
			if result.Error != nil {
				return nil, fmt.Errorf("async indexing failed: %w", result.Error)
			}
			totalIndexed += len(result.IDs)
		}

		results.Metrics["total_indexed"] = totalIndexed
	} else {
		// Synchronous indexing
		totalIndexed := 0
		for i := 0; i < scenario.DocumentCount; i += scenario.BatchSize {
			end := min(i+scenario.BatchSize, scenario.DocumentCount)
			batch := documents[i:end]

			ids, err := engine.IndexBatch(batch)
			if err != nil {
				return nil, fmt.Errorf("indexing failed: %w", err)
			}
			totalIndexed += len(ids)
		}

		results.Metrics["total_indexed"] = totalIndexed
	}

	indexDuration := time.Since(indexStart)
	results.IndexTime = indexDuration
	results.IndexThroughput = float64(scenario.DocumentCount) / indexDuration.Seconds()

	fmt.Printf("Indexing completed in %v (%.0f docs/sec)\n", indexDuration, results.IndexThroughput)

	// Measure search performance with concurrent users
	fmt.Printf("Running %d searches with %d concurrent users...\n", scenario.QueryCount, scenario.ConcurrentUsers)

	searchStart := time.Now()
	var searchWg sync.WaitGroup
	queryChannel := make(chan string, scenario.QueryCount)

	// Fill query channel
	for _, query := range queries {
		queryChannel <- query
	}
	close(queryChannel)

	// Launch concurrent search workers
	for i := 0; i < scenario.ConcurrentUsers; i++ {
		searchWg.Add(1)
		go func() {
			defer searchWg.Done()

			for query := range queryChannel {
				_, err := engine.Search(query, 10)
				if err != nil {
					log.Printf("Search error: %v", err)
				}
			}
		}()
	}

	searchWg.Wait()
	searchDuration := time.Since(searchStart)

	results.SearchLatency = searchDuration / time.Duration(scenario.QueryCount)
	results.SearchThroughput = float64(scenario.QueryCount) / searchDuration.Seconds()

	// Get engine stats
	stats := engine.Stats()
	results.Metrics["index_type"] = stats.IndexType
	results.Metrics["memory_mb"] = stats.MemoryUsageMB
	results.Metrics["num_documents"] = stats.NumDocuments

	fmt.Printf("Search completed: %v avg latency, %.0f QPS\n",
		results.SearchLatency, results.SearchThroughput)

	return results, nil
}

// BenchmarkResults stores benchmark results
type BenchmarkResults struct {
	Scenario         BenchmarkScenario
	IndexTime        time.Duration
	IndexThroughput  float64
	SearchLatency    time.Duration
	SearchThroughput float64
	Metrics          map[string]interface{}
}

// generateLargeCorpus generates a large corpus for testing
func generateLargeCorpus(size int) []string {
	baseTexts := []string{
		"machine learning algorithms", "deep learning neural networks", "artificial intelligence",
		"data science analytics", "cloud computing infrastructure", "web development frameworks",
		"database optimization", "microservices architecture", "DevOps automation",
		"cybersecurity protocols", "blockchain technology", "mobile app development",
		"distributed systems", "performance optimization", "scalability patterns",
		"API design principles", "containerization platforms", "CI/CD pipelines",
		"monitoring and observability", "load balancing strategies",
	}

	templates := []string{
		"Advanced %s techniques for enterprise applications",
		"Best practices in %s implementation and deployment",
		"Scalable %s solutions for high-traffic systems",
		"Performance optimization strategies for %s workloads",
		"Security considerations in %s architecture design",
		"Modern approaches to %s in cloud-native environments",
		"Integration patterns for %s in distributed systems",
		"Monitoring and debugging %s in production environments",
	}

	corpus := make([]string, size)
	for i := 0; i < size; i++ {
		template := templates[rand.Intn(len(templates))]
		topic := baseTexts[rand.Intn(len(baseTexts))]
		corpus[i] = fmt.Sprintf(template, topic)
	}

	return corpus
}

// generateTestQueries generates test queries
func generateTestQueries(count int) []string {
	queries := []string{
		"machine learning model optimization",
		"scalable microservices architecture",
		"cloud native application deployment",
		"database performance tuning",
		"API security best practices",
		"container orchestration platforms",
		"real-time data processing",
		"distributed system design patterns",
		"DevOps automation tools",
		"monitoring and alerting systems",
	}

	result := make([]string, count)
	for i := 0; i < count; i++ {
		result[i] = queries[i%len(queries)]
	}

	return result
}

func main() {
	fmt.Println("=== Gobed Search Engine Profiling Suite ===\n")

	// Profile configuration
	config := ProfileConfig{
		CPUProfile:  true,
		MemProfile:  true,
		HTTPPprof:   true,
		PProfPort:   "6060",
		ProfileTime: 2 * time.Minute,
		OutputDir:   "./profiles",
	}

	// Load model
	fmt.Println("Loading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("✓ Model loaded\n")

	// Initialize profiler
	profiler := NewProfileRunner(config)

	// Start profiling
	if err := profiler.StartProfiling(); err != nil {
		log.Fatalf("Failed to start profiling: %v", err)
	}

	// Define benchmark scenarios
	scenarios := []BenchmarkScenario{
		{
			Name:            "Small_Sync",
			DocumentCount:   5000,
			QueryCount:      100,
			ConcurrentUsers: 1,
			BatchSize:       1000,
			UseAsync:        false,
		},
		{
			Name:            "Medium_Sync",
			DocumentCount:   25000,
			QueryCount:      200,
			ConcurrentUsers: 4,
			BatchSize:       2000,
			UseAsync:        false,
		},
		{
			Name:            "Large_Sync",
			DocumentCount:   100000,
			QueryCount:      500,
			ConcurrentUsers: 8,
			BatchSize:       5000,
			UseAsync:        false,
		},
		{
			Name:            "Small_Async",
			DocumentCount:   5000,
			QueryCount:      100,
			ConcurrentUsers: 1,
			BatchSize:       500,
			UseAsync:        true,
		},
		{
			Name:            "Medium_Async",
			DocumentCount:   25000,
			QueryCount:      200,
			ConcurrentUsers: 4,
			BatchSize:       1000,
			UseAsync:        true,
		},
		{
			Name:            "Large_Async",
			DocumentCount:   100000,
			QueryCount:      500,
			ConcurrentUsers: 8,
			BatchSize:       2000,
			UseAsync:        true,
		},
	}

	// Run benchmark scenarios
	allResults := make([]*BenchmarkResults, 0, len(scenarios))

	for _, scenario := range scenarios {
		results, err := runBenchmarkScenario(scenario, model)
		if err != nil {
			log.Printf("Scenario %s failed: %v", scenario.Name, err)
			continue
		}
		allResults = append(allResults, results)

		// Force GC between scenarios
		runtime.GC()
		time.Sleep(1 * time.Second)
	}

	// Stop profiling
	if err := profiler.StopProfiling(); err != nil {
		log.Printf("Failed to stop profiling: %v", err)
	}

	// Print comprehensive results
	printProfileResults(allResults)

	fmt.Println("\n=== Profiling Instructions ===")
	fmt.Println("CPU Profile Analysis:")
	fmt.Printf("  go tool pprof %s/cpu_profile.pprof\n", config.OutputDir)
	fmt.Println("  Commands: top, list, web, svg")

	fmt.Println("\nMemory Profile Analysis:")
	fmt.Printf("  go tool pprof %s/mem_profile.pprof\n", config.OutputDir)
	fmt.Println("  Commands: top, list, web, svg")

	if config.HTTPPprof {
		fmt.Printf("\nLive Profiling (if server still running):\n")
		fmt.Printf("  go tool pprof http://localhost:%s/debug/pprof/profile\n", config.PProfPort)
		fmt.Printf("  go tool pprof http://localhost:%s/debug/pprof/heap\n", config.PProfPort)
	}

	fmt.Println("\n✓ Profiling suite completed successfully!")
}

// printProfileResults prints comprehensive benchmark results
func printProfileResults(results []*BenchmarkResults) {
	fmt.Println("\n=== PROFILING RESULTS SUMMARY ===\n")

	// Performance comparison table
	fmt.Println("| Scenario        | Docs    | Index Time | Index QPS | Search Latency | Search QPS | Memory MB | Index Type |")
	fmt.Println("|-----------------|---------|------------|-----------|----------------|------------|-----------|------------|")

	for _, r := range results {
		fmt.Printf("| %-15s | %7d | %10v | %9.0f | %14v | %10.0f | %9.1f | %-10s |\n",
			r.Scenario.Name,
			r.Scenario.DocumentCount,
			r.IndexTime.Round(time.Millisecond),
			r.IndexThroughput,
			r.SearchLatency,
			r.SearchThroughput,
			r.Metrics["memory_mb"],
			r.Metrics["index_type"])
	}

	// Async vs Sync comparison
	fmt.Println("\n=== ASYNC vs SYNC COMPARISON ===\n")

	syncResults := filterResults(results, false)
	asyncResults := filterResults(results, true)

	if len(syncResults) > 0 && len(asyncResults) > 0 {
		fmt.Println("Average Indexing Throughput:")
		syncAvg := averageIndexThroughput(syncResults)
		asyncAvg := averageIndexThroughput(asyncResults)
		improvement := (asyncAvg - syncAvg) / syncAvg * 100

		fmt.Printf("  Synchronous:  %.0f docs/sec\n", syncAvg)
		fmt.Printf("  Asynchronous: %.0f docs/sec\n", asyncAvg)
		fmt.Printf("  Improvement:  %.1f%%\n", improvement)

		fmt.Println("\nAverage Search Latency:")
		syncLatency := averageSearchLatency(syncResults)
		asyncLatency := averageSearchLatency(asyncResults)

		fmt.Printf("  Synchronous:  %v\n", syncLatency)
		fmt.Printf("  Asynchronous: %v\n", asyncLatency)
	}

	// Performance insights
	fmt.Println("\n=== PERFORMANCE INSIGHTS ===\n")

	for _, r := range results {
		fmt.Printf("%s:\n", r.Scenario.Name)

		if r.SearchLatency < time.Millisecond {
			fmt.Printf("  ✨ Sub-millisecond search achieved!\n")
		} else if r.SearchLatency < 2*time.Millisecond {
			fmt.Printf("  ✅ Excellent latency: %v\n", r.SearchLatency)
		} else if r.SearchLatency < 5*time.Millisecond {
			fmt.Printf("  ⚡ Good latency: %v\n", r.SearchLatency)
		} else {
			fmt.Printf("  ⚠️  High latency: %v\n", r.SearchLatency)
		}

		memoryMB := r.Metrics["memory_mb"].(float64)
		docsPerMB := float64(r.Scenario.DocumentCount) / memoryMB
		fmt.Printf("  📊 Memory efficiency: %.0f docs/MB\n", docsPerMB)

		fmt.Println()
	}
}

// Helper functions
func filterResults(results []*BenchmarkResults, async bool) []*BenchmarkResults {
	var filtered []*BenchmarkResults
	for _, r := range results {
		if r.Scenario.UseAsync == async {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

func averageIndexThroughput(results []*BenchmarkResults) float64 {
	if len(results) == 0 {
		return 0
	}

	total := 0.0
	for _, r := range results {
		total += r.IndexThroughput
	}
	return total / float64(len(results))
}

func averageSearchLatency(results []*BenchmarkResults) time.Duration {
	if len(results) == 0 {
		return 0
	}

	total := time.Duration(0)
	for _, r := range results {
		total += r.SearchLatency
	}
	return total / time.Duration(len(results))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
