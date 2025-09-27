package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lee101/gobed/bed/src"
)

// BenchmarkResult stores results from a benchmark run
type BenchmarkResult struct {
	Name           string        `json:"name"`
	NumChunks      int           `json:"num_chunks"`
	NumFiles       int           `json:"num_files"`
	NumQueries     int           `json:"num_queries"`
	IndexTime      time.Duration `json:"index_time_ms"`
	TotalTime      time.Duration `json:"total_time_ms"`
	QPS            float64       `json:"qps"`
	LatencyP50     float64       `json:"latency_p50_ms"`
	LatencyP95     float64       `json:"latency_p95_ms"`
	LatencyP99     float64       `json:"latency_p99_ms"`
	MemoryMB       float64       `json:"memory_mb"`
	Configuration  interface{}   `json:"configuration"`
}

// Benchmark manages benchmark execution
type Benchmark struct {
	outputDir string
	results   []BenchmarkResult
	mu        sync.Mutex
}

// NewBenchmark creates a new benchmark instance
func NewBenchmark(outputDir string) *Benchmark {
	os.MkdirAll(outputDir, 0755)
	return &Benchmark{
		outputDir: outputDir,
		results:   make([]BenchmarkResult, 0),
	}
}

// GenerateSyntheticCorpus creates test documents
func GenerateSyntheticCorpus(dir string, numFiles int, linesPerFile int) error {
	topics := []string{
		"machine learning", "database systems", "network protocols",
		"security algorithms", "optimization techniques", "distributed computing",
		"compiler design", "graphics rendering", "cryptography methods",
		"data structures",
	}

	subtopics := []string{
		"performance", "scalability", "reliability", "efficiency",
		"accuracy", "latency", "throughput", "memory usage",
		"concurrency", "parallelism",
	}

	for i := 0; i < numFiles; i++ {
		var content strings.Builder

		for j := 0; j < linesPerFile; j++ {
			topic := topics[(i+j)%len(topics)]
			subtopic := subtopics[(i*j)%len(subtopics)]

			line := fmt.Sprintf("Line %d: Research on %s focusing on %s improvements. ",
				j, topic, subtopic)
			line += "This involves analyzing various approaches and methodologies. "
			line += fmt.Sprintf("The implementation considers %s aspects.\n", subtopic)

			content.WriteString(line)
		}

		filename := fmt.Sprintf("doc_%04d.txt", i)
		filepath := filepath.Join(dir, filename)

		if err := ioutil.WriteFile(filepath, []byte(content.String()), 0644); err != nil {
			return err
		}
	}

	return nil
}

// RunIndexingBenchmark benchmarks indexing performance
func (b *Benchmark) RunIndexingBenchmark(config src.SearchConfig, corpusDir string, name string) (*src.GPUSearchIndex, time.Duration, error) {
	log.Printf("📊 Running indexing benchmark: %s", name)

	index, err := src.NewGPUSearchIndex(config)
	if err != nil {
		return nil, 0, err
	}

	ctx := context.Background()
	start := time.Now()

	if err := index.IndexDirectory(ctx, corpusDir, nil); err != nil {
		index.Close()
		return nil, 0, err
	}

	indexTime := time.Since(start)

	stats := index.GetStats()
	chunks := stats["num_chunks"].(int)
	files := stats["num_files"].(int)

	log.Printf("  ✅ Indexed %d chunks from %d files in %.2fs", chunks, files, indexTime.Seconds())
	log.Printf("  📈 Rate: %.0f chunks/sec", float64(chunks)/indexTime.Seconds())

	return index, indexTime, nil
}

// RunSearchBenchmark benchmarks search performance
func (b *Benchmark) RunSearchBenchmark(index *src.GPUSearchIndex, queries []string, k int) ([]float64, float64, error) {
	log.Printf("  🔍 Running search benchmark with %d queries", len(queries))

	// Warmup
	for i := 0; i < min(10, len(queries)); i++ {
		index.Search(queries[i], k)
	}

	// Measure latencies
	latencies := make([]float64, len(queries))
	start := time.Now()

	for i, query := range queries {
		queryStart := time.Now()
		_, err := index.Search(query, k)
		if err != nil {
			return nil, 0, err
		}
		latencies[i] = float64(time.Since(queryStart).Microseconds()) / 1000.0 // Convert to ms
	}

	totalTime := time.Since(start)
	qps := float64(len(queries)) / totalTime.Seconds()

	// Sort for percentiles
	sort.Float64s(latencies)

	return latencies, qps, nil
}

// RunBatchSearchBenchmark benchmarks batch search performance
func (b *Benchmark) RunBatchSearchBenchmark(index *src.GPUSearchIndex, batchSizes []int) {
	log.Printf("\n🔄 Running batch search benchmark")

	for _, batchSize := range batchSizes {
		queries := make([]string, batchSize)
		for i := range queries {
			queries[i] = fmt.Sprintf("search query about topic %d and subtopic %d", i%10, i%5)
		}

		// Warmup
		index.BatchSearch(queries[:min(10, len(queries))], 10)

		// Benchmark
		runs := 10
		var totalTime time.Duration

		for run := 0; run < runs; run++ {
			start := time.Now()
			_, err := index.BatchSearch(queries, 10)
			if err != nil {
				log.Printf("  ❌ Batch search failed: %v", err)
				continue
			}
			totalTime += time.Since(start)
		}

		avgTime := totalTime / time.Duration(runs)
		qps := float64(batchSize) / avgTime.Seconds()
		latencyMs := avgTime.Seconds() * 1000

		log.Printf("  Batch %4d: %.0f QPS, %.2fms total latency, %.3fms per query",
			batchSize, qps, latencyMs, latencyMs/float64(batchSize))
	}
}

// RunScalabilityTest tests performance at different scales
func (b *Benchmark) RunScalabilityTest() {
	log.Printf("\n🔬 Running Scalability Test")

	corpusSizes := []int{100, 500, 1000, 5000}
	config := src.DefaultSearchConfig()
	config.IVFClusters = 256
	config.MaxVectors = 100000

	for _, numFiles := range corpusSizes {
		log.Printf("\n  Testing with %d files...", numFiles)

		// Create corpus
		tempDir, err := ioutil.TempDir("", fmt.Sprintf("scale_%d", numFiles))
		if err != nil {
			log.Printf("Failed to create temp dir: %v", err)
			continue
		}
		defer os.RemoveAll(tempDir)

		if err := GenerateSyntheticCorpus(tempDir, numFiles, 50); err != nil {
			log.Printf("Failed to generate corpus: %v", err)
			continue
		}

		// Benchmark
		index, indexTime, err := b.RunIndexingBenchmark(config, tempDir,
			fmt.Sprintf("Scale-%d", numFiles))
		if err != nil {
			log.Printf("Indexing failed: %v", err)
			continue
		}

		// Search benchmark
		queries := make([]string, 100)
		for i := range queries {
			queries[i] = fmt.Sprintf("research on topic %d", i%20)
		}

		latencies, qps, err := b.RunSearchBenchmark(index, queries, 10)
		if err != nil {
			log.Printf("Search benchmark failed: %v", err)
			index.Close()
			continue
		}

		// Record results
		stats := index.GetStats()
		result := BenchmarkResult{
			Name:       fmt.Sprintf("Scalability-%d-files", numFiles),
			NumChunks:  stats["num_chunks"].(int),
			NumFiles:   stats["num_files"].(int),
			NumQueries: len(queries),
			IndexTime:  indexTime,
			QPS:        qps,
			LatencyP50: percentile(latencies, 0.50),
			LatencyP95: percentile(latencies, 0.95),
			LatencyP99: percentile(latencies, 0.99),
			MemoryMB:   stats["memory_mb"].(float64),
		}

		b.mu.Lock()
		b.results = append(b.results, result)
		b.mu.Unlock()

		index.Close()
	}
}

// RunParameterSweep tests different configuration parameters
func (b *Benchmark) RunParameterSweep(corpusDir string) {
	log.Printf("\n🔬 Running Parameter Sweep")

	chunkSizes := []int{128, 256, 512}
	ivfClusters := []int{64, 256, 1024}
	probeLists := []int{8, 32, 64}

	baseConfig := src.DefaultSearchConfig()
	baseConfig.MaxVectors = 50000

	for _, chunkSize := range chunkSizes {
		for _, ivf := range ivfClusters {
			for _, probe := range probeLists {
				config := baseConfig
				config.ChunkSize = chunkSize
				config.IVFClusters = ivf
				config.ProbeLists = probe

				name := fmt.Sprintf("chunk%d_ivf%d_probe%d", chunkSize, ivf, probe)
				log.Printf("\n  Testing configuration: %s", name)

				// Index
				index, indexTime, err := b.RunIndexingBenchmark(config, corpusDir, name)
				if err != nil {
					log.Printf("    ❌ Failed: %v", err)
					continue
				}

				// Search
				queries := make([]string, 100)
				for i := range queries {
					queries[i] = fmt.Sprintf("test query %d", i)
				}

				latencies, qps, err := b.RunSearchBenchmark(index, queries, 10)
				if err != nil {
					log.Printf("    ❌ Search failed: %v", err)
					index.Close()
					continue
				}

				// Record
				stats := index.GetStats()
				result := BenchmarkResult{
					Name:          name,
					NumChunks:     stats["num_chunks"].(int),
					NumFiles:      stats["num_files"].(int),
					NumQueries:    len(queries),
					IndexTime:     indexTime,
					QPS:           qps,
					LatencyP50:    percentile(latencies, 0.50),
					LatencyP95:    percentile(latencies, 0.95),
					LatencyP99:    percentile(latencies, 0.99),
					MemoryMB:      stats["memory_mb"].(float64),
					Configuration: config,
				}

				b.mu.Lock()
				b.results = append(b.results, result)
				b.mu.Unlock()

				index.Close()
			}
		}
	}
}

// RunConcurrentBenchmark tests concurrent search performance
func (b *Benchmark) RunConcurrentBenchmark(index *src.GPUSearchIndex, numWorkers int, duration time.Duration) {
	log.Printf("\n🔄 Running Concurrent Search Benchmark")
	log.Printf("  Workers: %d, Duration: %v", numWorkers, duration)

	var totalQueries int64
	var totalLatency int64
	latencies := make([]float64, 0, 10000)
	var mu sync.Mutex

	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for {
				select {
				case <-ctx.Done():
					return
				default:
					query := fmt.Sprintf("worker %d query %d", workerID, atomic.AddInt64(&totalQueries, 1))

					start := time.Now()
					_, err := index.Search(query, 10)
					latencyUs := time.Since(start).Microseconds()

					if err == nil {
						atomic.AddInt64(&totalLatency, latencyUs)

						mu.Lock()
						if len(latencies) < cap(latencies) {
							latencies = append(latencies, float64(latencyUs)/1000.0)
						}
						mu.Unlock()
					}
				}
			}
		}(i)
	}

	wg.Wait()

	queries := atomic.LoadInt64(&totalQueries)
	avgLatencyMs := float64(atomic.LoadInt64(&totalLatency)) / float64(queries) / 1000.0
	qps := float64(queries) / duration.Seconds()

	mu.Lock()
	sort.Float64s(latencies)
	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)
	p99 := percentile(latencies, 0.99)
	mu.Unlock()

	log.Printf("  ✅ Results:")
	log.Printf("     Total queries: %d", queries)
	log.Printf("     QPS: %.0f", qps)
	log.Printf("     Avg latency: %.2fms", avgLatencyMs)
	log.Printf("     P50 latency: %.2fms", p50)
	log.Printf("     P95 latency: %.2fms", p95)
	log.Printf("     P99 latency: %.2fms", p99)
}

// SaveResults saves benchmark results to JSON
func (b *Benchmark) SaveResults() error {
	outputFile := filepath.Join(b.outputDir, "benchmark_results.json")

	data, err := json.MarshalIndent(b.results, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(outputFile, data, 0644)
}

// PrintSummary prints benchmark summary
func (b *Benchmark) PrintSummary() {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("BENCHMARK SUMMARY")
	fmt.Println(strings.Repeat("=", 80))

	if len(b.results) == 0 {
		fmt.Println("No results to summarize")
		return
	}

	// Find best configurations
	var bestQPS, bestLatency, bestMemory BenchmarkResult
	bestQPS = b.results[0]
	bestLatency = b.results[0]
	bestMemory = b.results[0]

	for _, r := range b.results {
		if r.QPS > bestQPS.QPS {
			bestQPS = r
		}
		if r.LatencyP50 < bestLatency.LatencyP50 {
			bestLatency = r
		}
		if r.MemoryMB < bestMemory.MemoryMB && r.MemoryMB > 0 {
			bestMemory = r
		}
	}

	fmt.Printf("\n🏆 Best QPS: %s\n", bestQPS.Name)
	fmt.Printf("   - QPS: %.0f\n", bestQPS.QPS)
	fmt.Printf("   - P50 Latency: %.2fms\n", bestQPS.LatencyP50)

	fmt.Printf("\n🏆 Best Latency: %s\n", bestLatency.Name)
	fmt.Printf("   - P50 Latency: %.2fms\n", bestLatency.LatencyP50)
	fmt.Printf("   - QPS: %.0f\n", bestLatency.QPS)

	fmt.Printf("\n🏆 Most Memory Efficient: %s\n", bestMemory.Name)
	fmt.Printf("   - Memory: %.2fMB\n", bestMemory.MemoryMB)
	fmt.Printf("   - QPS: %.0f\n", bestMemory.QPS)

	// Overall statistics
	var totalQPS float64
	var totalLatency float64
	for _, r := range b.results {
		totalQPS += r.QPS
		totalLatency += r.LatencyP50
	}

	avgQPS := totalQPS / float64(len(b.results))
	avgLatency := totalLatency / float64(len(b.results))

	fmt.Printf("\n📊 Overall Statistics:\n")
	fmt.Printf("   - Average QPS: %.0f\n", avgQPS)
	fmt.Printf("   - Average P50 Latency: %.2fms\n", avgLatency)
	fmt.Printf("   - Configurations tested: %d\n", len(b.results))
	fmt.Printf("   - Results saved to: %s\n", b.outputDir)
}

// Helper functions

func percentile(data []float64, p float64) float64 {
	if len(data) == 0 {
		return 0
	}

	index := int(math.Ceil(float64(len(data))*p)) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(data) {
		index = len(data) - 1
	}

	return data[index]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	var (
		corpusDir   = flag.String("corpus", "", "Directory to index (will generate if empty)")
		outputDir   = flag.String("output", "./benchmark_results", "Output directory")
		testType    = flag.String("test", "all", "Test type: all|scale|params|concurrent|batch")
		numFiles    = flag.Int("files", 1000, "Number of files to generate")
		numQueries  = flag.Int("queries", 1000, "Number of test queries")
		numWorkers  = flag.Int("workers", 10, "Number of concurrent workers")
	)
	flag.Parse()

	// Initialize benchmark
	benchmark := NewBenchmark(*outputDir)

	// Setup corpus
	if *corpusDir == "" {
		tempDir, err := ioutil.TempDir("", "gpu_search_bench")
		if err != nil {
			log.Fatal(err)
		}
		defer os.RemoveAll(tempDir)

		log.Printf("📝 Generating synthetic corpus with %d files...", *numFiles)
		if err := GenerateSyntheticCorpus(tempDir, *numFiles, 100); err != nil {
			log.Fatal(err)
		}
		*corpusDir = tempDir
	}

	// Run benchmarks
	switch *testType {
	case "scale":
		benchmark.RunScalabilityTest()

	case "params":
		benchmark.RunParameterSweep(*corpusDir)

	case "concurrent":
		config := src.DefaultSearchConfig()
		config.MaxVectors = 100000

		index, _, err := benchmark.RunIndexingBenchmark(config, *corpusDir, "concurrent-test")
		if err != nil {
			log.Fatal(err)
		}
		defer index.Close()

		benchmark.RunConcurrentBenchmark(index, *numWorkers, 30*time.Second)

	case "batch":
		config := src.DefaultSearchConfig()
		config.MaxVectors = 100000

		index, _, err := benchmark.RunIndexingBenchmark(config, *corpusDir, "batch-test")
		if err != nil {
			log.Fatal(err)
		}
		defer index.Close()

		batchSizes := []int{1, 10, 50, 100, 500, 1000}
		benchmark.RunBatchSearchBenchmark(index, batchSizes)

	case "all":
		// Run all tests
		benchmark.RunScalabilityTest()
		benchmark.RunParameterSweep(*corpusDir)

		// Setup for concurrent and batch tests
		config := src.DefaultSearchConfig()
		config.MaxVectors = 100000

		index, _, err := benchmark.RunIndexingBenchmark(config, *corpusDir, "all-tests")
		if err != nil {
			log.Fatal(err)
		}
		defer index.Close()

		benchmark.RunConcurrentBenchmark(index, *numWorkers, 30*time.Second)

		batchSizes := []int{1, 10, 100, 1000}
		benchmark.RunBatchSearchBenchmark(index, batchSizes)

	default:
		log.Fatalf("Unknown test type: %s", *testType)
	}

	// Save results
	if err := benchmark.SaveResults(); err != nil {
		log.Printf("Failed to save results: %v", err)
	}

	// Print summary
	benchmark.PrintSummary()

	// Print GPU stats
	fmt.Printf("\n🖥️  System Information:\n")
	fmt.Printf("   - CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("   - Go version: %s\n", runtime.Version())
	fmt.Printf("   - CUDA available: %v\n", os.Getenv("CUDA_VISIBLE_DEVICES") != "-1")
}