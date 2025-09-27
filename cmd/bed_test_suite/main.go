package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"time"

	"github.com/fatih/color"
	"github.com/lee101/gobed"
)

// TestSuite runs comprehensive tests on the bed search implementation
type TestSuite struct {
	documents   []Document
	testQueries []string
	model       *gobed.EmbeddingModel
	results     map[string][]TestResult
	mu          sync.Mutex
}

type TestResult struct {
	Query       string
	NumResults  int
	SearchTime  time.Duration
	TopScore    float32
	Accuracy    float32
	MemoryUsed  int64
}

func NewTestSuite() *TestSuite {
	return &TestSuite{
		results: make(map[string][]TestResult),
		testQueries: []string{
			"Studio Ghibli",
			"anime",
			"Dragon Ball",
			"machine learning",
			"neural networks",
			"BERT GPT transformer",
			"CUDA GPU optimization",
			"time series forecasting",
			"reinforcement learning",
			"quantization",
		},
	}
}

// RunComprehensiveTests executes all test scenarios
func (ts *TestSuite) RunComprehensiveTests(dir string) {
	fmt.Println("\n=== BED COMPREHENSIVE TEST SUITE ===\n")

	// Load model
	fmt.Println("1. Loading embedding model...")
	startTime := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Printf("Warning: Using mock embeddings: %v", err)
	}
	ts.model = model
	fmt.Printf("   Model loaded in %.2fs\n\n", time.Since(startTime).Seconds())

	// Index documents
	fmt.Println("2. Indexing documents...")
	startTime = time.Now()
	ts.documents, err = ts.indexDirectory(dir)
	if err != nil {
		log.Fatalf("Failed to index: %v", err)
	}
	indexTime := time.Since(startTime)
	fmt.Printf("   Indexed %d documents in %.2fs\n", len(ts.documents), indexTime.Seconds())
	fmt.Printf("   Rate: %.0f docs/sec\n\n", float64(len(ts.documents))/indexTime.Seconds())

	// Test search quality
	fmt.Println("3. Testing search quality...")
	ts.testSearchQuality()

	// Test performance at different scales
	fmt.Println("\n4. Testing performance scaling...")
	ts.testPerformanceScaling()

	// Test memory efficiency
	fmt.Println("\n5. Testing memory efficiency...")
	ts.testMemoryEfficiency()

	// Test parallel processing
	fmt.Println("\n6. Testing parallel processing...")
	ts.testParallelProcessing()

	// Generate report
	fmt.Println("\n7. Generating performance report...")
	ts.generateReport()
}

func (ts *TestSuite) indexDirectory(dir string) ([]Document, error) {
	var documents []Document
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Use worker pool for parallel indexing
	numWorkers := runtime.NumCPU()
	fileChan := make(chan string, 100)

	// Worker function
	worker := func() {
		defer wg.Done()
		for path := range fileChan {
			docs, err := ts.indexFile(path)
			if err != nil {
				continue
			}
			mu.Lock()
			documents = append(documents, docs...)
			mu.Unlock()
		}
	}

	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker()
	}

	// Walk directory and send files to workers
	go func() {
		filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return nil
			}

			// Skip non-text files
			if !isTextFile(path) || info.Size() > 100*1024*1024 {
				return nil
			}

			fileChan <- path
			return nil
		})
		close(fileChan)
	}()

	wg.Wait()
	return documents, nil
}

func (ts *TestSuite) indexFile(path string) ([]Document, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var documents []Document
	scanner := bufio.NewScanner(file)
	lineNum := 1

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			lineNum++
			continue
		}

		embedding := ts.createEmbedding(line)
		doc := Document{
			FilePath:  path,
			LineNum:   lineNum,
			Content:   line,
			Embedding: embedding,
		}

		documents = append(documents, doc)
		lineNum++
	}

	return documents, scanner.Err()
}

func (ts *TestSuite) createEmbedding(text string) []int8 {
	if ts.model == nil {
		return mockEmbedding(text)
	}

	floatEmbed, err := ts.model.Encode(text)
	if err != nil {
		return mockEmbedding(text)
	}

	return quantizeEmbedding(floatEmbed)
}

func (ts *TestSuite) testSearchQuality() {
	for _, query := range ts.testQueries {
		fmt.Printf("   Testing: %s\n", query)

		// Create query embedding
		queryEmb := ts.createEmbedding(query)

		// Perform CPU search
		startTime := time.Now()
		results := ts.performCPUSearch(queryEmb, 10)
		searchTime := time.Since(startTime)

		// Calculate accuracy
		accuracy := ts.calculateAccuracy(query, results)

		// Store result
		ts.mu.Lock()
		ts.results["quality"] = append(ts.results["quality"], TestResult{
			Query:      query,
			NumResults: len(results),
			SearchTime: searchTime,
			TopScore:   results[0].Score,
			Accuracy:   accuracy,
		})
		ts.mu.Unlock()

		fmt.Printf("     Time: %.2fms, Score: %.3f, Accuracy: %.1f%%\n",
			float64(searchTime.Microseconds())/1000.0,
			results[0].Score,
			accuracy*100)
	}
}

func (ts *TestSuite) testPerformanceScaling() {
	scales := []int{100, 500, 1000, 5000, 10000}

	for _, scale := range scales {
		if scale > len(ts.documents) {
			break
		}

		subset := ts.documents[:scale]
		fmt.Printf("   Scale: %d documents\n", scale)

		// Test with multiple queries
		var totalTime time.Duration
		for i := 0; i < 10; i++ {
			query := ts.testQueries[i%len(ts.testQueries)]
			queryEmb := ts.createEmbedding(query)

			startTime := time.Now()
			ts.performSearchOnSubset(queryEmb, subset, 10)
			totalTime += time.Since(startTime)
		}

		avgTime := totalTime / 10
		fmt.Printf("     Avg search time: %.2fms\n", float64(avgTime.Microseconds())/1000.0)
		fmt.Printf("     Throughput: %.0f searches/sec\n", 1000000000.0/float64(avgTime.Nanoseconds()))
	}
}

func (ts *TestSuite) testMemoryEfficiency() {
	var m runtime.MemStats

	// Baseline memory
	runtime.GC()
	runtime.ReadMemStats(&m)
	baselineMem := m.Alloc

	// Index large dataset
	fmt.Printf("   Baseline memory: %.2f MB\n", float64(baselineMem)/(1024*1024))

	// Create embeddings
	embeddings := make([][]int8, 10000)
	for i := 0; i < 10000; i++ {
		embeddings[i] = mockEmbedding(fmt.Sprintf("test %d", i))
	}

	runtime.ReadMemStats(&m)
	afterIndexMem := m.Alloc
	fmt.Printf("   After 10K embeddings: %.2f MB\n", float64(afterIndexMem)/(1024*1024))
	fmt.Printf("   Memory per embedding: %.2f KB\n", float64(afterIndexMem-baselineMem)/(10000*1024))

	// Force GC and check memory
	embeddings = nil
	runtime.GC()
	runtime.ReadMemStats(&m)
	fmt.Printf("   After cleanup: %.2f MB\n", float64(m.Alloc)/(1024*1024))
}

func (ts *TestSuite) testParallelProcessing() {
	numWorkers := []int{1, 2, 4, 8, 16}
	query := ts.createEmbedding("test query")

	for _, workers := range numWorkers {
		if workers > runtime.NumCPU()*2 {
			break
		}

		var wg sync.WaitGroup
		startTime := time.Now()

		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < 100/workers; j++ {
					ts.performCPUSearch(query, 10)
				}
			}()
		}

		wg.Wait()
		totalTime := time.Since(startTime)

		fmt.Printf("   Workers: %d, Time: %.2fs, Throughput: %.0f searches/sec\n",
			workers,
			totalTime.Seconds(),
			100.0/totalTime.Seconds())
	}
}

func (ts *TestSuite) performCPUSearch(query []int8, k int) []SearchResult {
	dim := len(query)
	results := make([]SearchResult, 0, len(ts.documents))

	for _, doc := range ts.documents {
		if len(doc.Embedding) != dim {
			continue
		}

		score := cosineSimilarity(query, doc.Embedding)
		results = append(results, SearchResult{
			Document: &doc,
			Score:    score,
		})
	}

	// Sort by score
	sortResults(results)

	if len(results) > k {
		results = results[:k]
	}

	return results
}

func (ts *TestSuite) performSearchOnSubset(query []int8, docs []Document, k int) []SearchResult {
	dim := len(query)
	results := make([]SearchResult, 0, len(docs))

	for i := range docs {
		if len(docs[i].Embedding) != dim {
			continue
		}

		score := cosineSimilarity(query, docs[i].Embedding)
		results = append(results, SearchResult{
			Document: &docs[i],
			Score:    score,
		})
	}

	sortResults(results)

	if len(results) > k {
		results = results[:k]
	}

	return results
}

func (ts *TestSuite) calculateAccuracy(query string, results []SearchResult) float32 {
	queryLower := strings.ToLower(query)
	hits := 0

	for i, result := range results {
		contentLower := strings.ToLower(result.Document.Content)
		if strings.Contains(contentLower, queryLower) {
			// Weight by position (top results matter more)
			weight := 1.0 / float32(i+1)
			hits += int(weight * 100)
		}
	}

	return float32(hits) / float32(len(results)*10)
}

func (ts *TestSuite) generateReport() {
	fmt.Println("\n=== PERFORMANCE REPORT ===\n")

	// Quality results
	if qualityResults, ok := ts.results["quality"]; ok {
		fmt.Println("Search Quality:")
		var totalAccuracy float32
		var totalTime time.Duration

		for _, r := range qualityResults {
			totalAccuracy += r.Accuracy
			totalTime += r.SearchTime
		}

		fmt.Printf("  Average accuracy: %.1f%%\n", (totalAccuracy/float32(len(qualityResults)))*100)
		fmt.Printf("  Average search time: %.2fms\n", float64(totalTime.Microseconds())/float64(len(qualityResults))/1000.0)
	}

	// Memory usage
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("\nMemory Usage:\n")
	fmt.Printf("  Current: %.2f MB\n", float64(m.Alloc)/(1024*1024))
	fmt.Printf("  Total allocated: %.2f MB\n", float64(m.TotalAlloc)/(1024*1024))
	fmt.Printf("  GC cycles: %d\n", m.NumGC)

	// CPU info
	fmt.Printf("\nSystem Info:\n")
	fmt.Printf("  CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("  GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
}

// Helper functions

func cosineSimilarity(a, b []int8) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct int64
	var normA, normB int64

	for i := range a {
		ai := int64(a[i])
		bi := int64(b[i])

		dotProduct += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dotProduct) / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func sortResults(results []SearchResult) {
	// Quick sort implementation
	quickSort(results, 0, len(results)-1)
}

func quickSort(results []SearchResult, low, high int) {
	if low < high {
		pi := partition(results, low, high)
		quickSort(results, low, pi-1)
		quickSort(results, pi+1, high)
	}
}

func partition(results []SearchResult, low, high int) int {
	pivot := results[high].Score
	i := low - 1

	for j := low; j < high; j++ {
		if results[j].Score > pivot { // Descending order
			i++
			results[i], results[j] = results[j], results[i]
		}
	}
	results[i+1], results[high] = results[high], results[i+1]
	return i + 1
}

func mockEmbedding(text string) []int8 {
	const dim = 384
	embedding := make([]int8, dim)

	textLower := strings.ToLower(text)
	hash := 0
	for _, c := range textLower {
		hash = (hash*31 + int(c)) % 256
	}

	for i := 0; i < dim; i++ {
		embedding[i] = int8((hash + i) % 256 - 128)
	}

	return embedding
}

func quantizeEmbedding(embedding []float32) []int8 {
	minVal, maxVal := embedding[0], embedding[0]
	for _, v := range embedding {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	scale := (maxVal - minVal) / 255.0
	if scale == 0 {
		scale = 1.0
	}

	quantized := make([]int8, len(embedding))
	for i, v := range embedding {
		q := int((v - minVal) / scale)
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		quantized[i] = int8(q - 128)
	}

	return quantized
}

func isTextFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	textExts := map[string]bool{
		".txt": true, ".md": true, ".go": true, ".py": true,
		".js": true, ".ts": true, ".c": true, ".cpp": true,
		".h": true, ".java": true, ".rs": true, ".yaml": true,
		".json": true, ".xml": true, ".html": true, ".css": true,
	}
	return textExts[ext] || ext == ""
}

// ProfileCPU starts CPU profiling
func ProfileCPU(filename string) func() {
	f, err := os.Create(filename)
	if err != nil {
		log.Fatal(err)
	}

	pprof.StartCPUProfile(f)
	return func() {
		pprof.StopCPUProfile()
		f.Close()
	}
}