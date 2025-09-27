package main

import (
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

// #cgo LDFLAGS: -L. -lcuda_simple -L/usr/local/cuda/lib64 -lcudart
// #include <stdlib.h>
// extern void* simple_search_create(int max_docs, int dim);
// extern void simple_search_destroy(void* handle);
// extern int simple_search_add_vectors(void* handle, const signed char* docs, int num_docs);
// extern int simple_search_query(void* handle, const signed char* query, int k, int* indices, float* scores);
import "C"

type Document struct {
	FilePath  string
	LineNum   int
	Content   string
	Embedding []int8
}

var (
	benchModel   *FastModel
	benchQueries = []string{
		"anime",
		"Studio Ghibli films",
		"Dragon Ball Z",
		"neural networks machine learning",
		"semantic search optimization",
		"CUDA GPU acceleration performance",
	}
	benchTexts = []string{
		"anime is great",
		"Dragon Ball Z is a popular anime series",
		"Studio Ghibli makes beautiful anime films",
		"Neural networks are machine learning models",
		"CUDA provides GPU acceleration for compute workloads",
		"Semantic search finds meaning in text",
		"Optimization improves performance significantly",
		"Vectorized operations enable parallelism",
		"Memory access patterns affect cache performance",
		"Quantization reduces model size and improves speed",
	}
)

func init() {
	var err error
	benchModel, err = LoadFastModel("../../model/modelint8_512dim.safetensors", "../../model/tokenizer.json")
	if err != nil {
		panic("Failed to load benchmark model: " + err.Error())
	}
}

// MassiveScaleBenchmark runs performance tests with very large document sets
func runMassiveScaleBenchmark(name string, iterations int, benchFunc func()) {
	fmt.Printf("🚀 Running %s (%d iterations)...\n", name, iterations)

	// Warmup
	for i := 0; i < 3; i++ {
		benchFunc()
	}

	var m1, m2 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		benchFunc()
	}
	elapsed := time.Since(start)

	runtime.ReadMemStats(&m2)

	avgTime := elapsed / time.Duration(iterations)
	opsPerSec := float64(iterations) / elapsed.Seconds()
	allocBytes := m2.TotalAlloc - m1.TotalAlloc
	allocCount := m2.Mallocs - m1.Mallocs

	fmt.Printf("  ⚡ %s: %v/op (%.0f ops/sec)\n", name, avgTime, opsPerSec)
	fmt.Printf("     📊 Memory: %d bytes, %d allocs\n\n", allocBytes, allocCount)
}

// ScalableGPUSearch using simple CUDA library without restrictions
func ScalableGPUSearch(documents []*Document, query string, k int) ([]int32, []float32, error) {
	numDocs := len(documents)

	// Create GPU context sized for exactly what we need
	handle := C.simple_search_create(C.int(numDocs), C.int(512))
	defer C.simple_search_destroy(handle)

	// Prepare flat embeddings
	flatEmbeddings := make([]int8, numDocs*512)
	for i, doc := range documents {
		copy(flatEmbeddings[i*512:(i+1)*512], doc.Embedding)
	}

	// Add documents to GPU
	C.simple_search_add_vectors(
		handle,
		(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int(numDocs),
	)

	// Generate query embedding
	queryEmb, err := benchModel.EmbedInt8(query)
	if err != nil {
		return nil, nil, err
	}

	// Perform search
	indices := make([]int32, k)
	scores := make([]float32, k)

	C.simple_search_query(
		handle,
		(*C.schar)(unsafe.Pointer(&queryEmb[0])),
		C.int(k),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	return indices, scores, nil
}

func benchmarkMassiveScale() {
	fmt.Printf("=== Massive Scale Performance ===\n")

	// Test with increasingly large document sets
	docCounts := []int{1000, 5000, 10000, 25000, 50000, 100000}

	for _, numDocs := range docCounts {
		// Create large test dataset
		documents := make([]*Document, numDocs)
		for i := 0; i < numDocs; i++ {
			text := benchTexts[i%len(benchTexts)]
			emb, _ := benchModel.EmbedInt8(text)
			documents[i] = &Document{
				FilePath:  "test.txt",
				LineNum:   i + 1,
				Content:   text,
				Embedding: emb,
			}
		}

		// Benchmark scalable search
		iterations := max(1, 20/max(1, numDocs/10000)) // Fewer iterations for larger datasets
		runMassiveScaleBenchmark(fmt.Sprintf("Scalable Pipeline %d docs", numDocs), iterations, func() {
			query := benchQueries[0]
			_, _, _ = ScalableGPUSearch(documents, query, 10)
		})

		// Memory cleanup to prevent OOM
		documents = nil
		runtime.GC()
	}
}

func benchmarkMemoryEfficiency() {
	fmt.Printf("=== Memory Efficiency Analysis ===\n")

	docCounts := []int{1000, 10000, 50000, 100000}

	for _, numDocs := range docCounts {
		fmt.Printf("\n--- %d Documents Analysis ---\n", numDocs)

		// Measure peak memory usage
		var m1, m2, m3 runtime.MemStats

		// Baseline memory
		runtime.GC()
		runtime.ReadMemStats(&m1)

		// Create documents
		documents := make([]*Document, numDocs)
		for i := 0; i < numDocs; i++ {
			text := benchTexts[i%len(benchTexts)]
			emb, _ := benchModel.EmbedInt8(text)
			documents[i] = &Document{
				FilePath:  "test.txt",
				LineNum:   i + 1,
				Content:   text,
				Embedding: emb,
			}
		}

		runtime.ReadMemStats(&m2)

		// Perform search
		query := benchQueries[0]
		_, _, _ = ScalableGPUSearch(documents, query, 10)

		runtime.ReadMemStats(&m3)

		// Calculate memory usage
		docsMemory := m2.TotalAlloc - m1.TotalAlloc
		searchMemory := m3.TotalAlloc - m2.TotalAlloc
		totalMemory := m3.TotalAlloc - m1.TotalAlloc

		fmt.Printf("  Documents Memory: %.1f MB\n", float64(docsMemory)/1e6)
		fmt.Printf("  Search Memory: %.1f MB\n", float64(searchMemory)/1e6)
		fmt.Printf("  Total Memory: %.1f MB\n", float64(totalMemory)/1e6)
		fmt.Printf("  Memory per Doc: %.1f KB\n", float64(totalMemory)/float64(numDocs)/1e3)

		// Cleanup
		documents = nil
		runtime.GC()
	}
}

func benchmarkSearchQuality() {
	fmt.Printf("=== Search Quality Analysis ===\n")

	// Create test dataset with known relevant documents
	testData := []struct {
		content string
		query   string
		relevant bool
	}{
		{"machine learning algorithms are powerful", "machine learning", true},
		{"neural networks and deep learning", "machine learning", true},
		{"anime and manga culture in Japan", "machine learning", false},
		{"artificial intelligence research", "machine learning", true},
		{"cooking recipes for dinner", "machine learning", false},
		{"optimization techniques in software", "machine learning", false},
		{"CUDA GPU acceleration for ML", "machine learning", true},
		{"programming languages like Go", "machine learning", false},
	}

	documents := make([]*Document, len(testData))
	for i, data := range testData {
		emb, _ := benchModel.EmbedInt8(data.content)
		documents[i] = &Document{
			FilePath:  "test.txt",
			LineNum:   i + 1,
			Content:   data.content,
			Embedding: emb,
		}
	}

	query := "machine learning"
	indices, scores, _ := ScalableGPUSearch(documents, query, 5)

	fmt.Printf("Query: '%s'\n", query)
	fmt.Printf("Results:\n")

	relevantFound := 0
	for i := 0; i < min(5, len(indices)); i++ {
		idx := indices[i]
		if idx >= 0 && int(idx) < len(testData) {
			isRelevant := testData[idx].relevant
			if isRelevant {
				relevantFound++
			}
			fmt.Printf("  %d. [%s] %.1f - %s\n",
				i+1,
				map[bool]string{true: "✓", false: "✗"}[isRelevant],
				scores[i],
				testData[idx].content)
		}
	}

	precision := float64(relevantFound) / 5.0
	fmt.Printf("\nPrecision@5: %.2f (%d/5 relevant)\n\n", precision, relevantFound)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Printf("🔥 Massive Scale GPU Benchmark Suite\n")
	fmt.Printf("===================================\n\n")

	benchmarkSearchQuality()
	benchmarkMassiveScale()
	benchmarkMemoryEfficiency()

	fmt.Printf("🎯 Massive scale benchmark completed!\n")
}