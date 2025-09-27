package main

import (
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

// #cgo LDFLAGS: -L. -lcuda_unique_topk -L/usr/local/cuda/lib64 -lcudart -lcublas
// #include <stdlib.h>
// extern void* create_unique_topk_search(int max_docs, int dim, int max_k);
// extern void destroy_unique_topk_search(void* handle);
// extern void add_documents_topk(void* handle, const signed char* docs, int num_docs, int dim);
// extern int search_topk_unique(void* handle, const signed char* query, int dim, int k, int* out_indices, float* out_scores);
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

// ScalableBenchmark runs performance tests with dynamic pooling
func runScalableBenchmark(name string, iterations int, benchFunc func()) {
	fmt.Printf("🚀 Running %s (%d iterations)...\n", name, iterations)

	// Warmup
	for i := 0; i < 5; i++ {
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

func benchmarkScalableOperations() {
	fmt.Printf("=== Dynamic GPU Pool Performance ===\n")

	InitializeDynamicPools()

	// Test different document sizes
	sizes := []int{100, 1000, 5000, 10000, 20000, 50000}

	for _, size := range sizes {
		runScalableBenchmark(fmt.Sprintf("Dynamic GPU Context %d docs", size), 20, func() {
			handle := dynamicGPUPool.GetContextForSize(size)
			dynamicGPUPool.PutContext(handle, size)
		})

		runScalableBenchmark(fmt.Sprintf("Dynamic Memory Buffer %d docs", size), 100, func() {
			buffer := dynamicMemoryPool.GetBufferForSize(size * 512)
			dynamicMemoryPool.PutBuffer(buffer)
		})
	}
}

func benchmarkScalablePipeline() {
	fmt.Printf("=== Scalable End-to-End Pipeline ===\n")

	// Test with different document counts
	docCounts := []int{100, 500, 1000, 2000, 5000, 10000}

	for _, numDocs := range docCounts {
		// Create test documents
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

		// Benchmark dynamic search
		runScalableBenchmark(fmt.Sprintf("Dynamic Pipeline %d docs", numDocs), 20, func() {
			query := benchQueries[0]
			_, _, _ = OptimizedGPUSearchDynamic(documents, query, 10)
		})

		// Benchmark batch search for larger datasets
		if numDocs > 2000 {
			runScalableBenchmark(fmt.Sprintf("Batch Pipeline %d docs", numDocs), 20, func() {
				query := benchQueries[0]
				_, _, _ = BatchOptimizedSearch(documents, query, 10, 2000)
			})
		}
	}
}

func benchmarkMemoryScaling() {
	fmt.Printf("=== Memory Scaling Performance ===\n")

	InitializeDynamicPools()

	// Test memory pool scaling with different buffer sizes
	bufferSizes := []int{512, 1024, 5120, 10240, 25600, 51200}

	for _, size := range bufferSizes {
		runScalableBenchmark(fmt.Sprintf("Buffer allocation %d elements", size), 1000, func() {
			buffer := dynamicMemoryPool.GetBufferForSize(size)
			// Simulate some work
			for i := 0; i < min(size, 100); i++ {
				buffer[i] = int8(i % 256)
			}
			dynamicMemoryPool.PutBuffer(buffer)
		})
	}
}

func benchmarkGPUScaling() {
	fmt.Printf("=== GPU Context Scaling Performance ===\n")

	InitializeDynamicPools()

	// Test GPU context scaling with different capacities
	capacities := []int{1024, 2048, 5120, 10240, 20480, 40960}

	for _, capacity := range capacities {
		runScalableBenchmark(fmt.Sprintf("GPU context %d capacity", capacity), 100, func() {
			handle := dynamicGPUPool.GetContextForSize(capacity)

			// Simulate adding some documents
			testEmbeddings := make([]int8, min(capacity, 1000)*512)
			for i := range testEmbeddings {
				testEmbeddings[i] = int8(i % 256 - 128)
			}

			C.add_documents_topk(
				handle,
				(*C.schar)(unsafe.Pointer(&testEmbeddings[0])),
				C.int(min(capacity, 1000)),
				C.int(512),
			)

			dynamicGPUPool.PutContext(handle, capacity)
		})
	}
}

func main() {
	fmt.Printf("📈 Scalable GPU Performance Benchmark Suite\n")
	fmt.Printf("==========================================\n\n")

	benchmarkScalableOperations()
	benchmarkMemoryScaling()
	benchmarkGPUScaling()
	benchmarkScalablePipeline()

	fmt.Printf("🎯 Scalable benchmark suite completed!\n")
	fmt.Printf("\n=== Pool Statistics ===\n")

	if dynamicGPUPool != nil {
		fmt.Printf("GPU pools created: %d different sizes\n", len(dynamicGPUPool.pools))
	}
	if dynamicMemoryPool != nil {
		fmt.Printf("Memory pools created: %d different sizes\n", len(dynamicMemoryPool.pools))
	}
}