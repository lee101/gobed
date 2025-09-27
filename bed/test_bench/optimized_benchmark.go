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

// OptimizedBenchmark runs performance tests with pooling optimizations
func runOptimizedBenchmark(name string, iterations int, benchFunc func()) {
	fmt.Printf("🚀 Running %s (%d iterations)...\n", name, iterations)

	// Warmup
	for i := 0; i < 10; i++ {
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

func benchmarkOptimizedOperations() {
	fmt.Printf("=== Optimized GPU Operations ===\n")

	// Initialize pools once
	InitializePools()

	const numVectors = 10000
	const dim = 512

	// Create test embeddings
	embeddings := make([]int8, numVectors*dim)
	for i := range embeddings {
		embeddings[i] = int8(i % 256 - 128)
	}

	runOptimizedBenchmark("Pooled GPU Context", 1000, func() {
		handle := gpuPool.Get()
		gpuPool.Put(handle)
	})

	runOptimizedBenchmark("Pooled Memory Allocation", 10000, func() {
		buffer := embeddingPool.Get()
		embeddingPool.Put(buffer)
	})

	// Test batch operations with pooling
	runOptimizedBenchmark("Pooled Batch Operations", 100, func() {
		handle := gpuPool.Get()
		flatBuffer := flatDataPool.Get()

		// Copy data to flat buffer (limit to available space)
		copySize := min(len(flatBuffer), len(embeddings))
		copy(flatBuffer[:copySize], embeddings[:copySize])

		// Add to GPU (adjust count based on available buffer space)
		actualVectors := copySize / dim
		C.add_documents_topk(
			handle,
			(*C.schar)(unsafe.Pointer(&flatBuffer[0])),
			C.int(actualVectors),
			C.int(dim),
		)

		flatDataPool.Put(flatBuffer)
		gpuPool.Put(handle)
	})
}

func benchmarkOptimizedPipeline() {
	fmt.Printf("=== Optimized End-to-End Pipeline ===\n")

	const numDocs = 1000

	// Create test documents once
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

	// Baseline: Original approach
	runOptimizedBenchmark("Original Pipeline 1K docs", 50, func() {
		handle := C.create_unique_topk_search(C.int(numDocs+100), C.int(512), C.int(10))

		flatEmbeddings := make([]int8, numDocs*512)
		for j, doc := range documents {
			copy(flatEmbeddings[j*512:(j+1)*512], doc.Embedding)
		}

		C.add_documents_topk(
			handle,
			(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
			C.int(numDocs),
			C.int(512),
		)

		query := benchQueries[0]
		queryEmb, _ := benchModel.EmbedInt8(query)

		indices := make([]int32, 10)
		scores := make([]float32, 10)

		C.search_topk_unique(
			handle,
			(*C.schar)(unsafe.Pointer(&queryEmb[0])),
			C.int(512),
			C.int(10),
			(*C.int)(unsafe.Pointer(&indices[0])),
			(*C.float)(unsafe.Pointer(&scores[0])),
		)

		C.destroy_unique_topk_search(handle)
	})

	// Optimized: Using pools
	runOptimizedBenchmark("Optimized Pipeline 1K docs", 100, func() {
		query := benchQueries[0]
		_, _, _ = OptimizedGPUSearch(documents, query, 10)
	})

	// Test scaling with different document counts
	docCounts := []int{100, 500, 1000, 2000}
	for _, count := range docCounts {
		if count > len(documents) {
			continue
		}
		subDocs := documents[:count]
		runOptimizedBenchmark(fmt.Sprintf("Optimized Pipeline %d docs", count), 50, func() {
			query := benchQueries[0]
			_, _, _ = OptimizedGPUSearch(subDocs, query, 10)
		})
	}
}

func benchmarkMemoryPools() {
	fmt.Printf("=== Memory Pool Performance ===\n")

	InitializePools()

	runOptimizedBenchmark("Standard Allocation", 10000, func() {
		_ = make([]int8, 512)
	})

	runOptimizedBenchmark("Pooled Allocation", 10000, func() {
		buffer := embeddingPool.Get()
		embeddingPool.Put(buffer)
	})

	runOptimizedBenchmark("Large Buffer Standard", 1000, func() {
		_ = make([]int8, 1000*512)
	})

	runOptimizedBenchmark("Large Buffer Pooled", 1000, func() {
		buffer := flatDataPool.Get()
		flatDataPool.Put(buffer)
	})
}

func main() {
	fmt.Printf("⚡ Optimized GPU Performance Benchmark Suite\n")
	fmt.Printf("===========================================\n\n")

	benchmarkOptimizedOperations()
	benchmarkMemoryPools()
	benchmarkOptimizedPipeline()

	fmt.Printf("🎯 Optimized benchmark suite completed!\n")
}