package main

import (
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

// #cgo LDFLAGS: -L. -lcuda_unique_topk -L/usr/local/cuda/lib64 -lcudart -lcublas
// #include <stdlib.h>
// extern void* cuda_fast_search_create(int max_vectors, int dim);
// extern void cuda_fast_search_destroy(void* handle);
// extern int cuda_fast_search_add_vectors(void* handle, const signed char* vectors, int num_vectors);
// extern int cuda_fast_search_query(void* handle, const signed char* query, int k, int* indices, float* scores);
import "C"

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

// Benchmark runner with timing
func runBenchmark(name string, iterations int, benchFunc func()) {
	fmt.Printf("🔄 Running %s (%d iterations)...\n", name, iterations)

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

	fmt.Printf("  ✅ %s: %v/op (%.0f ops/sec)\n", name, avgTime, opsPerSec)
	fmt.Printf("     📊 Memory: %d bytes, %d allocs\n\n", allocBytes, allocCount)
}

// Benchmark tokenizer performance
func benchmarkTokenizer() {
	fmt.Printf("=== Tokenizer Benchmarks ===\n")

	tokenizer := benchModel.tokenizer

	runBenchmark("Tokenize Short", 10000, func() {
		_ = tokenizer.Tokenize("anime")
	})

	runBenchmark("Tokenize Medium", 5000, func() {
		_ = tokenizer.Tokenize("Studio Ghibli makes beautiful anime films")
	})

	runBenchmark("Tokenize Long", 1000, func() {
		_ = tokenizer.Tokenize("Neural networks are machine learning models that use CUDA GPU acceleration for fast training and inference")
	})
}

// Benchmark embedding generation
func benchmarkEmbedding() {
	fmt.Printf("=== Embedding Benchmarks ===\n")

	runBenchmark("EmbedInt8 Single", 1000, func() {
		text := benchQueries[0]
		_, err := benchModel.EmbedInt8(text)
		if err != nil {
			panic(err)
		}
	})

	runBenchmark("EmbedInt8 Batch", 100, func() {
		for _, text := range benchTexts {
			_, err := benchModel.EmbedInt8(text)
			if err != nil {
				panic(err)
			}
		}
	})
}

// Benchmark GPU operations
func benchmarkGPUOperations() {
	fmt.Printf("=== GPU Operations Benchmarks ===\n")

	const numVectors = 10000
	const dim = 512

	// Create test embeddings
	embeddings := make([]int8, numVectors*dim)
	for i := range embeddings {
		embeddings[i] = int8(i % 256 - 128)
	}

	queryEmb, _ := benchModel.EmbedInt8("anime test query")

	runBenchmark("GPU Create/Destroy", 1000, func() {
		handle := C.cuda_fast_search_create(C.int(numVectors), C.int(dim))
		if handle != nil {
			C.cuda_fast_search_destroy(handle)
		}
	})

	// Reuse handle for vector operations
	handle := C.cuda_fast_search_create(C.int(numVectors), C.int(dim))
	defer C.cuda_fast_search_destroy(handle)

	runBenchmark("GPU Add Vectors", 100, func() {
		C.cuda_fast_search_add_vectors(
			handle,
			(*C.schar)(unsafe.Pointer(&embeddings[0])),
			C.int(numVectors),
		)
	})

	// Add vectors once for query testing
	C.cuda_fast_search_add_vectors(
		handle,
		(*C.schar)(unsafe.Pointer(&embeddings[0])),
		C.int(numVectors),
	)

	indices := make([]int32, 10)
	scores := make([]float32, 10)

	runBenchmark("GPU Query", 1000, func() {
		C.cuda_fast_search_query(
			handle,
			(*C.schar)(unsafe.Pointer(&queryEmb[0])),
			C.int(10),
			(*C.int)(unsafe.Pointer(&indices[0])),
			(*C.float)(unsafe.Pointer(&scores[0])),
		)
	})
}

// Benchmark end-to-end pipeline
func benchmarkEndToEnd() {
	fmt.Printf("=== End-to-End Pipeline Benchmarks ===\n")

	const numDocs = 1000

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

	runBenchmark("Complete Pipeline 1K docs", 50, func() {
		// Setup GPU
		handle := C.cuda_fast_search_create(C.int(numDocs+100), C.int(512))

		// Add embeddings
		flatEmbeddings := make([]int8, numDocs*512)
		for j, doc := range documents {
			copy(flatEmbeddings[j*512:(j+1)*512], doc.Embedding)
		}

		C.cuda_fast_search_add_vectors(
			handle,
			(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
			C.int(numDocs),
		)

		// Query
		query := benchQueries[0]
		queryEmb, _ := benchModel.EmbedInt8(query)

		indices := make([]int32, 10)
		scores := make([]float32, 10)

		C.cuda_fast_search_query(
			handle,
			(*C.schar)(unsafe.Pointer(&queryEmb[0])),
			C.int(10),
			(*C.int)(unsafe.Pointer(&indices[0])),
			(*C.float)(unsafe.Pointer(&scores[0])),
		)

		C.cuda_fast_search_destroy(handle)
	})
}

// Benchmark memory operations
func benchmarkMemoryOperations() {
	fmt.Printf("=== Memory Operations Benchmarks ===\n")

	runBenchmark("Slice Allocation", 10000, func() {
		_ = make([]int8, 512)
	})

	src := make([]int8, 512)
	runBenchmark("Embedding Copy", 10000, func() {
		dst := make([]int8, 512)
		copy(dst, src)
	})

	result := make([]float32, 512)
	embedding := make([]int8, 512)
	scale := float32(0.1)

	runBenchmark("Vectorized Accumulation", 10000, func() {
		// Simulate vectorized accumulation from model.go
		for j := 0; j < 512; j += 4 {
			result[j] += float32(embedding[j]) * scale
			result[j+1] += float32(embedding[j+1]) * scale
			result[j+2] += float32(embedding[j+2]) * scale
			result[j+3] += float32(embedding[j+3]) * scale
		}
	})
}

func main() {
	fmt.Printf("🚀 GPU Performance Benchmark Suite\n")
	fmt.Printf("=" + string(make([]byte, 50)) + "\n\n")

	// Check GPU availability
	if info := GetGPUInfo(); info != nil {
		fmt.Printf("🔧 GPU: %s (SM %d.%d)\n", info.Name, info.Major, info.Minor)
		fmt.Printf("💾 Memory: %.1f GB total, %.1f GB free\n\n",
			float64(info.TotalMem)/1e9, float64(info.FreeMem)/1e9)
	}

	benchmarkTokenizer()
	benchmarkEmbedding()
	benchmarkGPUOperations()
	benchmarkEndToEnd()
	benchmarkMemoryOperations()

	fmt.Printf("🎉 Benchmark suite completed!\n")
}