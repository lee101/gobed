package main

/*
#cgo CPPFLAGS: -I../../gpu -I../../libtorch/include -I../../libtorch/include/torch/csrc/api/include
#cgo LDFLAGS: -L../../libtorch/lib -L../../gpu -L/usr/local/cuda-12.0/targets/x86_64-linux/lib -ltorch_cgo_wrapper -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lcudart -ldl
#include "torch_cgo_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"time"
	"unsafe"
)

type GPUBenchmarkConfig struct {
	VectorDim    int
	TrainingSize int
	IndexSize    int
	QueryCount   int
	K            int
	Name         string
}

type GPUBenchmarkResult struct {
	Config         GPUBenchmarkConfig
	TrainTime      time.Duration
	IndexTime      time.Duration
	SearchTime     time.Duration
	IndexRate      float64 // vectors/sec
	QueryRate      float64 // queries/sec
	AvgSearchTime  float64 // ms per query
	MemoryUsage    float64 // MB
	TotalTime      time.Duration
	UsingGPU       bool
}

func main() {
	fmt.Println("🚀 GPU LibTorch Comprehensive Benchmark")
	fmt.Println("=======================================")

	// Check system info
	version := C.GoString(C.torch_get_version())
	cudaAvailable := C.torch_cuda_is_available() != 0
	deviceCount := int(C.torch_cuda_device_count())

	fmt.Printf("📊 System Information:\n")
	fmt.Printf("   LibTorch version: %s\n", version)
	fmt.Printf("   CUDA available: %v\n", cudaAvailable)
	fmt.Printf("   Device count: %d\n", deviceCount)
	fmt.Printf("   Go version: %s\n", runtime.Version())
	fmt.Printf("   CPU cores: %d\n", runtime.NumCPU())

	if !cudaAvailable {
		log.Fatal("❌ CUDA not available - this benchmark requires GPU")
	}

	// Focus on 512D and 1024D as requested
	configs := []GPUBenchmarkConfig{
		// 512D benchmarks - Production scale
		{VectorDim: 512, TrainingSize: 5000, IndexSize: 50000, QueryCount: 1000, K: 10, Name: "512D-50K"},
		{VectorDim: 512, TrainingSize: 5000, IndexSize: 100000, QueryCount: 1000, K: 10, Name: "512D-100K"},
		
		// 1024D benchmarks - Large scale  
		{VectorDim: 1024, TrainingSize: 5000, IndexSize: 25000, QueryCount: 500, K: 10, Name: "1024D-25K"},
		{VectorDim: 1024, TrainingSize: 5000, IndexSize: 50000, QueryCount: 500, K: 10, Name: "1024D-50K"},
	}

	fmt.Printf("\n🚀 Running GPU Performance Benchmarks...\n")
	fmt.Printf("\n%-12s %-8s %-8s %-8s %-12s %-10s %-10s %-8s %-6s\n",
		"Config", "Dim", "Index", "Queries", "Index/sec", "Query/ms", "QPS", "Memory", "GPU")
	fmt.Println("------------------------------------------------------------------------------------")

	var results []GPUBenchmarkResult

	for i, config := range configs {
		fmt.Printf("\nRunning benchmark %d/%d: %s...\n", i+1, len(configs), config.Name)
		
		result := runGPUBenchmark(config)
		results = append(results, result)

		gpuStatus := "CPU"
		if result.UsingGPU {
			gpuStatus = "GPU"
		}

		fmt.Printf("%-12s %-8d %-8d %-8d %-12.0f %-10.2f %-10.0f %-8.1f %-6s\n",
			config.Name,
			config.VectorDim,
			config.IndexSize,
			config.QueryCount,
			result.IndexRate,
			result.AvgSearchTime,
			result.QueryRate,
			result.MemoryUsage,
			gpuStatus)
	}

	// Print detailed summary
	fmt.Println("\n📈 Detailed Performance Analysis:")
	fmt.Println("=================================")

	for i, result := range results {
		config := result.Config
		fmt.Printf("\n%d. %s (%dD vectors, %d indexed):\n", i+1, config.Name, config.VectorDim, config.IndexSize)
		fmt.Printf("   Training: %v (%d vectors)\n", result.TrainTime, config.TrainingSize)
		fmt.Printf("   Indexing: %v (%.0f vectors/sec)\n", result.IndexTime, result.IndexRate)
		fmt.Printf("   Search: %v (%d queries, %.2f ms/query, %.0f QPS)\n", 
			result.SearchTime, config.QueryCount, result.AvgSearchTime, result.QueryRate)
		fmt.Printf("   Memory: %.1f MB GPU\n", result.MemoryUsage)
		fmt.Printf("   Total: %v\n", result.TotalTime)
	}

	fmt.Println("\n✅ GPU Benchmark completed!")
}

func runGPUBenchmark(config GPUBenchmarkConfig) GPUBenchmarkResult {
	totalStart := time.Now()

	// Create indexer with GPU enabled
	cConfig := C.IndexConfig{
		vector_dim:        C.int(config.VectorDim),
		num_subquantizers: C.int(64),
		codebook_size:     C.int(256),
		ivf_clusters:      C.int(1024),
		probe_lists:       C.int(32),
		rerank_k:         C.int(200),
		device_id:        C.int(0), // Force GPU usage
	}

	handle := C.torch_indexer_create(cConfig)
	if handle == nil {
		log.Fatal("Failed to create indexer")
	}
	defer C.torch_indexer_destroy(handle)

	// Generate training data
	trainingData := generateVectors(config.TrainingSize, config.VectorDim)

	// Training benchmark
	trainStart := time.Now()
	result := C.torch_indexer_train(
		handle,
		(*C.schar)(unsafe.Pointer(&trainingData[0])),
		C.int(config.TrainingSize),
		C.int(config.VectorDim),
	)
	trainTime := time.Since(trainStart)

	if result == 0 {
		log.Fatal("Training failed")
	}

	// Generate index data
	indexData := generateVectors(config.IndexSize, config.VectorDim)

	// Indexing benchmark
	indexStart := time.Now()
	result = C.torch_indexer_add_vectors(
		handle,
		(*C.schar)(unsafe.Pointer(&indexData[0])),
		C.int(config.IndexSize),
		C.int(config.VectorDim),
	)
	indexTime := time.Since(indexStart)

	if result == 0 {
		log.Fatal("Indexing failed")
	}

	indexRate := float64(config.IndexSize) / indexTime.Seconds()

	// Generate queries (use vectors from index for ground truth)
	queries := make([][]int8, config.QueryCount)
	for i := 0; i < config.QueryCount; i++ {
		startIdx := rand.Intn(config.IndexSize) * config.VectorDim
		endIdx := startIdx + config.VectorDim
		queries[i] = make([]int8, config.VectorDim)
		copy(queries[i], indexData[startIdx:endIdx])
	}

	// Search benchmark
	searchStart := time.Now()
	
	for i := 0; i < config.QueryCount; i++ {
		searchResult := C.torch_indexer_search(
			handle,
			(*C.schar)(unsafe.Pointer(&queries[i][0])),
			C.int(config.VectorDim),
			C.int(config.K),
		)

		if searchResult.count == 0 {
			log.Printf("Query %d returned no results", i)
			continue
		}

		// Free search results
		C.torch_search_result_free(&searchResult)
	}
	
	searchTime := time.Since(searchStart)
	queryRate := float64(config.QueryCount) / searchTime.Seconds()
	avgSearchTime := searchTime.Seconds() * 1000 / float64(config.QueryCount) // ms

	// Get memory stats
	stats := C.torch_indexer_get_stats(handle)
	memoryUsage := float64(stats.gpu_memory_mb)
	usingGPU := memoryUsage > 0

	if memoryUsage == 0 {
		// Estimate CPU memory usage
		vectorSize := config.VectorDim * 1 // int8
		totalVectors := config.IndexSize
		memoryUsage = float64(totalVectors*vectorSize) / (1024 * 1024) // MB
	}

	totalTime := time.Since(totalStart)

	return GPUBenchmarkResult{
		Config:        config,
		TrainTime:     trainTime,
		IndexTime:     indexTime,
		SearchTime:    searchTime,
		IndexRate:     indexRate,
		QueryRate:     queryRate,
		AvgSearchTime: avgSearchTime,
		MemoryUsage:   memoryUsage,
		TotalTime:     totalTime,
		UsingGPU:      usingGPU,
	}
}

func generateVectors(count, dim int) []int8 {
	data := make([]int8, count*dim)
	for i := 0; i < len(data); i++ {
		data[i] = int8(rand.Intn(256) - 128)
	}
	return data
}