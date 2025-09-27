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

type BenchmarkConfig struct {
	VectorDim    int
	TrainingSize int
	IndexSize    int
	QueryCount   int
	K            int
}

type BenchmarkResult struct {
	Config         BenchmarkConfig
	TrainTime      time.Duration
	IndexTime      time.Duration
	SearchTime     time.Duration
	IndexRate      float64 // vectors/sec
	QueryRate      float64 // queries/sec
	MemoryUsage    float64 // MB
	TotalTime      time.Duration
}

func main() {
	fmt.Println(" LibTorch Performance Benchmark")
	fmt.Println("==================================")

	// Check system info
	version := C.GoString(C.torch_get_version())
	cudaAvailable := C.torch_cuda_is_available() != 0
	deviceCount := int(C.torch_cuda_device_count())

	fmt.Printf(" System Information:\n")
	fmt.Printf("   LibTorch version: %s\n", version)
	fmt.Printf("   CUDA available: %v\n", cudaAvailable)
	fmt.Printf("   Device count: %d\n", deviceCount)
	fmt.Printf("   Go version: %s\n", runtime.Version())
	fmt.Printf("   CPU cores: %d\n", runtime.NumCPU())

	// Benchmark configurations
	configs := []BenchmarkConfig{
		{VectorDim: 128, TrainingSize: 1000, IndexSize: 10000, QueryCount: 100, K: 10},
		{VectorDim: 256, TrainingSize: 2000, IndexSize: 25000, QueryCount: 200, K: 10},
		{VectorDim: 512, TrainingSize: 5000, IndexSize: 50000, QueryCount: 500, K: 10},
		{VectorDim: 768, TrainingSize: 5000, IndexSize: 75000, QueryCount: 1000, K: 10},
		{VectorDim: 1024, TrainingSize: 5000, IndexSize: 100000, QueryCount: 1000, K: 10},
	}

	fmt.Printf("\n Running Performance Benchmarks...\n")
	fmt.Printf("\n%-8s %-10s %-10s %-8s %-12s %-10s %-10s %-8s\n",
		"Dim", "Index", "Queries", "K", "Index/sec", "Query/ms", "QPS", "Memory")
	fmt.Println("--------------------------------------------------------------------------------")

	var results []BenchmarkResult

	for _, config := range configs {
		result := runBenchmark(config)
		results = append(results, result)

		fmt.Printf("%-8d %-10d %-10d %-8d %-12.0f %-10.2f %-10.0f %-8.1f\n",
			config.VectorDim,
			config.IndexSize,
			config.QueryCount,
			config.K,
			result.IndexRate,
			result.SearchTime.Seconds()*1000/float64(config.QueryCount),
			result.QueryRate,
			result.MemoryUsage)
	}

	// Print summary
	fmt.Println("\n Performance Summary:")
	for i, result := range results {
		config := result.Config
		fmt.Printf("\n%d. %dD vectors (%d indexed):\n", i+1, config.VectorDim, config.IndexSize)
		fmt.Printf("   Training: %v (%d vectors)\n", result.TrainTime, config.TrainingSize)
		fmt.Printf("   Indexing: %v (%.0f vectors/sec)\n", result.IndexTime, result.IndexRate)
		fmt.Printf("   Search: %v (%d queries, %.0f QPS)\n", result.SearchTime, config.QueryCount, result.QueryRate)
		fmt.Printf("   Memory: %.1f MB\n", result.MemoryUsage)
		fmt.Printf("   Total: %v\n", result.TotalTime)
	}

	// Find best performance
	bestIndexRate := 0.0
	bestQueryRate := 0.0
	bestConfig := BenchmarkConfig{}

	for _, result := range results {
		if result.IndexRate > bestIndexRate {
			bestIndexRate = result.IndexRate
			bestConfig = result.Config
		}
		if result.QueryRate > bestQueryRate {
			bestQueryRate = result.QueryRate
		}
	}

	fmt.Printf("\n Best Performance:\n")
	fmt.Printf("   Indexing: %.0f vectors/sec (%dD)\n", bestIndexRate, bestConfig.VectorDim)
	fmt.Printf("   Search: %.0f QPS\n", bestQueryRate)

	// Scalability estimates
	fmt.Println("\n Scalability Estimates:")
	fmt.Println("   With current performance:")

	scales := []int{1000000, 10000000, 100000000}
	for _, scale := range scales {
		indexTime := float64(scale) / bestIndexRate
		fmt.Printf("   %8d vectors: ~%.1f seconds to index\n", scale, indexTime)
	}

	fmt.Println("\n Benchmark completed!")
}

func runBenchmark(config BenchmarkConfig) BenchmarkResult {
	totalStart := time.Now()

	// Create indexer
	cConfig := C.IndexConfig{
		vector_dim:        C.int(config.VectorDim),
		num_subquantizers: C.int(64),
		codebook_size:     C.int(256),
		ivf_clusters:      C.int(1024),
		probe_lists:       C.int(32),
		rerank_k:         C.int(200),
		device_id:        C.int(0),
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

	// Generate queries
	queries := make([][]int8, config.QueryCount)
	for i := 0; i < config.QueryCount; i++ {
		// Extract vector from flat array
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

	// Get memory stats
	stats := C.torch_indexer_get_stats(handle)
	memoryUsage := float64(stats.gpu_memory_mb)
	if memoryUsage == 0 {
		// Estimate CPU memory usage
		vectorSize := config.VectorDim * 1 // int8
		totalVectors := config.IndexSize
		memoryUsage = float64(totalVectors*vectorSize) / (1024 * 1024) // MB
	}

	totalTime := time.Since(totalStart)

	return BenchmarkResult{
		Config:      config,
		TrainTime:   trainTime,
		IndexTime:   indexTime,
		SearchTime:  searchTime,
		IndexRate:   indexRate,
		QueryRate:   queryRate,
		MemoryUsage: memoryUsage,
		TotalTime:   totalTime,
	}
}

func generateVectors(count, dim int) []int8 {
	data := make([]int8, count*dim)
	for i := 0; i < len(data); i++ {
		data[i] = int8(rand.Intn(256) - 128)
	}
	return data
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}