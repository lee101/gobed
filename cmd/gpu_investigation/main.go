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
	"runtime"
	"strings"
	"time"
	"unsafe"
)

func main() {
	fmt.Println(" Deep GPU Investigation - LibTorch CUDA Analysis")
	fmt.Println(strings.Repeat("=", 60))

	// Basic system info
	fmt.Printf(" System Information:\n")
	fmt.Printf("   Go: %s\n", runtime.Version())
	fmt.Printf("   GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	// LibTorch info
	version := C.GoString(C.torch_get_version())
	fmt.Printf("   LibTorch: %s\n", version)

	// CUDA investigation
	fmt.Printf("\n CUDA Investigation:\n")
	cudaAvailable := C.torch_cuda_is_available() != 0
	deviceCount := int(C.torch_cuda_device_count())

	fmt.Printf("   CUDA Available: %v\n", cudaAvailable)
	fmt.Printf("   Device Count: %d\n", deviceCount)

	if !cudaAvailable {
		fmt.Println(" CUDA not available - investigating why...")
		// Continue anyway to test our manual CUDA operations
	}

	// Test different configurations
	testConfigurations := []struct {
		name     string
		deviceID int
		vecDim   int
		vecCount int
	}{
		{"Small GPU Test", 0, 128, 1000},
		{"Medium GPU Test", 0, 384, 5000},
		{"Large GPU Test", 0, 512, 10000},
		{"CPU Fallback", -1, 512, 10000},
	}

	for i, config := range testConfigurations {
		fmt.Printf("\n🧪 Test %d: %s\n", i+1, config.name)
		fmt.Println(strings.Repeat("-", 40))

		testConfiguration(config.deviceID, config.vecDim, config.vecCount)
	}

	fmt.Println("\n Investigation complete!")
}

func testConfiguration(deviceID, vecDim, vecCount int) {
	// Create indexer
	cConfig := C.IndexConfig{
		vector_dim:        C.int(vecDim),
		num_subquantizers: C.int(32),
		codebook_size:     C.int(256),
		ivf_clusters:      C.int(128),
		probe_lists:       C.int(8),
		rerank_k:          C.int(50),
		device_id:         C.int(deviceID),
	}

	fmt.Printf("    Config: dim=%d, vectors=%d, device=%d\n", vecDim, vecCount, deviceID)

	handle := C.torch_indexer_create(cConfig)
	if handle == nil {
		fmt.Println("    Failed to create indexer")
		return
	}
	defer C.torch_indexer_destroy(handle)

	// Generate test data
	fmt.Printf("   🔄 Generating %d test vectors...\n", vecCount)
	testData := generateTestVectors(vecCount, vecDim)

	// Training phase
	trainingSize := min(vecCount/10, 1000) // Use 10% for training, max 1000
	fmt.Printf("   🎓 Training with %d vectors...\n", trainingSize)

	trainStart := time.Now()
	result := C.torch_indexer_train(
		handle,
		(*C.schar)(unsafe.Pointer(&testData[0])),
		C.int(trainingSize),
		C.int(vecDim),
	)
	trainTime := time.Since(trainStart)

	if result == 0 {
		fmt.Println("    Training failed")
		return
	}
	fmt.Printf("    Training: %v\n", trainTime)

	// Indexing phase
	fmt.Printf("   📚 Indexing %d vectors...\n", vecCount)

	indexStart := time.Now()
	result = C.torch_indexer_add_vectors(
		handle,
		(*C.schar)(unsafe.Pointer(&testData[0])),
		C.int(vecCount),
		C.int(vecDim),
	)
	indexTime := time.Since(indexStart)

	if result == 0 {
		fmt.Println("    Indexing failed")
		return
	}

	indexRate := float64(vecCount) / indexTime.Seconds()
	fmt.Printf("    Indexing: %v (%.0f vec/sec)\n", indexTime, indexRate)

	// Search phase
	fmt.Printf("    Testing search performance...\n")

	numQueries := 100
	queryStart := time.Now()

	for i := 0; i < numQueries; i++ {
		// Use random vector from our dataset as query
		queryIdx := (i * 17) % vecCount // Deterministic but varied
		queryVector := testData[queryIdx*vecDim : (queryIdx+1)*vecDim]

		searchResult := C.torch_indexer_search(
			handle,
			(*C.schar)(unsafe.Pointer(&queryVector[0])),
			C.int(vecDim),
			C.int(10),
		)

		if searchResult.count > 0 {
			C.torch_search_result_free(&searchResult)
		}
	}

	searchTime := time.Since(queryStart)
	avgSearchTime := searchTime.Nanoseconds() / int64(numQueries) / 1000 // microseconds
	qps := float64(numQueries) / searchTime.Seconds()

	fmt.Printf("    Search: %d queries in %v (%.0f μs/query, %.0f QPS)\n",
		numQueries, searchTime, float64(avgSearchTime), qps)

	// Get detailed stats
	stats := C.torch_indexer_get_stats(handle)
	fmt.Printf("    Memory: %.1f MB GPU, %d vectors indexed\n",
		float64(stats.gpu_memory_mb), int(stats.num_vectors))
	fmt.Printf("    GPU Usage: %v\n", stats.gpu_memory_mb > 0)
}

func generateTestVectors(count, dim int) []int8 {
	data := make([]int8, count*dim)

	// Generate more realistic patterns
	for i := 0; i < count; i++ {
		// Create vector with some structure
		basePattern := i % 10 // Create 10 different base patterns

		for j := 0; j < dim; j++ {
			// Mix deterministic pattern with some randomness
			val := (basePattern*17 + j*3 + (i/10)*7) % 256
			data[i*dim+j] = int8(val - 128)
		}
	}

	return data
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
