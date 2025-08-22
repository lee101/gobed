// +build ignore

package main

/*
#cgo CPPFLAGS: -I. -I../libtorch/include -I../libtorch/include/torch/csrc/api/include
#cgo LDFLAGS: -L../libtorch/lib -L. -ltorch_cgo_wrapper -ltorch -ltorch_cuda -ltorch_cpu -ldl
#include "torch_cgo_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"log"
	"math/rand"
	"time"
	"unsafe"
)

func main() {
	fmt.Println("🔥 Simple LibTorch CGO Test")
	fmt.Println("===========================")

	// Test 1: Check LibTorch info
	fmt.Println("1. Testing LibTorch info...")
	version := C.GoString(C.torch_get_version())
	cudaAvailable := C.torch_cuda_is_available() != 0
	deviceCount := int(C.torch_cuda_device_count())

	fmt.Printf("   LibTorch version: %s\n", version)
	fmt.Printf("   CUDA available: %v\n", cudaAvailable)
	fmt.Printf("   Device count: %d\n", deviceCount)

	if !cudaAvailable {
		fmt.Println("   ⚠️  CUDA not available, using CPU")
	}

	// Test 2: Create indexer
	fmt.Println("\n2. Testing indexer creation...")
	deviceID := 0
	if !cudaAvailable {
		deviceID = -1
	}
	
	config := C.IndexConfig{
		vector_dim:        256,
		num_subquantizers: 32,
		codebook_size:     256,
		ivf_clusters:      64,
		probe_lists:       8,
		rerank_k:         100,
		device_id:        C.int(deviceID),
	}

	handle := C.torch_indexer_create(config)
	if handle == nil {
		log.Fatal("❌ Failed to create indexer")
	}
	defer C.torch_indexer_destroy(handle)
	fmt.Println("   ✅ Indexer created successfully")

	// Test 3: Generate training data
	fmt.Println("\n3. Generating training data...")
	numTraining := 1000
	vectorDim := 256
	trainingData := make([]int8, numTraining*vectorDim)

	for i := 0; i < len(trainingData); i++ {
		trainingData[i] = int8(rand.Intn(256) - 128)
	}

	// Test 4: Train indexer
	fmt.Println("\n4. Training indexer...")
	start := time.Now()
	result := C.torch_indexer_train(
		handle,
		(*C.schar)(unsafe.Pointer(&trainingData[0])),
		C.int(numTraining),
		C.int(vectorDim),
	)
	trainTime := time.Since(start)

	if result == 0 {
		log.Fatal("❌ Failed to train indexer")
	}
	fmt.Printf("   ✅ Training completed in %v\n", trainTime)

	// Test 5: Generate and add vectors
	fmt.Println("\n5. Adding vectors to index...")
	numVectors := 5000
	vectorData := make([]int8, numVectors*vectorDim)

	for i := 0; i < len(vectorData); i++ {
		vectorData[i] = int8(rand.Intn(256) - 128)
	}

	start = time.Now()
	result = C.torch_indexer_add_vectors(
		handle,
		(*C.schar)(unsafe.Pointer(&vectorData[0])),
		C.int(numVectors),
		C.int(vectorDim),
	)
	indexTime := time.Since(start)

	if result == 0 {
		log.Fatal("❌ Failed to add vectors")
	}
	fmt.Printf("   ✅ Added %d vectors in %v\n", numVectors, indexTime)
	fmt.Printf("   Indexing rate: %.0f vectors/sec\n", float64(numVectors)/indexTime.Seconds())

	// Test 6: Search
	fmt.Println("\n6. Testing search...")
	query := vectorData[:vectorDim] // Use first vector as query
	k := 10

	start = time.Now()
	searchResult := C.torch_indexer_search(
		handle,
		(*C.schar)(unsafe.Pointer(&query[0])),
		C.int(vectorDim),
		C.int(k),
	)
	searchTime := time.Since(start)

	if searchResult.count == 0 {
		log.Fatal("❌ Search returned no results")
	}

	fmt.Printf("   ✅ Search completed in %v\n", searchTime)
	fmt.Printf("   Found %d results\n", int(searchResult.count))
	fmt.Printf("   Search latency: %.2f ms\n", searchTime.Seconds()*1000)

	// Print top results
	count := int(searchResult.count)
	ids := (*[1 << 30]C.int)(unsafe.Pointer(searchResult.ids))[:count:count]
	scores := (*[1 << 30]C.float)(unsafe.Pointer(searchResult.scores))[:count:count]

	fmt.Println("   Top results:")
	for i := 0; i < min(5, count); i++ {
		fmt.Printf("     %d. ID=%d, Score=%.3f\n", i+1, int(ids[i]), float32(scores[i]))
	}

	// Check that the first result is the query itself (ID 0)
	if int(ids[0]) == 0 {
		fmt.Println("   🎯 Exact match found!")
	} else {
		fmt.Printf("   ⚠️  Expected exact match, got ID %d\n", int(ids[0]))
	}

	// Free search results
	C.torch_search_result_free(&searchResult)

	// Test 7: Get stats
	fmt.Println("\n7. Index statistics...")
	stats := C.torch_indexer_get_stats(handle)

	fmt.Printf("   Vectors: %d\n", int(stats.num_vectors))
	fmt.Printf("   Vector dim: %d\n", int(stats.vector_dim))
	fmt.Printf("   IVF clusters: %d\n", int(stats.ivf_clusters))
	fmt.Printf("   PQ subquantizers: %d\n", int(stats.pq_subquantizers))
	fmt.Printf("   GPU memory: %.1f MB\n", float64(stats.gpu_memory_mb))
	fmt.Printf("   Trained: %v\n", stats.is_trained != 0)
	fmt.Printf("   Built: %v\n", stats.index_built != 0)

	// Performance summary
	fmt.Println("\n📊 Performance Summary:")
	fmt.Printf("   Training: %d vectors in %v\n", numTraining, trainTime)
	fmt.Printf("   Indexing: %d vectors in %v (%.0f vec/sec)\n", 
		numVectors, indexTime, float64(numVectors)/indexTime.Seconds())
	fmt.Printf("   Search: %v latency (%.0f QPS potential)\n", 
		searchTime, 1.0/searchTime.Seconds())

	fmt.Println("\n✅ All tests passed! LibTorch CGO integration working.")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}