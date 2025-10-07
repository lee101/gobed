// +build cagra

package gobed

/*
#cgo CFLAGS: -I./
#cgo LDFLAGS: -L./ -lcagra_wrapper -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuvs -ldl -lstdc++ -lm
#include "cagra_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// CAGRAIndex provides NVIDIA cuVS CAGRA-based ultra-fast approximate nearest neighbor search
// Targets sub-millisecond search latency for million+ vectors
type CAGRAIndex struct {
	handle       *C.cagra_wrapper_t
	config       CAGRAConfig
	mutex        sync.RWMutex

	// State
	isBuilt      bool
	numVectors   int
	vectorDim    int

	// Performance tracking
	buildTime    time.Duration
	avgSearchTime time.Duration
	searchCount  int64

	// Caching
	cachePath    string
	cacheEnabled bool
}

// CAGRAConfig configures CAGRA for optimal performance
type CAGRAConfig struct {
	MaxVectors    int     // Maximum vectors to support
	VectorDim     int     // Vector dimension (512 for gobed)
	GraphDegree   int     // Graph connectivity (32-128, higher = better quality)
	MaxIterations int     // Search iterations (32-128, higher = better recall)
	CachePath     string  // Path for persistent index storage

	// Performance targets
	TargetLatencyUs  int  // Target search latency in microseconds (1000 = 1ms)
	TargetRecall     float32 // Target recall (0.9-0.99)
}

// DefaultCAGRAConfig returns configuration optimized for ultra-fast search
func DefaultCAGRAConfig() CAGRAConfig {
	config := CAGRAConfig{
		MaxVectors:      1000000, // Support 1M vectors
		VectorDim:       512,     // gobed dimension
		GraphDegree:     64,      // Balance speed/quality
		MaxIterations:   64,      // Balance speed/recall
		TargetLatencyUs: 1000,    // <1ms target
		TargetRecall:    0.95,    // 95% recall target
	}
	config.CachePath = BuildCAGRACachePath("default", config.VectorDim, config.GraphDegree, config.MaxVectors)
	return config
}

// FastCAGRAConfig returns configuration optimized for speed over quality
func FastCAGRAConfig() CAGRAConfig {
	config := DefaultCAGRAConfig()
	config.GraphDegree = 32        // Faster build/search
	config.MaxIterations = 32      // Faster search
	config.TargetLatencyUs = 500   // <0.5ms target
	config.TargetRecall = 0.90     // 90% recall
	config.CachePath = BuildCAGRACachePath("fast", config.VectorDim, config.GraphDegree, config.MaxVectors)
	return config
}

// QualityCAGRAConfig returns configuration optimized for quality over speed
func QualityCAGRAConfig() CAGRAConfig {
	config := DefaultCAGRAConfig()
	config.GraphDegree = 128       // Better quality
	config.MaxIterations = 128     // Better recall
	config.TargetLatencyUs = 2000  // <2ms target
	config.TargetRecall = 0.99     // 99% recall
	config.CachePath = BuildCAGRACachePath("quality", config.VectorDim, config.GraphDegree, config.MaxVectors)
	return config
}

// NewCAGRAIndex creates a new CAGRA-based index
func NewCAGRAIndex(config CAGRAConfig) (*CAGRAIndex, error) {
	// Check if cuVS/CAGRA is available
	if !isCAGRAAvailable() {
		return nil, fmt.Errorf("CAGRA/cuVS not available - install cuVS library")
	}

	// Create CAGRA wrapper
	handle := C.cagra_create(C.int(config.MaxVectors), C.int(config.VectorDim))
	if handle == nil {
		return nil, fmt.Errorf("failed to create CAGRA index")
	}

	index := &CAGRAIndex{
		handle:       handle,
		config:       config,
		vectorDim:    config.VectorDim,
		cacheEnabled: config.CachePath != "",
		cachePath:    config.CachePath,
	}

	// Set finalizer
	runtime.SetFinalizer(index, (*CAGRAIndex).destroy)

	log.Printf("🚀 CAGRA Index created: max_vectors=%d, dim=%d, graph_degree=%d",
		config.MaxVectors, config.VectorDim, config.GraphDegree)

	return index, nil
}

// BuildIndex builds CAGRA index from int8 quantized vectors
func (idx *CAGRAIndex) BuildIndex(vectors []simd.Vec512, scales []float32) error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	n := len(vectors)
	if n == 0 {
		return fmt.Errorf("no vectors to index")
	}

	if len(scales) != n {
		return fmt.Errorf("scales count mismatch: %d vectors, %d scales", n, len(scales))
	}

	log.Printf("🔨 Building CAGRA index: %d vectors", n)
	start := time.Now()

	// Flatten vectors for C API
	flatVectors := make([]int8, n*idx.vectorDim)
	for i, vec := range vectors {
		for j := 0; j < idx.vectorDim; j++ {
			flatVectors[i*idx.vectorDim+j] = vec[j]
		}
	}

	// Build index
	result := C.cagra_build_index(
		idx.handle,
		(*C.int8_t)(unsafe.Pointer(&flatVectors[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int(n),
	)

	if result != 0 {
		return fmt.Errorf("CAGRA index build failed: error code %d", result)
	}

	idx.buildTime = time.Since(start)
	idx.isBuilt = true
	idx.numVectors = n

	buildThroughput := float64(n) / idx.buildTime.Seconds()

	log.Printf("✅ CAGRA index built: %v (%.0f vectors/sec)", idx.buildTime, buildThroughput)

	// Save to cache if enabled
	if idx.cacheEnabled {
		go idx.saveToCache()
	}

	return nil
}

// Search performs ultra-fast k-nearest neighbor search
func (idx *CAGRAIndex) Search(query simd.Vec512, queryScale float32, k int) ([]SearchResult, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	if !idx.isBuilt {
		return nil, fmt.Errorf("index not built")
	}

	start := time.Now()

	// Allocate result buffers
	neighbors := make([]uint32, k)
	distances := make([]float32, k)

	// Perform search
	result := C.cagra_search(
		idx.handle,
		(*C.int8_t)(unsafe.Pointer(&query[0])),
		C.float(queryScale),
		C.int(k),
		(*C.uint32_t)(unsafe.Pointer(&neighbors[0])),
		(*C.float)(unsafe.Pointer(&distances[0])),
	)

	if result != 0 {
		return nil, fmt.Errorf("CAGRA search failed")
	}

	searchTime := time.Since(start)

	// Update performance tracking
	idx.searchCount++
	if idx.searchCount == 1 {
		idx.avgSearchTime = searchTime
	} else {
		// Exponential moving average
		alpha := 0.1
		idx.avgSearchTime = time.Duration(
			float64(idx.avgSearchTime)*(1-alpha) + float64(searchTime)*alpha,
		)
	}

	// Convert to SearchResult format
	results := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		results[i] = SearchResult{
			ID:         int(neighbors[i]),
			Similarity: distances[i],
		}
	}

	// Log performance for first few searches
	if idx.searchCount <= 5 {
		log.Printf("🔍 CAGRA search #%d: %.3fms (target: %.1fms)",
			idx.searchCount,
			float64(searchTime.Microseconds())/1000.0,
			float64(idx.config.TargetLatencyUs)/1000.0)
	}

	return results, nil
}

// SearchBatch performs batch search for maximum throughput
func (idx *CAGRAIndex) SearchBatch(queries []simd.Vec512, queryScales []float32, k int) ([][]SearchResult, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	if !idx.isBuilt {
		return nil, fmt.Errorf("index not built")
	}

	nQueries := len(queries)
	if nQueries == 0 {
		return [][]SearchResult{}, nil
	}

	if len(queryScales) != nQueries {
		return nil, fmt.Errorf("query scales count mismatch")
	}

	start := time.Now()

	// Flatten queries
	flatQueries := make([]int8, nQueries*idx.vectorDim)
	for i, query := range queries {
		for j := 0; j < idx.vectorDim; j++ {
			flatQueries[i*idx.vectorDim+j] = query[j]
		}
	}

	// Allocate result buffers
	neighbors := make([]uint32, nQueries*k)
	distances := make([]float32, nQueries*k)

	// Perform batch search
	result := C.cagra_search_batch(
		idx.handle,
		(*C.int8_t)(unsafe.Pointer(&flatQueries[0])),
		(*C.float)(unsafe.Pointer(&queryScales[0])),
		C.int(nQueries),
		C.int(k),
		(*C.uint32_t)(unsafe.Pointer(&neighbors[0])),
		(*C.float)(unsafe.Pointer(&distances[0])),
	)

	if result != 0 {
		return nil, fmt.Errorf("CAGRA batch search failed")
	}

	batchTime := time.Since(start)
	avgPerQuery := batchTime / time.Duration(nQueries)
	qps := float64(nQueries) / batchTime.Seconds()

	log.Printf("📊 CAGRA batch: %d queries in %v (%.2fms/query, %.0f QPS)",
		nQueries, batchTime, float64(avgPerQuery.Microseconds())/1000.0, qps)

	// Convert results
	results := make([][]SearchResult, nQueries)
	for i := 0; i < nQueries; i++ {
		results[i] = make([]SearchResult, k)
		for j := 0; j < k; j++ {
			idx := i*k + j
			results[i][j] = SearchResult{
				ID:         int(neighbors[idx]),
				Similarity: distances[idx],
			}
		}
	}

	return results, nil
}

// LoadFromCache loads pre-built index from cache
func (idx *CAGRAIndex) LoadFromCache() error {
	if !idx.cacheEnabled {
		return fmt.Errorf("caching not enabled")
	}

	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	log.Printf("📂 Loading CAGRA index from cache: %s", idx.cachePath)
	start := time.Now()

	cPath := C.CString(idx.cachePath)
	defer C.free(unsafe.Pointer(cPath))

	result := C.cagra_deserialize_index(idx.handle, cPath)
	if result != 0 {
		return fmt.Errorf("failed to load CAGRA index from cache")
	}

	loadTime := time.Since(start)
	idx.isBuilt = true

	log.Printf("✅ CAGRA index loaded from cache in %v", loadTime)

	return nil
}

// saveToCache saves index to cache asynchronously
func (idx *CAGRAIndex) saveToCache() {
	if !idx.cacheEnabled || !idx.isBuilt {
		return
	}

	log.Printf("💾 Saving CAGRA index to cache: %s", idx.cachePath)

	cPath := C.CString(idx.cachePath)
	defer C.free(unsafe.Pointer(cPath))

	result := C.cagra_serialize_index(idx.handle, cPath)
	if result != 0 {
		log.Printf("⚠️  Failed to save CAGRA index to cache")
		return
	}

	log.Printf("✅ CAGRA index saved to cache")
}

// GetStats returns performance statistics
func (idx *CAGRAIndex) GetStats() CAGRAStats {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	var cStats C.cagra_stats_t
	C.cagra_get_stats(idx.handle, &cStats)

	return CAGRAStats{
		NumVectors:       idx.numVectors,
		VectorDim:        idx.vectorDim,
		BuildTimeMs:      float64(cStats.build_time_ms),
		AvgSearchTimeMs:  float64(idx.avgSearchTime.Microseconds()) / 1000.0,
		SearchCount:      idx.searchCount,
		MemoryBytes:      int64(cStats.index_memory_bytes),
		IsBuilt:          idx.isBuilt,
		CacheEnabled:     idx.cacheEnabled,
	}
}

// CAGRAStats contains performance statistics
type CAGRAStats struct {
	NumVectors       int
	VectorDim        int
	BuildTimeMs      float64
	AvgSearchTimeMs  float64
	SearchCount      int64
	MemoryBytes      int64
	IsBuilt          bool
	CacheEnabled     bool
}

// destroy cleans up CAGRA resources
func (idx *CAGRAIndex) destroy() {
	if idx.handle != nil {
		C.cagra_destroy(idx.handle)
		idx.handle = nil
	}
}

// Close explicitly releases resources
func (idx *CAGRAIndex) Close() {
	idx.destroy()
}

// BuildCAGRACachePath generates CAGRA cache file path with dev/prod/test separation
func BuildCAGRACachePath(namespace string, vectorDim, graphDegree, count int) string {
	baseDir := "/mnt/fast/tmp/netwrck_gobed_indexes"

	if os.Getenv("DEV") == "true" || os.Getenv("DEBUG") == "true" {
		baseDir += "_dev"
	}

	if os.Getenv("TESTING") == "true" {
		baseDir = "/mnt/fast/tmp/test_indexes"
	}

	if err := os.MkdirAll(baseDir, 0755); err != nil {
		log.Printf("Warning: Failed to create CAGRA cache directory %s: %v", baseDir, err)
	}

	filename := fmt.Sprintf("%s_cagra_%d_%d_%d.cagrabin", namespace, vectorDim, graphDegree, count)
	return filepath.Join(baseDir, filename)
}

// isCAGRAAvailable checks if cuVS/CAGRA is available
func isCAGRAAvailable() bool {
	// This would check for cuVS library availability
	// For now, assume it's available if built with cagra tag
	return true
}

// SearchResult represents a single search result
type SearchResult struct {
	ID         int
	Similarity float32
}