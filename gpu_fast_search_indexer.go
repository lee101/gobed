// +build gpu

package gobed

/*
#cgo CFLAGS: -I./gpu
#cgo LDFLAGS: -L./gpu -lgpu_fast_search -L/usr/local/cuda-12.9/lib64 -lcudart -lcublas -Wl,-rpath,./gpu -Wl,-rpath,/usr/local/cuda-12.9/lib64
#include <stdlib.h>

// Ultra-fast GPU search optimized for 1ms inference
typedef struct {
    void* handle;
    void* search_cache;    // Pre-computed search structures
    void* gpu_index;       // Index optimized for search
    int nlist;
    int vector_dim;
    int total_vectors;
} gpu_fast_search_t;

// Index persistence and loading
gpu_fast_search_t* gpu_fast_search_create(int nlist, int nprobe, int vector_dim);
int gpu_fast_search_load_from_file(gpu_fast_search_t* searcher, const char* index_file);
int gpu_fast_search_save_to_file(gpu_fast_search_t* searcher, const char* index_file);
void gpu_fast_search_destroy(gpu_fast_search_t* searcher);

// Ultra-fast bulk building (10-20s target)
int gpu_fast_search_build_index(gpu_fast_search_t* searcher, const signed char* vectors, 
                                const float* scales, const int* ids, int num_vectors);

// Pre-computation for 1ms search
int gpu_fast_search_precompute_structures(gpu_fast_search_t* searcher);
int gpu_fast_search_warm_cache(gpu_fast_search_t* searcher, const signed char* common_queries, int num_queries);

// Ultra-fast search operations (<1ms target)
int gpu_fast_search_single(gpu_fast_search_t* searcher, const signed char* query, float scale,
                          int k, int* result_ids, float* result_scores);
                          
int gpu_fast_search_batch(gpu_fast_search_t* searcher, const signed char* queries, const float* scales,
                         int num_queries, int k, int* result_ids, float* result_scores);

// Search optimization
int gpu_fast_search_set_search_params(gpu_fast_search_t* searcher, int nprobe, int use_cache);
int gpu_fast_search_optimize_for_latency(gpu_fast_search_t* searcher);

// Performance monitoring
double gpu_fast_search_get_avg_latency_us(gpu_fast_search_t* searcher);
int gpu_fast_search_get_cache_hit_rate(gpu_fast_search_t* searcher);
*/
import "C"
import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// GPUFastSearchIndexer optimized for 1ms inference-time search
type GPUFastSearchIndexer struct {
	handle         *C.gpu_fast_search_t
	config         FastSearchConfig
	mutex          sync.RWMutex
	
	// Search performance tracking
	searchCount    int64
	totalLatencyUs int64
	
	// Index state
	isBuilt        bool
	isPersisted    bool
	indexPath      string
	
	// Search cache
	queryCache     *QueryCache
	warmupQueries  [][]int8
}

// FastSearchConfig optimizes for ultra-fast search
type FastSearchConfig struct {
	NList           int     // Fewer clusters for faster search
	NProbe          int     // Optimized probe count for 1ms target
	VectorDim       int
	MaxLatencyUs    int     // Target max latency in microseconds (1000 = 1ms)
	EnableCache     bool    // Query result caching
	WarmupQueries   int     // Number of common queries to pre-cache
	IndexPath       string  // Path for persistent index storage
	
	// Search-optimized settings
	PrecomputeStructures bool // Pre-compute search acceleration structures
	OptimizeForLatency   bool // Prioritize latency over accuracy
}

// DefaultFastSearchConfig returns config optimized for 1ms search
func DefaultFastSearchConfig() FastSearchConfig {
	return FastSearchConfig{
		NList:               256,  // Fewer clusters = faster search
		NProbe:              8,    // Aggressive pruning for speed
		VectorDim:           512,
		MaxLatencyUs:        1000, // 1ms target
		EnableCache:         true,
		WarmupQueries:       1000,
		PrecomputeStructures: true,
		OptimizeForLatency:  true,
	}
}

// NewGPUFastSearchIndexer creates indexer optimized for search speed
func NewGPUFastSearchIndexer(config FastSearchConfig) (*GPUFastSearchIndexer, error) {
	if !IsCUDAAvailable() {
		return nil, fmt.Errorf("CUDA not available")
	}

	handle := C.gpu_fast_search_create(
		C.int(config.NList),
		C.int(config.NProbe),
		C.int(config.VectorDim),
	)
	if handle == nil {
		return nil, fmt.Errorf("failed to create fast search indexer")
	}

	indexer := &GPUFastSearchIndexer{
		handle: handle,
		config: config,
	}
	
	if config.EnableCache {
		indexer.queryCache = NewQueryCache(10000) // Cache 10K queries
	}
	
	runtime.SetFinalizer(indexer, (*GPUFastSearchIndexer).destroy)
	
	log.Printf("Fast Search Indexer created: %d clusters, %d probe, target %dμs", 
		config.NList, config.NProbe, config.MaxLatencyUs)
	
	return indexer, nil
}

// BuildIndexFast builds index optimized for fast search (10-20s target)
func (idx *GPUFastSearchIndexer) BuildIndexFast(vectors []int8, scales []float32, ids []int) error {
	if len(vectors) != len(scales)*idx.config.VectorDim {
		return fmt.Errorf("vector data size mismatch")
	}
	
	idx.mutex.Lock()
	defer idx.mutex.Unlock()
	
	log.Printf("Building fast search index: %d vectors", len(scales))
	start := time.Now()
	
	result := C.gpu_fast_search_build_index(
		idx.handle,
		(*C.schar)(unsafe.Pointer(&vectors[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		(*C.int)(unsafe.Pointer(&ids[0])),
		C.int(len(scales)),
	)
	
	if result == 0 {
		return fmt.Errorf("failed to build GPU index")
	}
	
	buildTime := time.Since(start)
	
	// Pre-compute search structures for ultra-fast search
	if idx.config.PrecomputeStructures {
		log.Printf("Pre-computing search acceleration structures...")
		preStart := time.Now()
		
		C.gpu_fast_search_precompute_structures(idx.handle)
		
		preTime := time.Since(preStart)
		log.Printf("Pre-computation completed in %v", preTime)
	}
	
	// Optimize for target latency
	if idx.config.OptimizeForLatency {
		C.gpu_fast_search_optimize_for_latency(idx.handle)
		C.gpu_fast_search_set_search_params(idx.handle, C.int(idx.config.NProbe), 1)
	}
	
	idx.isBuilt = true
	
	log.Printf("Index built successfully: %v total, target search: %dμs", 
		buildTime, idx.config.MaxLatencyUs)
	
	return nil
}

// LoadIndex loads pre-built index from disk (ultra-fast startup)
func (idx *GPUFastSearchIndexer) LoadIndex(indexPath string) error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()
	
	log.Printf("Loading pre-built index from %s", indexPath)
	start := time.Now()
	
	cPath := C.CString(indexPath)
	defer C.free(unsafe.Pointer(cPath))
	
	result := C.gpu_fast_search_load_from_file(idx.handle, cPath)
	if result == 0 {
		return fmt.Errorf("failed to load index from %s", indexPath)
	}
	
	// Optimize for fast search after loading
	if idx.config.OptimizeForLatency {
		C.gpu_fast_search_optimize_for_latency(idx.handle)
	}
	
	loadTime := time.Since(start)
	idx.isBuilt = true
	idx.isPersisted = true
	idx.indexPath = indexPath
	
	log.Printf("Index loaded successfully in %v", loadTime)
	return nil
}

// SaveIndex persists index to disk for fast loading
func (idx *GPUFastSearchIndexer) SaveIndex(indexPath string) error {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	
	if !idx.isBuilt {
		return fmt.Errorf("index not built yet")
	}
	
	// Ensure directory exists
	if err := ensureDir(filepath.Dir(indexPath)); err != nil {
		return fmt.Errorf("failed to create index directory: %w", err)
	}
	
	log.Printf("Saving index to %s", indexPath)
	start := time.Now()
	
	cPath := C.CString(indexPath)
	defer C.free(unsafe.Pointer(cPath))
	
	result := C.gpu_fast_search_save_to_file(idx.handle, cPath)
	if result == 0 {
		return fmt.Errorf("failed to save index to %s", indexPath)
	}
	
	saveTime := time.Since(start)
	idx.isPersisted = true
	idx.indexPath = indexPath
	
	log.Printf("Index saved successfully in %v", saveTime)
	return nil
}

// WarmupCache pre-loads common queries for sub-millisecond response
func (idx *GPUFastSearchIndexer) WarmupCache(commonQueries [][]int8) error {
	if len(commonQueries) == 0 {
		return nil
	}
	
	log.Printf("Warming up search cache with %d common queries...", len(commonQueries))
	
	// Flatten queries for C API
	flatQueries := make([]int8, len(commonQueries)*idx.config.VectorDim)
	for i, query := range commonQueries {
		if len(query) != idx.config.VectorDim {
			return fmt.Errorf("query %d dimension mismatch", i)
		}
		copy(flatQueries[i*idx.config.VectorDim:(i+1)*idx.config.VectorDim], query)
	}
	
	result := C.gpu_fast_search_warm_cache(
		idx.handle,
		(*C.schar)(unsafe.Pointer(&flatQueries[0])),
		C.int(len(commonQueries)),
	)
	
	if result == 0 {
		return fmt.Errorf("cache warmup failed")
	}
	
	idx.warmupQueries = commonQueries
	log.Printf("Cache warmed up successfully")
	return nil
}

// SearchSingle performs ultra-fast single query search (<1ms target)
func (idx *GPUFastSearchIndexer) SearchSingle(query []int8, scale float32, k int) ([]SearchResult, error) {
	if !idx.isBuilt {
		return nil, fmt.Errorf("index not built")
	}
	
	if len(query) != idx.config.VectorDim {
		return nil, fmt.Errorf("query dimension mismatch")
	}
	
	// Check cache first
	if idx.config.EnableCache && idx.queryCache != nil {
		if cached := idx.queryCache.Get(query, k); cached != nil {
			return cached, nil
		}
	}
	
	// Time the search for performance monitoring
	start := time.Now()
	
	// Allocate result buffers
	resultIDs := make([]int, k)
	resultScores := make([]float32, k)
	
	result := C.gpu_fast_search_single(
		idx.handle,
		(*C.schar)(unsafe.Pointer(&query[0])),
		C.float(scale),
		C.int(k),
		(*C.int)(unsafe.Pointer(&resultIDs[0])),
		(*C.float)(unsafe.Pointer(&resultScores[0])),
	)
	
	latencyUs := time.Since(start).Microseconds()
	
	// Update performance stats
	idx.mutex.Lock()
	idx.searchCount++
	idx.totalLatencyUs += latencyUs
	idx.mutex.Unlock()
	
	if result == 0 {
		return nil, fmt.Errorf("GPU search failed")
	}
	
	// Convert to SearchResult format
	results := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		results[i] = SearchResult{
			ID:         resultIDs[i],
			Similarity: resultScores[i],
		}
	}
	
	// Cache results
	if idx.config.EnableCache && idx.queryCache != nil {
		idx.queryCache.Set(query, k, results)
	}
	
	// Warn if target latency exceeded
	if latencyUs > int64(idx.config.MaxLatencyUs) {
		log.Printf("Warning: Search latency %dμs exceeded target %dμs", 
			latencyUs, idx.config.MaxLatencyUs)
	}
	
	return results, nil
}

// SearchBatch performs batch search with optimizations
func (idx *GPUFastSearchIndexer) SearchBatch(queries [][]int8, scales []float32, k int) ([][]SearchResult, error) {
	if !idx.isBuilt {
		return nil, fmt.Errorf("index not built")
	}
	
	numQueries := len(queries)
	if len(scales) != numQueries {
		return nil, fmt.Errorf("scales count mismatch")
	}
	
	// Flatten queries
	flatQueries := make([]int8, numQueries*idx.config.VectorDim)
	for i, query := range queries {
		if len(query) != idx.config.VectorDim {
			return nil, fmt.Errorf("query %d dimension mismatch", i)
		}
		copy(flatQueries[i*idx.config.VectorDim:(i+1)*idx.config.VectorDim], query)
	}
	
	start := time.Now()
	
	// Allocate result buffers
	resultIDs := make([]int, numQueries*k)
	resultScores := make([]float32, numQueries*k)
	
	result := C.gpu_fast_search_batch(
		idx.handle,
		(*C.schar)(unsafe.Pointer(&flatQueries[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int(numQueries),
		C.int(k),
		(*C.int)(unsafe.Pointer(&resultIDs[0])),
		(*C.float)(unsafe.Pointer(&resultScores[0])),
	)
	
	latencyUs := time.Since(start).Microseconds()
	avgLatencyUs := latencyUs / int64(numQueries)
	
	// Update performance stats
	idx.mutex.Lock()
	idx.searchCount += int64(numQueries)
	idx.totalLatencyUs += latencyUs
	idx.mutex.Unlock()
	
	if result == 0 {
		return nil, fmt.Errorf("GPU batch search failed")
	}
	
	// Convert to SearchResult format
	results := make([][]SearchResult, numQueries)
	for i := 0; i < numQueries; i++ {
		results[i] = make([]SearchResult, k)
		for j := 0; j < k; j++ {
			idx := i*k + j
			results[i][j] = SearchResult{
				ID:         resultIDs[idx],
				Similarity: resultScores[idx],
			}
		}
	}
	
	log.Printf("Batch search: %d queries, %dμs total, %dμs avg", 
		numQueries, latencyUs, avgLatencyUs)
	
	return results, nil
}

// GetSearchStats returns performance statistics
func (idx *GPUFastSearchIndexer) GetSearchStats() SearchStats {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	
	var avgLatencyUs float64
	if idx.searchCount > 0 {
		avgLatencyUs = float64(idx.totalLatencyUs) / float64(idx.searchCount)
	}
	
	// Get GPU-side stats
	cacheHitRate := int(C.gpu_fast_search_get_cache_hit_rate(idx.handle))
	gpuAvgLatencyUs := float64(C.gpu_fast_search_get_avg_latency_us(idx.handle))
	
	return SearchStats{
		TotalSearches:   idx.searchCount,
		AvgLatencyUs:    avgLatencyUs,
		GPULatencyUs:    gpuAvgLatencyUs,
		CacheHitRate:    float64(cacheHitRate) / 100.0,
		TargetLatencyUs: idx.config.MaxLatencyUs,
		TargetMet:       avgLatencyUs <= float64(idx.config.MaxLatencyUs),
	}
}

// OptimizeForInference performs final optimizations for production search
func (idx *GPUFastSearchIndexer) OptimizeForInference() error {
	if !idx.isBuilt {
		return fmt.Errorf("index not built")
	}
	
	log.Printf("Optimizing for inference performance...")
	
	// GPU-side optimizations
	C.gpu_fast_search_optimize_for_latency(idx.handle)
	
	// Set aggressive search parameters for speed
	C.gpu_fast_search_set_search_params(idx.handle, C.int(idx.config.NProbe), 1)
	
	log.Printf("Inference optimization completed")
	return nil
}

// CheckIndexExists checks if persistent index exists
func (idx *GPUFastSearchIndexer) CheckIndexExists(indexPath string) bool {
	_, err := os.Stat(indexPath)
	return err == nil
}

// destroy cleans up resources
func (idx *GPUFastSearchIndexer) destroy() {
	if idx.handle != nil {
		C.gpu_fast_search_destroy(idx.handle)
		idx.handle = nil
	}
}

// Close explicitly releases resources
func (idx *GPUFastSearchIndexer) Close() {
	idx.destroy()
}

// SearchStats contains search performance metrics
type SearchStats struct {
	TotalSearches   int64   `json:"total_searches"`
	AvgLatencyUs    float64 `json:"avg_latency_us"`
	GPULatencyUs    float64 `json:"gpu_latency_us"`
	CacheHitRate    float64 `json:"cache_hit_rate"`
	TargetLatencyUs int     `json:"target_latency_us"`
	TargetMet       bool    `json:"target_met"`
}

// QueryCache provides result caching for ultra-fast repeated queries
type QueryCache struct {
	cache map[string][]SearchResult
	mutex sync.RWMutex
	maxSize int
}

// NewQueryCache creates a new query cache
func NewQueryCache(maxSize int) *QueryCache {
	return &QueryCache{
		cache:   make(map[string][]SearchResult),
		maxSize: maxSize,
	}
}

// Get retrieves cached search results
func (qc *QueryCache) Get(query []int8, k int) []SearchResult {
	qc.mutex.RLock()
	defer qc.mutex.RUnlock()
	
	key := fmt.Sprintf("%v_%d", query, k)
	if results, exists := qc.cache[key]; exists {
		return results
	}
	return nil
}

// Set stores search results in cache
func (qc *QueryCache) Set(query []int8, k int, results []SearchResult) {
	qc.mutex.Lock()
	defer qc.mutex.Unlock()
	
	key := fmt.Sprintf("%v_%d", query, k)
	qc.cache[key] = results
	
	// Simple eviction if cache too large
	if len(qc.cache) > qc.maxSize {
		// Remove oldest entries (simplified)
		count := 0
		for k := range qc.cache {
			if count > qc.maxSize/10 { // Remove 10% of cache
				break
			}
			delete(qc.cache, k)
			count++
		}
	}
}

// Helper functions
func ensureDir(dir string) error {
	return os.MkdirAll(dir, 0755)
}