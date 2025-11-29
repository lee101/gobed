package search

import (
	"fmt"
	"sync"
	"time"

	"github.com/lee101/gobed/pkg/ann/flat"
	"github.com/lee101/gobed/pkg/ann/hnsw"
	"github.com/lee101/gobed/pkg/ann/ivf"
	"github.com/lee101/gobed/pkg/ann/pq"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// Engine is the main search engine combining IVF-HNSW-PQ with reranking
type Engine struct {
	// Configuration
	config Config

	// Components
	flatIndex  *flat.FlatIndex      // For small datasets
	ivfIndex   *ivf.IVFIndex        // IVF for partitioning
	hnswRouter *hnsw.HNSW           // HNSW for fast centroid routing
	pq         *pq.ProductQuantizer // Product quantizer
	pqCodes    [][]uint8            // PQ encoded vectors
	rawVectors []simd.Vec512        // Original int8 vectors for reranking
	scales     []float32            // Vector scales
	ids        []int                // External IDs

	// State
	trained bool
	size    int
	mu      sync.RWMutex

	// Object pools for reducing allocations
	candidatePool *sync.Pool
	resultPool    *sync.Pool
}

// Config holds engine configuration
type Config struct {
	// Index type threshold
	MaxFlatSize int // Use flat index below this size (default: 50000)

	// IVF parameters
	NList  int // Number of clusters (default: 4096)
	NProbe int // Number of clusters to search (default: 8)

	// PQ parameters
	M     int // Number of subquantizers (default: 64)
	NBits int // Bits per subquantizer (default: 8)

	// HNSW parameters
	HNSWEnabled bool // Use HNSW for centroid routing (default: true)
	HNSWM       int  // HNSW connections (default: 16)
	HNSWEfC     int  // HNSW construction parameter (default: 200)

	// Search parameters
	RerankSize  int  // Number of candidates to rerank (default: 128)
	UseParallel bool // Use parallel search (default: true)
}

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	return Config{
		MaxFlatSize: 1500, // Tuned default based on recent optimization runs
		NList:       4096,
		NProbe:      8,
		M:           64,
		NBits:       8,
		HNSWEnabled: true,
		HNSWM:       16,
		HNSWEfC:     200,
		RerankSize:  128,
		UseParallel: true,
	}
}

// NewEngine creates a new search engine
func NewEngine(config Config) *Engine {
	// Validate and set defaults
	if config.MaxFlatSize <= 0 {
		config.MaxFlatSize = 1500
	}
	if config.NList <= 0 {
		config.NList = 4096
	}
	if config.NProbe <= 0 {
		config.NProbe = 8
	}
	if config.M <= 0 {
		config.M = 64
	}
	if config.NBits <= 0 {
		config.NBits = 8
	}
	if config.RerankSize <= 0 {
		config.RerankSize = 128
	}

	// Pre-allocate with estimated capacity
	initialCap := config.MaxFlatSize
	if initialCap <= 0 {
		initialCap = 1500
	}

	e := &Engine{
		config:     config,
		flatIndex:  flat.NewFlatIndex(config.MaxFlatSize),
		rawVectors: make([]simd.Vec512, 0, initialCap),
		scales:     make([]float32, 0, initialCap),
		ids:        make([]int, 0, initialCap),
	}

	// Initialize object pools
	e.candidatePool = &sync.Pool{
		New: func() interface{} {
			slice := make([]int, 0, config.NProbe*100)
			return &slice
		},
	}
	e.resultPool = &sync.Pool{
		New: func() interface{} {
			slice := make([]SearchResult, 0, config.RerankSize)
			return &slice
		},
	}

	return e
}

// Train trains the index on sample data
func (e *Engine) Train(vectors []simd.Vec512, scales []float32) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	n := len(vectors)
	if n == 0 {
		return fmt.Errorf("no training data provided")
	}

	// For small datasets, no training needed
	if n <= e.config.MaxFlatSize {
		e.trained = true
		return nil
	}

	// Only print if actually training, not when loading from cache
	if !e.trained {
		fmt.Printf("Indexing %d vectors...\n", n)
	}
	start := time.Now()

	// Initialize IVF
	e.ivfIndex = ivf.NewIVFIndex(e.config.NList, e.config.NProbe)
	e.ivfIndex.Train(vectors, scales)

	// Train PQ on residuals (only if we have enough data)
	if e.config.M > 0 && n >= e.config.M*256 {
		// Convert int8 to float32 for PQ training
		floatVectors := make([][]float32, n)
		for i := range vectors {
			floatVectors[i] = dequantizeVector(&vectors[i], scales[i])
		}

		e.pq = pq.NewProductQuantizer(512, e.config.M, 256)
		e.pq.Train(floatVectors)
	}

	// Build HNSW on centroids if enabled
	if e.config.HNSWEnabled && e.ivfIndex != nil {
		e.hnswRouter = hnsw.NewHNSW(e.config.HNSWM, e.config.HNSWEfC)

		// Add centroids to HNSW
		for i, centroid := range e.ivfIndex.KMeans.Centroids {
			e.hnswRouter.Add(centroid, e.ivfIndex.KMeans.Scales[i], i)
		}
	}

	e.trained = true
	fmt.Printf("Training completed in %v\n", time.Since(start))

	return nil
}

// Add adds a vector to the index
func (e *Engine) Add(vec simd.Vec512, scale float32, id int) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.trained {
		// Auto-train on first batch if needed
		if e.size == 0 {
			e.trained = true
		}
	}

	// Store raw vector for reranking
	e.rawVectors = append(e.rawVectors, vec)
	e.scales = append(e.scales, scale)
	e.ids = append(e.ids, id)

	// Add to appropriate index
	if e.size < e.config.MaxFlatSize {
		e.flatIndex.Add(vec, scale, id)
	} else {
		if e.ivfIndex == nil {
			return fmt.Errorf("index not trained for large dataset")
		}

		e.ivfIndex.Add(vec, scale, id)

		// Encode with PQ if enabled
		if e.pq != nil && e.pq.Trained {
			floatVec := dequantizeVector(&vec, scale)
			codes := e.pq.Encode(floatVec)
			e.pqCodes = append(e.pqCodes, codes)
		}
	}

	e.size++
	return nil
}

// AddBatch adds multiple vectors efficiently
func (e *Engine) AddBatch(vectors []simd.Vec512, scales []float32, ids []int) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	n := len(vectors)
	if n == 0 {
		return nil
	}

	// Auto-train if needed
	if !e.trained && e.size == 0 {
		if n > e.config.MaxFlatSize {
			// Need to train for large dataset
			e.mu.Unlock()
			err := e.Train(vectors[:min(n, 100000)], scales[:min(n, 100000)])
			e.mu.Lock()
			if err != nil {
				return err
			}
		} else {
			e.trained = true
		}
	}

	// Store raw vectors
	e.rawVectors = append(e.rawVectors, vectors...)
	e.scales = append(e.scales, scales...)
	e.ids = append(e.ids, ids...)

	// Add to index
	// For large datasets, always use IVF if it exists (was trained)
	if e.ivfIndex != nil && e.trained {
		// Use IVF index for large datasets
		e.ivfIndex.AddBatch(vectors, scales, ids)

		// Batch encode with PQ
		if e.pq != nil && e.pq.Trained {
			floatVectors := make([][]float32, n)
			for i := range vectors {
				floatVectors[i] = dequantizeVector(&vectors[i], scales[i])
			}
			codes := e.pq.EncodeBatch(floatVectors)
			e.pqCodes = append(e.pqCodes, codes...)
		}
	} else if e.size+n <= e.config.MaxFlatSize {
		// Use flat index for small datasets
		e.flatIndex.AddBatch(vectors, scales, ids)
	} else {
		// This shouldn't happen - large dataset without trained IVF
		return fmt.Errorf("index not properly trained for large dataset (size=%d, n=%d, MaxFlatSize=%d, trained=%v, ivfIndex=%v)",
			e.size, n, e.config.MaxFlatSize, e.trained, e.ivfIndex != nil)
	}

	e.size += n
	return nil
}

// Search performs k-NN search
func (e *Engine) Search(query *simd.Vec512, k int) ([]SearchResult, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.size == 0 {
		return nil, nil
	}

	// Use flat index for small datasets
	if e.size <= e.config.MaxFlatSize {
		var flatResults []flat.SearchResult
		if e.config.UseParallel {
			flatResults = e.flatIndex.SearchTopKParallel(query, k)
		} else {
			flatResults = e.flatIndex.SearchTopK(query, k)
		}

		// Convert results using pool
		resultsPtr := e.resultPool.Get().(*[]SearchResult)
		results := (*resultsPtr)[:0]

		// Ensure capacity
		if cap(results) < len(flatResults) {
			results = make([]SearchResult, 0, len(flatResults))
		}

		for _, r := range flatResults {
			results = append(results, SearchResult{
				ID:       r.ID,
				Score:    float32(r.Score),
				Distance: float32(-r.Score), // Convert similarity to distance
			})
		}
		defer func() {
			*resultsPtr = results
			e.resultPool.Put(resultsPtr)
		}()
		return results, nil
	}

	// Use IVF-HNSW-PQ for large datasets
	if e.ivfIndex == nil {
		return nil, fmt.Errorf("index not trained")
	}

	// Step 1: Route to clusters using HNSW or k-means
	var clusters []int
	if e.hnswRouter != nil {
		// Use HNSW for fast routing
		hnswResults := e.hnswRouter.Search(query, e.config.NProbe)
		clusters = make([]int, 0, len(hnswResults))
		for _, r := range hnswResults {
			clusters = append(clusters, r.ID)
		}
	} else {
		// Fall back to k-means
		clusters = e.ivfIndex.KMeans.PredictMultiple(query, e.config.NProbe)
	}

	// Step 2: Collect candidates from selected clusters
	candidates := e.collectCandidates(clusters)

	// Step 3: Score with PQ if available, otherwise use exact distance
	var topCandidates []int
	if e.pq != nil && e.pq.Trained {
		topCandidates = e.scorePQCandidates(query, candidates, e.config.RerankSize)
	} else {
		topCandidates = e.scoreExactCandidates(query, candidates, e.config.RerankSize)
	}

	// Step 4: Rerank top candidates with exact SIMD distance
	results := e.rerankCandidates(query, topCandidates, k)

	return results, nil
}

// collectCandidates collects vector indices from selected clusters
func (e *Engine) collectCandidates(clusters []int) []int {
	// Estimate total size first to avoid map reallocations
	totalSize := 0
	for _, cluster := range clusters {
		if cluster >= 0 && cluster < len(e.ivfIndex.Lists) {
			totalSize += len(e.ivfIndex.Lists[cluster])
		}
	}

	// Use a map with pre-allocated size hint
	candidateSet := make(map[int]bool, totalSize)

	for _, cluster := range clusters {
		if cluster >= 0 && cluster < len(e.ivfIndex.Lists) {
			e.ivfIndex.ListLocks[cluster].RLock()
			for _, idx := range e.ivfIndex.Lists[cluster] {
				candidateSet[idx] = true
			}
			e.ivfIndex.ListLocks[cluster].RUnlock()
		}
	}

	// Get candidates from pool
	candidatesPtr := e.candidatePool.Get().(*[]int)
	candidates := (*candidatesPtr)[:0]

	// Ensure capacity
	if cap(candidates) < len(candidateSet) {
		candidates = make([]int, 0, len(candidateSet))
	}

	for idx := range candidateSet {
		candidates = append(candidates, idx)
	}

	// Return to pool later (caller must handle)
	return candidates
}

// scorePQCandidates scores candidates using PQ and returns top R
func (e *Engine) scorePQCandidates(query *simd.Vec512, candidates []int, R int) []int {
	if len(candidates) == 0 {
		return nil
	}

	// Dequantize query for PQ
	queryFloat := dequantizeVector(query, 1.0)

	// Compute ADC table
	adcTable := e.pq.ComputeADCTable(queryFloat)

	// Score all candidates
	type scorePair struct {
		idx   int
		score float32
	}

	scores := make([]scorePair, len(candidates))
	for i, idx := range candidates {
		scores[i] = scorePair{
			idx:   idx,
			score: adcTable.Distance(e.pqCodes[idx]),
		}
	}

	// Partial sort to get top R
	if R > len(scores) {
		R = len(scores)
	}

	for i := 0; i < R; i++ {
		minIdx := i
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score < scores[minIdx].score {
				minIdx = j
			}
		}
		scores[i], scores[minIdx] = scores[minIdx], scores[i]
	}

	// Extract top R indices
	topIndices := make([]int, R)
	for i := 0; i < R; i++ {
		topIndices[i] = scores[i].idx
	}

	return topIndices
}

// scoreExactCandidates scores candidates with exact distance
func (e *Engine) scoreExactCandidates(query *simd.Vec512, candidates []int, R int) []int {
	if len(candidates) == 0 {
		return nil
	}

	type scorePair struct {
		idx   int
		score int32
	}

	scores := make([]scorePair, len(candidates))
	for i, idx := range candidates {
		scores[i] = scorePair{
			idx:   idx,
			score: simd.Dot512(query, &e.rawVectors[idx]),
		}
	}

	// Partial sort to get top R (higher score is better for dot product)
	if R > len(scores) {
		R = len(scores)
	}

	for i := 0; i < R; i++ {
		maxIdx := i
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[maxIdx].score {
				maxIdx = j
			}
		}
		scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
	}

	topIndices := make([]int, R)
	for i := 0; i < R; i++ {
		topIndices[i] = scores[i].idx
	}

	return topIndices
}

// rerankCandidates performs exact reranking on top candidates
func (e *Engine) rerankCandidates(query *simd.Vec512, candidates []int, k int) []SearchResult {
	if len(candidates) == 0 {
		return nil
	}

	type scorePair struct {
		idx   int
		score int32
	}

	scores := make([]scorePair, len(candidates))
	for i, idx := range candidates {
		scores[i] = scorePair{
			idx:   idx,
			score: simd.Dot512(query, &e.rawVectors[idx]),
		}
	}

	// Sort by score (descending)
	for i := 0; i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	// Return top k
	if k > len(scores) {
		k = len(scores)
	}

	results := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		idx := scores[i].idx
		results[i] = SearchResult{
			ID:       e.ids[idx],
			Score:    float32(scores[i].score),
			Distance: float32(-scores[i].score),
		}
	}

	return results
}

// SearchResult represents a search result
type SearchResult struct {
	ID       int
	Score    float32
	Distance float32
}

// Stats returns index statistics
func (e *Engine) Stats() IndexStats {
	e.mu.RLock()
	defer e.mu.RUnlock()

	stats := IndexStats{
		Size:        e.size,
		IndexType:   "flat",
		MemoryUsage: int64(e.size * 512), // Approximate
	}

	if e.size > e.config.MaxFlatSize {
		stats.IndexType = "ivf-hnsw-pq"
		if e.ivfIndex != nil {
			stats.NLists = e.config.NList
			stats.ListSizes = e.ivfIndex.GetListSizes()
		}
		if e.pq != nil {
			stats.PQEnabled = true
			stats.PQM = e.config.M
			stats.PQBits = e.config.NBits
		}
		if e.hnswRouter != nil {
			stats.HNSWEnabled = true
		}
	}

	return stats
}

// IndexStats contains index statistics
type IndexStats struct {
	Size        int
	IndexType   string
	MemoryUsage int64
	NLists      int
	ListSizes   []int
	PQEnabled   bool
	PQM         int
	PQBits      int
	HNSWEnabled bool
}

// SetTrained sets the trained flag (used when loading from cache)
func (e *Engine) SetTrained(trained bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.trained = trained
}

// Helper functions

func dequantizeVector(vec *simd.Vec512, scale float32) []float32 {
	result := make([]float32, 512)
	for i := 0; i < 512; i++ {
		result[i] = float32(vec[i]) * scale
	}
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
