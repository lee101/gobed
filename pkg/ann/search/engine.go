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

	// Per-query scratch buffers
	scratchPool sync.Pool
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

	// Always create IVF when Train is explicitly called (caller knows IVF is needed)
	// The check for small datasets should happen at the caller level

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

		results := make([]SearchResult, len(flatResults))
		for i, r := range flatResults {
			results[i] = SearchResult{
				ID:       r.ID,
				Score:    float32(r.Score),
				Distance: float32(-r.Score), // Convert similarity to distance
			}
		}
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

	return e.searchClusters(query, clusters, k), nil
}

// searchClusters scores candidates from the given clusters and returns the top k.
func (e *Engine) searchClusters(query *simd.Vec512, clusters []int, k int) []SearchResult {
	sc := e.getScratch()
	defer e.scratchPool.Put(sc)

	// Step 2: Collect candidates from selected clusters
	candidates := e.collectCandidates(sc, clusters)

	// Step 3/4: PQ shortlist then exact rerank, or exact top-k directly
	var results []SearchResult
	if e.pq != nil && e.pq.Trained {
		top := e.scorePQCandidates(sc, query, candidates, e.config.RerankSize)
		results = e.rerankCandidates(sc, query, top, k)
	} else {
		// Exact top-R sorted by score then position; its first k equal the
		// exact rerank of that shortlist.
		h := e.scoreExactTop(sc, query, candidates, e.config.RerankSize)
		if k < len(h) {
			h = h[:k]
		}
		results = e.toResults(h, candidates)
	}

	return results
}

// searchScratch holds per-query buffers reused across searches.
type searchScratch struct {
	seen  []uint32
	gen   uint32
	cands []int
	exact []scored[int32]
	pqs   []scored[float32]
}

func (e *Engine) getScratch() *searchScratch {
	if v := e.scratchPool.Get(); v != nil {
		return v.(*searchScratch)
	}
	return &searchScratch{}
}

// collectCandidates collects deduplicated vector indices from selected clusters
// in cluster order, then list order. The returned slice aliases sc.cands.
func (e *Engine) collectCandidates(sc *searchScratch, clusters []int) []int {
	n := len(e.rawVectors)
	if len(sc.seen) < n {
		sc.seen = make([]uint32, n+n/4)
		sc.gen = 0
	}
	sc.gen++
	if sc.gen == 0 {
		clear(sc.seen)
		sc.gen = 1
	}
	gen, seen := sc.gen, sc.seen
	cands := sc.cands[:0]
	for _, cluster := range clusters {
		if cluster < 0 || cluster >= len(e.ivfIndex.Lists) {
			continue
		}
		e.ivfIndex.ListLocks[cluster].RLock()
		for _, idx := range e.ivfIndex.Lists[cluster] {
			if uint(idx) >= uint(len(seen)) {
				grown := make([]uint32, idx+1+idx/4)
				copy(grown, seen)
				seen = grown
				sc.seen = grown
			}
			if seen[idx] != gen {
				seen[idx] = gen
				cands = append(cands, idx)
			}
		}
		e.ivfIndex.ListLocks[cluster].RUnlock()
	}
	sc.cands = cands
	return cands
}

// scorePQCandidates scores candidates using PQ and returns top R (lowest distance,
// ties broken by candidate position).
func (e *Engine) scorePQCandidates(sc *searchScratch, query *simd.Vec512, candidates []int, R int) []int {
	if len(candidates) == 0 || R <= 0 {
		return nil
	}
	adcTable := e.pq.ComputeADCTable(dequantizeVector(query, 1.0))
	h := sc.pqs[:0]
	for i, idx := range candidates {
		h = topRPush(h, R, scored[float32]{s: -adcTable.Distance(e.pqCodes[idx]), pos: int32(i)})
	}
	topRSort(h)
	sc.pqs = h
	top := make([]int, len(h))
	for i := range h {
		top[i] = candidates[h[i].pos]
	}
	return top
}

// scoreExactTop scores candidates with exact dot product and returns the top R
// sorted by score desc, ties broken by candidate position. Aliases sc.exact.
func (e *Engine) scoreExactTop(sc *searchScratch, query *simd.Vec512, candidates []int, R int) []scored[int32] {
	if len(candidates) == 0 || R <= 0 {
		return nil
	}
	h := sc.exact[:0]
	for i, idx := range candidates {
		h = topRPush(h, R, scored[int32]{s: simd.Dot512(query, &e.rawVectors[idx]), pos: int32(i)})
	}
	topRSort(h)
	sc.exact = h
	return h
}

// scoreExactCandidates scores candidates with exact distance
func (e *Engine) scoreExactCandidates(sc *searchScratch, query *simd.Vec512, candidates []int, R int) []int {
	h := e.scoreExactTop(sc, query, candidates, R)
	if h == nil {
		return nil
	}
	top := make([]int, len(h))
	for i := range h {
		top[i] = candidates[h[i].pos]
	}
	return top
}

// rerankCandidates performs exact reranking on top candidates
func (e *Engine) rerankCandidates(sc *searchScratch, query *simd.Vec512, candidates []int, k int) []SearchResult {
	h := e.scoreExactTop(sc, query, candidates, k)
	return e.toResults(h, candidates)
}

func (e *Engine) toResults(h []scored[int32], candidates []int) []SearchResult {
	if len(h) == 0 {
		return nil
	}
	results := make([]SearchResult, len(h))
	for i, p := range h {
		results[i] = SearchResult{
			ID:       e.ids[candidates[p.pos]],
			Score:    float32(p.s),
			Distance: float32(-p.s),
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
