package ivf

import (
	"sync"

	"github.com/lee101/gobed/pkg/ann/flat"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// IVFIndexOptimized is an optimized inverted file index for approximate search
type IVFIndexOptimized struct {
	NList     int              // Number of inverted lists
	NProbe    int              // Number of lists to search
	KMeans    *KMeansOptimized // Optimized K-means clusterer for routing
	Lists     [][]int          // Inverted lists (vector indices)
	Vectors   []simd.Vec512    // All vectors
	Scales    []float32        // Vector scales
	IDs       []int            // External IDs
	ListLocks []sync.RWMutex   // Per-list locks for concurrent access
	Trained   bool

	// Pre-allocated buffers for better performance
	candidateBuffer []int // Reused candidate buffer
	tempAssignments []int // Reused assignment buffer
}

// NewIVFIndexOptimized creates a new optimized IVF index
func NewIVFIndexOptimized(nlist, nprobe int) *IVFIndexOptimized {
	return &IVFIndexOptimized{
		NList:           nlist,
		NProbe:          nprobe,
		KMeans:          NewKMeansOptimized(nlist, 25),
		Lists:           make([][]int, nlist),
		ListLocks:       make([]sync.RWMutex, nlist),
		candidateBuffer: make([]int, 0, nprobe*200), // Pre-allocate reasonable buffer
	}
}

// Train trains the index on a sample of vectors using optimized k-means
func (idx *IVFIndexOptimized) Train(vectors []simd.Vec512, scales []float32) {
	if len(vectors) < idx.NList {
		panic("not enough training data for IVF")
	}

	// Train optimized k-means
	idx.KMeans.Fit(vectors, scales)
	idx.Trained = true

	// Initialize empty lists with pre-allocated capacity
	estimatedSize := len(vectors) * 2 / idx.NList // Estimate based on training data
	for i := range idx.Lists {
		idx.Lists[i] = make([]int, 0, estimatedSize)
	}
}

// Add adds a vector to the index (optimized with SIMD distance computation)
func (idx *IVFIndexOptimized) Add(vec simd.Vec512, scale float32, id int) {
	if !idx.Trained {
		panic("index must be trained before adding vectors")
	}

	// Store vector
	vecIdx := len(idx.Vectors)
	idx.Vectors = append(idx.Vectors, vec)
	idx.Scales = append(idx.Scales, scale)
	idx.IDs = append(idx.IDs, id)

	// Assign to nearest cluster using optimized prediction
	cluster := idx.KMeans.Predict(&vec)

	// Add to inverted list
	idx.ListLocks[cluster].Lock()
	idx.Lists[cluster] = append(idx.Lists[cluster], vecIdx)
	idx.ListLocks[cluster].Unlock()
}

// AddBatchOptimized adds multiple vectors with enhanced batching optimizations
func (idx *IVFIndexOptimized) AddBatchOptimized(vectors []simd.Vec512, scales []float32, ids []int) {
	if !idx.Trained {
		panic("index must be trained before adding vectors")
	}

	n := len(vectors)
	startIdx := len(idx.Vectors)

	// Store all vectors (pre-allocate if needed)
	if cap(idx.Vectors) < len(idx.Vectors)+n {
		newVectors := make([]simd.Vec512, len(idx.Vectors), len(idx.Vectors)+n*2)
		copy(newVectors, idx.Vectors)
		idx.Vectors = newVectors

		newScales := make([]float32, len(idx.Scales), len(idx.Scales)+n*2)
		copy(newScales, idx.Scales)
		idx.Scales = newScales

		newIDs := make([]int, len(idx.IDs), len(idx.IDs)+n*2)
		copy(newIDs, idx.IDs)
		idx.IDs = newIDs
	}

	idx.Vectors = append(idx.Vectors, vectors...)
	idx.Scales = append(idx.Scales, scales...)
	idx.IDs = append(idx.IDs, ids...)

	// Assign to clusters using optimized parallel prediction
	if cap(idx.tempAssignments) < n {
		idx.tempAssignments = make([]int, n)
	} else {
		idx.tempAssignments = idx.tempAssignments[:n]
	}

	// Use parallel assignment for large batches
	if n > 500 {
		idx.assignToClustersBatch(vectors, idx.tempAssignments)
	} else {
		// Sequential for smaller batches (better cache locality)
		for i := 0; i < n; i++ {
			idx.tempAssignments[i] = idx.KMeans.Predict(&vectors[i])
		}
	}

	// Group by cluster with pre-allocated slices
	clusterGroups := make([][]int, idx.NList)
	for i := range clusterGroups {
		clusterGroups[i] = make([]int, 0, n/idx.NList+10) // Pre-allocate reasonable size
	}

	for i, cluster := range idx.tempAssignments {
		clusterGroups[cluster] = append(clusterGroups[cluster], startIdx+i)
	}

	// Add to inverted lists in parallel for large updates
	if n > 1000 {
		var wg sync.WaitGroup
		for cluster, indices := range clusterGroups {
			if len(indices) > 0 {
				wg.Add(1)
				go func(cluster int, indices []int) {
					defer wg.Done()
					idx.ListLocks[cluster].Lock()
					idx.Lists[cluster] = append(idx.Lists[cluster], indices...)
					idx.ListLocks[cluster].Unlock()
				}(cluster, indices)
			}
		}
		wg.Wait()
	} else {
		// Sequential for smaller batches
		for cluster, indices := range clusterGroups {
			if len(indices) > 0 {
				idx.ListLocks[cluster].Lock()
				idx.Lists[cluster] = append(idx.Lists[cluster], indices...)
				idx.ListLocks[cluster].Unlock()
			}
		}
	}
}

// assignToClustersBatch assigns vectors to clusters in parallel
func (idx *IVFIndexOptimized) assignToClustersBatch(vectors []simd.Vec512, assignments []int) {
	n := len(vectors)
	numWorkers := 8
	chunkSize := (n + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				assignments[i] = idx.KMeans.Predict(&vectors[i])
			}
		}(start, end)
	}
	wg.Wait()
}

// SearchOptimized performs optimized approximate search
func (idx *IVFIndexOptimized) SearchOptimized(query *simd.Vec512, k int) []flat.SearchResult {
	if !idx.Trained {
		panic("index must be trained before searching")
	}

	// Find nprobe nearest clusters using optimized prediction
	clusters := idx.KMeans.PredictMultiple(query, idx.NProbe)

	// Reset and collect candidates from selected lists
	idx.candidateBuffer = idx.candidateBuffer[:0]

	for _, cluster := range clusters {
		idx.ListLocks[cluster].RLock()
		idx.candidateBuffer = append(idx.candidateBuffer, idx.Lists[cluster]...)
		idx.ListLocks[cluster].RUnlock()
	}

	// Compute exact distances for candidates using optimized reranking
	return idx.rerankCandidatesOptimized(query, idx.candidateBuffer, k)
}

// SearchOptimizedWithStats performs optimized search and returns statistics
func (idx *IVFIndexOptimized) SearchOptimizedWithStats(query *simd.Vec512, k int) ([]flat.SearchResult, SearchStats) {
	stats := SearchStats{}

	if !idx.Trained {
		panic("index must be trained before searching")
	}

	// Find nprobe nearest clusters
	clusters := idx.KMeans.PredictMultiple(query, idx.NProbe)
	stats.ClustersSearched = len(clusters)

	// Collect candidates
	idx.candidateBuffer = idx.candidateBuffer[:0]
	for _, cluster := range clusters {
		idx.ListLocks[cluster].RLock()
		idx.candidateBuffer = append(idx.candidateBuffer, idx.Lists[cluster]...)
		idx.ListLocks[cluster].RUnlock()
	}
	stats.CandidatesScored = len(idx.candidateBuffer)

	// Rerank
	results := idx.rerankCandidatesOptimized(query, idx.candidateBuffer, k)
	stats.ResultsReturned = len(results)

	return results, stats
}

// rerankCandidatesOptimized computes exact distances with optimizations
func (idx *IVFIndexOptimized) rerankCandidatesOptimized(query *simd.Vec512, candidates []int, k int) []flat.SearchResult {
	if len(candidates) == 0 {
		return nil
	}

	// Use flat search on candidates - but consider implementing inline reranking for better performance
	tempIndex := flat.NewFlatIndex(len(candidates))
	for _, vecIdx := range candidates {
		tempIndex.Add(idx.Vectors[vecIdx], idx.Scales[vecIdx], idx.IDs[vecIdx])
	}

	return tempIndex.SearchTopK(query, k)
}

// SetNProbe updates the number of lists to search
func (idx *IVFIndexOptimized) SetNProbe(nprobe int) {
	if nprobe > idx.NList {
		nprobe = idx.NList
	}
	if nprobe < 1 {
		nprobe = 1
	}
	idx.NProbe = nprobe
}

// Size returns the number of vectors in the index
func (idx *IVFIndexOptimized) Size() int {
	return len(idx.Vectors)
}

// GetListSizes returns the size of each inverted list
func (idx *IVFIndexOptimized) GetListSizes() []int {
	sizes := make([]int, idx.NList)
	for i := range idx.Lists {
		idx.ListLocks[i].RLock()
		sizes[i] = len(idx.Lists[i])
		idx.ListLocks[i].RUnlock()
	}
	return sizes
}

// GetMemoryUsage estimates memory usage in bytes
func (idx *IVFIndexOptimized) GetMemoryUsage() int64 {
	if !idx.Trained {
		return 0
	}

	var totalMemory int64

	// Vectors: 512 bytes per vector
	totalMemory += int64(len(idx.Vectors)) * 512

	// Scales: 4 bytes per vector
	totalMemory += int64(len(idx.Scales)) * 4

	// IDs: 8 bytes per vector (assuming 64-bit int)
	totalMemory += int64(len(idx.IDs)) * 8

	// Inverted lists: estimate 8 bytes per entry
	for i := range idx.Lists {
		idx.ListLocks[i].RLock()
		totalMemory += int64(len(idx.Lists[i])) * 8
		idx.ListLocks[i].RUnlock()
	}

	// K-means centroids: 512 bytes per centroid + scales
	totalMemory += int64(idx.KMeans.K) * (512 + 4)

	return totalMemory
}

// GetIndexStats returns comprehensive index statistics
func (idx *IVFIndexOptimized) GetIndexStats() IndexStats {
	if !idx.Trained {
		return IndexStats{}
	}

	sizes := idx.GetListSizes()

	// Calculate statistics
	totalVectors := 0
	minSize := sizes[0]
	maxSize := sizes[0]

	for _, size := range sizes {
		totalVectors += size
		if size < minSize {
			minSize = size
		}
		if size > maxSize {
			maxSize = size
		}
	}

	avgSize := float64(totalVectors) / float64(len(sizes))

	// Calculate standard deviation
	var variance float64
	for _, size := range sizes {
		diff := float64(size) - avgSize
		variance += diff * diff
	}
	variance /= float64(len(sizes))
	stdDev := variance // Simplified, not taking square root

	return IndexStats{
		TotalVectors:    totalVectors,
		NumLists:        idx.NList,
		MinListSize:     minSize,
		MaxListSize:     maxSize,
		AvgListSize:     avgSize,
		ListImbalance:   stdDev / (avgSize * avgSize), // Normalized variance
		MemoryUsage:     idx.GetMemoryUsage(),
		TrainingVectors: len(idx.KMeans.Centroids),
	}
}

// IndexStats contains comprehensive index statistics
type IndexStats struct {
	TotalVectors    int
	NumLists        int
	MinListSize     int
	MaxListSize     int
	AvgListSize     float64
	ListImbalance   float64 // Higher = more imbalanced
	MemoryUsage     int64   // Bytes
	TrainingVectors int
}
