package ivf

import (
	"sync"

	"github.com/lee101/gobed/pkg/ann/flat"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// IVFIndex is an inverted file index for approximate search
type IVFIndex struct {
	NList     int            // Number of inverted lists
	NProbe    int            // Number of lists to search
	KMeans    *KMeans        // K-means clusterer for routing
	Lists     [][]int        // Inverted lists (vector indices)
	Vectors   []simd.Vec512  // All vectors
	Scales    []float32      // Vector scales
	IDs       []int          // External IDs
	ListLocks []sync.RWMutex // Per-list locks for concurrent access
	Trained   bool
}

// NewIVFIndex creates a new IVF index
func NewIVFIndex(nlist, nprobe int) *IVFIndex {
	return &IVFIndex{
		NList:     nlist,
		NProbe:    nprobe,
		KMeans:    NewKMeans(nlist, 25),
		Lists:     make([][]int, nlist),
		ListLocks: make([]sync.RWMutex, nlist),
	}
}

// Train trains the index on a sample of vectors
func (idx *IVFIndex) Train(vectors []simd.Vec512, scales []float32) {
	if len(vectors) < idx.NList {
		panic("not enough training data for IVF")
	}

	// Train k-means
	idx.KMeans.Fit(vectors, scales)
	idx.Trained = true

	// Initialize empty lists with capacity hint
	avgListSize := len(vectors) / idx.NList
	for i := range idx.Lists {
		idx.Lists[i] = make([]int, 0, avgListSize*2) // 2x for headroom
	}
}

// Restore installs previously trained centroids and inverted lists without
// retraining. vectors/scales/ids back the list entries (list values index them).
func (idx *IVFIndex) Restore(centroids []simd.Vec512, centroidScales []float32, lists [][]int, vectors []simd.Vec512, scales []float32, ids []int) {
	idx.NList = len(centroids)
	idx.KMeans.SetCentroids(centroids, centroidScales)
	idx.Lists = lists
	idx.ListLocks = make([]sync.RWMutex, len(lists))
	idx.Vectors = vectors
	idx.Scales = scales
	idx.IDs = ids
	for k, l := range lists {
		idx.KMeans.Counts[k] = len(l)
	}
	idx.Trained = true
}

// Add adds a vector to the index
func (idx *IVFIndex) Add(vec simd.Vec512, scale float32, id int) {
	if !idx.Trained {
		panic("index must be trained before adding vectors")
	}

	// Store vector
	vecIdx := len(idx.Vectors)
	idx.Vectors = append(idx.Vectors, vec)
	idx.Scales = append(idx.Scales, scale)
	idx.IDs = append(idx.IDs, id)

	// Assign to nearest cluster
	cluster := idx.KMeans.Predict(&vec)

	// Add to inverted list
	idx.ListLocks[cluster].Lock()
	idx.Lists[cluster] = append(idx.Lists[cluster], vecIdx)
	idx.ListLocks[cluster].Unlock()
}

// AddBatch adds multiple vectors efficiently
func (idx *IVFIndex) AddBatch(vectors []simd.Vec512, scales []float32, ids []int) {
	if !idx.Trained {
		panic("index must be trained before adding vectors")
	}

	n := len(vectors)
	startIdx := len(idx.Vectors)

	// Store all vectors
	idx.Vectors = append(idx.Vectors, vectors...)
	idx.Scales = append(idx.Scales, scales...)
	idx.IDs = append(idx.IDs, ids...)

	// Assign to clusters in parallel
	assignments := make([]int, n)

	parallelRange(n, func(_, start, end int) {
		buf := make([]int32, idx.KMeans.K)
		for i := start; i < end; i++ {
			if len(idx.KMeans.norms) == len(idx.KMeans.Centroids) {
				assignments[i] = idx.KMeans.nearest(&vectors[i], buf)
			} else {
				assignments[i] = idx.KMeans.Predict(&vectors[i])
			}
		}
	})

	// Count per cluster first for pre-allocation
	clusterCounts := make([]int, idx.NList)
	for _, cluster := range assignments {
		clusterCounts[cluster]++
	}

	// Group by cluster with pre-allocated sizes
	clusterGroups := make([][]int, idx.NList)
	for i := range clusterGroups {
		if clusterCounts[i] > 0 {
			clusterGroups[i] = make([]int, 0, clusterCounts[i])
		}
	}

	for i, cluster := range assignments {
		clusterGroups[cluster] = append(clusterGroups[cluster], startIdx+i)
	}

	// Add to inverted lists - batch lock operations
	for cluster, indices := range clusterGroups {
		if len(indices) > 0 {
			idx.ListLocks[cluster].Lock()
			// Check capacity and grow if needed
			if cap(idx.Lists[cluster])-len(idx.Lists[cluster]) < len(indices) {
				newCap := len(idx.Lists[cluster]) + len(indices)*2
				newList := make([]int, len(idx.Lists[cluster]), newCap)
				copy(newList, idx.Lists[cluster])
				idx.Lists[cluster] = newList
			}
			idx.Lists[cluster] = append(idx.Lists[cluster], indices...)
			idx.ListLocks[cluster].Unlock()
		}
	}
}

// Search performs approximate search
func (idx *IVFIndex) Search(query *simd.Vec512, k int) []flat.SearchResult {
	if !idx.Trained {
		panic("index must be trained before searching")
	}

	// Find nprobe nearest clusters
	clusters := idx.KMeans.PredictMultiple(query, idx.NProbe)

	// Pre-calculate total capacity needed
	totalCap := 0
	for _, cluster := range clusters {
		if cluster >= 0 && cluster < len(idx.Lists) {
			totalCap += len(idx.Lists[cluster])
		}
	}

	// Collect candidates from selected lists
	candidates := make([]int, 0, totalCap)

	for _, cluster := range clusters {
		idx.ListLocks[cluster].RLock()
		candidates = append(candidates, idx.Lists[cluster]...)
		idx.ListLocks[cluster].RUnlock()
	}

	// Compute exact distances for candidates
	return idx.rerankCandidates(query, candidates, k)
}

// SearchWithStats performs search and returns statistics
func (idx *IVFIndex) SearchWithStats(query *simd.Vec512, k int) ([]flat.SearchResult, SearchStats) {
	stats := SearchStats{}

	if !idx.Trained {
		panic("index must be trained before searching")
	}

	// Find nprobe nearest clusters
	clusters := idx.KMeans.PredictMultiple(query, idx.NProbe)
	stats.ClustersSearched = len(clusters)

	// Collect candidates
	candidates := make([]int, 0)
	for _, cluster := range clusters {
		idx.ListLocks[cluster].RLock()
		candidates = append(candidates, idx.Lists[cluster]...)
		idx.ListLocks[cluster].RUnlock()
	}
	stats.CandidatesScored = len(candidates)

	// Rerank
	results := idx.rerankCandidates(query, candidates, k)
	stats.ResultsReturned = len(results)

	return results, stats
}

// rerankCandidates computes exact distances and returns top-k
func (idx *IVFIndex) rerankCandidates(query *simd.Vec512, candidates []int, k int) []flat.SearchResult {
	if len(candidates) == 0 {
		return nil
	}

	// Use flat search on candidates
	tempIndex := flat.NewFlatIndex(len(candidates))
	for _, vecIdx := range candidates {
		tempIndex.Add(idx.Vectors[vecIdx], idx.Scales[vecIdx], idx.IDs[vecIdx])
	}

	return tempIndex.SearchTopK(query, k)
}

// SetNProbe updates the number of lists to search
func (idx *IVFIndex) SetNProbe(nprobe int) {
	if nprobe > idx.NList {
		nprobe = idx.NList
	}
	if nprobe < 1 {
		nprobe = 1
	}
	idx.NProbe = nprobe
}

// Size returns the number of vectors in the index
func (idx *IVFIndex) Size() int {
	return len(idx.Vectors)
}

// GetListSizes returns the size of each inverted list
func (idx *IVFIndex) GetListSizes() []int {
	sizes := make([]int, idx.NList)
	for i := range idx.Lists {
		idx.ListLocks[i].RLock()
		sizes[i] = len(idx.Lists[i])
		idx.ListLocks[i].RUnlock()
	}
	return sizes
}

// SearchStats contains search statistics
type SearchStats struct {
	ClustersSearched int
	CandidatesScored int
	ResultsReturned  int
}
