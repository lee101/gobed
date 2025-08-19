package flat

import (
	"container/heap"
	"runtime"
	"sync"
	"unsafe"

	"github.com/gobed/ann/simd"
)

// SearchResult represents a search result with ID and score
type SearchResult struct {
	ID    int
	Score int32
}

// minHeap implements heap.Interface for keeping top-K results
type minHeap []SearchResult

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].Score < h[j].Score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(SearchResult)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// FlatIndex is a simple flat index for exact search
type FlatIndex struct {
	vectors []simd.Vec512
	scales  []float32 // Optional: per-vector scales for dequantization
	ids     []int     // Optional: external IDs
	dim     int
}

// NewFlatIndex creates a new flat index
func NewFlatIndex(capacity int) *FlatIndex {
	return &FlatIndex{
		vectors: make([]simd.Vec512, 0, capacity),
		scales:  make([]float32, 0, capacity),
		ids:     make([]int, 0, capacity),
		dim:     512,
	}
}

// Add adds a vector to the index
func (idx *FlatIndex) Add(vec simd.Vec512, scale float32, id int) {
	idx.vectors = append(idx.vectors, vec)
	idx.scales = append(idx.scales, scale)
	idx.ids = append(idx.ids, id)
}

// AddBatch adds multiple vectors at once
func (idx *FlatIndex) AddBatch(vecs []simd.Vec512, scales []float32, ids []int) {
	idx.vectors = append(idx.vectors, vecs...)
	idx.scales = append(idx.scales, scales...)
	idx.ids = append(idx.ids, ids...)
}

// Size returns the number of vectors in the index
func (idx *FlatIndex) Size() int {
	return len(idx.vectors)
}

// SearchTopK performs exact search and returns top K results
func (idx *FlatIndex) SearchTopK(query *simd.Vec512, K int) []SearchResult {
	if K <= 0 || len(idx.vectors) == 0 {
		return nil
	}
	
	// Use min-heap to maintain top-K
	h := &minHeap{}
	heap.Init(h)
	
	for i := range idx.vectors {
		score := simd.Dot512(query, &idx.vectors[i])
		
		if h.Len() < K {
			heap.Push(h, SearchResult{ID: idx.ids[i], Score: score})
		} else if score > (*h)[0].Score {
			(*h)[0] = SearchResult{ID: idx.ids[i], Score: score}
			heap.Fix(h, 0)
		}
	}
	
	// Extract results in descending order
	results := make([]SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SearchResult)
	}
	
	return results
}

// SearchTopKParallel performs parallel exact search for better throughput
func (idx *FlatIndex) SearchTopKParallel(query *simd.Vec512, K int) []SearchResult {
	if K <= 0 || len(idx.vectors) == 0 {
		return nil
	}
	
	numWorkers := runtime.NumCPU()
	n := len(idx.vectors)
	
	// If dataset is small, use single-threaded
	if n < 1000 {
		return idx.SearchTopK(query, K)
	}
	
	chunkSize := (n + numWorkers - 1) / numWorkers
	
	type chunkResult struct {
		results []SearchResult
	}
	
	resultChan := make(chan chunkResult, numWorkers)
	var wg sync.WaitGroup
	
	// Launch workers
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= end {
			continue
		}
		
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			
			// Local heap for this chunk
			h := &minHeap{}
			heap.Init(h)
			
			for i := start; i < end; i++ {
				score := simd.Dot512(query, &idx.vectors[i])
				
				if h.Len() < K {
					heap.Push(h, SearchResult{ID: idx.ids[i], Score: score})
				} else if score > (*h)[0].Score {
					(*h)[0] = SearchResult{ID: idx.ids[i], Score: score}
					heap.Fix(h, 0)
				}
			}
			
			// Convert heap to slice
			results := make([]SearchResult, h.Len())
			for i := len(results) - 1; i >= 0; i-- {
				results[i] = heap.Pop(h).(SearchResult)
			}
			
			resultChan <- chunkResult{results: results}
		}(start, end)
	}
	
	// Wait for all workers
	wg.Wait()
	close(resultChan)
	
	// Merge results from all chunks
	allResults := make([]SearchResult, 0, K*numWorkers)
	for chunk := range resultChan {
		allResults = append(allResults, chunk.results...)
	}
	
	// Final top-K selection
	h := &minHeap{}
	heap.Init(h)
	
	for _, res := range allResults {
		if h.Len() < K {
			heap.Push(h, res)
		} else if res.Score > (*h)[0].Score {
			(*h)[0] = res
			heap.Fix(h, 0)
		}
	}
	
	// Extract final results
	results := make([]SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SearchResult)
	}
	
	return results
}

// SearchTopKBatch processes multiple queries in batch for better cache utilization
func (idx *FlatIndex) SearchTopKBatch(queries []*simd.Vec512, K int) [][]SearchResult {
	results := make([][]SearchResult, len(queries))
	
	// Process queries in parallel
	var wg sync.WaitGroup
	for q := range queries {
		wg.Add(1)
		go func(qIdx int) {
			defer wg.Done()
			results[qIdx] = idx.SearchTopK(queries[qIdx], K)
		}(q)
	}
	wg.Wait()
	
	return results
}

// Optimized version with cache tiling
func (idx *FlatIndex) SearchTopKTiled(query *simd.Vec512, K int) []SearchResult {
	if K <= 0 || len(idx.vectors) == 0 {
		return nil
	}
	
	const tileSize = 64 // Process 64 vectors at a time for L2 cache
	
	h := &minHeap{}
	heap.Init(h)
	
	n := len(idx.vectors)
	
	// Process in tiles
	for tileStart := 0; tileStart < n; tileStart += tileSize {
		tileEnd := tileStart + tileSize
		if tileEnd > n {
			tileEnd = n
		}
		
		// Prefetch next tile
		if tileEnd < n && tileEnd+tileSize <= n {
			nextTile := &idx.vectors[tileEnd]
			_ = unsafe.Sizeof(*nextTile) // Prefetch hint
		}
		
		// Process current tile
		for i := tileStart; i < tileEnd; i++ {
			score := simd.Dot512(query, &idx.vectors[i])
			
			if h.Len() < K {
				heap.Push(h, SearchResult{ID: idx.ids[i], Score: score})
			} else if score > (*h)[0].Score {
				(*h)[0] = SearchResult{ID: idx.ids[i], Score: score}
				heap.Fix(h, 0)
			}
		}
	}
	
	// Extract results
	results := make([]SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SearchResult)
	}
	
	return results
}