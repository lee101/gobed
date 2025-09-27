package hnsw

import (
	"container/heap"
	"math"
	"math/rand"
	"sync"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// HNSW implements Hierarchical Navigable Small World graph
type HNSW struct {
	M              int     // Max number of connections per node
	MMax           int     // Max connections for layer > 0
	MMax0          int     // Max connections for layer 0
	efConstruction int     // Size of dynamic candidate list
	ef             int     // Size of search candidate list
	ml             float64 // Normalization factor for level assignment
	seed           int64   // Random seed

	nodes      []Node       // Graph nodes
	entryPoint int          // Entry point for search
	distFunc   DistanceFunc // Distance function
	mu         sync.RWMutex // Global lock

	// Object pools
	visitedPool *sync.Pool
	heapPool    *sync.Pool
}

// Node represents a node in the HNSW graph
type Node struct {
	ID        int
	Vector    simd.Vec512
	Scale     float32
	Level     int
	Neighbors [][]int // Neighbors at each level
}

// DistanceFunc computes distance between two vectors
type DistanceFunc func(a, b *simd.Vec512) float32

// NewHNSW creates a new HNSW index
func NewHNSW(M, efConstruction int) *HNSW {
	mMax := M
	mMax0 := M * 2
	ml := 1.0 / math.Log(2.0)

	h := &HNSW{
		M:              M,
		MMax:           mMax,
		MMax0:          mMax0,
		efConstruction: efConstruction,
		ef:             efConstruction,
		ml:             ml,
		seed:           rand.Int63(),
		nodes:          make([]Node, 0, 1000), // Pre-allocate
		entryPoint:     -1,
		distFunc:       l2Distance,
	}

	// Initialize pools
	h.visitedPool = &sync.Pool{
		New: func() interface{} {
			return make(map[int]bool, 1000)
		},
	}
	h.heapPool = &sync.Pool{
		New: func() interface{} {
			h := make(CandidateHeap, 0, efConstruction)
			return &h
		},
	}

	return h
}

// SetEf sets the search parameter ef
func (h *HNSW) SetEf(ef int) {
	h.ef = ef
}

// SetDistanceFunc sets the distance function
func (h *HNSW) SetDistanceFunc(f DistanceFunc) {
	h.distFunc = f
}

// Add inserts a new vector into the graph
func (h *HNSW) Add(vec simd.Vec512, scale float32, id int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if len(h.nodes) >= id+1 {
		// Grow with extra capacity to reduce reallocations
		newCap := (id + 1) * 2
		if newCap < len(h.nodes)+100 {
			newCap = len(h.nodes) + 100
		}
		newNodes := make([]Node, id+1, newCap)
		copy(newNodes, h.nodes)
		h.nodes = newNodes
	}

	// Assign level
	level := h.getRandomLevel()

	node := Node{
		ID:        id,
		Vector:    vec,
		Scale:     scale,
		Level:     level,
		Neighbors: make([][]int, level+1),
	}

	// Initialize neighbor lists with capacity
	maxNeighbors := h.MMax
	if level == 0 {
		maxNeighbors = h.MMax0
	}
	for l := 0; l <= level; l++ {
		cap := maxNeighbors
		if l == 0 {
			cap = h.MMax0
		}
		node.Neighbors[l] = make([]int, 0, cap)
	}

	if h.entryPoint == -1 {
		// First node
		h.nodes = append(h.nodes, node)
		h.entryPoint = 0
		return
	}

	// Find neighbors using a greedy search
	candidates := h.searchLayer(&vec, h.entryPoint, 1, level)

	// Insert node
	nodeIdx := len(h.nodes)
	h.nodes = append(h.nodes, node)

	// Connect to neighbors at all levels
	for lc := level; lc >= 0; lc-- {
		candidates = h.searchLayer(&vec, h.entryPoint, h.efConstruction, lc)

		m := h.MMax
		if lc == 0 {
			m = h.MMax0
		}

		neighbors := h.getNeighbors(candidates, m)
		h.nodes[nodeIdx].Neighbors[lc] = neighbors

		// Add bidirectional links
		for _, neighbor := range neighbors {
			h.nodes[neighbor].Neighbors[lc] = append(h.nodes[neighbor].Neighbors[lc], nodeIdx)

			// Prune connections if needed
			maxConn := h.MMax
			if lc == 0 {
				maxConn = h.MMax0
			}

			if len(h.nodes[neighbor].Neighbors[lc]) > maxConn {
				h.pruneConnections(neighbor, lc, maxConn)
			}
		}
	}

	// Update entry point if necessary
	if level > h.nodes[h.entryPoint].Level {
		h.entryPoint = nodeIdx
	}
}

// Search finds k nearest neighbors
func (h *HNSW) Search(query *simd.Vec512, k int) []SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == -1 {
		return nil
	}

	ep := h.entryPoint

	// Search from top layer to layer 0
	for level := h.nodes[ep].Level; level > 0; level-- {
		nearest := h.searchLayerSingle(query, ep, 1, level)
		if len(nearest) > 0 {
			ep = nearest[0].ID
		}
	}

	// Search at layer 0
	candidates := h.searchLayer(query, ep, max(h.ef, k), 0)

	// Return top k
	results := make([]SearchResult, 0, k)
	for i := 0; i < k && i < len(candidates); i++ {
		results = append(results, SearchResult{
			ID:       candidates[i].ID,
			Distance: candidates[i].Distance,
		})
	}

	return results
}

// searchLayer searches for nearest neighbors at a specific layer
func (h *HNSW) searchLayer(query *simd.Vec512, ep int, ef int, layer int) []Candidate {
	// Get from pools
	visited := h.visitedPool.Get().(map[int]bool)
	for k := range visited {
		delete(visited, k) // Clear the map
	}
	defer h.visitedPool.Put(visited)

	candidates := h.heapPool.Get().(*CandidateHeap)
	*candidates = (*candidates)[:0]
	heap.Init(candidates)
	defer func() {
		*candidates = (*candidates)[:0]
		h.heapPool.Put(candidates)
	}()

	w := h.heapPool.Get().(*CandidateHeap)
	*w = (*w)[:0]
	heap.Init(w)
	defer func() {
		*w = (*w)[:0]
		h.heapPool.Put(w)
	}()

	dist := h.distFunc(query, &h.nodes[ep].Vector)
	heap.Push(candidates, Candidate{ID: ep, Distance: -dist}) // Negative for max heap
	heap.Push(w, Candidate{ID: ep, Distance: dist})
	visited[ep] = true

	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(Candidate)
		current.Distance = -current.Distance // Convert back

		if current.Distance > (*w)[0].Distance {
			break
		}

		// Check neighbors
		neighbors := h.nodes[current.ID].Neighbors[layer]
		for _, neighbor := range neighbors {
			if !visited[neighbor] {
				visited[neighbor] = true
				dist := h.distFunc(query, &h.nodes[neighbor].Vector)

				if dist < (*w)[0].Distance || w.Len() < ef {
					heap.Push(candidates, Candidate{ID: neighbor, Distance: -dist})
					heap.Push(w, Candidate{ID: neighbor, Distance: dist})

					if w.Len() > ef {
						heap.Pop(w)
					}
				}
			}
		}
	}

	// Convert heap to slice
	results := make([]Candidate, w.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(w).(Candidate)
	}

	return results
}

// searchLayerSingle searches for a single nearest neighbor
func (h *HNSW) searchLayerSingle(query *simd.Vec512, ep int, ef int, layer int) []Candidate {
	return h.searchLayer(query, ep, ef, layer)
}

// getNeighbors selects M neighbors from candidates
func (h *HNSW) getNeighbors(candidates []Candidate, m int) []int {
	if len(candidates) <= m {
		neighbors := make([]int, len(candidates))
		for i, c := range candidates {
			neighbors[i] = c.ID
		}
		return neighbors
	}

	// Select m nearest
	neighbors := make([]int, m)
	for i := 0; i < m; i++ {
		neighbors[i] = candidates[i].ID
	}

	return neighbors
}

// pruneConnections prunes excess connections using a heuristic
func (h *HNSW) pruneConnections(nodeIdx int, level int, maxConn int) {
	neighbors := h.nodes[nodeIdx].Neighbors[level]
	if len(neighbors) <= maxConn {
		return
	}

	// Sort by distance
	node := &h.nodes[nodeIdx]
	candidates := make([]Candidate, len(neighbors))
	for i, n := range neighbors {
		dist := h.distFunc(&node.Vector, &h.nodes[n].Vector)
		candidates[i] = Candidate{ID: n, Distance: dist}
	}

	// Sort candidates
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].Distance < candidates[i].Distance {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Keep only maxConn nearest
	h.nodes[nodeIdx].Neighbors[level] = h.nodes[nodeIdx].Neighbors[level][:maxConn]
	for i := 0; i < maxConn; i++ {
		h.nodes[nodeIdx].Neighbors[level][i] = candidates[i].ID
	}
}

// getRandomLevel returns a random level according to the HNSW paper
func (h *HNSW) getRandomLevel() int {
	level := 0
	for rand.Float64() < 0.5 {
		level++
	}
	return level
}

// Candidate represents a search candidate
type Candidate struct {
	ID       int
	Distance float32
}

// CandidateHeap implements a min-heap of candidates
type CandidateHeap []Candidate

func (h CandidateHeap) Len() int           { return len(h) }
func (h CandidateHeap) Less(i, j int) bool { return h[i].Distance < h[j].Distance }
func (h CandidateHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *CandidateHeap) Push(x interface{}) {
	*h = append(*h, x.(Candidate))
}

func (h *CandidateHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// SearchResult represents a search result
type SearchResult struct {
	ID       int
	Distance float32
}

// Distance functions

func l2Distance(a, b *simd.Vec512) float32 {
	return float32(simd.L2Squared512(a, b))
}

func dotDistance(a, b *simd.Vec512) float32 {
	// Negative dot product for similarity (higher is better)
	return -float32(simd.Dot512(a, b))
}

func cosineDistance(a, b *simd.Vec512) float32 {
	dot := simd.Dot512(a, b)
	// Assuming vectors are normalized
	return 1.0 - float32(dot)/(127.0*127.0*512)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
