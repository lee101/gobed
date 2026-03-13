package src

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/lee101/gobed"
)

// hnswEmbedder defines the interface for embedding models used by HNSW
type hnswEmbedder interface {
	EmbedFast(text string) ([]float32, func())
	Close() error
}

// HNSWSearcher provides HNSW-accelerated semantic search
type HNSWSearcher struct {
	model     hnswEmbedder
	documents []Document

	// HNSW graph
	hnsw      *HNSWFloat32

	// For incremental updates
	mu sync.RWMutex
}

// HNSWFloat32 implements HNSW for float32 vectors
type HNSWFloat32 struct {
	M              int
	MMax           int
	MMax0          int
	efConstruction int
	ef             int
	ml             float64

	nodes      []hnswNode
	entryPoint int
	mu         sync.RWMutex
}

type hnswNode struct {
	ID        int
	Vector    []float32
	InvNorm   float32
	Level     int
	Neighbors [][]int
}

type hnswCandidate struct {
	id   int
	dist float32
}

type hnswMaxHeap []hnswCandidate
type hnswMinHeap []hnswCandidate

func (h hnswMaxHeap) Len() int           { return len(h) }
func (h hnswMaxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h hnswMaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *hnswMaxHeap) Push(x any)        { *h = append(*h, x.(hnswCandidate)) }
func (h *hnswMaxHeap) Pop() any          { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

func (h hnswMinHeap) Len() int           { return len(h) }
func (h hnswMinHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h hnswMinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *hnswMinHeap) Push(x any)        { *h = append(*h, x.(hnswCandidate)) }
func (h *hnswMinHeap) Pop() any          { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

// NewHNSWFloat32 creates an HNSW index
// M: max connections per node (16-64 typical)
// efConstruction: search width during construction (100-200 typical)
func NewHNSWFloat32(M, efConstruction int) *HNSWFloat32 {
	return &HNSWFloat32{
		M:              M,
		MMax:           M,
		MMax0:          M * 2,
		efConstruction: efConstruction,
		ef:             efConstruction,
		ml:             1.0 / math.Log(float64(M)),
		nodes:          make([]hnswNode, 0, 10000),
		entryPoint:     -1,
	}
}

// SetEf sets search ef parameter
func (h *HNSWFloat32) SetEf(ef int) {
	h.ef = ef
}

// Add inserts a vector
func (h *HNSWFloat32) Add(vec []float32, id int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Compute inverse norm
	var normSq float32
	for _, v := range vec {
		normSq += v * v
	}
	invNorm := float32(1.0 / math.Sqrt(float64(normSq)))

	level := h.randomLevel()

	node := hnswNode{
		ID:        id,
		Vector:    vec,
		InvNorm:   invNorm,
		Level:     level,
		Neighbors: make([][]int, level+1),
	}

	for l := 0; l <= level; l++ {
		cap := h.MMax
		if l == 0 {
			cap = h.MMax0
		}
		node.Neighbors[l] = make([]int, 0, cap)
	}

	if h.entryPoint == -1 {
		h.nodes = append(h.nodes, node)
		h.entryPoint = 0
		return
	}

	// Search for nearest neighbors at each level
	ep := h.entryPoint

	// Traverse from top to insertion level
	for lc := h.nodes[ep].Level; lc > level; lc-- {
		nearest := h.searchLayerSingle(vec, invNorm, ep, lc)
		if nearest >= 0 {
			ep = nearest
		}
	}

	nodeIdx := len(h.nodes)
	h.nodes = append(h.nodes, node)

	// Connect at each level
	for lc := minInt(level, h.nodes[ep].Level); lc >= 0; lc-- {
		candidates := h.searchLayer(vec, invNorm, ep, h.efConstruction, lc)

		m := h.MMax
		if lc == 0 {
			m = h.MMax0
		}

		neighbors := h.selectNeighbors(candidates, m)
		h.nodes[nodeIdx].Neighbors[lc] = neighbors

		// Bidirectional links
		for _, neighbor := range neighbors {
			h.nodes[neighbor].Neighbors[lc] = append(h.nodes[neighbor].Neighbors[lc], nodeIdx)

			maxConn := h.MMax
			if lc == 0 {
				maxConn = h.MMax0
			}
			if len(h.nodes[neighbor].Neighbors[lc]) > maxConn {
				h.pruneConnections(neighbor, lc, maxConn)
			}
		}

		if lc > 0 && len(candidates) > 0 {
			ep = candidates[0].id
		}
	}

	if level > h.nodes[h.entryPoint].Level {
		h.entryPoint = nodeIdx
	}
}

// Search finds k nearest neighbors
func (h *HNSWFloat32) Search(query []float32, k int) []hnswCandidate {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == -1 || len(h.nodes) == 0 {
		return nil
	}

	// Compute query inverse norm
	var normSq float32
	for _, v := range query {
		normSq += v * v
	}
	invNorm := float32(1.0 / math.Sqrt(float64(normSq)))

	ep := h.entryPoint

	// Traverse from top layer
	for level := h.nodes[ep].Level; level > 0; level-- {
		nearest := h.searchLayerSingle(query, invNorm, ep, level)
		if nearest >= 0 {
			ep = nearest
		}
	}

	// Search at layer 0
	ef := h.ef
	if ef < k {
		ef = k
	}
	candidates := h.searchLayer(query, invNorm, ep, ef, 0)

	if len(candidates) > k {
		candidates = candidates[:k]
	}

	return candidates
}

func (h *HNSWFloat32) searchLayer(query []float32, queryInvNorm float32, ep int, ef int, level int) []hnswCandidate {
	visited := make(map[int]bool)

	candidates := &hnswMinHeap{}
	heap.Init(candidates)

	results := &hnswMaxHeap{}
	heap.Init(results)

	dist := h.cosineDist(query, queryInvNorm, ep)
	heap.Push(candidates, hnswCandidate{id: ep, dist: dist})
	heap.Push(results, hnswCandidate{id: ep, dist: dist})
	visited[ep] = true

	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(hnswCandidate)

		if results.Len() > 0 && current.dist > (*results)[0].dist {
			break
		}

		if level >= len(h.nodes[current.id].Neighbors) {
			continue
		}

		for _, neighbor := range h.nodes[current.id].Neighbors[level] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true

			dist := h.cosineDist(query, queryInvNorm, neighbor)

			if results.Len() < ef || dist < (*results)[0].dist {
				heap.Push(candidates, hnswCandidate{id: neighbor, dist: dist})
				heap.Push(results, hnswCandidate{id: neighbor, dist: dist})

				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	// Convert to slice sorted by distance
	result := make([]hnswCandidate, results.Len())
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = heap.Pop(results).(hnswCandidate)
	}

	return result
}

func (h *HNSWFloat32) searchLayerSingle(query []float32, queryInvNorm float32, ep int, level int) int {
	current := ep
	currentDist := h.cosineDist(query, queryInvNorm, current)

	for {
		changed := false

		if level >= len(h.nodes[current].Neighbors) {
			break
		}

		for _, neighbor := range h.nodes[current].Neighbors[level] {
			dist := h.cosineDist(query, queryInvNorm, neighbor)
			if dist < currentDist {
				current = neighbor
				currentDist = dist
				changed = true
			}
		}

		if !changed {
			break
		}
	}

	return current
}

func (h *HNSWFloat32) cosineDist(query []float32, queryInvNorm float32, nodeIdx int) float32 {
	node := &h.nodes[nodeIdx]

	// Unrolled dot product (8 elements at a time for 512 dims)
	var dot float32
	for i := 0; i < 512; i += 8 {
		dot += query[i]*node.Vector[i] + query[i+1]*node.Vector[i+1] +
			query[i+2]*node.Vector[i+2] + query[i+3]*node.Vector[i+3] +
			query[i+4]*node.Vector[i+4] + query[i+5]*node.Vector[i+5] +
			query[i+6]*node.Vector[i+6] + query[i+7]*node.Vector[i+7]
	}

	// Cosine similarity -> distance (1 - sim)
	sim := dot * queryInvNorm * node.InvNorm
	return 1.0 - sim
}

func (h *HNSWFloat32) selectNeighbors(candidates []hnswCandidate, m int) []int {
	if len(candidates) <= m {
		result := make([]int, len(candidates))
		for i, c := range candidates {
			result[i] = c.id
		}
		return result
	}

	result := make([]int, m)
	for i := 0; i < m; i++ {
		result[i] = candidates[i].id
	}
	return result
}

func (h *HNSWFloat32) pruneConnections(nodeIdx int, level int, maxConn int) {
	neighbors := h.nodes[nodeIdx].Neighbors[level]
	if len(neighbors) <= maxConn {
		return
	}

	// Sort by distance to node
	node := &h.nodes[nodeIdx]
	candidates := make([]hnswCandidate, len(neighbors))
	for i, n := range neighbors {
		candidates[i] = hnswCandidate{
			id:   n,
			dist: h.cosineDist(node.Vector, node.InvNorm, n),
		}
	}

	// Simple sort
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].dist < candidates[i].dist {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	h.nodes[nodeIdx].Neighbors[level] = make([]int, maxConn)
	for i := 0; i < maxConn; i++ {
		h.nodes[nodeIdx].Neighbors[level][i] = candidates[i].id
	}
}

func (h *HNSWFloat32) randomLevel() int {
	level := 0
	for rand.Float64() < 1.0/float64(h.M) && level < 16 {
		level++
	}
	return level
}

// NewHNSWSearcher creates an HNSW-backed searcher
func NewHNSWSearcher() (*HNSWSearcher, error) {
	// Try SimpleInt8Model512 first (has EmbedFast)
	simpleModel, err1 := gobed.LoadSimpleInt8Model512()
	if err1 == nil {
		return &HNSWSearcher{
			model:     simpleModel,
			documents: make([]Document, 0, 10000),
			hnsw:      NewHNSWFloat32(32, 200),
		}, nil
	}

	// Fallback to Int8Model512
	int8Model, err2 := gobed.LoadInt8Model512()
	if err2 == nil {
		return &HNSWSearcher{
			model:     &int8ModelEmbedder{int8Model},
			documents: make([]Document, 0, 10000),
			hnsw:      NewHNSWFloat32(32, 200),
		}, nil
	}

	return nil, fmt.Errorf("no suitable model: simple=%v, int8=%v", err1, err2)
}

// int8ModelEmbedder wraps Int8EmbeddingModel512 to implement hnswEmbedder
type int8ModelEmbedder struct {
	model *gobed.Int8EmbeddingModel512
}

func (e *int8ModelEmbedder) EmbedFast(text string) ([]float32, func()) {
	emb, _ := e.model.Embed(text)
	return emb, func() {}
}

func (e *int8ModelEmbedder) Close() error {
	return nil
}

// AddDocument adds a document to the index
func (h *HNSWSearcher) AddDocument(doc Document) error {
	emb, release := h.model.EmbedFast(doc.Content)
	defer release()

	embCopy := make([]float32, len(emb))
	copy(embCopy, emb)

	h.mu.Lock()
	doc.ID = len(h.documents)
	h.documents = append(h.documents, doc)
	h.mu.Unlock()

	h.hnsw.Add(embCopy, doc.ID)
	return nil
}

// BatchAddDocuments adds documents in parallel
func (h *HNSWSearcher) BatchAddDocuments(docs []Document, numWorkers int) error {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	type result struct {
		doc Document
		emb []float32
	}

	results := make([]result, len(docs))
	var wg sync.WaitGroup
	chunkSize := (len(docs) + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * chunkSize
			end := start + chunkSize
			if end > len(docs) {
				end = len(docs)
			}

			for i := start; i < end; i++ {
				emb, release := h.model.EmbedFast(docs[i].Content)
				embCopy := make([]float32, len(emb))
				copy(embCopy, emb)
				release()

				results[i] = result{doc: docs[i], emb: embCopy}
			}
		}(w)
	}

	wg.Wait()

	// Sequential insert into HNSW (required for graph consistency)
	h.mu.Lock()
	for i := range results {
		results[i].doc.ID = len(h.documents)
		h.documents = append(h.documents, results[i].doc)
		h.hnsw.Add(results[i].emb, results[i].doc.ID)
	}
	h.mu.Unlock()

	return nil
}

// Search performs HNSW-accelerated search
func (h *HNSWSearcher) Search(query string, topK int, threshold float32) ([]SearchMatch, error) {
	h.hnsw.SetEf(maxInt(topK*2, 100))

	qEmb, release := h.model.EmbedFast(query)
	defer release()

	qCopy := make([]float32, len(qEmb))
	copy(qCopy, qEmb)

	candidates := h.hnsw.Search(qCopy, topK*2)

	h.mu.RLock()
	defer h.mu.RUnlock()

	results := make([]SearchMatch, 0, topK)
	for _, c := range candidates {
		sim := 1.0 - c.dist
		if sim >= threshold {
			results = append(results, SearchMatch{
				Document:   h.documents[c.id],
				Similarity: sim,
			})
			if len(results) >= topK {
				break
			}
		}
	}

	return results, nil
}

// NumDocuments returns the document count
func (h *HNSWSearcher) NumDocuments() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.documents)
}

// Close releases resources
func (h *HNSWSearcher) Close() error {
	return h.model.Close()
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
