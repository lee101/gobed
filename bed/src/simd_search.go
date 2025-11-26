package src

import (
	"container/heap"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/ann/simd"
)

// OptimizedSearchEngine provides SIMD-accelerated semantic search
// It can use either int8 quantized embeddings or float32 embeddings
type OptimizedSearchEngine struct {
	// Model (either int8 or float32)
	int8Model    *gobed.Int8EmbeddingModel512
	float32Model *gobed.EmbeddingModel
	useInt8      bool

	// Contiguous storage for cache-friendly int8 access
	int8Embeddings []int8
	int8Scales     []float32

	// Float32 embeddings storage (when int8 not available)
	float32Embeddings []float32
	embeddingDim      int

	// Document metadata
	documents []Document
	numDocs   int

	mu sync.RWMutex
}

// Int8SearchEngine is an alias for backwards compatibility
type Int8SearchEngine = OptimizedSearchEngine

// NewInt8SearchEngine creates a new SIMD-accelerated search engine
// Falls back to float32 model if int8 model unavailable
func NewInt8SearchEngine() (*OptimizedSearchEngine, error) {
	return NewOptimizedSearchEngine()
}

// NewOptimizedSearchEngine creates a new optimized search engine
func NewOptimizedSearchEngine() (*OptimizedSearchEngine, error) {
	engine := &OptimizedSearchEngine{
		documents: make([]Document, 0),
	}

	// Try int8 model first (preferred for SIMD)
	if int8Model, err := gobed.LoadInt8Model512(); err == nil {
		engine.int8Model = int8Model
		engine.useInt8 = true
		engine.int8Embeddings = make([]int8, 0)
		engine.int8Scales = make([]float32, 0)
		engine.embeddingDim = 512
		return engine, nil
	}

	// Fallback to float32 model
	float32Model, err := gobed.LoadModel()
	if err != nil {
		return nil, err
	}

	engine.float32Model = float32Model
	engine.useInt8 = false
	engine.float32Embeddings = make([]float32, 0)
	engine.embeddingDim = float32Model.EmbedDim

	return engine, nil
}

// searchResult for the heap
type searchResult struct {
	docID      int
	similarity float32
}

// minHeap implements a min-heap for top-k selection
type minHeap []searchResult

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].similarity < h[j].similarity }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(searchResult)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// AddDocument adds a document with an int8 embedding
func (e *OptimizedSearchEngine) AddDocument(doc Document, embedding *gobed.Int8Result512) {
	if !e.useInt8 {
		return // Can't add int8 to float32 engine
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	doc.ID = e.numDocs
	e.documents = append(e.documents, doc)
	e.int8Embeddings = append(e.int8Embeddings, embedding.Vector...)
	e.int8Scales = append(e.int8Scales, embedding.Scale)
	e.numDocs++
}

// AddDocumentFloat32 adds a document with a float32 embedding
func (e *OptimizedSearchEngine) AddDocumentFloat32(doc Document, embedding []float32) {
	e.mu.Lock()
	defer e.mu.Unlock()

	doc.ID = e.numDocs
	e.documents = append(e.documents, doc)
	e.float32Embeddings = append(e.float32Embeddings, embedding...)
	e.numDocs++
}

// Search performs optimized parallel search
func (e *OptimizedSearchEngine) Search(query string, topK int, threshold float32) ([]SearchMatch, error) {
	if e.useInt8 {
		return e.searchInt8(query, topK, threshold)
	}
	return e.searchFloat32(query, topK, threshold)
}

// searchInt8 performs SIMD-accelerated search with int8 embeddings
func (e *OptimizedSearchEngine) searchInt8(query string, topK int, threshold float32) ([]SearchMatch, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.numDocs == 0 {
		return nil, nil
	}

	// Embed query
	queryResult, err := e.int8Model.EmbedInt8(query)
	if err != nil {
		return nil, err
	}

	// Convert query to Vec512 for SIMD
	var queryVec simd.Vec512
	copy(queryVec[:], queryResult.Vector)

	// Parallel search across CPU cores
	numWorkers := runtime.NumCPU()
	if numWorkers > e.numDocs {
		numWorkers = e.numDocs
	}

	chunkSize := (e.numDocs + numWorkers - 1) / numWorkers
	workerResults := make([]minHeap, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			start := workerID * chunkSize
			end := start + chunkSize
			if end > e.numDocs {
				end = e.numDocs
			}

			h := make(minHeap, 0, topK+1)
			heap.Init(&h)

			queryScale := queryResult.Scale
			queryNormSq := computeNormSquaredInt8(&queryVec)

			for i := start; i < end; i++ {
				docOffset := i * 512
				var docVec simd.Vec512
				copy(docVec[:], e.int8Embeddings[docOffset:docOffset+512])
				docScale := e.int8Scales[i]

				// SIMD dot product
				dotInt := simd.Dot512(&queryVec, &docVec)
				docNormSq := computeNormSquaredInt8(&docVec)

				// Cosine similarity with scales
				scaledDot := float32(dotInt) * queryScale * docScale
				scaledNorm1 := float32(queryNormSq) * queryScale * queryScale
				scaledNorm2 := float32(docNormSq) * docScale * docScale

				var similarity float32
				if scaledNorm1 > 0 && scaledNorm2 > 0 {
					similarity = scaledDot / fastSqrt(scaledNorm1*scaledNorm2)
				}

				if similarity >= threshold {
					if h.Len() < topK {
						heap.Push(&h, searchResult{docID: i, similarity: similarity})
					} else if similarity > h[0].similarity {
						heap.Pop(&h)
						heap.Push(&h, searchResult{docID: i, similarity: similarity})
					}
				}
			}

			workerResults[workerID] = h
		}(w)
	}

	wg.Wait()

	// Merge and return results
	return e.mergeResults(workerResults, topK), nil
}

// searchFloat32 performs parallel search with float32 embeddings
func (e *OptimizedSearchEngine) searchFloat32(query string, topK int, threshold float32) ([]SearchMatch, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.numDocs == 0 {
		return nil, nil
	}

	// Embed query
	queryEmb, err := e.float32Model.Encode(query)
	if err != nil {
		return nil, err
	}

	// Parallel search
	numWorkers := runtime.NumCPU()
	if numWorkers > e.numDocs {
		numWorkers = e.numDocs
	}

	chunkSize := (e.numDocs + numWorkers - 1) / numWorkers
	workerResults := make([]minHeap, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			start := workerID * chunkSize
			end := start + chunkSize
			if end > e.numDocs {
				end = e.numDocs
			}

			h := make(minHeap, 0, topK+1)
			heap.Init(&h)

			dim := e.embeddingDim

			for i := start; i < end; i++ {
				docOffset := i * dim
				docEmb := e.float32Embeddings[docOffset : docOffset+dim]

				similarity := cosineSimFloat32(queryEmb, docEmb)

				if similarity >= threshold {
					if h.Len() < topK {
						heap.Push(&h, searchResult{docID: i, similarity: similarity})
					} else if similarity > h[0].similarity {
						heap.Pop(&h)
						heap.Push(&h, searchResult{docID: i, similarity: similarity})
					}
				}
			}

			workerResults[workerID] = h
		}(w)
	}

	wg.Wait()

	return e.mergeResults(workerResults, topK), nil
}

// mergeResults merges worker heaps into final results
func (e *OptimizedSearchEngine) mergeResults(workerResults []minHeap, topK int) []SearchMatch {
	finalHeap := make(minHeap, 0, topK+1)
	heap.Init(&finalHeap)

	for _, wr := range workerResults {
		for _, r := range wr {
			if finalHeap.Len() < topK {
				heap.Push(&finalHeap, r)
			} else if r.similarity > finalHeap[0].similarity {
				heap.Pop(&finalHeap)
				heap.Push(&finalHeap, r)
			}
		}
	}

	// Extract in descending order
	results := make([]SearchMatch, finalHeap.Len())
	for i := len(results) - 1; i >= 0; i-- {
		r := heap.Pop(&finalHeap).(searchResult)
		results[i] = SearchMatch{
			Document:   e.documents[r.docID],
			Similarity: r.similarity,
		}
	}

	return results
}

// BatchAddDocuments adds multiple documents efficiently
func (e *OptimizedSearchEngine) BatchAddDocuments(docs []Document, numWorkers int) error {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	if e.useInt8 {
		return e.batchAddInt8(docs, numWorkers)
	}
	return e.batchAddFloat32(docs, numWorkers)
}

func (e *OptimizedSearchEngine) batchAddInt8(docs []Document, numWorkers int) error {
	type result struct {
		doc       Document
		embedding *gobed.Int8Result512
		err       error
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
				emb, err := e.int8Model.EmbedInt8(docs[i].Content)
				results[i] = result{doc: docs[i], embedding: emb, err: err}
			}
		}(w)
	}

	wg.Wait()

	e.mu.Lock()
	defer e.mu.Unlock()

	for _, r := range results {
		if r.err == nil && r.embedding != nil {
			r.doc.ID = e.numDocs
			e.documents = append(e.documents, r.doc)
			e.int8Embeddings = append(e.int8Embeddings, r.embedding.Vector...)
			e.int8Scales = append(e.int8Scales, r.embedding.Scale)
			e.numDocs++
		}
	}

	return nil
}

func (e *OptimizedSearchEngine) batchAddFloat32(docs []Document, numWorkers int) error {
	type result struct {
		doc       Document
		embedding []float32
		err       error
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
				emb, err := e.float32Model.Encode(docs[i].Content)
				results[i] = result{doc: docs[i], embedding: emb, err: err}
			}
		}(w)
	}

	wg.Wait()

	e.mu.Lock()
	defer e.mu.Unlock()

	for _, r := range results {
		if r.err == nil && r.embedding != nil {
			r.doc.ID = e.numDocs
			e.documents = append(e.documents, r.doc)
			e.float32Embeddings = append(e.float32Embeddings, r.embedding...)
			e.numDocs++
		}
	}

	return nil
}

// NumDocuments returns the number of indexed documents
func (e *OptimizedSearchEngine) NumDocuments() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.numDocs
}

// Clear removes all documents from the index
func (e *OptimizedSearchEngine) Clear() {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.int8Embeddings = e.int8Embeddings[:0]
	e.int8Scales = e.int8Scales[:0]
	e.float32Embeddings = e.float32Embeddings[:0]
	e.documents = e.documents[:0]
	e.numDocs = 0
}

// UseInt8 returns whether int8 mode is active
func (e *OptimizedSearchEngine) UseInt8() bool {
	return e.useInt8
}

// Helper functions

func computeNormSquaredInt8(v *simd.Vec512) int32 {
	var sum int32
	for i := 0; i < 512; i += 8 {
		sum += int32(v[i]) * int32(v[i])
		sum += int32(v[i+1]) * int32(v[i+1])
		sum += int32(v[i+2]) * int32(v[i+2])
		sum += int32(v[i+3]) * int32(v[i+3])
		sum += int32(v[i+4]) * int32(v[i+4])
		sum += int32(v[i+5]) * int32(v[i+5])
		sum += int32(v[i+6]) * int32(v[i+6])
		sum += int32(v[i+7]) * int32(v[i+7])
	}
	return sum
}

func fastSqrt(x float32) float32 {
	if x <= 0 {
		return 0
	}
	i := *(*uint32)(unsafe.Pointer(&x))
	i = 0x5f3759df - (i >> 1)
	y := *(*float32)(unsafe.Pointer(&i))
	y = y * (1.5 - (x*0.5)*y*y)
	return 1.0 / y
}

func cosineSimFloat32(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float32
	// Unroll by 4
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		dot += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		normA += a[i]*a[i] + a[i+1]*a[i+1] + a[i+2]*a[i+2] + a[i+3]*a[i+3]
		normB += b[i]*b[i] + b[i+1]*b[i+1] + b[i+2]*b[i+2] + b[i+3]*b[i+3]
	}
	for ; i < n; i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / float32(math.Sqrt(float64(normA*normB)))
}
