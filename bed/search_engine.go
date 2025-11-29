package main

import (
	"log"
	"time"
	"unsafe"
)

// SearchEngine interface for different search implementations
type SearchEngine interface {
	AddDocuments(embeddings []int8, numDocs, dim int)
	Search(query []int8, k int) (indices []int32, scores []float32, numResults int)
	Destroy()
	GetMode() string
}

// GPUSearchEngine using CUDA
type GPUSearchEngine struct {
	handle unsafe.Pointer
}

func NewGPUSearchEngine(maxDocs, dim, maxK int) (*GPUSearchEngine, error) {
	handle := C.create_unique_topk_search(
		C.int(maxDocs),
		C.int(dim),
		C.int(maxK),
	)
	if handle == nil {
		return nil, fmt.Errorf("failed to create GPU search index")
	}
	return &GPUSearchEngine{handle: handle}, nil
}

func (g *GPUSearchEngine) AddDocuments(embeddings []int8, numDocs, dim int) {
	C.add_documents_topk(
		g.handle,
		(*C.schar)(unsafe.Pointer(&embeddings[0])),
		C.int(numDocs),
		C.int(dim),
	)
}

func (g *GPUSearchEngine) Search(query []int8, k int) ([]int32, []float32, int) {
	indices := make([]int32, k)
	scores := make([]float32, k)

	numResults := int(C.search_topk_unique(
		g.handle,
		(*C.schar)(unsafe.Pointer(&query[0])),
		C.int(len(query)),
		C.int(k),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	))

	return indices, scores, numResults
}

func (g *GPUSearchEngine) Destroy() {
	if g.handle != nil {
		C.destroy_unique_topk_search(g.handle)
	}
}

func (g *GPUSearchEngine) GetMode() string {
	return "GPU"
}

// CPUSearchEngine using brute-force search
type CPUSearchEngine struct {
	embeddings []int8
	numDocs    int
	dim        int
}

func NewCPUSearchEngine(maxDocs, dim, maxK int) *CPUSearchEngine {
	return &CPUSearchEngine{
		dim: dim,
	}
}

func (c *CPUSearchEngine) AddDocuments(embeddings []int8, numDocs, dim int) {
	c.embeddings = embeddings
	c.numDocs = numDocs
	c.dim = dim
}

func (c *CPUSearchEngine) Search(query []int8, k int) ([]int32, []float32, int) {
	indices := make([]int32, k)
	scores := make([]float32, k)

	numResults := cpuSearch(query, c.embeddings, c.numDocs, c.dim, k, indices, scores)

	return indices, scores, numResults
}

func (c *CPUSearchEngine) Destroy() {
	// Nothing to clean up for CPU engine
}

func (c *CPUSearchEngine) GetMode() string {
	return "CPU"
}

// CreateSearchEngine auto-detects GPU and creates appropriate engine
func CreateSearchEngine(numDocs, dim int, forceGPU, forceCPU, debug bool) SearchEngine {
	maxK := 100

	// Determine which engine to use
	if forceCPU {
		if debug {
			log.Println("Using CPU engine (forced)")
		}
		return NewCPUSearchEngine(numDocs+100, dim, maxK)
	}

	if forceGPU || (CheckGPUAvailable() && numDocs > 100) {
		engine, err := NewGPUSearchEngine(numDocs+100, dim, maxK)
		if err == nil {
			if debug {
				log.Printf("Using GPU engine for %d documents", numDocs)
			}
			return engine
		}
		if debug {
			log.Printf("GPU initialization failed: %v, falling back to CPU", err)
		}
	}

	if debug {
		log.Printf("Using CPU engine for %d documents", numDocs)
	}
	return NewCPUSearchEngine(numDocs+100, dim, maxK)
}