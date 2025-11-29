package main

// #cgo LDFLAGS: -L. -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda_search
// #include <stdlib.h>
// extern void* create_max_performance_index(int max_vectors, int dim, int max_batch_size);
// extern void destroy_max_performance_index(void* handle);
// extern int add_vectors_cuda(void* handle, const signed char* vectors, const float* scales, int num_vectors);
// extern int search_batch_cuda(void* handle, const signed char* queries, int num_queries, int k,
//                              int* result_indices, float* result_scores);
import "C"
import (
	"fmt"
	"unsafe"
	"sync"
)

// GPUSearchIndex wraps the CUDA search implementation
type GPUSearchIndex struct {
	handle   unsafe.Pointer
	dim      int
	capacity int
	count    int
	mu       sync.RWMutex
}

// NewGPUSearchIndex creates a new GPU-accelerated search index
func NewGPUSearchIndex(maxVectors, dim int) (*GPUSearchIndex, error) {
	handle := C.create_max_performance_index(
		C.int(maxVectors),
		C.int(dim),
		C.int(10000), // max batch size
	)

	if handle == nil {
		return nil, fmt.Errorf("failed to create GPU index")
	}

	return &GPUSearchIndex{
		handle:   handle,
		dim:      dim,
		capacity: maxVectors,
		count:    0,
	}, nil
}

// AddVectors adds int8 quantized vectors to the GPU index
func (g *GPUSearchIndex) AddVectors(vectors []int8, scales []float32) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	numVectors := len(scales)
	if len(vectors) != numVectors*g.dim {
		return fmt.Errorf("invalid vector dimensions")
	}

	if g.count+numVectors > g.capacity {
		return fmt.Errorf("index capacity exceeded")
	}

	ret := C.add_vectors_cuda(
		g.handle,
		(*C.schar)(unsafe.Pointer(&vectors[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int(numVectors),
	)

	if ret != 0 {
		return fmt.Errorf("failed to add vectors to GPU")
	}

	g.count += numVectors
	return nil
}

// Search performs k-NN search on GPU
func (g *GPUSearchIndex) Search(queries []int8, k int) ([]int, []float32, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	numQueries := len(queries) / g.dim
	if len(queries) != numQueries*g.dim {
		return nil, nil, fmt.Errorf("invalid query dimensions")
	}

	indices := make([]int32, numQueries*k)
	scores := make([]float32, numQueries*k)

	ret := C.search_batch_cuda(
		g.handle,
		(*C.schar)(unsafe.Pointer(&queries[0])),
		C.int(numQueries),
		C.int(k),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	if ret != 0 {
		return nil, nil, fmt.Errorf("GPU search failed")
	}

	// Convert int32 to int
	intIndices := make([]int, len(indices))
	for i, idx := range indices {
		intIndices[i] = int(idx)
	}

	return intIndices, scores, nil
}

// Destroy releases GPU resources
func (g *GPUSearchIndex) Destroy() {
	if g.handle != nil {
		C.destroy_max_performance_index(g.handle)
		g.handle = nil
	}
}

// QuantizeEmbedding converts float32 embedding to int8 with scale
func QuantizeEmbedding(embedding []float32) ([]int8, float32) {
	// Find min/max for quantization
	minVal, maxVal := embedding[0], embedding[0]
	for _, v := range embedding {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Quantize to int8
	scale := (maxVal - minVal) / 255.0
	if scale == 0 {
		scale = 1.0
	}

	quantized := make([]int8, len(embedding))
	for i, v := range embedding {
		q := int((v - minVal) / scale)
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		quantized[i] = int8(q - 128)
	}

	return quantized, scale
}