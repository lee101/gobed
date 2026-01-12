// +build gpu

package gobed

/*
#cgo CFLAGS: -I./gpu
#cgo LDFLAGS: -L./gpu -lgpu_cagra -L/usr/local/cuda-12.9/lib64 -lcudart -lcublas -Wl,-rpath,./gpu -Wl,-rpath,/usr/local/cuda-12.9/lib64
#include <stdlib.h>

typedef struct gpu_cagra_t gpu_cagra_t;

gpu_cagra_t* gpu_cagra_create(int nlist, int degree, int vector_dim);
int gpu_cagra_build(gpu_cagra_t* h,
                    const signed char* vectors, const float* scales, const int* ids,
                    const int* list_ids, int num_vectors);
int gpu_cagra_search_lists(gpu_cagra_t* h,
                           const signed char* query, float qscale,
                           const int* list_ids, int nprobe, int k,
                           int* result_ids, float* result_scores);
void gpu_cagra_destroy(gpu_cagra_t* h);
*/
import "C"

import (
    "fmt"
    "unsafe"

    "github.com/lee101/gobed/pkg/ann/ivf"
    "github.com/lee101/gobed/pkg/ann/simd"
)

// GPUCagraConfig configures the GPU graph search (CAGRA-style) indexer.
type GPUCagraConfig struct {
    NList     int
    Degree    int
    NProbe    int
    VectorDim int
}

func DefaultGPUCagraConfig() GPUCagraConfig {
    return GPUCagraConfig{ NList: 1024, Degree: 32, NProbe: 8, VectorDim: 512 }
}

// GPUCagraIndexer builds CPU IVF routing + GPU-restricted candidate search.
type GPUCagraIndexer struct {
    handle *C.gpu_cagra_t
    cfg    GPUCagraConfig
    km     *ivf.KMeans
}

// NewGPUCagraIndexer creates the indexer.
func NewGPUCagraIndexer(cfg GPUCagraConfig) (*GPUCagraIndexer, error) {
    if !IsCUDAAvailable() { return nil, fmt.Errorf("CUDA not available") }
    h := C.gpu_cagra_create(C.int(cfg.NList), C.int(cfg.Degree), C.int(cfg.VectorDim))
    if h == nil { return nil, fmt.Errorf("failed to create gpu cagra indexer") }
    return &GPUCagraIndexer{ handle: h, cfg: cfg }, nil
}

// Build trains IVF on CPU and uploads list-grouped data to GPU for fast list-restricted search.
func (g *GPUCagraIndexer) Build(vectors []simd.Vec512, scales []float32, ids []int) error {
    if len(vectors) == 0 || len(vectors) != len(scales) || len(ids) != len(vectors) {
        return fmt.Errorf("invalid input sizes")
    }
    km := ivf.NewKMeans(g.cfg.NList, 25)
    km.Fit(vectors, scales)
    // Assign vectors to lists
    listIDs := make([]int32, len(vectors))
    for i := range vectors {
        lid := km.Predict(&vectors[i])
        listIDs[i] = int32(lid)
    }
    // Flatten inputs
    flatVecs := make([]int8, len(vectors)*g.cfg.VectorDim)
    for i := range vectors { copy(flatVecs[i*g.cfg.VectorDim:(i+1)*g.cfg.VectorDim], vectors[i][:]) }
    // cgo pointers
    res := C.gpu_cagra_build(
        g.handle,
        (*C.schar)(unsafe.Pointer(&flatVecs[0])),
        (*C.float)(unsafe.Pointer(&scales[0])),
        (*C.int)(unsafe.Pointer(&ids[0])),
        (*C.int)(unsafe.Pointer(&listIDs[0])),
        C.int(len(vectors)),
    )
    if int(res) == 0 { return fmt.Errorf("gpu cagra build failed") }
    g.km = km
    return nil
}

// Search returns top-k ids and scores using IVF routing on CPU and GPU list-constrained scoring.
func (g *GPUCagraIndexer) Search(q *simd.Vec512, qScale float32, k int) ([]int32, []float32, error) {
    if g.km == nil { return nil, nil, fmt.Errorf("index not built") }
    // Choose lists via CPU KMeans
    lists := g.km.PredictMultiple(q, g.cfg.NProbe)
    if len(lists) == 0 { return nil, nil, fmt.Errorf("no lists predicted") }
    // prepare c arrays
    cLists := make([]C.int, len(lists))
    for i, l := range lists { cLists[i] = C.int(l) }
    // query buffer
    qFlat := (*[512]int8)(unsafe.Pointer(q))
    ids := make([]int32, k)
    scores := make([]float32, k)
    res := C.gpu_cagra_search_lists(
        g.handle,
        (*C.schar)(unsafe.Pointer(&qFlat[0])),
        C.float(qScale),
        (*C.int)(unsafe.Pointer(&cLists[0])),
        C.int(len(cLists)),
        C.int(k),
        (*C.int)(unsafe.Pointer(&ids[0])),
        (*C.float)(unsafe.Pointer(&scores[0])),
    )
    if int(res) == 0 { return nil, nil, fmt.Errorf("gpu cagra search failed") }
    return ids, scores, nil
}

func (g *GPUCagraIndexer) Close() { if g.handle != nil { C.gpu_cagra_destroy(g.handle); g.handle = nil } }

