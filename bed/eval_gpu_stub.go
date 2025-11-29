//go:build !gpu

package bed

import (
    "fmt"
    gobed "github.com/lee101/gobed"
)

type EvalGPUResult struct {
    K, NumQueries int
    P50SearchMs float64
    P95SearchMs float64
    P50EndToEndMs float64
    P95EndToEndMs float64
    QPS float64
    RecallAtK float64
}

func RunEvalGPU(model *gobed.EmbeddingModel, docs []gobed.Document, base EvalConfig, graph gobed.GPUCagraConfig) (EvalGPUResult, error) {
    return EvalGPUResult{}, fmt.Errorf("GPU eval not available: build with -tags gpu and ensure CUDA libs are built")
}
