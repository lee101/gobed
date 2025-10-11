//go:build !cagra
// +build !cagra

package gobed

import (
    "fmt"
    "log"
    "os"
    "path/filepath"
)

// CAGRAConfig captures the subset of configuration fields that tests rely on.
// The full implementation lives behind the cagra build tag.
type CAGRAConfig struct {
    MaxVectors      int
    VectorDim       int
    GraphDegree     int
    MaxIterations   int
    CachePath       string
    TargetLatencyUs int
    TargetRecall    float32
}

// DefaultCAGRAConfig mirrors the exported API so callers do not need build tags.
func DefaultCAGRAConfig() CAGRAConfig {
    cfg := CAGRAConfig{
        MaxVectors:      1_000_000,
        VectorDim:       512,
        GraphDegree:     64,
        MaxIterations:   64,
        TargetLatencyUs: 1000,
        TargetRecall:    0.95,
    }
    cfg.CachePath = BuildCAGRACachePath("default", cfg.VectorDim, cfg.GraphDegree, cfg.MaxVectors)
    return cfg
}

// FastCAGRAConfig returns a speed-optimized configuration without requiring CUDA.
func FastCAGRAConfig() CAGRAConfig {
    cfg := DefaultCAGRAConfig()
    cfg.GraphDegree = 32
    cfg.MaxIterations = 32
    cfg.TargetLatencyUs = 500
    cfg.TargetRecall = 0.90
    cfg.CachePath = BuildCAGRACachePath("fast", cfg.VectorDim, cfg.GraphDegree, cfg.MaxVectors)
    return cfg
}

// QualityCAGRAConfig returns a quality-focused configuration for callers that expect it.
func QualityCAGRAConfig() CAGRAConfig {
    cfg := DefaultCAGRAConfig()
    cfg.GraphDegree = 128
    cfg.MaxIterations = 128
    cfg.TargetLatencyUs = 2000
    cfg.TargetRecall = 0.99
    cfg.CachePath = BuildCAGRACachePath("quality", cfg.VectorDim, cfg.GraphDegree, cfg.MaxVectors)
    return cfg
}

// BuildCAGRACachePath generates the cache path used in tests.
func BuildCAGRACachePath(namespace string, vectorDim, graphDegree, count int) string {
    baseDir := "/mnt/fast/tmp/netwrck_gobed_indexes"

    if os.Getenv("DEV") == "true" || os.Getenv("DEBUG") == "true" {
        baseDir += "_dev"
    }

    if os.Getenv("TESTING") == "true" {
        baseDir = "/mnt/fast/tmp/test_indexes"
    }

    if err := os.MkdirAll(baseDir, 0o755); err != nil {
        log.Printf("Warning: failed to create CAGRA cache directory %s: %v", baseDir, err)
    }

    filename := fmt.Sprintf("%s_cagra_%d_%d_%d.cagrabin", namespace, vectorDim, graphDegree, count)
    return filepath.Join(baseDir, filename)
}
