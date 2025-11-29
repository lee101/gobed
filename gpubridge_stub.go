//go:build !gpu

package gobed

// GPUCagraConfig is defined in gpu builds; provide a stub for non-GPU builds
type GPUCagraConfig struct {
    NList     int
    Degree    int
    NProbe    int
    VectorDim int
}

