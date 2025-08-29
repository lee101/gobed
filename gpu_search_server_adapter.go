// +build !gpu

package gobed

import "github.com/lee101/gobed/ann/simd"

// createGPUIndexConfig creates IndexConfig for non-GPU builds
func createGPUIndexConfig(deviceID int) IndexConfig {
	return IndexConfig{
		VectorDim:        512,
		NumSubquantizers: 8,
		CodebookSize:     256,
		IVFClusters:      1024,
		ProbeLists:       64,
		RerankK:          1000,
		DeviceID:         deviceID,
	}
}

// gpuIndexerSearch performs search with proper type conversion for non-GPU builds
func (s *GPUSearchServer) gpuIndexerSearch(embedding *EmbedInt8Result, k int) ([]int, []float32, error) {
	var vec512 simd.Vec512
	for i, v := range embedding.Vector {
		if i < 512 {
			vec512[i] = v
		}
	}
	return s.gpuIndexer.Search(vec512, embedding.Scale, k)
}

// gpuIndexerAddVectors adds vectors with proper signature for non-GPU builds
func (s *GPUSearchServer) gpuIndexerAddVectors(validEmbeddings [][]int8, validScales []float32) error {
	return s.gpuIndexer.AddVectors(validEmbeddings)
}