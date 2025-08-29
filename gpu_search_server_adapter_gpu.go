// +build gpu

package gobed

// createGPUIndexConfig creates IndexConfig for GPU builds
func createGPUIndexConfig(deviceID int) IndexConfig {
	return IndexConfig{
		VectorDim: 512,   // gobed embedding dimension (updated to 512)
		VocabSize: 30522, // BERT vocab size
		EmbedDim:  512,   // gobed embedding dimension (updated to 512)
		DeviceID:  deviceID,
	}
}

// gpuIndexerSearch performs search with proper type conversion for GPU builds
func (s *GPUSearchServer) gpuIndexerSearch(embedding *EmbedInt8Result, k int) ([]int, []float32, error) {
	int8Vec := make([]int8, len(embedding.Vector))
	for i, v := range embedding.Vector {
		int8Vec[i] = v
	}
	indices32, scores, err := s.gpuIndexer.Search(int8Vec, embedding.Scale, k)
	if err != nil {
		return nil, nil, err
	}
	
	// Convert int32 to int
	indices := make([]int, len(indices32))
	for i, idx := range indices32 {
		indices[i] = int(idx)
	}
	return indices, scores, nil
}

// gpuIndexerAddVectors adds vectors with proper signature for GPU builds
func (s *GPUSearchServer) gpuIndexerAddVectors(validEmbeddings [][]int8, validScales []float32) error {
	return s.gpuIndexer.AddVectors(validEmbeddings, validScales)
}