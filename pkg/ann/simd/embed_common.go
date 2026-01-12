package simd

// embedAccumFallback is the pure Go implementation
func embedAccumFallback(embedding []int8, scale float32, result []float32) {
	n := len(embedding)
	if n > len(result) {
		n = len(result)
	}
	// Unrolled for better performance
	i := 0
	for ; i+7 < n; i += 8 {
		result[i] += float32(embedding[i]) * scale
		result[i+1] += float32(embedding[i+1]) * scale
		result[i+2] += float32(embedding[i+2]) * scale
		result[i+3] += float32(embedding[i+3]) * scale
		result[i+4] += float32(embedding[i+4]) * scale
		result[i+5] += float32(embedding[i+5]) * scale
		result[i+6] += float32(embedding[i+6]) * scale
		result[i+7] += float32(embedding[i+7]) * scale
	}
	for ; i < n; i++ {
		result[i] += float32(embedding[i]) * scale
	}
}

// embedAccumBatchFallback accumulates multiple embeddings
func embedAccumBatchFallback(embeddings [][]int8, scales []float32, result []float32) {
	for i, emb := range embeddings {
		if i >= len(scales) {
			break
		}
		embedAccumFallback(emb, scales[i], result)
	}
}

// EmbedTokensIntoSIMD is the optimized embedding function using SIMD
// Returns number of valid tokens processed
func EmbedTokensIntoSIMD(embeddings [][]int8, scales []float32, tokens []int16, result []float32) int {
	if len(result) != 512 {
		return 0
	}

	// Clear result
	ClearF32(result)

	validTokens := 0
	vocabLen := len(embeddings)

	// Collect valid embeddings and scales
	validEmbs := make([][]int8, 0, len(tokens))
	validScales := make([]float32, 0, len(tokens))

	for _, token := range tokens {
		if token < 0 || int(token) >= vocabLen {
			continue
		}
		validEmbs = append(validEmbs, embeddings[token])
		validScales = append(validScales, scales[token])
		validTokens++
	}

	if validTokens == 0 {
		return 0
	}

	// Use batch SIMD accumulation
	EmbedAccumBatch(validEmbs, validScales, result)

	// Average
	if validTokens > 1 {
		ScaleF32(result, 1.0/float32(validTokens))
	}

	return validTokens
}
