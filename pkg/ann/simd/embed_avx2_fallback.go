//go:build !amd64 || noasm

package simd

// EmbedAccum accumulates embedding * scale into result (fallback)
func EmbedAccum(embedding []int8, scale float32, result []float32) {
	embedAccumFallback(embedding, scale, result)
}

// EmbedAccumBatch accumulates multiple embeddings into result (fallback)
func EmbedAccumBatch(embeddings [][]int8, scales []float32, result []float32) {
	embedAccumBatchFallback(embeddings, scales, result)
}

// ClearF32 zeros float32 values (fallback)
func ClearF32(buf []float32) {
	for i := range buf {
		buf[i] = 0
	}
}

// ScaleF32 multiplies float32 values by scale (fallback)
func ScaleF32(buf []float32, scale float32) {
	for i := range buf {
		buf[i] *= scale
	}
}
