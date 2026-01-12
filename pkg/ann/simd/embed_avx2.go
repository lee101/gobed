//go:build amd64 && !noasm

package simd

// embed_accum_avx2 accumulates int8 embedding * scale into float32 result (512 dims)
//
//go:noescape
func embed_accum_avx2(embedding *int8, scale float32, result *float32)

// embed_accum_batch_avx2 accumulates multiple embeddings with scales into result
//
//go:noescape
func embed_accum_batch_avx2(embeddings **int8, scales *float32, result *float32, count int)

// clear_f32_avx2 zeros 512 float32 values
//
//go:noescape
func clear_f32_avx2(buf *float32)

// scale_f32_avx2 multiplies 512 float32 values by scale
//
//go:noescape
func scale_f32_avx2(buf *float32, scale float32)

// EmbedAccum accumulates embedding * scale into result using AVX2
func EmbedAccum(embedding []int8, scale float32, result []float32) {
	if len(embedding) != 512 || len(result) != 512 {
		embedAccumFallback(embedding, scale, result)
		return
	}
	embed_accum_avx2(&embedding[0], scale, &result[0])
}

// EmbedAccumBatch accumulates multiple embeddings into result using AVX2
func EmbedAccumBatch(embeddings [][]int8, scales []float32, result []float32) {
	if len(result) != 512 {
		return
	}
	if len(embeddings) == 0 {
		return
	}

	// Build pointer array
	ptrs := make([]*int8, len(embeddings))
	for i := range embeddings {
		if len(embeddings[i]) != 512 {
			embedAccumBatchFallback(embeddings, scales, result)
			return
		}
		ptrs[i] = &embeddings[i][0]
	}

	embed_accum_batch_avx2(&ptrs[0], &scales[0], &result[0], len(embeddings))
}

// ClearF32 zeros 512 float32 values using AVX2
func ClearF32(buf []float32) {
	if len(buf) != 512 {
		for i := range buf {
			buf[i] = 0
		}
		return
	}
	clear_f32_avx2(&buf[0])
}

// ScaleF32 multiplies 512 float32 values by scale using AVX2
func ScaleF32(buf []float32, scale float32) {
	if len(buf) != 512 {
		for i := range buf {
			buf[i] *= scale
		}
		return
	}
	scale_f32_avx2(&buf[0], scale)
}
