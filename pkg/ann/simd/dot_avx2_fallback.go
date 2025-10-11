//go:build !amd64 || noasm

package simd

import "unsafe"

// Fallback implementation for non-AMD64 or no-assembly builds
func dot512_i8_avx2(a, b *Vec512) int32 {
	// Use generic implementation on non-x86 platforms
	return dot512_generic(a, b)
}

func dot512_i8_avx2_alt(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}

func l2squared512_i8_avx2(a, b *Vec512) int32 {
	return l2squared512_generic(a, b)
}

func l2squared512_i8_neon(a, b *Vec512) int32 {
	return l2squared512_generic(a, b)
}

func sumabsdiff512_i8_avx2(a, b *Vec512) int32 {
	return SumAbsDiff512(a, b)
}

func dot512_i8_batch_avx2(query *Vec512, vectors *Vec512, scores *int32, count int) {
	if count <= 0 {
		return
	}
	vecSlice := unsafe.Slice(vectors, count)
	scoreSlice := unsafe.Slice(scores, count)
	dot512_batch_generic(query, vecSlice, scoreSlice)
}
