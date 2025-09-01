// +build !amd64 noasm

package simd

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