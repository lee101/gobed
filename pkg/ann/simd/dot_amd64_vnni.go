//go:build amd64 && !noasm

package simd

import "golang.org/x/sys/cpu"

// VNNI implementation - falls back to AVX2 if actual VNNI not available
func dot512_i8_vnni(a, b *Vec512) int32 {
	// For now, we don't have pure assembly VNNI implementation
	// so we fall back to AVX2 which is still much faster than generic
	if cpu.X86.HasAVX2 {
		return dot512_i8_avx2(a, b)
	}
	return dot512_generic(a, b)
}

// Stub for ARM function on AMD64
func dot512_i8_sdot(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}

// Stub for ARM function on AMD64
func l2squared512_i8_neon(a, b *Vec512) int32 {
	return l2squared512_generic(a, b)
}
