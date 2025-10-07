//go:build amd64 && !noasm
// +build amd64,!noasm

package simd

// Assembly implementations for AVX2
//
//go:noescape
func dot512_i8_avx2(a, b *Vec512) int32

//go:noescape
func dot512_i8_avx2_alt(a, b *Vec512) int32

//go:noescape
func l2squared512_i8_avx2(a, b *Vec512) int32

//go:noescape
func sumabsdiff512_i8_avx2(a, b *Vec512) int32
