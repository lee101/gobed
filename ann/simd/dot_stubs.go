//go:build (!amd64 && !arm64) || !cgo
// +build !amd64,!arm64 !cgo

package simd

// Stub implementations for architectures without SIMD or when CGO is disabled

func dot512_i8_vnni(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}

func dot512_i8_sdot(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}