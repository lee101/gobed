//go:build !amd64 || noasm

package simd

import "github.com/lee101/gobed/internal/i8simd"

// dot512_i8_sdot is the NEON SDOT kernel on arm64: Go assembly in
// internal/i8simd, so a call costs a normal Go call instead of a cgo
// transition (~15ns on Apple M4; Dot512 is 8.7ns, ~3.6x faster than the
// old cgo NEON kernel). It falls back to
// unrolled Go on CPUs without FEAT_DotProd and on other architectures.
func dot512_i8_sdot(a, b *Vec512) int32 {
	return i8simd.Dot512((*[512]int8)(a), (*[512]int8)(b))
}
