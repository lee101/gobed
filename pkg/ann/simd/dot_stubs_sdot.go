//go:build (!amd64 || noasm) && (!arm64 || !cgo)

package simd

func dot512_i8_sdot(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}
