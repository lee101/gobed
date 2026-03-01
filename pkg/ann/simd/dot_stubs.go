//go:build !amd64 || noasm

package simd

func dot512_i8_vnni(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}
