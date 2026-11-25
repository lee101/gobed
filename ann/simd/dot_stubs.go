//go:build !amd64 || noasm

package simd

// The VNNI kernel exists only on amd64 (dot_amd64_vnni.go); everything else
// dispatches to it never, but still has to link.
func dot512_i8_vnni(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}
