//go:build (!amd64 || noasm) && (!arm64 || !cgo)

package simd

// Generic sdot for builds that have neither the amd64 stub (dot_amd64_vnni.go)
// nor the arm64 cgo NEON kernel (dot_arm64_sdot.go).
func dot512_i8_sdot(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}
