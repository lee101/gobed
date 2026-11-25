//go:build !(cgo && (amd64 || gobed_cgo_int8))

package gobed

// No cgo int8 kernels in this build; the pure-Go ones are used everywhere.
const (
	haveCgoInt8     = false
	useCgoInt8      = false
	cgoInt8Runnable = false
)

func quantizeRowInt8Cgo(in []float32, out []int8, scale float32, zeroPoint int8) {
	quantizeRowInt8Go(in, out, scale, zeroPoint)
}

func embedInt8Cgo(flat []int8, tokenIDs []int, dim, vocab int, out []uint8, scale float32) {
	embedInt8Go(flat, tokenIDs, dim, vocab, out, scale)
}

func cosineInt8Cgo(a, b []uint8) float32 { return cosineInt8Go(a, b) }
