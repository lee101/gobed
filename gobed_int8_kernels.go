package gobed

import (
	"math"

	"github.com/lee101/gobed/internal/i8simd"
)

// Int8 kernel dispatch. useCgoInt8 is a build-time constant: true only for
// cgo builds on amd64 (AVX-512 C kernels). Everywhere else the pure-Go
// kernels run: NEON SDOT Go assembly on arm64 (no cgo transition), unrolled
// Go otherwise. Results match the scalar C kernels bit for bit (see
// gobed_int8_kernels_test.go); quantization of exact .5 ties differs from the
// AVX-512 C path, which rounds ties to even instead of away from zero.

func quantizeRowInt8(in []float32, out []int8, scale float32, zeroPoint int8) {
	if useCgoInt8 {
		quantizeRowInt8Cgo(in, out, scale, zeroPoint)
		return
	}
	quantizeRowInt8Go(in, out, scale, zeroPoint)
}

func embedInt8(flat []int8, tokenIDs []int, dim, vocab int, out []uint8, scale float32) {
	if useCgoInt8 {
		embedInt8Cgo(flat, tokenIDs, dim, vocab, out, scale)
		return
	}
	embedInt8Go(flat, tokenIDs, dim, vocab, out, scale)
}

func cosineInt8(a, b []uint8) float32 {
	if useCgoInt8 {
		return cosineInt8Cgo(a, b)
	}
	return cosineInt8Go(a, b)
}

// quantizeRowInt8Go mirrors the scalar C quantize_weights_avx512:
// q = clamp(roundf(x*scale) + zeroPoint, -128, 127), roundf = ties away from 0.
func quantizeRowInt8Go(in []float32, out []int8, scale float32, zeroPoint int8) {
	n := min(len(in), len(out))
	in, out = in[:n], out[:n]
	for i, x := range in {
		r := math.Round(float64(float32(x * scale)))
		// Clamp before the int conversion; the C code's (int32_t) cast of an
		// out-of-range float is undefined, so any saturating choice is valid.
		if r < -1024 {
			r = -1024
		} else if r > 1024 {
			r = 1024
		}
		q := int32(r) + int32(zeroPoint)
		if q < -128 {
			q = -128
		} else if q > 127 {
			q = 127
		}
		out[i] = int8(q)
	}
}

// embedInt8Go mirrors compute_embedding_int8_avx512: mean-pool the int8 rows
// of valid token IDs, then map each dim to uint8 via ((mean/scale)+1)*127.5.
func embedInt8Go(flat []int8, tokenIDs []int, dim, vocab int, out []uint8, scale float32) {
	var stack [1024]int32
	var acc []int32
	if dim <= len(stack) {
		acc = stack[:dim]
	} else {
		acc = make([]int32, dim)
	}
	var offStack [256]int
	offs := offStack[:0]
	valid := 0
	for _, id := range tokenIDs {
		if id < 0 || id >= vocab {
			continue
		}
		if len(offs) == cap(offs) {
			i8simd.AccumulateRows(acc, flat, offs)
			offs = offs[:0]
		}
		offs = append(offs, id*dim)
		valid++
	}
	i8simd.AccumulateRows(acc, flat, offs)
	out = out[:dim]
	if valid == 0 {
		for i := range out {
			out[i] = 128
		}
		return
	}
	i8simd.FinishU8(out, acc, 1.0/float32(valid), scale)
}

// cosineInt8Go mirrors cosine_similarity_int8_avx512's scalar formula:
// dot / (sqrtf(na) * sqrtf(nb)) in float32. Sums are exact int64.
func cosineInt8Go(a, b []uint8) float32 {
	dot, na, nb := i8simd.CenteredDotNorms(a, b)
	if na == 0 || nb == 0 {
		return 0
	}
	sa := float32(math.Sqrt(float64(float32(na))))
	sb := float32(math.Sqrt(float64(float32(nb))))
	return float32(dot) / float32(sa*sb)
}
