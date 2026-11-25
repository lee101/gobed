// Package i8simd holds the int8 vector kernels shared by gobed's root package
// and ann/simd. On arm64 they are Go-assembly NEON SDOT kernels, called like
// any Go function (no cgo transition); elsewhere they are unrolled pure Go.
package i8simd

// maxBlocksPerCall bounds one assembly call so its int32 sums cannot overflow:
// 2048 blocks * 32 elements * 16384 (max product) = 2^30.
const maxBlocksPerCall = 2048

// Dot512 returns the dot product of two 512-element int8 vectors.
func Dot512(a, b *[512]int8) int32 {
	if HasSDOT {
		return dot512SDOT(a, b)
	}
	return Dot512Go(a, b)
}

// CenteredDotNorms treats a and b as uint8 values offset by 128 (x-128 in
// [-128,127]) and returns sum(a*b), sum(a*a), sum(b*b) exactly. It uses the
// first min(len(a), len(b)) elements.
func CenteredDotNorms(a, b []uint8) (dot, na, nb int64) {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	if HasSDOT {
		for len(a) >= 32 {
			blocks := min(len(a)/32, maxBlocksPerCall)
			d, x, y := centeredBlocks32(&a[0], &b[0], blocks)
			dot, na, nb = dot+int64(d), na+int64(x), nb+int64(y)
			a, b = a[blocks*32:], b[blocks*32:]
		}
	}
	d, x, y := CenteredDotNormsGo(a, b)
	return dot + d, na + x, nb + y
}

// AccumulateS8 adds row[i] (sign-extended) into acc[i] for i < len(row).
// len(acc) must be >= len(row).
func AccumulateS8(acc []int32, row []int8) {
	acc = acc[:len(row)]
	if HasSDOT && len(row) >= 16 {
		blocks := len(row) / 16
		accumS8(&acc[0], &row[0], blocks)
		acc, row = acc[blocks*16:], row[blocks*16:]
	}
	AccumulateS8Go(acc, row)
}

// AccumulateRows adds the rows flat[off:off+len(acc)] for every off in offs
// into acc (sign-extended). It is the mean-pooling inner loop of an int8
// embedding lookup: offs are token_id*dim. Panics if a row is out of range.
func AccumulateRows(acc []int32, flat []int8, offs []int) {
	dim := len(acc)
	for _, off := range offs {
		_ = flat[off : off+dim] // bounds-check every row before any assembly
	}
	if HasSDOT && dim >= 64 {
		nblk := dim / 64
		for rest := offs; len(rest) > 0; {
			k := min(len(rest), 256) // int16 lanes stay exact up to 256 rows
			accumRows64(&acc[0], &flat[0], &rest[0], k, nblk)
			rest = rest[k:]
		}
		if done := nblk * 64; done < dim {
			for _, off := range offs {
				AccumulateS8Go(acc[done:], flat[off+done:off+dim])
			}
		}
		return
	}
	for _, off := range offs {
		AccumulateS8(acc, flat[off:off+dim])
	}
}

// FinishU8 sets out[i] = uint8(clamp(((float32(acc[i])*inv)/scale+1)*127.5,
// 0, 255)) with every step rounded to float32 (no FMA) and truncation toward
// zero: the mean-pool-to-uint8 step of gobed's int8 embedding. NaN inputs
// (scale == 0) give unspecified bytes.
func FinishU8(out []uint8, acc []int32, inv, scale float32) {
	out = out[:len(acc)]
	if HasSDOT && len(acc) >= 16 {
		blocks := len(acc) / 16
		finishU8(&out[0], &acc[0], blocks, inv, scale)
		out, acc = out[blocks*16:], acc[blocks*16:]
	}
	FinishU8Go(out, acc, inv, scale)
}

// FinishU8Go is the portable kernel behind FinishU8.
func FinishU8Go(out []uint8, acc []int32, inv, scale float32) {
	out = out[:len(acc)]
	for i, s := range acc {
		mean := float32(float32(s) * inv)
		v := float32(float32(float32(mean/scale)+1.0) * 127.5)
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		out[i] = uint8(v)
	}
}

// Dot512Go is the portable unrolled kernel behind Dot512.
func Dot512Go(a, b *[512]int8) int32 {
	var s0, s1, s2, s3 int32
	for i := 0; i < 512; i += 8 {
		x := (*[8]int8)(a[i : i+8])
		y := (*[8]int8)(b[i : i+8])
		s0 += int32(x[0])*int32(y[0]) + int32(x[4])*int32(y[4])
		s1 += int32(x[1])*int32(y[1]) + int32(x[5])*int32(y[5])
		s2 += int32(x[2])*int32(y[2]) + int32(x[6])*int32(y[6])
		s3 += int32(x[3])*int32(y[3]) + int32(x[7])*int32(y[7])
	}
	return s0 + s1 + s2 + s3
}

// CenteredDotNormsGo is the portable unrolled kernel behind CenteredDotNorms.
// Inner sums are int32 over chunks of at most 32768 elements (|sum| <= 2^29),
// so the result is exact for any length.
func CenteredDotNormsGo(a, b []uint8) (dot, na, nb int64) {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	for len(a) >= 8 {
		chunk := min(len(a), 32768) &^ 7
		var d, x, y int32
		for i := 0; i < chunk; i += 8 {
			p := (*[8]uint8)(a[i : i+8])
			q := (*[8]uint8)(b[i : i+8])
			a0, a1, a2, a3 := int32(int8(p[0]^0x80)), int32(int8(p[1]^0x80)), int32(int8(p[2]^0x80)), int32(int8(p[3]^0x80))
			a4, a5, a6, a7 := int32(int8(p[4]^0x80)), int32(int8(p[5]^0x80)), int32(int8(p[6]^0x80)), int32(int8(p[7]^0x80))
			b0, b1, b2, b3 := int32(int8(q[0]^0x80)), int32(int8(q[1]^0x80)), int32(int8(q[2]^0x80)), int32(int8(q[3]^0x80))
			b4, b5, b6, b7 := int32(int8(q[4]^0x80)), int32(int8(q[5]^0x80)), int32(int8(q[6]^0x80)), int32(int8(q[7]^0x80))
			d += a0*b0 + a1*b1 + a2*b2 + a3*b3 + a4*b4 + a5*b5 + a6*b6 + a7*b7
			x += a0*a0 + a1*a1 + a2*a2 + a3*a3 + a4*a4 + a5*a5 + a6*a6 + a7*a7
			y += b0*b0 + b1*b1 + b2*b2 + b3*b3 + b4*b4 + b5*b5 + b6*b6 + b7*b7
		}
		dot, na, nb = dot+int64(d), na+int64(x), nb+int64(y)
		a, b = a[chunk:], b[chunk:]
	}
	for i := range a {
		x, y := int64(a[i])-128, int64(b[i])-128
		dot += x * y
		na += x * x
		nb += y * y
	}
	return dot, na, nb
}

// AccumulateS8Go is the portable kernel behind AccumulateS8.
func AccumulateS8Go(acc []int32, row []int8) {
	acc = acc[:len(row)]
	i := 0
	for ; i+8 <= len(row); i += 8 {
		r := (*[8]int8)(row[i : i+8])
		c := (*[8]int32)(acc[i : i+8])
		c[0] += int32(r[0])
		c[1] += int32(r[1])
		c[2] += int32(r[2])
		c[3] += int32(r[3])
		c[4] += int32(r[4])
		c[5] += int32(r[5])
		c[6] += int32(r[6])
		c[7] += int32(r[7])
	}
	for ; i < len(row); i++ {
		acc[i] += int32(row[i])
	}
}
