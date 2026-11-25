//go:build goexperiment.simd && amd64

package i8simd

import "simd/archsimd"

// Go 1.27 native SIMD (GOEXPERIMENT=simd) amd64 versions, benchmark-only.
// AVX-512BW when available (VPMOVSXBW zmm + VPMADDWD zmm), else AVX2.
// Compile-checked on arm64 only; run on the amd64 box to get numbers.

func init() {
	if archsimd.X86.AVX512() {
		archsimdDot512 = dot512Archsimd512
		archsimdCentered = chunked(centeredArchsimd512)
	} else if archsimd.X86.AVX2() {
		archsimdDot512 = dot512Archsimd256
		archsimdCentered = chunked(centeredArchsimd256)
	}
}

func sum8(v archsimd.Int32x8) int32 {
	var out [8]int32
	v.StoreArray(&out)
	return out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7]
}

func sum16(v archsimd.Int32x16) int32 {
	var out [16]int32
	v.StoreArray(&out)
	var s int32
	for _, x := range out {
		s += x
	}
	return s
}

func dot512Archsimd256(a, b *[512]int8) int32 {
	var acc0, acc1 archsimd.Int32x8
	for i := 0; i < 512; i += 32 {
		a0 := archsimd.LoadInt8x16Array((*[16]int8)(a[i : i+16])).ExtendToInt16()
		b0 := archsimd.LoadInt8x16Array((*[16]int8)(b[i : i+16])).ExtendToInt16()
		a1 := archsimd.LoadInt8x16Array((*[16]int8)(a[i+16 : i+32])).ExtendToInt16()
		b1 := archsimd.LoadInt8x16Array((*[16]int8)(b[i+16 : i+32])).ExtendToInt16()
		acc0 = acc0.Add(a0.DotProductPairs(b0))
		acc1 = acc1.Add(a1.DotProductPairs(b1))
	}
	return sum8(acc0.Add(acc1))
}

func dot512Archsimd512(a, b *[512]int8) int32 {
	var acc0, acc1 archsimd.Int32x16
	for i := 0; i < 512; i += 64 {
		a0 := archsimd.LoadInt8x32Array((*[32]int8)(a[i : i+32])).ExtendToInt16()
		b0 := archsimd.LoadInt8x32Array((*[32]int8)(b[i : i+32])).ExtendToInt16()
		a1 := archsimd.LoadInt8x32Array((*[32]int8)(a[i+32 : i+64])).ExtendToInt16()
		b1 := archsimd.LoadInt8x32Array((*[32]int8)(b[i+32 : i+64])).ExtendToInt16()
		acc0 = acc0.Add(a0.DotProductPairs(b0))
		acc1 = acc1.Add(a1.DotProductPairs(b1))
	}
	return sum16(acc0.Add(acc1))
}

func centeredArchsimd256(a, b []uint8) (dot, na, nb int64) {
	n := min(len(a), len(b))
	m := archsimd.BroadcastUint8x16(0x80)
	var d, x, y archsimd.Int32x8
	i := 0
	for ; i+16 <= n; i += 16 {
		va := archsimd.LoadUint8x16Array((*[16]uint8)(a[i : i+16])).Xor(m).BitsToInt8().ExtendToInt16()
		vb := archsimd.LoadUint8x16Array((*[16]uint8)(b[i : i+16])).Xor(m).BitsToInt8().ExtendToInt16()
		d = d.Add(va.DotProductPairs(vb))
		x = x.Add(va.DotProductPairs(va))
		y = y.Add(vb.DotProductPairs(vb))
	}
	td, tx, ty := CenteredDotNormsGo(a[i:n], b[i:n])
	return int64(sum8(d)) + td, int64(sum8(x)) + tx, int64(sum8(y)) + ty
}

func centeredArchsimd512(a, b []uint8) (dot, na, nb int64) {
	n := min(len(a), len(b))
	m := archsimd.BroadcastUint8x32(0x80)
	var d, x, y archsimd.Int32x16
	i := 0
	for ; i+32 <= n; i += 32 {
		va := archsimd.LoadUint8x32Array((*[32]uint8)(a[i : i+32])).Xor(m).BitsToInt8().ExtendToInt16()
		vb := archsimd.LoadUint8x32Array((*[32]uint8)(b[i : i+32])).Xor(m).BitsToInt8().ExtendToInt16()
		d = d.Add(va.DotProductPairs(vb))
		x = x.Add(va.DotProductPairs(va))
		y = y.Add(vb.DotProductPairs(vb))
	}
	td, tx, ty := CenteredDotNormsGo(a[i:n], b[i:n])
	return int64(sum16(d)) + td, int64(sum16(x)) + tx, int64(sum16(y)) + ty
}
