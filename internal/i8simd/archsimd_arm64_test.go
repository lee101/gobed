//go:build goexperiment.simd && arm64

package i8simd

import "simd/archsimd"

// Go 1.27 native SIMD (GOEXPERIMENT=simd, package simd/archsimd) versions,
// kept in test files for benchmarking only. archsimd's arm64 surface has no
// SDOT, so int8 dot products widen with SMULL + SXTL, ~4x the instructions.

func init() {
	archsimdDot512 = dot512Archsimd
	archsimdCentered = chunked(centeredArchsimd)
}

func dot512Archsimd(a, b *[512]int8) int32 {
	var acc0, acc1 archsimd.Int32x4
	for i := 0; i < 512; i += 16 {
		va := archsimd.LoadInt8x16Array((*[16]int8)(a[i : i+16]))
		vb := archsimd.LoadInt8x16Array((*[16]int8)(b[i : i+16]))
		lo := va.MulWidenLo(vb)
		hi := va.HiToLo().MulWidenLo(vb.HiToLo())
		acc0 = acc0.Add(lo.ExtendLo4ToInt32()).Add(lo.HiToLo().ExtendLo4ToInt32())
		acc1 = acc1.Add(hi.ExtendLo4ToInt32()).Add(hi.HiToLo().ExtendLo4ToInt32())
	}
	return sum4(acc0.Add(acc1))
}

func sum4(v archsimd.Int32x4) int32 {
	var out [4]int32
	v.StoreArray(&out)
	return out[0] + out[1] + out[2] + out[3]
}

func mulWide(x, y archsimd.Int8x16) archsimd.Int32x4 {
	lo := x.MulWidenLo(y)
	hi := x.HiToLo().MulWidenLo(y.HiToLo())
	return lo.ExtendLo4ToInt32().Add(lo.HiToLo().ExtendLo4ToInt32()).
		Add(hi.ExtendLo4ToInt32()).Add(hi.HiToLo().ExtendLo4ToInt32())
}

func centeredArchsimd(a, b []uint8) (dot, na, nb int64) {
	n := min(len(a), len(b))
	m := archsimd.BroadcastUint8x16(0x80)
	var d, x, y archsimd.Int32x4
	i := 0
	for ; i+16 <= n; i += 16 {
		va := archsimd.LoadUint8x16Array((*[16]uint8)(a[i : i+16])).Xor(m).BitsToInt8()
		vb := archsimd.LoadUint8x16Array((*[16]uint8)(b[i : i+16])).Xor(m).BitsToInt8()
		d = d.Add(mulWide(va, vb))
		x = x.Add(mulWide(va, va))
		y = y.Add(mulWide(vb, vb))
	}
	td, tx, ty := CenteredDotNormsGo(a[i:n], b[i:n])
	return int64(sum4(d)) + td, int64(sum4(x)) + tx, int64(sum4(y)) + ty
}
