//go:build !arm64 || noasm

package i8simd

// HasSDOT reports whether the NEON SDOT kernels are used.
const HasSDOT = false

func dot512SDOT(a, b *[512]int8) int32 { return Dot512Go(a, b) }

func centeredBlocks32(a, b *uint8, n int) (dot, na, nb int32) { panic("unreachable") }

func accumS8(acc *int32, row *int8, n int) { panic("unreachable") }

func accumRows64(acc *int32, base *int8, offs *int, ntok int, nblk int) { panic("unreachable") }

func finishU8(out *uint8, acc *int32, n int, inv, scale float32) { panic("unreachable") }
