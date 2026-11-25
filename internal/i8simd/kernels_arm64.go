//go:build arm64 && !noasm

package i8simd

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// HasSDOT reports whether the NEON SDOT kernels are used.
// Every Apple Silicon core has FEAT_DotProd, but x/sys/cpu does not detect
// features on darwin, so trust GOOS there.
var HasSDOT = cpu.ARM64.HasASIMDDP || runtime.GOOS == "darwin" || runtime.GOOS == "ios"

// Implemented in kernels_arm64.s. SDOT is emitted via WORD because the Go
// assembler has no NEON SDOT mnemonic.

//go:noescape
func dot512SDOT(a, b *[512]int8) int32

// centeredBlocks32 processes n (>0) 32-byte blocks of a and b.
//
//go:noescape
func centeredBlocks32(a, b *uint8, n int) (dot, na, nb int32)

// accumS8 adds n (>0) 16-element blocks of row into acc.
//
//go:noescape
func accumS8(acc *int32, row *int8, n int)

// accumRows64 adds, for each of nblk (>0) 64-column blocks, the ntok
// (1..256) int8 rows at base+offs[t] into acc.
//
//go:noescape
func accumRows64(acc *int32, base *int8, offs *int, ntok int, nblk int)

// finishU8 maps n (>0) blocks of 16 pooled sums to uint8; see FinishU8.
//
//go:noescape
func finishU8(out *uint8, acc *int32, n int, inv, scale float32)
