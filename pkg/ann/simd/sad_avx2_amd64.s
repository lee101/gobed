//go:build amd64 && !noasm
// +build amd64,!noasm

#include "textflag.h"

// func sumabsdiff512_i8_avx2(a, b *Vec512) int32
// Computes sum of absolute differences of two 512-d int8 vectors using AVX2
TEXT ·sumabsdiff512_i8_avx2(SB), NOSPLIT, $0-24
    MOVQ a+0(FP), AX    // a pointer
    MOVQ b+8(FP), BX    // b pointer

    // Zero int32 accumulators
    VPXOR Y0, Y0, Y0    // acc0
    VPXOR Y1, Y1, Y1    // acc1
    VPXOR Y2, Y2, Y2    // acc2
    VPXOR Y3, Y3, Y3    // acc3

    MOVQ $4, CX         // 4 iterations * 128 bytes = 512 bytes

loop:
    // Load 32 bytes from each vector (4 times per iteration)
    // Block 0
    VMOVDQU (AX), Y4        // a0
    VMOVDQU (BX), Y5        // b0
    // Sign-extend bytes to words (lower/upper halves)
    VPMOVSXBW X4, Y6        // a0_low -> words
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7        // a0_high -> words

    VPMOVSXBW X5, Y8        // b0_low -> words
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9        // b0_high -> words

    // Lower half: abs(a-b) in words
    VPSUBW Y8, Y6, Y10      // a_low - b_low
    VPABSW Y10, Y10         // |diff|
    // Expand to int32 and accumulate
    VPMOVSXWD X10, Y11      // lower 8 words -> dwords
    VEXTRACTI128 $1, Y10, X12
    VPMOVSXWD X12, Y13      // upper 8 words -> dwords
    VPADDD Y11, Y0, Y0
    VPADDD Y13, Y1, Y1

    // Upper half: abs(a-b)
    VPSUBW Y9, Y7, Y10      // a_high - b_high
    VPABSW Y10, Y10
    VPMOVSXWD X10, Y11
    VEXTRACTI128 $1, Y10, X12
    VPMOVSXWD X12, Y13
    VPADDD Y11, Y2, Y2
    VPADDD Y13, Y3, Y3

    // Block 1
    VMOVDQU 32(AX), Y4
    VMOVDQU 32(BX), Y5

    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7

    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9

    VPSUBW Y8, Y6, Y10
    VPABSW Y10, Y10
    VPMOVSXWD X10, Y11
    VEXTRACTI128 $1, Y10, X12
    VPMOVSXWD X12, Y13
    VPADDD Y11, Y0, Y0
    VPADDD Y13, Y1, Y1

    VPSUBW Y9, Y7, Y10
    VPABSW Y10, Y10
    VPMOVSXWD X10, Y11
    VEXTRACTI128 $1, Y10, X12
    VPMOVSXWD X12, Y13
    VPADDD Y11, Y2, Y2
    VPADDD Y13, Y3, Y3

    // Block 2
    VMOVDQU 64(AX), Y4
    VMOVDQU 64(BX), Y5

    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7

    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9

    VPSUBW Y8, Y6, Y10
    VPABSW Y10, Y10
    VPMOVSXWD X10, Y11
    VEXTRACTI128 $1, Y10, X12
    VPMOVSXWD X12, Y13
    VPADDD Y11, Y0, Y0
    VPADDD Y13, Y1, Y1

    VPSUBW Y9, Y7, Y10
    VPABSW Y10, Y10
    VPMOVSXWD X10, Y11
    VEXTRACTI128 $1, Y10, X12
    VPMOVSXWD X12, Y13
    VPADDD Y11, Y2, Y2
    VPADDD Y13, Y3, Y3

    // Block 3
    VMOVDQU 96(AX), Y4
    VMOVDQU 96(BX), Y5

    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7

    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9

    VPSUBW Y8, Y6, Y10
    VPABSW Y10, Y10
    VPMOVSXWD X10, Y11
    VEXTRACTI128 $1, Y10, X12
    VPMOVSXWD X12, Y13
    VPADDD Y11, Y0, Y0
    VPADDD Y13, Y1, Y1

    VPSUBW Y9, Y7, Y10
    VPABSW Y10, Y10
    VPMOVSXWD X10, Y11
    VEXTRACTI128 $1, Y10, X12
    VPMOVSXWD X12, Y13
    VPADDD Y11, Y2, Y2
    VPADDD Y13, Y3, Y3

    // Advance pointers
    ADDQ $128, AX
    ADDQ $128, BX

    DECQ CX
    JNZ loop

    // Sum accumulators
    VPADDD Y0, Y1, Y0
    VPADDD Y2, Y3, Y2
    VPADDD Y0, Y2, Y0

    // Horizontal sum Y0 to scalar
    VEXTRACTI128 $1, Y0, X1
    VPADDD X0, X1, X0
    VPSHUFD $0x4E, X0, X1
    VPADDD X0, X1, X0
    VPSHUFD $0xB1, X0, X1
    VPADDD X0, X1, X0

    VMOVD X0, AX
    MOVL AX, ret+16(FP)

    VZEROUPPER
    RET

