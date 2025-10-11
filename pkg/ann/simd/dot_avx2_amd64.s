// +build amd64,!noasm

#include "textflag.h"

// func dot512_i8_avx2(a, b *Vec512) int32
// Computes dot product of two 512-dimensional int8 vectors using AVX2
TEXT ·dot512_i8_avx2(SB), NOSPLIT, $0-24
    MOVQ a+0(FP), AX    // Load pointer to a
    MOVQ b+8(FP), BX    // Load pointer to b
    
    // Initialize accumulators
    VPXOR Y0, Y0, Y0    // Y0 = accumulator 1
    VPXOR Y1, Y1, Y1    // Y1 = accumulator 2
    VPXOR Y2, Y2, Y2    // Y2 = accumulator 3
    VPXOR Y3, Y3, Y3    // Y3 = accumulator 4
    
    // Process 128 bytes (128 int8s) per iteration
    // 512 bytes total = 4 iterations
    MOVQ $4, CX         // Loop counter
    
loop:
    // Load 32 bytes from each vector
    VMOVDQU (AX), Y4        // Load 32 bytes from a[i:i+32]
    VMOVDQU (BX), Y5        // Load 32 bytes from b[i:i+32]
    
    // Convert int8 to int16 (sign extend)
    VPMOVSXBW X4, Y6        // Lower 16 bytes of a -> Y6 (as int16)
    VEXTRACTI128 $1, Y4, X7 // Extract upper 16 bytes of a
    VPMOVSXBW X7, Y7        // Upper 16 bytes of a -> Y7 (as int16)
    
    VPMOVSXBW X5, Y8        // Lower 16 bytes of b -> Y8 (as int16)
    VEXTRACTI128 $1, Y5, X9 // Extract upper 16 bytes of b
    VPMOVSXBW X9, Y9        // Upper 16 bytes of b -> Y9 (as int16)
    
    // Multiply and accumulate (int16 * int16 -> int32)
    VPMADDWD Y6, Y8, Y10    // Multiply lower pairs and add adjacent
    VPMADDWD Y7, Y9, Y11    // Multiply upper pairs and add adjacent
    
    VPADDD Y10, Y0, Y0      // Accumulate to Y0
    VPADDD Y11, Y1, Y1      // Accumulate to Y1
    
    // Process next 32 bytes
    VMOVDQU 32(AX), Y4     // Load next 32 bytes from a
    VMOVDQU 32(BX), Y5     // Load next 32 bytes from b
    
    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7
    
    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9
    
    VPMADDWD Y6, Y8, Y10
    VPMADDWD Y7, Y9, Y11
    
    VPADDD Y10, Y2, Y2      // Accumulate to Y2
    VPADDD Y11, Y3, Y3      // Accumulate to Y3
    
    // Process next 32 bytes
    VMOVDQU 64(AX), Y4
    VMOVDQU 64(BX), Y5
    
    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7
    
    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9
    
    VPMADDWD Y6, Y8, Y10
    VPMADDWD Y7, Y9, Y11
    
    VPADDD Y10, Y0, Y0
    VPADDD Y11, Y1, Y1
    
    // Process last 32 bytes of this iteration
    VMOVDQU 96(AX), Y4
    VMOVDQU 96(BX), Y5
    
    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7
    
    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9
    
    VPMADDWD Y6, Y8, Y10
    VPMADDWD Y7, Y9, Y11
    
    VPADDD Y10, Y2, Y2
    VPADDD Y11, Y3, Y3
    
    // Advance pointers
    ADDQ $128, AX
    ADDQ $128, BX
    
    DECQ CX
    JNZ loop
    
    // Sum all accumulators
    VPADDD Y0, Y1, Y0       // Y0 = Y0 + Y1
    VPADDD Y2, Y3, Y2       // Y2 = Y2 + Y3
    VPADDD Y0, Y2, Y0       // Y0 = Y0 + Y2 (all sums)
    
    // Horizontal sum of Y0
    VEXTRACTI128 $1, Y0, X1 // Extract upper 128 bits
    VPADDD X0, X1, X0       // Add upper and lower halves
    VPSHUFD $0x4E, X0, X1   // Shuffle: swap 64-bit halves
    VPADDD X0, X1, X0       // Add
    VPSHUFD $0xB1, X0, X1   // Shuffle: swap 32-bit pairs
    VPADDD X0, X1, X0       // Add
    
    // Extract final result
    VMOVD X0, AX            // Move result to AX
    MOVL AX, ret+16(FP)     // Store as return value
    
    VZEROUPPER              // Clear upper YMM registers
    RET

// func dot512_i8_batch_avx2(query *Vec512, vectors *Vec512, scores *int32, count int)
TEXT ·dot512_i8_batch_avx2(SB), NOSPLIT, $0-32
    MOVQ query+0(FP), SI      // Base pointer to query vector
    MOVQ vectors+8(FP), DI    // Base pointer to dataset vectors
    MOVQ scores+16(FP), R8    // Pointer to output scores
    MOVQ count+24(FP), CX     // Number of vectors to process

    TESTQ CX, CX
    JLE batch_done

batch_loop:
    MOVQ SI, R9               // Reset query pointer
    MOVQ DI, R10              // Current vector pointer

    VPXOR Y0, Y0, Y0
    VPXOR Y1, Y1, Y1
    VPXOR Y2, Y2, Y2
    VPXOR Y3, Y3, Y3

    MOVQ $4, DX               // 512 bytes / 128 bytes per iteration

batch_inner:
    VMOVDQU (R9), Y4
    VMOVDQU (R10), Y5

    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7

    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9

    VPMADDWD Y6, Y8, Y10
    VPMADDWD Y7, Y9, Y11

    VPADDD Y10, Y0, Y0
    VPADDD Y11, Y1, Y1

    VMOVDQU 32(R9), Y4
    VMOVDQU 32(R10), Y5

    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7

    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9

    VPMADDWD Y6, Y8, Y10
    VPMADDWD Y7, Y9, Y11

    VPADDD Y10, Y2, Y2
    VPADDD Y11, Y3, Y3

    VMOVDQU 64(R9), Y4
    VMOVDQU 64(R10), Y5

    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7

    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9

    VPMADDWD Y6, Y8, Y10
    VPMADDWD Y7, Y9, Y11

    VPADDD Y10, Y0, Y0
    VPADDD Y11, Y1, Y1

    VMOVDQU 96(R9), Y4
    VMOVDQU 96(R10), Y5

    VPMOVSXBW X4, Y6
    VEXTRACTI128 $1, Y4, X7
    VPMOVSXBW X7, Y7

    VPMOVSXBW X5, Y8
    VEXTRACTI128 $1, Y5, X9
    VPMOVSXBW X9, Y9

    VPMADDWD Y6, Y8, Y10
    VPMADDWD Y7, Y9, Y11

    VPADDD Y10, Y2, Y2
    VPADDD Y11, Y3, Y3

    ADDQ $128, R9
    ADDQ $128, R10

    DECQ DX
    JNZ batch_inner

    VPADDD Y0, Y1, Y0
    VPADDD Y2, Y3, Y2
    VPADDD Y0, Y2, Y0

    VEXTRACTI128 $1, Y0, X1
    VPADDD X0, X1, X0
    VPSHUFD $0x4E, X0, X1
    VPADDD X0, X1, X0
    VPSHUFD $0xB1, X0, X1
    VPADDD X0, X1, X0

    VMOVD X0, AX
    MOVL AX, (R8)           // Store score

    MOVQ R10, DI            // Advance to next vector
    ADDQ $4, R8             // Advance score pointer

    DECQ CX
    JNZ batch_loop

batch_done:
    VZEROUPPER
    RET

// Alternative implementation for processors without VPMADDWD
// func dot512_i8_avx2_alt(a, b *Vec512) int32
TEXT ·dot512_i8_avx2_alt(SB), NOSPLIT, $0-24
    MOVQ a+0(FP), AX    // Load pointer to a
    MOVQ b+8(FP), BX    // Load pointer to b
    
    // Initialize accumulator
    VPXOR Y0, Y0, Y0    // Y0 = 0 (accumulator)
    
    // Process 32 bytes at a time (32 int8s)
    // 512 bytes total = 16 iterations
    MOVQ $16, CX        // Loop counter
    
loop_alt:
    // Load 32 bytes from each vector
    VMOVDQU (AX), Y1    // Load 32 bytes from a
    VMOVDQU (BX), Y2    // Load 32 bytes from b
    
    // Unpack and multiply using PMULLW (multiply words)
    // Process lower 16 bytes
    VPUNPCKLBW Y1, Y1, Y3   // Unpack low bytes of a (sign extend)
    VPUNPCKLBW Y2, Y2, Y4   // Unpack low bytes of b (sign extend)
    VPSRAW $8, Y3, Y3       // Sign extend by arithmetic shift
    VPSRAW $8, Y4, Y4       // Sign extend by arithmetic shift
    VPMULLW Y3, Y4, Y5      // Multiply as int16
    
    // Process upper 16 bytes
    VPUNPCKHBW Y1, Y1, Y6   // Unpack high bytes of a
    VPUNPCKHBW Y2, Y2, Y7   // Unpack high bytes of b
    VPSRAW $8, Y6, Y6       // Sign extend
    VPSRAW $8, Y7, Y7       // Sign extend
    VPMULLW Y6, Y7, Y8      // Multiply as int16
    
    // Convert int16 results to int32 and accumulate
    VPMOVSXWD X5, Y9        // Lower part of lower result
    VEXTRACTI128 $1, Y5, X10
    VPMOVSXWD X10, Y10      // Upper part of lower result
    
    VPMOVSXWD X8, Y11       // Lower part of upper result
    VEXTRACTI128 $1, Y8, X12
    VPMOVSXWD X12, Y12      // Upper part of upper result
    
    // Accumulate all
    VPADDD Y9, Y0, Y0
    VPADDD Y10, Y0, Y0
    VPADDD Y11, Y0, Y0
    VPADDD Y12, Y0, Y0
    
    // Advance pointers
    ADDQ $32, AX
    ADDQ $32, BX
    
    DECQ CX
    JNZ loop_alt
    
    // Horizontal sum of Y0
    VEXTRACTI128 $1, Y0, X1
    VPADDD X0, X1, X0
    VPSHUFD $0x4E, X0, X1
    VPADDD X0, X1, X0
    VPSHUFD $0xB1, X0, X1
    VPADDD X0, X1, X0
    
    // Extract final result
    VMOVD X0, AX
    MOVL AX, ret+16(FP)
    
    VZEROUPPER
    RET
