// +build amd64,!noasm

#include "textflag.h"

// func embed_accum_avx2(embedding *int8, scale float32, result *float32)
// Accumulates int8 embedding * scale into float32 result buffer (512 dims)
// This is the hot path for EmbedTokensInto
TEXT ·embed_accum_avx2(SB), NOSPLIT, $0-20
    MOVQ embedding+0(FP), SI    // Source int8 embedding
    MOVSS scale+8(FP), X0       // Scale factor
    MOVQ result+12(FP), DI      // Destination float32 buffer

    // Broadcast scale to all 8 lanes of Y0
    VBROADCASTSS X0, Y0         // Y0 = [scale, scale, ..., scale] x8

    // Process 512 dimensions in 16 iterations of 32 int8s each
    MOVQ $16, CX                // Loop counter

embed_loop:
    // Load 32 int8 values
    VMOVDQU (SI), Y1            // Y1 = 32 int8 values

    // Convert lower 16 int8 -> int16 -> int32 -> float32
    VPMOVSXBW X1, Y2            // Y2 = lower 16 as int16
    VPMOVSXWD X2, Y3            // Y3 = lower 8 as int32
    VEXTRACTI128 $1, Y2, X4     // X4 = upper 8 int16 of lower half
    VPMOVSXWD X4, Y4            // Y4 = next 8 as int32

    VCVTDQ2PS Y3, Y3            // Y3 = 8 floats
    VCVTDQ2PS Y4, Y4            // Y4 = 8 floats

    // Multiply by scale
    VMULPS Y3, Y0, Y3           // Y3 *= scale
    VMULPS Y4, Y0, Y4           // Y4 *= scale

    // Add to result
    VADDPS (DI), Y3, Y3         // Y3 += result[0:8]
    VADDPS 32(DI), Y4, Y4       // Y4 += result[8:16]
    VMOVUPS Y3, (DI)            // Store result[0:8]
    VMOVUPS Y4, 32(DI)          // Store result[8:16]

    // Convert upper 16 int8 -> int16 -> int32 -> float32
    VEXTRACTI128 $1, Y1, X5     // X5 = upper 16 int8
    VPMOVSXBW X5, Y5            // Y5 = upper 16 as int16
    VPMOVSXWD X5, Y6            // Y6 = lower 8 of upper as int32
    VEXTRACTI128 $1, Y5, X7     // X7 = upper 8 int16 of upper half
    VPMOVSXWD X7, Y7            // Y7 = upper 8 as int32

    VCVTDQ2PS Y6, Y6            // Y6 = 8 floats
    VCVTDQ2PS Y7, Y7            // Y7 = 8 floats

    // Multiply by scale
    VMULPS Y6, Y0, Y6           // Y6 *= scale
    VMULPS Y7, Y0, Y7           // Y7 *= scale

    // Add to result
    VADDPS 64(DI), Y6, Y6       // Y6 += result[16:24]
    VADDPS 96(DI), Y7, Y7       // Y7 += result[24:32]
    VMOVUPS Y6, 64(DI)          // Store result[16:24]
    VMOVUPS Y7, 96(DI)          // Store result[24:32]

    // Advance pointers
    ADDQ $32, SI                // embedding += 32
    ADDQ $128, DI               // result += 32 floats (128 bytes)

    DECQ CX
    JNZ embed_loop

    VZEROUPPER
    RET

// func embed_accum_batch_avx2(embeddings **int8, scales *float32, result *float32, count int)
// Batch version: accumulates multiple embeddings with their scales into result
// More efficient for multiple tokens as it keeps result in registers longer
TEXT ·embed_accum_batch_avx2(SB), NOSPLIT, $0-32
    MOVQ embeddings+0(FP), R8   // Pointer to array of embedding pointers
    MOVQ scales+8(FP), R9       // Pointer to scales array
    MOVQ result+16(FP), DI      // Destination float32 buffer
    MOVQ count+24(FP), CX       // Number of embeddings

    TESTQ CX, CX
    JLE batch_done

token_loop:
    MOVQ (R8), SI               // Get current embedding pointer
    VBROADCASTSS (R9), Y0       // Broadcast current scale

    // Process all 512 dims for this token
    MOVQ $16, DX                // 512 / 32 = 16 iterations
    MOVQ DI, R10                // Save result base

dim_loop:
    VMOVDQU (SI), Y1            // Load 32 int8

    // Lower 16 int8 -> float32
    VPMOVSXBW X1, Y2
    VPMOVSXWD X2, Y3
    VEXTRACTI128 $1, Y2, X4
    VPMOVSXWD X4, Y4
    VCVTDQ2PS Y3, Y3
    VCVTDQ2PS Y4, Y4
    VMULPS Y3, Y0, Y3
    VMULPS Y4, Y0, Y4
    VADDPS (R10), Y3, Y3
    VADDPS 32(R10), Y4, Y4
    VMOVUPS Y3, (R10)
    VMOVUPS Y4, 32(R10)

    // Upper 16 int8 -> float32
    VEXTRACTI128 $1, Y1, X5
    VPMOVSXBW X5, Y5
    VPMOVSXWD X5, Y6
    VEXTRACTI128 $1, Y5, X7
    VPMOVSXWD X7, Y7
    VCVTDQ2PS Y6, Y6
    VCVTDQ2PS Y7, Y7
    VMULPS Y6, Y0, Y6
    VMULPS Y7, Y0, Y7
    VADDPS 64(R10), Y6, Y6
    VADDPS 96(R10), Y7, Y7
    VMOVUPS Y6, 64(R10)
    VMOVUPS Y7, 96(R10)

    ADDQ $32, SI
    ADDQ $128, R10

    DECQ DX
    JNZ dim_loop

    // Next token
    ADDQ $8, R8                 // Next embedding pointer
    ADDQ $4, R9                 // Next scale
    DECQ CX
    JNZ token_loop

batch_done:
    VZEROUPPER
    RET

// func clear_f32_avx2(buf *float32)
// Zero 512 float32 values using AVX2
TEXT ·clear_f32_avx2(SB), NOSPLIT, $0-8
    MOVQ buf+0(FP), DI
    VPXOR Y0, Y0, Y0            // Y0 = 0

    // 512 floats = 2048 bytes = 64 YMM stores
    MOVQ $64, CX

clear_loop:
    VMOVUPS Y0, (DI)
    ADDQ $32, DI
    DECQ CX
    JNZ clear_loop

    VZEROUPPER
    RET

// func scale_f32_avx2(buf *float32, scale float32)
// Multiply 512 float32 values by scale (for averaging)
TEXT ·scale_f32_avx2(SB), NOSPLIT, $0-12
    MOVQ buf+0(FP), DI
    MOVSS scale+8(FP), X0
    VBROADCASTSS X0, Y0         // Broadcast scale

    MOVQ $64, CX                // 512 floats / 8 per YMM = 64

scale_loop:
    VMOVUPS (DI), Y1
    VMULPS Y1, Y0, Y1
    VMOVUPS Y1, (DI)
    ADDQ $32, DI
    DECQ CX
    JNZ scale_loop

    VZEROUPPER
    RET
