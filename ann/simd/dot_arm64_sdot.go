//go:build arm64 && cgo
// +build arm64,cgo

package simd

// #cgo CFLAGS: -O3 -march=armv8.4-a+dotprod -Wall
// #include <arm_neon.h>
// #include <stdint.h>
//
// static inline int32_t dot512_i8_sdot_c(const int8_t* a, const int8_t* b) {
//   int32x4_t acc0 = vdupq_n_s32(0);
//   int32x4_t acc1 = vdupq_n_s32(0);
//   int32x4_t acc2 = vdupq_n_s32(0);
//   int32x4_t acc3 = vdupq_n_s32(0);
//   
//   // Process 512 bytes, unrolled by 4 for better pipelining
//   for (int i = 0; i < 512; i += 64) {
//     // Load and accumulate 4x16 bytes
//     int8x16_t va0 = vld1q_s8(a + i);
//     int8x16_t vb0 = vld1q_s8(b + i);
//     acc0 = vdotq_s32(acc0, va0, vb0);
//     
//     int8x16_t va1 = vld1q_s8(a + i + 16);
//     int8x16_t vb1 = vld1q_s8(b + i + 16);
//     acc1 = vdotq_s32(acc1, va1, vb1);
//     
//     int8x16_t va2 = vld1q_s8(a + i + 32);
//     int8x16_t vb2 = vld1q_s8(b + i + 32);
//     acc2 = vdotq_s32(acc2, va2, vb2);
//     
//     int8x16_t va3 = vld1q_s8(a + i + 48);
//     int8x16_t vb3 = vld1q_s8(b + i + 48);
//     acc3 = vdotq_s32(acc3, va3, vb3);
//   }
//   
//   // Combine accumulators
//   acc0 = vaddq_s32(acc0, acc1);
//   acc2 = vaddq_s32(acc2, acc3);
//   acc0 = vaddq_s32(acc0, acc2);
//   
//   // Horizontal sum
//   int32x2_t sum_low = vadd_s32(vget_low_s32(acc0), vget_high_s32(acc0));
//   int32x2_t sum_final = vpadd_s32(sum_low, sum_low);
//   
//   return vget_lane_s32(sum_final, 0);
// }
import "C"
import "unsafe"

func dot512_i8_sdot(a, b *Vec512) int32 {
	return int32(C.dot512_i8_sdot_c((*C.int8_t)(unsafe.Pointer(a)), (*C.int8_t)(unsafe.Pointer(b))))
}