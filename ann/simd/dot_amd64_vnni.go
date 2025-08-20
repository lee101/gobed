//go:build amd64 && cgo
// +build amd64,cgo

package simd

// #cgo CFLAGS: -O3 -mavx512vnni -mavx512bw -mavx512f -Wall
// #include <immintrin.h>
// #include <stdint.h>
//
// static inline int32_t dot512_i8_vnni_c(const int8_t* a, const int8_t* b) {
//   // VPDPBUSD computes u8*s8, so we need to transform one operand
//   // We use: sum(a*b) = sum((a+128)*b) - 128*sum(b)
//   __m512i acc = _mm512_setzero_si512();
//   __m512i bias = _mm512_set1_epi8((char)0x80);
//   
//   int32_t sum_b = 0;
//   
//   // Process 512 bytes as 8 chunks of 64B
//   for (int i = 0; i < 512; i += 64) {
//     __m512i va = _mm512_loadu_si512((const __m512i*)(a + i));
//     __m512i vb = _mm512_loadu_si512((const __m512i*)(b + i));
//     
//     // Accumulate sum(b) for bias correction
//     // Simple scalar accumulation for correctness
//     for (int j = 0; j < 64; ++j) {
//       sum_b += (int32_t)b[i + j];
//     }
//     
//     // Transform a to unsigned: a' = a ^ 0x80 (equivalent to +128)
//     __m512i va_u = _mm512_xor_si512(va, bias);
//     
//     // Accumulate dot product: acc += dot(u8(va_u), s8(vb))
//     // VPDPBUSD accumulates 4-byte groups into 32-bit lanes
//     acc = _mm512_dpbusd_epi32(acc, va_u, vb);
//   }
//   
//   // Horizontal sum of acc (16 x int32)
//   int32_t tmp[16];
//   _mm512_storeu_si512((__m512i*)tmp, acc);
//   int64_t sum = 0;
//   for (int i = 0; i < 16; ++i) {
//     sum += tmp[i];
//   }
//   
//   // Apply bias correction: subtract 128 * sum(b)
//   sum -= 128LL * (int64_t)sum_b;
//   
//   return (int32_t)sum;
// }
//
// // Optimized version with better sum_b calculation
// static inline int32_t dot512_i8_vnni_fast(const int8_t* a, const int8_t* b) {
//   __m512i acc = _mm512_setzero_si512();
//   __m512i bias = _mm512_set1_epi8((char)0x80);
//   
//   for (int i = 0; i < 512; i += 64) {
//     __m512i va = _mm512_loadu_si512((const __m512i*)(a + i));
//     __m512i vb = _mm512_loadu_si512((const __m512i*)(b + i));
//     
//     // Transform a to unsigned
//     __m512i va_u = _mm512_xor_si512(va, bias);
//     
//     // Accumulate dot product
//     acc = _mm512_dpbusd_epi32(acc, va_u, vb);
//   }
//   
//   // Horizontal sum
//   int32_t tmp[16];
//   _mm512_storeu_si512((__m512i*)tmp, acc);
//   int64_t sum = 0;
//   for (int i = 0; i < 16; ++i) {
//     sum += tmp[i];
//   }
//   
//   // Calculate sum_b separately for simplicity
//   int32_t sum_b = 0;
//   for (int i = 0; i < 512; ++i) {
//     sum_b += (int32_t)b[i];
//   }
//   
//   sum -= 128LL * (int64_t)sum_b;
//   return (int32_t)sum;
// }
import "C"
import "unsafe"

func dot512_i8_vnni(a, b *Vec512) int32 {
	return int32(C.dot512_i8_vnni_fast((*C.int8_t)(unsafe.Pointer(a)), (*C.int8_t)(unsafe.Pointer(b))))
}

// Stub for ARM function on AMD64
func dot512_i8_sdot(a, b *Vec512) int32 {
	return dot512_generic(a, b)
}