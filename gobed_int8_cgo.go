//go:build cgo && (amd64 || gobed_cgo_int8)

package gobed

// // AVX-512 flags only on amd64: clang rejects them on arm64. On other
// // architectures this file is built only with -tags gobed_cgo_int8 (for
// // benchmarking against the pure-Go kernels) and uses the scalar #else branch.
// #cgo amd64 CFLAGS: -mavx512f -mavx512bw -mavx512vl -O3 -march=native
// #cgo !amd64 CFLAGS: -O3
// #cgo LDFLAGS: -lm
// #include <stdint.h>
// #include <stdlib.h>
// #include <string.h>
// #include <math.h>
//
// #if defined(__x86_64__) && defined(__AVX512F__)
// #include <immintrin.h>
//
// // Quantize float32 to int8 with scale and zero point
// void quantize_weights_avx512(const float* input, int8_t* output, int size, float scale, int8_t zero_point) {
//     int i;
//     for (i = 0; i <= size - 16; i += 16) {
//         __m512 vals = _mm512_loadu_ps(&input[i]);
//         __m512 scaled = _mm512_mul_ps(vals, _mm512_set1_ps(scale));
//         __m512 rounded = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
//         __m512i as_int = _mm512_cvtps_epi32(rounded);
//         __m512i with_zp = _mm512_add_epi32(as_int, _mm512_set1_epi32(zero_point));
//
//         // Clamp to int8 range
//         __m512i clamped = _mm512_max_epi32(_mm512_set1_epi32(-128),
//                           _mm512_min_epi32(with_zp, _mm512_set1_epi32(127)));
//
//         // Pack to int8
//         __m128i packed = _mm512_cvtsepi32_epi8(clamped);
//         _mm_storeu_si128((__m128i*)&output[i], packed);
//     }
//
//     // Handle remaining elements
//     for (; i < size; i++) {
//         float scaled = input[i] * scale;
//         int32_t quantized = (int32_t)roundf(scaled) + zero_point;
//         if (quantized < -128) quantized = -128;
//         if (quantized > 127) quantized = 127;
//         output[i] = (int8_t)quantized;
//     }
// }
//
// // Compute embedding with INT8 weights using AVX-512
// void compute_embedding_int8_avx512(
//     const int8_t* weights,  // [vocab_size, embed_dim]
//     const int* token_ids,
//     int num_tokens,
//     int embed_dim,
//     int vocab_size,
//     uint8_t* output,        // Output in 0-255 range
//     float scale,
//     uint8_t zero_point
// ) {
//     // Initialize accumulator (using int32 for intermediate sums)
//     int32_t* accumulator = (int32_t*)calloc(embed_dim, sizeof(int32_t));
//
//     int valid_tokens = 0;
//
//     for (int t = 0; t < num_tokens; t++) {
//         int token_id = token_ids[t];
//         if (token_id >= 0 && token_id < vocab_size) {
//             const int8_t* weight_row = &weights[token_id * embed_dim];
//
//             // SIMD addition
//             int d;
//             for (d = 0; d <= embed_dim - 16; d += 16) {
//                 __m128i w = _mm_loadu_si128((__m128i*)&weight_row[d]);
//                 __m512i w_extended = _mm512_cvtepi8_epi32(w);
//                 __m512i acc = _mm512_loadu_si512(&accumulator[d]);
//                 acc = _mm512_add_epi32(acc, w_extended);
//                 _mm512_storeu_si512(&accumulator[d], acc);
//             }
//
//             // Handle remaining elements
//             for (; d < embed_dim; d++) {
//                 accumulator[d] += weight_row[d];
//             }
//
//             valid_tokens++;
//         }
//     }
//
//     // Mean pooling and convert to 0-255 range
//     if (valid_tokens > 0) {
//         float inv_tokens = 1.0f / valid_tokens;
//
//         for (int d = 0; d < embed_dim; d++) {
//             // Mean pooling
//             float mean = accumulator[d] * inv_tokens;
//
//             // Dequantize to float
//             float dequantized = mean / scale;
//
//             // Convert from [-1, 1] to [0, 255]
//             // Assuming original range is approximately [-1, 1]
//             float normalized = (dequantized + 1.0f) * 127.5f;
//
//             // Clamp and convert to uint8
//             if (normalized < 0) normalized = 0;
//             if (normalized > 255) normalized = 255;
//             output[d] = (uint8_t)normalized;
//         }
//     } else {
//         // Return zero embedding (128 = middle of 0-255 range)
//         memset(output, 128, embed_dim);
//     }
//
//     free(accumulator);
// }
//
// // Cosine similarity of uint8 vectors centered at 128 (x-128 in [-128,127]).
// // x ^ 0x80 is x-128 as int8; sign-extend to int16 and use vpmaddwd, which
// // is exact (|pair sum| <= 2*128*128). The previous version fed the signed
// // centered values to _mm512_maddubs_epi16, which treats its first operand as
// // unsigned and saturates, so norms came out negative and results were NaN or
// // wrong on every AVX-512 machine.
// float cosine_similarity_int8_avx512(const uint8_t* a, const uint8_t* b, int size) {
//     const __m256i offset = _mm256_set1_epi8((char)0x80);
//     __m512i dot_product = _mm512_setzero_si512();
//     __m512i norm_a = _mm512_setzero_si512();
//     __m512i norm_b = _mm512_setzero_si512();
//
//     int i;
//     for (i = 0; i <= size - 32; i += 32) {
//         __m512i va = _mm512_cvtepi8_epi16(_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)&a[i]), offset));
//         __m512i vb = _mm512_cvtepi8_epi16(_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)&b[i]), offset));
//         dot_product = _mm512_add_epi32(dot_product, _mm512_madd_epi16(va, vb));
//         norm_a = _mm512_add_epi32(norm_a, _mm512_madd_epi16(va, va));
//         norm_b = _mm512_add_epi32(norm_b, _mm512_madd_epi16(vb, vb));
//     }
//
//     int32_t dot = _mm512_reduce_add_epi32(dot_product);
//     int32_t na = _mm512_reduce_add_epi32(norm_a);
//     int32_t nb = _mm512_reduce_add_epi32(norm_b);
//
//     for (; i < size; i++) {
//         int8_t va = (int8_t)(a[i] - 128);
//         int8_t vb = (int8_t)(b[i] - 128);
//         dot += va * vb;
//         na += va * va;
//         nb += vb * vb;
//     }
//
//     if (na == 0 || nb == 0) return 0.0f;
//
//     return dot / (sqrtf(na) * sqrtf(nb));
// }
//
// #else // portable scalar fallbacks (same signatures and semantics)
//
// void quantize_weights_avx512(const float* input, int8_t* output, int size, float scale, int8_t zero_point) {
//     for (int i = 0; i < size; i++) {
//         int32_t q = (int32_t)roundf(input[i] * scale) + zero_point;
//         if (q < -128) q = -128;
//         if (q > 127) q = 127;
//         output[i] = (int8_t)q;
//     }
// }
//
// void compute_embedding_int8_avx512(const int8_t* weights, const int* token_ids, int num_tokens,
//                                    int embed_dim, int vocab_size, uint8_t* output,
//                                    float scale, uint8_t zero_point) {
//     int32_t* acc = (int32_t*)calloc(embed_dim, sizeof(int32_t));
//     int valid = 0;
//     for (int t = 0; t < num_tokens; t++) {
//         int id = token_ids[t];
//         if (id < 0 || id >= vocab_size) continue;
//         const int8_t* row = &weights[id * embed_dim];
//         for (int d = 0; d < embed_dim; d++) acc[d] += row[d];
//         valid++;
//     }
//     if (valid > 0) {
//         float inv = 1.0f / valid;
//         for (int d = 0; d < embed_dim; d++) {
//             float n = ((acc[d] * inv) / scale + 1.0f) * 127.5f;
//             if (n < 0) n = 0;
//             if (n > 255) n = 255;
//             output[d] = (uint8_t)n;
//         }
//     } else {
//         memset(output, 128, embed_dim);
//     }
//     free(acc);
// }
//
// float cosine_similarity_int8_avx512(const uint8_t* a, const uint8_t* b, int size) {
//     int32_t dot = 0, na = 0, nb = 0;
//     for (int i = 0; i < size; i++) {
//         int8_t va = (int8_t)(a[i] - 128), vb = (int8_t)(b[i] - 128);
//         dot += va * vb; na += va * va; nb += vb * vb;
//     }
//     if (na == 0 || nb == 0) return 0.0f;
//     return dot / (sqrtf(na) * sqrtf(nb));
// }
// #endif
import "C"

import (
	"runtime"
	"unsafe"

	"golang.org/x/sys/cpu"
)

// The C file is compiled with AVX-512 enabled on amd64, so its kernels may only
// run on CPUs that have it: an AVX2-only Xeon (e.g. Broadwell, the prod GPU box)
// gets SIGILL. Dispatch to them only when the CPU reports AVX-512 F/BW/VL;
// otherwise the pure-Go kernels in gobed_int8_kernels.go run. Elsewhere the
// Go-assembly / pure-Go kernels are faster and need no cgo.
const haveCgoInt8 = true

var (
	hasAVX512Int8 = cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW && cpu.X86.HasAVX512VL
	// useCgoInt8: production dispatch goes to the C kernels.
	useCgoInt8 = runtime.GOARCH == "amd64" && hasAVX512Int8
	// cgoInt8Runnable: the C kernels can be called at all (tests, benchmarks).
	// Off amd64 this file only builds with -tags gobed_cgo_int8, using scalar C.
	cgoInt8Runnable = runtime.GOARCH != "amd64" || hasAVX512Int8
)

func quantizeRowInt8Cgo(in []float32, out []int8, scale float32, zeroPoint int8) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	C.quantize_weights_avx512(
		(*C.float)(unsafe.Pointer(&in[0])),
		(*C.int8_t)(unsafe.Pointer(&out[0])),
		C.int(n),
		C.float(scale),
		C.int8_t(zeroPoint),
	)
}

func embedInt8Cgo(flat []int8, tokenIDs []int, dim, vocab int, out []uint8, scale float32) {
	cTokenIDs := make([]C.int, len(tokenIDs))
	for i, id := range tokenIDs {
		cTokenIDs[i] = C.int(id)
	}
	C.compute_embedding_int8_avx512(
		(*C.int8_t)(unsafe.Pointer(&flat[0])),
		(*C.int)(&cTokenIDs[0]),
		C.int(len(tokenIDs)),
		C.int(dim),
		C.int(vocab),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		C.float(scale),
		C.uint8_t(128), // zero point for output
	)
}

func cosineInt8Cgo(a, b []uint8) float32 {
	return float32(C.cosine_similarity_int8_avx512(
		(*C.uint8_t)(unsafe.Pointer(&a[0])),
		(*C.uint8_t)(unsafe.Pointer(&b[0])),
		C.int(len(a)),
	))
}
