package gobed

import (
	"fmt"
	"math"
	"os"
	"time"
	"unsafe"
)

// #cgo CFLAGS: -mavx512f -mavx512bw -mavx512vl -O3 -march=native
// #cgo LDFLAGS: -lm
// #include <immintrin.h>
// #include <stdint.h>
// #include <string.h>
// #include <math.h>
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
// // Compute cosine similarity between two INT8 vectors
// float cosine_similarity_int8_avx512(const uint8_t* a, const uint8_t* b, int size) {
//     __m512i dot_product = _mm512_setzero_si512();
//     __m512i norm_a = _mm512_setzero_si512();
//     __m512i norm_b = _mm512_setzero_si512();
//     
//     int i;
//     for (i = 0; i <= size - 64; i += 64) {
//         __m512i va = _mm512_loadu_si512(&a[i]);
//         __m512i vb = _mm512_loadu_si512(&b[i]);
//         
//         // Subtract 128 to center around 0
//         __m512i offset = _mm512_set1_epi8(128);
//         va = _mm512_sub_epi8(va, offset);
//         vb = _mm512_sub_epi8(vb, offset);
//         
//         // Compute dot product and norms (using maddubs for efficiency)
//         __m512i prod = _mm512_maddubs_epi16(va, vb);
//         __m512i sqr_a = _mm512_maddubs_epi16(va, va);
//         __m512i sqr_b = _mm512_maddubs_epi16(vb, vb);
//         
//         // Accumulate
//         dot_product = _mm512_add_epi32(dot_product, _mm512_madd_epi16(prod, _mm512_set1_epi16(1)));
//         norm_a = _mm512_add_epi32(norm_a, _mm512_madd_epi16(sqr_a, _mm512_set1_epi16(1)));
//         norm_b = _mm512_add_epi32(norm_b, _mm512_madd_epi16(sqr_b, _mm512_set1_epi16(1)));
//     }
//     
//     // Reduce SIMD registers
//     int32_t dot = _mm512_reduce_add_epi32(dot_product);
//     int32_t na = _mm512_reduce_add_epi32(norm_a);
//     int32_t nb = _mm512_reduce_add_epi32(norm_b);
//     
//     // Handle remaining elements
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
import "C"

// EmbeddingModelInt8 provides INT8 quantized embeddings with SIMD acceleration
type EmbeddingModelInt8 struct {
	VocabSize       int
	EmbedDim        int
	weightsInt8     [][]int8    // Quantized weights
	weightsFloat32  [][]float32 // Original weights for comparison
	referenceTokens map[string]TokenData
	scale           float32
	zeroPoint       int8
	useInt8         bool // Flag to enable/disable INT8 mode
}

// LoadModelInt8 loads the model with INT8 quantization support
func LoadModelInt8(useInt8 bool) (*EmbeddingModelInt8, error) {
	fmt.Printf("🔄 Loading model with INT8=%v and SIMD support...\n", useInt8)
	start := time.Now()

	// Load safetensors weights
	safetensorsPath := "model/real_model.safetensors"
	if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
		safetensorsPath = "../../model/real_model.safetensors"
		if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
			safetensorsPath = "./model/real_model.safetensors"
		}
	}

	weightsFloat32, vocabSize, embedDim, err := loadRealSafetensors(safetensorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load safetensors: %v", err)
	}

	// Load reference tokens - reuse the function from gobed.go
	tokensPath := "model/real_reference_tokens.json"
	if _, err := os.Stat(tokensPath); os.IsNotExist(err) {
		tokensPath = "../../model/real_reference_tokens.json"
		if _, err := os.Stat(tokensPath); os.IsNotExist(err) {
			tokensPath = "./model/real_reference_tokens.json"
		}
	}
	// For now, use a simple map - in production would share with main package
	referenceTokens := make(map[string]TokenData)
	// We'll only use pre-tokenized texts from the benchmark anyway

	model := &EmbeddingModelInt8{
		VocabSize:       vocabSize,
		EmbedDim:        embedDim,
		weightsFloat32:  weightsFloat32,
		referenceTokens: referenceTokens,
		useInt8:         useInt8,
	}

	if useInt8 {
		// Quantize weights to INT8
		fmt.Println("⚡ Quantizing weights to INT8...")
		model.quantizeWeights()
	}

	loadTime := time.Since(start)
	fmt.Printf("✅ Model loaded in %v (vocab: %d, dims: %d, INT8: %v)\n", 
		loadTime, vocabSize, embedDim, useInt8)
	return model, nil
}

// quantizeWeights converts float32 weights to INT8 with proper scaling
func (m *EmbeddingModelInt8) quantizeWeights() {
	// Find min/max across all weights for optimal quantization
	var minVal, maxVal float32
	minVal = math.MaxFloat32
	maxVal = -math.MaxFloat32

	for i := 0; i < m.VocabSize; i++ {
		for j := 0; j < m.EmbedDim; j++ {
			val := m.weightsFloat32[i][j]
			if val < minVal {
				minVal = val
			}
			if val > maxVal {
				maxVal = val
			}
		}
	}

	// Calculate scale and zero point for symmetric quantization
	m.scale = (maxVal - minVal) / 255.0
	m.zeroPoint = int8(-128 - int(minVal/m.scale))

	fmt.Printf("📊 Quantization params: scale=%.6f, zero_point=%d, range=[%.3f, %.3f]\n",
		m.scale, m.zeroPoint, minVal, maxVal)

	// Allocate INT8 weights
	m.weightsInt8 = make([][]int8, m.VocabSize)
	for i := range m.weightsInt8 {
		m.weightsInt8[i] = make([]int8, m.EmbedDim)
	}

	// Quantize using SIMD
	for i := 0; i < m.VocabSize; i++ {
		C.quantize_weights_avx512(
			(*C.float)(unsafe.Pointer(&m.weightsFloat32[i][0])),
			(*C.int8_t)(unsafe.Pointer(&m.weightsInt8[i][0])),
			C.int(m.EmbedDim),
			C.float(m.scale),
			C.int8_t(m.zeroPoint),
		)
		
		if i%5000 == 0 {
			fmt.Printf("  Quantized %d/%d rows\n", i, m.VocabSize)
		}
	}
}

// Encode converts text to INT8 embedding (0-255 range)
func (m *EmbeddingModelInt8) Encode(text string) ([]uint8, error) {
	// Get token IDs (reuse existing tokenization)
	tokenData, exists := m.referenceTokens[text]
	if !exists {
		return nil, fmt.Errorf("text not in reference tokens: %s", text)
	}

	if m.useInt8 {
		return m.computeEmbeddingInt8(tokenData.TokenIDs)
	} else {
		return m.computeEmbeddingFloat32AsInt8(tokenData.TokenIDs)
	}
}

// computeEmbeddingInt8 uses SIMD-accelerated INT8 computation
func (m *EmbeddingModelInt8) computeEmbeddingInt8(tokenIDs []int) ([]uint8, error) {
	result := make([]uint8, m.EmbedDim)
	
	if len(tokenIDs) == 0 {
		// Return middle value for empty input
		for i := range result {
			result[i] = 128
		}
		return result, nil
	}

	// Convert token IDs to C array
	cTokenIDs := make([]C.int, len(tokenIDs))
	for i, id := range tokenIDs {
		cTokenIDs[i] = C.int(id)
	}

	// Flatten INT8 weights for C function
	flatWeights := make([]int8, m.VocabSize*m.EmbedDim)
	for i := 0; i < m.VocabSize; i++ {
		copy(flatWeights[i*m.EmbedDim:], m.weightsInt8[i])
	}

	// Call SIMD function
	C.compute_embedding_int8_avx512(
		(*C.int8_t)(unsafe.Pointer(&flatWeights[0])),
		(*C.int)(&cTokenIDs[0]),
		C.int(len(tokenIDs)),
		C.int(m.EmbedDim),
		C.int(m.VocabSize),
		(*C.uint8_t)(unsafe.Pointer(&result[0])),
		C.float(m.scale),
		C.uint8_t(128), // zero point for output
	)

	return result, nil
}

// computeEmbeddingFloat32AsInt8 computes with float32 then converts to INT8
func (m *EmbeddingModelInt8) computeEmbeddingFloat32AsInt8(tokenIDs []int) ([]uint8, error) {
	// Use float32 computation
	embedding := make([]float32, m.EmbedDim)
	validTokens := 0

	for _, tokenID := range tokenIDs {
		if tokenID >= 0 && tokenID < m.VocabSize {
			weightRow := m.weightsFloat32[tokenID]
			for i := 0; i < m.EmbedDim; i++ {
				embedding[i] += weightRow[i]
			}
			validTokens++
		}
	}

	// Mean pooling
	if validTokens > 0 {
		invTokens := 1.0 / float32(validTokens)
		for i := range embedding {
			embedding[i] *= invTokens
		}
	}

	// Convert to 0-255 range
	result := make([]uint8, m.EmbedDim)
	for i, val := range embedding {
		// Assuming embeddings are roughly in [-1, 1] range
		// Map to [0, 255]
		normalized := (val + 1.0) * 127.5
		if normalized < 0 {
			normalized = 0
		} else if normalized > 255 {
			normalized = 255
		}
		result[i] = uint8(normalized)
	}

	return result, nil
}

// CosineSimilarityInt8 computes similarity between INT8 vectors using SIMD
func CosineSimilarityInt8(a, b []uint8) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	// Use SIMD-accelerated similarity
	return float32(C.cosine_similarity_int8_avx512(
		(*C.uint8_t)(unsafe.Pointer(&a[0])),
		(*C.uint8_t)(unsafe.Pointer(&b[0])),
		C.int(len(a)),
	))
}

// CosineSimilarityInt8Fallback is a pure Go fallback for systems without AVX-512
func CosineSimilarityInt8Fallback(a, b []uint8) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	var dotProduct, normA, normB int64
	
	for i := 0; i < len(a); i++ {
		// Center around 0 by subtracting 128
		aVal := int16(a[i]) - 128
		bVal := int16(b[i]) - 128
		
		dotProduct += int64(aVal) * int64(bVal)
		normA += int64(aVal) * int64(aVal)
		normB += int64(bVal) * int64(bVal)
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return float32(dotProduct) / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}