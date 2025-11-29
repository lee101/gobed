//go:build amd64 && !noasm
// +build amd64,!noasm

package simd

// dot512_batch_avx2 computes batched dot products using AVX2 assembly.
func dot512_batch_avx2(query *Vec512, vectors []Vec512, scores []int32) []int32 {
	n := len(vectors)
	if n == 0 {
		return scores[:0]
	}
	if cap(scores) < n {
		scores = make([]int32, n)
	} else {
		scores = scores[:n]
	}
	dot512_i8_batch_avx2(query, &vectors[0], &scores[0], n)
	return scores
}
