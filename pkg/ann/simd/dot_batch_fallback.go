//go:build !amd64 || noasm
// +build !amd64 noasm

package simd

func dot512_batch_avx2(query *Vec512, vectors []Vec512, scores []int32) []int32 {
	return dot512_batch_generic(query, vectors, scores)
}
