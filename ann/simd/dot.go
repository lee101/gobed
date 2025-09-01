package simd

import (
	"golang.org/x/sys/cpu"
)

// Vec512 represents a 512-dimensional int8 vector
type Vec512 [512]int8

// Dot512 returns the dot product of two 512-d int8 vectors with runtime dispatch
func Dot512(a, b *Vec512) int32 {
	switch {
	case cpu.X86.HasAVX512VNNI || (cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW):
		return dot512_i8_vnni(a, b)
	case cpu.X86.HasAVX2:
		return dot512_i8_avx2(a, b)
	case cpu.ARM64.HasASIMDDP: // ARM Dot Product
		return dot512_i8_sdot(a, b)
	default:
		// Unrolled pure Go fallback
		return dot512_generic(a, b)
	}
}

// DotN computes dot product for arbitrary length int8 vectors
func DotN(a, b []int8) int32 {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	n := len(a)
	if n == 512 {
		return Dot512((*Vec512)(a), (*Vec512)(b))
	}

	// For non-512 dimensions, use generic implementation
	var sum int32
	// Unroll by 8 for better performance
	i := 0
	for ; i <= n-8; i += 8 {
		sum += int32(a[i]) * int32(b[i])
		sum += int32(a[i+1]) * int32(b[i+1])
		sum += int32(a[i+2]) * int32(b[i+2])
		sum += int32(a[i+3]) * int32(b[i+3])
		sum += int32(a[i+4]) * int32(b[i+4])
		sum += int32(a[i+5]) * int32(b[i+5])
		sum += int32(a[i+6]) * int32(b[i+6])
		sum += int32(a[i+7]) * int32(b[i+7])
	}
	// Handle remainder
	for ; i < n; i++ {
		sum += int32(a[i]) * int32(b[i])
	}
	return sum
}

// Cosine computes cosine similarity between normalized int8 vectors
// Assumes vectors are pre-normalized and quantized
func Cosine512(a, b *Vec512, scaleA, scaleB float32) float32 {
	dot := Dot512(a, b)
	// Dequantize the dot product
	return float32(dot) * scaleA * scaleB
}

// L2Squared512 computes squared L2 distance between int8 vectors (optimized)
func L2Squared512(a, b *Vec512) int32 {
	// Use optimized generic implementation for now
	return l2squared512_generic(a, b)
}

// l2squared512_generic is the optimized generic implementation
func l2squared512_generic(a, b *Vec512) int32 {
	var sum int32
	// Unroll by 16 for better performance
	for i := 0; i < 512; i += 16 {
		diff0 := int32(a[i]) - int32(b[i])
		diff1 := int32(a[i+1]) - int32(b[i+1])
		diff2 := int32(a[i+2]) - int32(b[i+2])
		diff3 := int32(a[i+3]) - int32(b[i+3])
		diff4 := int32(a[i+4]) - int32(b[i+4])
		diff5 := int32(a[i+5]) - int32(b[i+5])
		diff6 := int32(a[i+6]) - int32(b[i+6])
		diff7 := int32(a[i+7]) - int32(b[i+7])
		diff8 := int32(a[i+8]) - int32(b[i+8])
		diff9 := int32(a[i+9]) - int32(b[i+9])
		diff10 := int32(a[i+10]) - int32(b[i+10])
		diff11 := int32(a[i+11]) - int32(b[i+11])
		diff12 := int32(a[i+12]) - int32(b[i+12])
		diff13 := int32(a[i+13]) - int32(b[i+13])
		diff14 := int32(a[i+14]) - int32(b[i+14])
		diff15 := int32(a[i+15]) - int32(b[i+15])
		
		sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
		sum += diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7
		sum += diff8*diff8 + diff9*diff9 + diff10*diff10 + diff11*diff11
		sum += diff12*diff12 + diff13*diff13 + diff14*diff14 + diff15*diff15
	}
	return sum
}

// Generic fallback implementation
func dot512_generic(a, b *Vec512) int32 {
	var sum int32
	// Unroll by 16 for better performance
	for i := 0; i < 512; i += 16 {
		sum += int32(a[i]) * int32(b[i])
		sum += int32(a[i+1]) * int32(b[i+1])
		sum += int32(a[i+2]) * int32(b[i+2])
		sum += int32(a[i+3]) * int32(b[i+3])
		sum += int32(a[i+4]) * int32(b[i+4])
		sum += int32(a[i+5]) * int32(b[i+5])
		sum += int32(a[i+6]) * int32(b[i+6])
		sum += int32(a[i+7]) * int32(b[i+7])
		sum += int32(a[i+8]) * int32(b[i+8])
		sum += int32(a[i+9]) * int32(b[i+9])
		sum += int32(a[i+10]) * int32(b[i+10])
		sum += int32(a[i+11]) * int32(b[i+11])
		sum += int32(a[i+12]) * int32(b[i+12])
		sum += int32(a[i+13]) * int32(b[i+13])
		sum += int32(a[i+14]) * int32(b[i+14])
		sum += int32(a[i+15]) * int32(b[i+15])
	}
	return sum
}
