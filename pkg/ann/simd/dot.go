package simd

import (
    "golang.org/x/sys/cpu"
    "sync"
    "unsafe"
)

// Vec512 represents a 512-dimensional int8 vector
type Vec512 [512]int8

// CPU feature detection cache
var (
	dotImpl     func(*Vec512, *Vec512) int32
	dotImplOnce sync.Once
)

// initDotImpl initializes the dot product implementation based on CPU features
func initDotImpl() {
	switch {
	case cpu.X86.HasAVX512VNNI || (cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW):
		dotImpl = dot512_i8_vnni
	case cpu.X86.HasAVX2:
		dotImpl = dot512_i8_avx2
	case cpu.ARM64.HasASIMDDP:
		dotImpl = dot512_i8_sdot
	default:
		dotImpl = dot512_generic
	}
}

// Dot512 returns the dot product of two 512-d int8 vectors with cached dispatch
func Dot512(a, b *Vec512) int32 {
	dotImplOnce.Do(initDotImpl)
	return dotImpl(a, b)
}

// DotN computes dot product for arbitrary length int8 vectors
func DotN(a, b []int8) int32 {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	n := len(a)
	if n == 512 {
		// Avoid allocation by using unsafe cast
		return Dot512((*Vec512)(unsafe.Pointer(&a[0])), (*Vec512)(unsafe.Pointer(&b[0])))
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
	var sum0, sum1, sum2, sum3 int32
	// Unroll by 16 with multiple accumulators for ILP
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

		sum0 += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
		sum1 += diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7
		sum2 += diff8*diff8 + diff9*diff9 + diff10*diff10 + diff11*diff11
		sum3 += diff12*diff12 + diff13*diff13 + diff14*diff14 + diff15*diff15
	}
	return sum0 + sum1 + sum2 + sum3
}

// Generic fallback implementation
func dot512_generic(a, b *Vec512) int32 {
	var sum0, sum1, sum2, sum3 int32
	// Unroll by 16 with multiple accumulators for ILP
	for i := 0; i < 512; i += 16 {
		sum0 += int32(a[i]) * int32(b[i])
		sum0 += int32(a[i+1]) * int32(b[i+1])
		sum0 += int32(a[i+2]) * int32(b[i+2])
		sum0 += int32(a[i+3]) * int32(b[i+3])

		sum1 += int32(a[i+4]) * int32(b[i+4])
		sum1 += int32(a[i+5]) * int32(b[i+5])
		sum1 += int32(a[i+6]) * int32(b[i+6])
		sum1 += int32(a[i+7]) * int32(b[i+7])

		sum2 += int32(a[i+8]) * int32(b[i+8])
		sum2 += int32(a[i+9]) * int32(b[i+9])
		sum2 += int32(a[i+10]) * int32(b[i+10])
		sum2 += int32(a[i+11]) * int32(b[i+11])

		sum3 += int32(a[i+12]) * int32(b[i+12])
		sum3 += int32(a[i+13]) * int32(b[i+13])
		sum3 += int32(a[i+14]) * int32(b[i+14])
		sum3 += int32(a[i+15]) * int32(b[i+15])
	}
	return sum0 + sum1 + sum2 + sum3
}

// SumAbsDiff512 computes sum of absolute differences (L1) between two int8 vectors (512-d)
// Returns the total L1 distance as int32. Uses AVX2 when available.
func SumAbsDiff512(a, b *Vec512) int32 {
    if cpu.X86.HasAVX2 {
        return sumabsdiff512_i8_avx2(a, b)
    }
    return sumabsdiff512_generic(a, b)
}

func sumabsdiff512_generic(a, b *Vec512) int32 {
    var sum int32
    // Unroll by 16 for better performance
    for i := 0; i < 512; i += 16 {
        d0 := int32(a[i]) - int32(b[i])
        d1 := int32(a[i+1]) - int32(b[i+1])
        d2 := int32(a[i+2]) - int32(b[i+2])
        d3 := int32(a[i+3]) - int32(b[i+3])
        d4 := int32(a[i+4]) - int32(b[i+4])
        d5 := int32(a[i+5]) - int32(b[i+5])
        d6 := int32(a[i+6]) - int32(b[i+6])
        d7 := int32(a[i+7]) - int32(b[i+7])
        d8 := int32(a[i+8]) - int32(b[i+8])
        d9 := int32(a[i+9]) - int32(b[i+9])
        d10 := int32(a[i+10]) - int32(b[i+10])
        d11 := int32(a[i+11]) - int32(b[i+11])
        d12 := int32(a[i+12]) - int32(b[i+12])
        d13 := int32(a[i+13]) - int32(b[i+13])
        d14 := int32(a[i+14]) - int32(b[i+14])
        d15 := int32(a[i+15]) - int32(b[i+15])

        if d0 < 0 { d0 = -d0 };   sum += d0
        if d1 < 0 { d1 = -d1 };   sum += d1
        if d2 < 0 { d2 = -d2 };   sum += d2
        if d3 < 0 { d3 = -d3 };   sum += d3
        if d4 < 0 { d4 = -d4 };   sum += d4
        if d5 < 0 { d5 = -d5 };   sum += d5
        if d6 < 0 { d6 = -d6 };   sum += d6
        if d7 < 0 { d7 = -d7 };   sum += d7
        if d8 < 0 { d8 = -d8 };   sum += d8
        if d9 < 0 { d9 = -d9 };   sum += d9
        if d10 < 0 { d10 = -d10 }; sum += d10
        if d11 < 0 { d11 = -d11 }; sum += d11
        if d12 < 0 { d12 = -d12 }; sum += d12
        if d13 < 0 { d13 = -d13 }; sum += d13
        if d14 < 0 { d14 = -d14 }; sum += d14
        if d15 < 0 { d15 = -d15 }; sum += d15
    }
    return sum
}

// AvgAbsDiff512 returns the average absolute difference across 512 dims as float32
func AvgAbsDiff512(a, b *Vec512) float32 {
    return float32(SumAbsDiff512(a, b)) / 512.0
}
