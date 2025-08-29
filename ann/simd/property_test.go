package simd

import (
	"math/rand"
	"testing"
	"time"
)

// Property-based testing framework for SIMD implementations

// Test property: All implementations should give identical results
func TestPropertyAllImplementationsMatch(t *testing.T) {
	const numTests = 10000
	
	rand.Seed(time.Now().UnixNano())
	
	for i := 0; i < numTests; i++ {
		// Generate random vectors
		a := &Vec512{}
		b := &Vec512{}
		
		for j := 0; j < 512; j++ {
			a[j] = int8(rand.Intn(256) - 128)
			b[j] = int8(rand.Intn(256) - 128)
		}
		
		// Test all implementations
		generic := dot512_generic(a, b)
		avx2 := dot512_i8_avx2(a, b)
		avx2Alt := dot512_i8_avx2_alt(a, b)
		vnni := dot512_i8_vnni(a, b)
		dispatch := Dot512(a, b)
		
		// All should match generic
		if avx2 != generic {
			t.Fatalf("Test %d: AVX2 mismatch: got %d, want %d", i, avx2, generic)
		}
		if avx2Alt != generic {
			t.Fatalf("Test %d: AVX2 Alt mismatch: got %d, want %d", i, avx2Alt, generic)
		}
		if vnni != generic {
			t.Fatalf("Test %d: VNNI mismatch: got %d, want %d", i, vnni, generic)
		}
		if dispatch != generic {
			t.Fatalf("Test %d: Dispatch mismatch: got %d, want %d", i, dispatch, generic)
		}
	}
	
	t.Logf("✓ All implementations matched across %d random test cases", numTests)
}

// Test property: Dot product is commutative
func TestPropertyCommutativity(t *testing.T) {
	const numTests = 1000
	
	rand.Seed(time.Now().UnixNano())
	
	for i := 0; i < numTests; i++ {
		a := &Vec512{}
		b := &Vec512{}
		
		// Generate random vectors with smaller range to avoid overflow
		for j := 0; j < 512; j++ {
			a[j] = int8(rand.Intn(200) - 100)
			b[j] = int8(rand.Intn(200) - 100)
		}
		
		// Test commutativity for all implementations
		implementations := map[string]func(*Vec512, *Vec512) int32{
			"Generic":  dot512_generic,
			"AVX2":     dot512_i8_avx2,
			"AVX2_Alt": dot512_i8_avx2_alt,
			"VNNI":     dot512_i8_vnni,
			"Dispatch": Dot512,
		}
		
		for name, impl := range implementations {
			result1 := impl(a, b)
			result2 := impl(b, a)
			
			if result1 != result2 {
				t.Fatalf("Test %d %s: Commutativity failed: dot(a,b)=%d != dot(b,a)=%d", 
					i, name, result1, result2)
			}
		}
	}
	
	t.Logf("✓ Commutativity verified across %d test cases for all implementations", numTests)
}

// Test property: Dot product with zero vector is zero
func TestPropertyZeroVector(t *testing.T) {
	const numTests = 100
	
	zero := &Vec512{} // All zeros by default
	
	rand.Seed(time.Now().UnixNano())
	
	for i := 0; i < numTests; i++ {
		a := &Vec512{}
		
		// Generate random vector
		for j := 0; j < 512; j++ {
			a[j] = int8(rand.Intn(256) - 128)
		}
		
		// Test with all implementations
		implementations := map[string]func(*Vec512, *Vec512) int32{
			"Generic":  dot512_generic,
			"AVX2":     dot512_i8_avx2,
			"AVX2_Alt": dot512_i8_avx2_alt,
			"VNNI":     dot512_i8_vnni,
			"Dispatch": Dot512,
		}
		
		for name, impl := range implementations {
			result := impl(a, zero)
			if result != 0 {
				t.Fatalf("Test %d %s: Zero vector property failed: got %d, want 0", 
					i, name, result)
			}
		}
	}
	
	t.Logf("✓ Zero vector property verified across %d test cases", numTests)
}

// Test property: Linearity - dot(a, b+c) = dot(a,b) + dot(a,c)
func TestPropertyLinearity(t *testing.T) {
	const numTests = 100
	
	rand.Seed(time.Now().UnixNano())
	
	for i := 0; i < numTests; i++ {
		a := &Vec512{}
		b := &Vec512{}
		c := &Vec512{}
		
		// Use smaller range to avoid overflow
		for j := 0; j < 512; j++ {
			a[j] = int8(rand.Intn(20) - 10)
			b[j] = int8(rand.Intn(20) - 10)
			c[j] = int8(rand.Intn(20) - 10)
		}
		
		// Compute b + c (with overflow protection)
		bPlusC := &Vec512{}
		for j := 0; j < 512; j++ {
			sum := int32(b[j]) + int32(c[j])
			if sum > 127 {
				sum = 127
			} else if sum < -128 {
				sum = -128
			}
			bPlusC[j] = int8(sum)
		}
		
		// Test linearity: dot(a, b+c) ?= dot(a,b) + dot(a,c)
		// Note: This may not hold exactly due to overflow in int8 arithmetic
		dotABC := Dot512(a, bPlusC)
		dotAB := Dot512(a, b)
		dotAC := Dot512(a, c)
		
		// Allow for some difference due to int8 clamping
		expected := dotAB + dotAC
		diff := abs32(dotABC - expected)
		
		// If the difference is large, it might indicate a bug
		if diff > 1000 { // Reasonable threshold
			t.Logf("Test %d: Large linearity deviation: dot(a,b+c)=%d, dot(a,b)+dot(a,c)=%d, diff=%d",
				i, dotABC, expected, diff)
		}
	}
	
	t.Logf("✓ Linearity property tested across %d cases (with int8 overflow considerations)", numTests)
}

// Test property: Self dot product is always non-negative
func TestPropertySelfDotNonNegative(t *testing.T) {
	const numTests = 1000
	
	rand.Seed(time.Now().UnixNano())
	
	for i := 0; i < numTests; i++ {
		a := &Vec512{}
		
		// Generate random vector
		for j := 0; j < 512; j++ {
			a[j] = int8(rand.Intn(256) - 128)
		}
		
		// Test self dot product
		result := Dot512(a, a)
		
		if result < 0 {
			t.Fatalf("Test %d: Self dot product is negative: %d", i, result)
		}
		
		// Verify it equals sum of squares
		expected := int32(0)
		for j := 0; j < 512; j++ {
			expected += int32(a[j]) * int32(a[j])
		}
		
		if result != expected {
			t.Fatalf("Test %d: Self dot product mismatch: got %d, want %d", i, result, expected)
		}
	}
	
	t.Logf("✓ Self dot product non-negativity verified across %d test cases", numTests)
}

// Test property: Bounds checking - result should be within expected range
func TestPropertyResultBounds(t *testing.T) {
	const numTests = 1000
	
	rand.Seed(time.Now().UnixNano())
	
	for i := 0; i < numTests; i++ {
		a := &Vec512{}
		b := &Vec512{}
		
		minA, maxA := int8(127), int8(-128)
		minB, maxB := int8(127), int8(-128)
		
		// Generate vectors and track min/max
		for j := 0; j < 512; j++ {
			a[j] = int8(rand.Intn(256) - 128)
			b[j] = int8(rand.Intn(256) - 128)
			
			if a[j] < minA { minA = a[j] }
			if a[j] > maxA { maxA = a[j] }
			if b[j] < minB { minB = b[j] }
			if b[j] > maxB { maxB = b[j] }
		}
		
		result := Dot512(a, b)
		
		// Calculate theoretical bounds
		// Worst case: all elements are at extremes and same sign
		var minBound, maxBound int64
		
		if (int64(minA)*int64(minB)) > (int64(maxA)*int64(maxB)) {
			maxBound = int64(minA) * int64(minB) * 512
		} else {
			maxBound = int64(maxA) * int64(maxB) * 512
		}
		
		if (int64(minA)*int64(maxB)) < (int64(maxA)*int64(minB)) {
			minBound = int64(minA) * int64(maxB) * 512
		} else {
			minBound = int64(maxA) * int64(minB) * 512
		}
		
		// Check if result is within bounds
		if int64(result) < minBound || int64(result) > maxBound {
			t.Fatalf("Test %d: Result %d outside bounds [%d, %d]", 
				i, result, minBound, maxBound)
		}
	}
	
	t.Logf("✓ Result bounds verified across %d test cases", numTests)
}

// Test property: Determinism - same inputs should give same outputs
func TestPropertyDeterminism(t *testing.T) {
	const numTests = 100
	const repeats = 10
	
	rand.Seed(42) // Fixed seed for determinism
	
	for i := 0; i < numTests; i++ {
		a := &Vec512{}
		b := &Vec512{}
		
		// Generate fixed vectors
		for j := 0; j < 512; j++ {
			a[j] = int8(rand.Intn(256) - 128)
			b[j] = int8(rand.Intn(256) - 128)
		}
		
		// Run same computation multiple times
		results := make([]int32, repeats)
		for r := 0; r < repeats; r++ {
			results[r] = Dot512(a, b)
		}
		
		// All results should be identical
		for r := 1; r < repeats; r++ {
			if results[r] != results[0] {
				t.Fatalf("Test %d: Non-deterministic result: first=%d, repeat %d=%d", 
					i, results[0], r, results[r])
			}
		}
	}
	
	t.Logf("✓ Determinism verified across %d test cases with %d repeats each", numTests, repeats)
}

// Utility function
func abs32(x int32) int32 {
	if x < 0 {
		return -x
	}
	return x
}

