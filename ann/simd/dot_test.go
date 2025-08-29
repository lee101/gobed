package simd

import (
	"fmt"
	"math/rand"
	"testing"
	"golang.org/x/sys/cpu"
)

// Test that all implementations produce the same result
func TestDot512Consistency(t *testing.T) {
	// Create test vectors
	a := &Vec512{}
	b := &Vec512{}
	
	// Fill with test data
	rand.Seed(42)
	for i := 0; i < 512; i++ {
		a[i] = int8(rand.Intn(256) - 128)
		b[i] = int8(rand.Intn(256) - 128)
	}
	
	// Get generic result as baseline
	genericResult := dot512_generic(a, b)
	
	// Test AVX2 implementation (always test, fallback will handle it)
	avx2Result := dot512_i8_avx2(a, b)
	if avx2Result != genericResult {
		t.Errorf("AVX2 result mismatch: got %d, want %d", avx2Result, genericResult)
	}
	
	// Test VNNI implementation (always test, fallback will handle it)
	vnniResult := dot512_i8_vnni(a, b)
	if vnniResult != genericResult {
		t.Errorf("VNNI result mismatch: got %d, want %d", vnniResult, genericResult)
	}
	
	// Test alternative AVX2
	avx2AltResult := dot512_i8_avx2_alt(a, b)
	if avx2AltResult != genericResult {
		t.Errorf("AVX2 alt result mismatch: got %d, want %d", avx2AltResult, genericResult)
	}
	
	// Test edge cases
	testCases := []struct {
		name string
		init func(a, b *Vec512)
	}{
		{
			name: "All zeros",
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					a[i] = 0
					b[i] = 0
				}
			},
		},
		{
			name: "All ones",
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					a[i] = 1
					b[i] = 1
				}
			},
		},
		{
			name: "Alternating signs",
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					if i%2 == 0 {
						a[i] = 127
						b[i] = 127
					} else {
						a[i] = -128
						b[i] = -128
					}
				}
			},
		},
		{
			name: "Max values",
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					a[i] = 127
					b[i] = 127
				}
			},
		},
		{
			name: "Min values",
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					a[i] = -128
					b[i] = -128
				}
			},
		},
		{
			name: "Mixed patterns",
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					a[i] = int8((i * 31) % 256 - 128)
					b[i] = int8((i * 17) % 256 - 128)
				}
			},
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := &Vec512{}
			b := &Vec512{}
			tc.init(a, b)
			
			genericResult := dot512_generic(a, b)
			avx2Result := dot512_i8_avx2(a, b)
			vnniResult := dot512_i8_vnni(a, b)
			avx2AltResult := dot512_i8_avx2_alt(a, b)
			
			if avx2Result != genericResult {
				t.Errorf("AVX2 result mismatch for %s: got %d, want %d", tc.name, avx2Result, genericResult)
			}
			if vnniResult != genericResult {
				t.Errorf("VNNI result mismatch for %s: got %d, want %d", tc.name, vnniResult, genericResult)
			}
			if avx2AltResult != genericResult {
				t.Errorf("AVX2 alt result mismatch for %s: got %d, want %d", tc.name, avx2AltResult, genericResult)
			}
		})
	}
}

// Test that runtime dispatch works correctly
func TestRuntimeDispatch(t *testing.T) {
	a := &Vec512{}
	b := &Vec512{}
	
	// Fill with test data
	for i := 0; i < 512; i++ {
		a[i] = int8(i % 128)
		b[i] = int8((i * 3) % 128)
	}
	
	genericResult := dot512_generic(a, b)
	dispatchedResult := Dot512(a, b)
	
	if dispatchedResult != genericResult {
		t.Errorf("Runtime dispatch result mismatch: got %d, want %d", dispatchedResult, genericResult)
	}
	
	// Print which implementation was used (for debugging)
	t.Logf("CPU features: AVX512VNNI=%t, AVX512F=%t, AVX512BW=%t, AVX2=%t", 
		cpu.X86.HasAVX512VNNI, cpu.X86.HasAVX512F, cpu.X86.HasAVX512BW, cpu.X86.HasAVX2)
}

func BenchmarkDot512(b *testing.B) {
	va := &Vec512{}
	vb := &Vec512{}
	
	rand.Seed(42)
	for i := 0; i < 512; i++ {
		va[i] = int8(rand.Intn(256) - 128)
		vb[i] = int8(rand.Intn(256) - 128)
	}
	
	b.Run("Generic", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = dot512_generic(va, vb)
		}
	})
	
	b.Run("AVX2_Assembly", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = dot512_i8_avx2(va, vb)
		}
	})
	
	b.Run("AVX2_Alt", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = dot512_i8_avx2_alt(va, vb)
		}
	})
	
	b.Run("VNNI_CGO", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = dot512_i8_vnni(va, vb)
		}
	})
	
	b.Run("Auto_Dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Dot512(va, vb)
		}
	})
}

// Benchmark different vector sizes for DotN
func BenchmarkDotN(b *testing.B) {
	sizes := []int{128, 256, 512, 1024, 2048}
	
	for _, size := range sizes {
		a := make([]int8, size)
		bb := make([]int8, size)
		
		for i := 0; i < size; i++ {
			a[i] = int8(rand.Intn(256) - 128)
			bb[i] = int8(rand.Intn(256) - 128)
		}
		
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = DotN(a, bb)
			}
		})
	}
}

// Test overflow scenarios
func TestDot512Overflow(t *testing.T) {
	testCases := []struct {
		name string
		init func(a, b *Vec512)
		expectedRange [2]int32 // min, max expected values
	}{
		{
			name: "Maximum positive overflow risk",
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					a[i] = 127
					b[i] = 127
				}
			},
			expectedRange: [2]int32{127*127*512, 127*127*512}, // 8,257,536
		},
		{
			name: "Maximum negative overflow risk", 
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					a[i] = -128
					b[i] = -128
				}
			},
			expectedRange: [2]int32{128*128*512, 128*128*512}, // 8,388,608
		},
		{
			name: "Mixed signs - cancel out",
			init: func(a, b *Vec512) {
				for i := 0; i < 512; i++ {
					if i%2 == 0 {
						a[i] = 127
						b[i] = -128
					} else {
						a[i] = -128
						b[i] = 127
					}
				}
			},
			expectedRange: [2]int32{-127*128*512, -127*128*512}, // -8,323,072
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := &Vec512{}
			b := &Vec512{}
			tc.init(a, b)
			
			// Test all implementations
			genericResult := dot512_generic(a, b)
			avx2Result := dot512_i8_avx2(a, b)
			vnniResult := dot512_i8_vnni(a, b)
			avx2AltResult := dot512_i8_avx2_alt(a, b)
			
			// Verify all implementations agree
			if avx2Result != genericResult {
				t.Errorf("AVX2 mismatch: got %d, want %d", avx2Result, genericResult)
			}
			if vnniResult != genericResult {
				t.Errorf("VNNI mismatch: got %d, want %d", vnniResult, genericResult)
			}
			if avx2AltResult != genericResult {
				t.Errorf("AVX2 alt mismatch: got %d, want %d", avx2AltResult, genericResult)
			}
			
			// Verify result is in expected range
			if genericResult < tc.expectedRange[0] || genericResult > tc.expectedRange[1] {
				t.Errorf("Result %d outside expected range [%d, %d]", 
					genericResult, tc.expectedRange[0], tc.expectedRange[1])
			}
			
			t.Logf("Result for %s: %d", tc.name, genericResult)
		})
	}
}

// Test alignment scenarios
func TestDot512Alignment(t *testing.T) {
	// Create unaligned test data by embedding in larger arrays
	largeA := make([]int8, 512+64) // Add padding
	largeB := make([]int8, 512+64)
	
	rand.Seed(12345)
	for i := 0; i < len(largeA); i++ {
		largeA[i] = int8(rand.Intn(256) - 128)
		largeB[i] = int8(rand.Intn(256) - 128)
	}
	
	// Test different alignments
	alignments := []int{0, 1, 4, 8, 16, 32}
	
	for _, offset := range alignments {
		t.Run(fmt.Sprintf("Offset_%d", offset), func(t *testing.T) {
			// Extract Vec512 at different offsets
			a := (*Vec512)(largeA[offset : offset+512])
			b := (*Vec512)(largeB[offset : offset+512])
			
			genericResult := dot512_generic(a, b)
			avx2Result := dot512_i8_avx2(a, b)
			vnniResult := dot512_i8_vnni(a, b)
			
			if avx2Result != genericResult {
				t.Errorf("AVX2 alignment issue at offset %d: got %d, want %d", 
					offset, avx2Result, genericResult)
			}
			if vnniResult != genericResult {
				t.Errorf("VNNI alignment issue at offset %d: got %d, want %d", 
					offset, vnniResult, genericResult)
			}
		})
	}
}

// Stress test with many random vectors
func TestDot512StressRandom(t *testing.T) {
	const numTests = 1000
	
	rand.Seed(42)
	
	for i := 0; i < numTests; i++ {
		a := &Vec512{}
		b := &Vec512{}
		
		// Generate random vectors
		for j := 0; j < 512; j++ {
			a[j] = int8(rand.Intn(256) - 128)
			b[j] = int8(rand.Intn(256) - 128)
		}
		
		// Test all implementations
		genericResult := dot512_generic(a, b)
		avx2Result := dot512_i8_avx2(a, b)
		vnniResult := dot512_i8_vnni(a, b)
		avx2AltResult := dot512_i8_avx2_alt(a, b)
		
		if avx2Result != genericResult {
			t.Fatalf("Test %d: AVX2 mismatch: got %d, want %d", i, avx2Result, genericResult)
		}
		if vnniResult != genericResult {
			t.Fatalf("Test %d: VNNI mismatch: got %d, want %d", i, vnniResult, genericResult)
		}
		if avx2AltResult != genericResult {
			t.Fatalf("Test %d: AVX2 alt mismatch: got %d, want %d", i, avx2AltResult, genericResult)
		}
	}
	
	t.Logf("Successfully tested %d random vector pairs", numTests)
}

// Test mathematical properties
func TestDot512Properties(t *testing.T) {
	rand.Seed(789)
	
	// Generate test vectors
	a := &Vec512{}
	b := &Vec512{}
	c := &Vec512{}
	
	for i := 0; i < 512; i++ {
		a[i] = int8(rand.Intn(100) - 50) // Smaller range to avoid overflow in tests
		b[i] = int8(rand.Intn(100) - 50)
		c[i] = int8(rand.Intn(100) - 50)
	}
	
	t.Run("Commutativity", func(t *testing.T) {
		// dot(a, b) == dot(b, a)
		result1 := Dot512(a, b)
		result2 := Dot512(b, a)
		
		if result1 != result2 {
			t.Errorf("Commutativity failed: dot(a,b)=%d != dot(b,a)=%d", result1, result2)
		}
	})
	
	t.Run("Zero vector", func(t *testing.T) {
		zero := &Vec512{}
		// All elements are already 0 by default
		
		result := Dot512(a, zero)
		if result != 0 {
			t.Errorf("Zero vector test failed: expected 0, got %d", result)
		}
	})
	
	t.Run("Self dot product", func(t *testing.T) {
		// dot(a, a) should be sum of squares
		result := Dot512(a, a)
		
		// Calculate expected sum of squares
		expected := int32(0)
		for i := 0; i < 512; i++ {
			expected += int32(a[i]) * int32(a[i])
		}
		
		if result != expected {
			t.Errorf("Self dot product failed: got %d, want %d", result, expected)
		}
	})
}

// Test DotN function with various sizes
func TestDotNSizes(t *testing.T) {
	sizes := []int{1, 8, 16, 32, 64, 128, 256, 512, 768, 1024}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size_%d", size), func(t *testing.T) {
			a := make([]int8, size)
			b := make([]int8, size)
			
			// Fill with deterministic pattern
			for i := 0; i < size; i++ {
				a[i] = int8((i * 7) % 256 - 128)
				b[i] = int8((i * 11) % 256 - 128)
			}
			
			result := DotN(a, b)
			
			// Calculate expected result manually
			expected := int32(0)
			for i := 0; i < size; i++ {
				expected += int32(a[i]) * int32(b[i])
			}
			
			if result != expected {
				t.Errorf("DotN size %d failed: got %d, want %d", size, result, expected)
			}
		})
	}
}

// Test error conditions
func TestDotNErrors(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5}
	
	// Test length mismatch should panic
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for length mismatch, but didn't panic")
		}
	}()
	
	_ = DotN(a, b)
}

// Cosine similarity tests
func TestCosine512(t *testing.T) {
	a := &Vec512{}
	b := &Vec512{}
	
	// Create normalized-like vectors
	for i := 0; i < 512; i++ {
		a[i] = int8((i % 64) - 32)
		b[i] = int8((i % 48) - 24)
	}
	
	scaleA := float32(0.01)
	scaleB := float32(0.02)
	
	result := Cosine512(a, b, scaleA, scaleB)
	
	// Verify it's using the dot product correctly
	expectedDot := Dot512(a, b)
	expectedCosine := float32(expectedDot) * scaleA * scaleB
	
	if result != expectedCosine {
		t.Errorf("Cosine similarity mismatch: got %f, want %f", result, expectedCosine)
	}
}

// L2 distance tests
func TestL2Squared512(t *testing.T) {
	a := &Vec512{}
	b := &Vec512{}
	
	// Simple test case
	for i := 0; i < 512; i++ {
		a[i] = int8(i % 128)
		b[i] = int8((i + 1) % 128)
	}
	
	result := L2Squared512(a, b)
	
	// Calculate expected L2 squared distance
	expected := int32(0)
	for i := 0; i < 512; i++ {
		diff := int32(a[i]) - int32(b[i])
		expected += diff * diff
	}
	
	if result != expected {
		t.Errorf("L2 squared distance mismatch: got %d, want %d", result, expected)
	}
	
	// Test identical vectors should give 0
	result2 := L2Squared512(a, a)
	if result2 != 0 {
		t.Errorf("L2 squared distance of identical vectors should be 0, got %d", result2)
	}
}

