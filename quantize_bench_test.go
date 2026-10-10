package gobed

import (
	"math"
	"math/rand"
	"testing"
)

// quantizeVector512Ref is the original implementation, kept as the correctness
// oracle for the optimized version.
func quantizeVector512Ref(vec []float32) *Int8Result512 {
	maxAbs := float32(0)
	for _, v := range vec {
		if abs := float32(math.Abs(float64(v))); abs > maxAbs {
			maxAbs = abs
		}
	}
	scale := maxAbs / 127.0
	if scale == 0 {
		scale = 1.0
	}
	result := make([]int8, len(vec))
	for i, v := range vec {
		quantized := int8(math.Round(float64(v / scale)))
		if quantized > 127 {
			quantized = 127
		} else if quantized < -128 {
			quantized = -128
		}
		result[i] = quantized
	}
	return &Int8Result512{Vector: result, Scale: scale}
}

func randVec(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64()) * 0.37
	}
	return v
}

// TestQuantizeMatchesReference pins the optimized quantizer to the original
// byte-for-byte across many random vectors and edge cases.
func TestQuantizeMatchesReference(t *testing.T) {
	cases := [][]float32{
		make([]float32, 512),            // all zeros -> scale 1.0 path
		{1, -1, 0, 127, -128, 0.5, -0.5}, // exact .5 rounding boundaries
		{-0.0, 1e-30, -1e-30},
	}
	for s := int64(0); s < 200; s++ {
		cases = append(cases, randVec(512, s))
	}

	for ci, vec := range cases {
		want := quantizeVector512Ref(vec)
		got := quantizeVector512(vec)

		if want.Scale != got.Scale {
			t.Fatalf("case %d: scale mismatch: want %v got %v", ci, want.Scale, got.Scale)
		}
		if len(want.Vector) != len(got.Vector) {
			t.Fatalf("case %d: length mismatch", ci)
		}
		for i := range want.Vector {
			if want.Vector[i] != got.Vector[i] {
				t.Fatalf("case %d: index %d: want %d got %d (input %v scale %v)",
					ci, i, want.Vector[i], got.Vector[i], vec[i], want.Scale)
			}
		}
	}
}

func BenchmarkQuantizeReference(b *testing.B) {
	vec := randVec(512, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = quantizeVector512Ref(vec)
	}
}

func BenchmarkQuantizeOptimized(b *testing.B) {
	vec := randVec(512, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = quantizeVector512(vec)
	}
}
