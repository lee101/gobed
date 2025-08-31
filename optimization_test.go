package gobed

import (
	"runtime"
	"testing"
)

// TestMemoryPooling verifies buffer pool reduces allocations
func TestMemoryPooling(t *testing.T) {
	// Measure allocations without pool
	var m1, m2 runtime.MemStats
	
	runtime.GC()
	runtime.ReadMemStats(&m1)
	
	// Allocate without pool
	for i := 0; i < 1000; i++ {
		_ = make([]int, 512)
	}
	
	runtime.ReadMemStats(&m2)
	allocsWithoutPool := m2.Mallocs - m1.Mallocs
	
	// Measure allocations with pool
	runtime.GC()
	runtime.ReadMemStats(&m1)
	
	for i := 0; i < 1000; i++ {
		buf := GetTokenBuffer()
		// Simulate some usage
		for j := 0; j < 10; j++ {
			buf = append(buf, j)
		}
		PutTokenBuffer(buf)
	}
	
	runtime.ReadMemStats(&m2)
	allocsWithPool := m2.Mallocs - m1.Mallocs
	
	reduction := float64(allocsWithoutPool-allocsWithPool) / float64(allocsWithoutPool) * 100
	t.Logf("Memory pool reduced allocations by %.1f%% (%d -> %d)", 
		reduction, allocsWithoutPool, allocsWithPool)
	
	// Pool should reduce allocations significantly
	if allocsWithPool >= allocsWithoutPool {
		t.Error("Buffer pool did not reduce allocations")
	}
}

// TestBatchSizing verifies batch size calculation
func TestBatchSizing(t *testing.T) {
	cpuBatch := GetOptimalBatchSize()
	t.Logf("Optimal CPU batch size: %d", cpuBatch)
	
	if cpuBatch < 32 || cpuBatch > 2048 {
		t.Errorf("CPU batch size out of expected range: %d", cpuBatch)
	}
	
	// Test power of 2 rounding
	tests := []struct {
		input    int
		expected int
	}{
		{100, 128},
		{200, 256},
		{500, 512},
		{1000, 1024},
	}
	
	for _, tt := range tests {
		result := nearestPowerOf2(tt.input)
		if result != tt.expected {
			t.Errorf("nearestPowerOf2(%d) = %d, want %d", tt.input, result, tt.expected)
		}
	}
}

// TestFastQuantization verifies quantization works correctly
func TestFastQuantization(t *testing.T) {
	// Create test embedding
	embedding := make([]float32, 512)
	for i := range embedding {
		embedding[i] = float32(i) / 512.0
	}
	
	quantized, scale := FastQuantize(embedding)
	
	if len(quantized) != len(embedding) {
		t.Errorf("Quantized length mismatch: got %d, want %d", len(quantized), len(embedding))
	}
	
	if scale <= 0 {
		t.Errorf("Invalid scale: %f", scale)
	}
	
	// Verify values are in int8 range
	for i, v := range quantized {
		if v < 0 || v > 127 {
			t.Errorf("Quantized value out of range at index %d: %d", i, v)
		}
	}
	
	// Return buffer to pool
	PutInt8Buffer(quantized)
}

// BenchmarkBufferPool measures buffer pool performance
func BenchmarkBufferPool(b *testing.B) {
	b.Run("WithPool", func(b *testing.B) {
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				buf := GetTokenBuffer()
				for i := 0; i < 100; i++ {
					buf = append(buf, i)
				}
				PutTokenBuffer(buf)
			}
		})
	})
	
	b.Run("WithoutPool", func(b *testing.B) {
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				buf := make([]int, 0, 512)
				for i := 0; i < 100; i++ {
					buf = append(buf, i)
				}
			}
		})
	})
}

// BenchmarkQuantizationSimple compares quantization methods
func BenchmarkQuantizationSimple(b *testing.B) {
	embedding := make([]float32, 1024)
	for i := range embedding {
		embedding[i] = float32(i) / 1024.0
	}
	
	b.Run("FastQuantize", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			quantized, _ := FastQuantize(embedding)
			PutInt8Buffer(quantized)
		}
	})
}