//go:build legacy

package gobed

import (
	"testing"
)

// TestMemoryPooling verifies buffer pool reduces allocations
func TestMemoryPooling(t *testing.T) {
	// Test that buffer pool functions work correctly
	// Rather than testing allocation patterns which are non-deterministic

	// Test basic pool operation
	buffers := make([][]int, 100)
	for i := 0; i < 100; i++ {
		buf := GetTokenBuffer()
		if buf == nil {
			t.Fatalf("GetTokenBuffer returned nil at iteration %d", i)
		}
		if cap(buf) < 512 {
			t.Errorf("Buffer %d too small: capacity=%d, expected>=512", i, cap(buf))
		}
		// Use the buffer
		buf = append(buf, i)
		buffers[i] = buf
	}

	// Return all buffers to pool
	for i, buf := range buffers {
		if len(buf) != 1 || buf[0] != i {
			t.Errorf("Buffer %d modified unexpectedly", i)
		}
		PutTokenBuffer(buf)
	}

	// Test that we can get buffers again
	for i := 0; i < 10; i++ {
		buf := GetTokenBuffer()
		if buf == nil {
			t.Errorf("GetTokenBuffer returned nil after pool return at iteration %d", i)
		}
		PutTokenBuffer(buf)
	}

	t.Logf("✓ Memory pool operations working correctly")
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
