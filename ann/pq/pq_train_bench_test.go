package pq

import (
	"math/rand"
	"testing"
	"time"
)

// makeTrainingVectors builds a clustered synthetic set so k-means has real
// structure to find rather than uniform noise.
func makeTrainingVectors(n, d, clusters int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range centers {
		centers[c] = make([]float32, d)
		for j := range centers[c] {
			centers[c][j] = rng.Float32()*2 - 1
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(clusters)]
		v := make([]float32, d)
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64())*0.1
		}
		out[i] = v
	}
	return out
}

// TestTrainTiming reports wall time for PQ training at M=64/KSub=256.
// Train requires n >= KSub*M (16384), which is why the production index
// (n=10000) skips PQ entirely and spends its time in IVF k-means instead.
func TestTrainTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("long")
	}
	vectors := makeTrainingVectors(20000, 512, 64, 42)
	pq := NewProductQuantizer(512, 64, 256)

	start := time.Now()
	pq.Train(vectors)
	elapsed := time.Since(start)

	t.Logf("PQ Train(10000x512, M=64, KSub=256) took %v", elapsed)
	if !pq.Trained {
		t.Fatal("not trained")
	}
}

// TestEncodeQuantizationError guards against the optimizations degrading
// codebook quality: mean reconstruction error must stay small.
func TestEncodeQuantizationError(t *testing.T) {
	// n must be >= KSub*M (4096) or Train panics.
	vectors := makeTrainingVectors(8000, 128, 32, 7)
	pq := NewProductQuantizer(128, 16, 256)
	pq.Train(vectors)

	var total float64
	for _, v := range vectors {
		codes := pq.Encode(v)
		var se float64
		for m := 0; m < pq.M; m++ {
			c := pq.Codebooks[m][codes[m]]
			for d := 0; d < pq.DSub; d++ {
				diff := float64(v[m*pq.DSub+d] - c[d])
				se += diff * diff
			}
		}
		total += se
	}
	mean := total / float64(len(vectors))
	t.Logf("mean squared reconstruction error: %.6f", mean)

	// Clustered data with sigma=0.1 per dim: a working quantizer lands well
	// under the variance of the raw data.
	if mean > 1.0 {
		t.Errorf("reconstruction error too high: %.6f", mean)
	}
}
