package ivf

import (
	"math/rand"
	"testing"

	"github.com/lee101/gobed/pkg/ann/simd"
)

func TestInitializeCentroidsKeepsScalePairing(t *testing.T) {
	vectors := make([]simd.Vec512, 64)
	scales := make([]float32, len(vectors))
	for i := range vectors {
		vectors[i][0] = int8(i - 32)
		vectors[i][1] = int8(i%7 - 3)
		scales[i] = float32(i + 1)
	}

	km := NewKMeans(24, 1)
	km.initializeCentroidsPlusPlus(vectors, scales)

	for i, centroid := range km.Centroids {
		matched := false
		for j := range vectors {
			if centroid == vectors[j] && km.Scales[i] == scales[j] {
				matched = true
				break
			}
		}
		if !matched {
			t.Fatalf("centroid %d lost its scale pairing", i)
		}
	}
}

func TestInitializeOptimizedCentroidsKeepsScalePairing(t *testing.T) {
	vectors := make([]simd.Vec512, 64)
	scales := make([]float32, len(vectors))
	for i := range vectors {
		vectors[i][0] = int8(i - 32)
		vectors[i][1] = int8(i%7 - 3)
		scales[i] = float32(i + 1)
	}

	km := NewKMeansOptimized(24, 1)
	km.initializeCentroidsOptimized(vectors, scales)

	for i, centroid := range km.Centroids {
		matched := false
		for j := range vectors {
			if centroid == vectors[j] && km.Scales[i] == scales[j] {
				matched = true
				break
			}
		}
		if !matched {
			t.Fatalf("centroid %d lost its scale pairing", i)
		}
	}
}

func BenchmarkInitializeCentroidsPlusPlus(b *testing.B) {
	vectors := make([]simd.Vec512, 4096)
	scales := make([]float32, len(vectors))
	rng := rand.New(rand.NewSource(1))
	for i := range vectors {
		for j := range vectors[i] {
			vectors[i][j] = int8(rng.Intn(255) - 127)
		}
		scales[i] = 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rand.Seed(1)
		NewKMeans(256, 1).initializeCentroidsPlusPlus(vectors, scales)
	}
}

func BenchmarkInitializeCentroidsOptimized(b *testing.B) {
	vectors := make([]simd.Vec512, 4096)
	scales := make([]float32, len(vectors))
	rng := rand.New(rand.NewSource(1))
	for i := range vectors {
		for j := range vectors[i] {
			vectors[i][j] = int8(rng.Intn(255) - 127)
		}
		scales[i] = 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rand.Seed(1)
		NewKMeansOptimized(256, 1).initializeCentroidsOptimized(vectors, scales)
	}
}
