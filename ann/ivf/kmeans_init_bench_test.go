package ivf

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

func makeVecs(n int, clusters int, seed int64) ([]simd.Vec512, []float32) {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][512]int8, clusters)
	for c := range centers {
		for j := 0; j < 512; j++ {
			centers[c][j] = int8(rng.Intn(255) - 128)
		}
	}
	vecs := make([]simd.Vec512, n)
	scales := make([]float32, n)
	for i := range vecs {
		c := centers[rng.Intn(clusters)]
		for j := 0; j < 512; j++ {
			v := int(c[j]) + rng.Intn(21) - 10
			if v > 127 {
				v = 127
			} else if v < -128 {
				v = -128
			}
			vecs[i][j] = int8(v)
		}
		scales[i] = 1.0
	}
	return vecs, scales
}

// inertia is the sum of squared distances from each point to its nearest
// centroid: the objective k-means minimises. Lower is better.
func inertia(km *KMeans, vecs []simd.Vec512) float64 {
	total := 0.0
	for i := range vecs {
		best := float32(math.MaxFloat32)
		for k := 0; k < km.K; k++ {
			if d := km.squaredDistance(&vecs[i], &km.Centroids[k]); d < best {
				best = d
			}
		}
		total += float64(best)
	}
	return total / float64(len(vecs))
}

// TestKMeansProductionShape times the shape that took 1h3m47s in prod:
// n=10000, K=NList=4096, 512-dim int8.
func TestKMeansProductionShape(t *testing.T) {
	vecs, scales := makeVecs(10000, 64, 11)
	km := NewKMeans(4096, 25)

	start := time.Now()
	km.Fit(vecs, scales)
	elapsed := time.Since(start)

	t.Logf("KMeans.Fit(n=10000, K=4096, dim=512) took %v", elapsed)
	t.Logf("mean inertia: %.2f", inertia(km, vecs))

	used := 0
	for _, c := range km.Counts {
		if c > 0 {
			used++
		}
	}
	t.Logf("non-empty clusters: %d/%d", used, km.K)

	if elapsed > 10*time.Minute {
		t.Errorf("Fit still slow: %v", elapsed)
	}
	if used < km.K/10 {
		t.Errorf("clustering degenerate: only %d/%d clusters used", used, km.K)
	}
}

// TestKMeansSmallQuality checks the incremental k-means++ still produces a
// sane clustering on a shape where structure is easy to verify.
func TestKMeansSmallQuality(t *testing.T) {
	vecs, scales := makeVecs(2000, 16, 3)
	km := NewKMeans(64, 25)
	km.Fit(vecs, scales)

	got := inertia(km, vecs)
	t.Logf("mean inertia (n=2000, K=64): %.2f", got)

	// Random centroids on this data give a far higher inertia; a working
	// k-means lands well below that baseline.
	rng := rand.New(rand.NewSource(99))
	baseline := NewKMeans(64, 25)
	for k := 0; k < baseline.K; k++ {
		baseline.Centroids[k] = vecs[rng.Intn(len(vecs))]
	}
	base := inertia(baseline, vecs)
	t.Logf("random-centroid baseline inertia: %.2f", base)

	if got > base {
		t.Errorf("k-means worse than random init: %.2f vs %.2f", got, base)
	}
}
