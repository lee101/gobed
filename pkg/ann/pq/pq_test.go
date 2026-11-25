package pq

import (
	"math/rand"
	"testing"
)

func clustered(n, d, clusters int, seed int64) [][]float32 {
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

func TestEncodeQuantizationError(t *testing.T) {
	vectors := clustered(8000, 128, 32, 7)
	pq := NewProductQuantizer(128, 16, 256)
	pq.Train(vectors)

	var total float64
	for _, v := range vectors {
		rec := pq.Decode(pq.Encode(v))
		for d := range v {
			diff := float64(v[d] - rec[d])
			total += diff * diff
		}
	}
	mean := total / float64(len(vectors))
	t.Logf("mean squared reconstruction error: %.6f", mean)
	if mean > 1.0 {
		t.Errorf("reconstruction error too high: %.6f", mean)
	}

	codes := pq.EncodeBatch(vectors[:100])
	for i := range codes {
		single := pq.Encode(vectors[i])
		for m := range single {
			if codes[i][m] != single[m] {
				t.Fatalf("EncodeBatch differs from Encode")
			}
		}
	}
}

func TestIPTableApproximatesDot(t *testing.T) {
	vectors := clustered(4000, 64, 16, 3)
	pq := NewProductQuantizer(64, 8, 256)
	pq.Train(vectors)
	q := vectors[0]
	table := pq.ComputeIPTable(q)
	for _, v := range vectors[1:50] {
		rec := pq.Decode(pq.Encode(v))
		var want float32
		for d := range q {
			want += q[d] * rec[d]
		}
		got := table.Distance(pq.Encode(v))
		if diff := got - want; diff > 1e-3 || diff < -1e-3 {
			t.Fatalf("IP table %f, decoded dot %f", got, want)
		}
	}
}
