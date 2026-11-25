package ivf

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// KMeans performs k-means clustering on int8 vectors
type KMeans struct {
	K         int           // Number of clusters
	MaxIters  int           // Maximum iterations
	Centroids []simd.Vec512 // Cluster centroids (quantized)
	Scales    []float32     // Centroid scales for dequantization
	Counts    []int         // Number of points per cluster

	// norms[k] = ||Centroids[k]||^2, so nearest-centroid search is one batched
	// SIMD dot product per point: argmin ||x-c||^2 = argmin ||c||^2 - 2x.c.
	norms   []int32
	bufPool sync.Pool
}

// NewKMeans creates a new k-means clusterer
func NewKMeans(k int, maxIters int) *KMeans {
	if maxIters <= 0 {
		maxIters = 25
	}
	return &KMeans{
		K:         k,
		MaxIters:  maxIters,
		Centroids: make([]simd.Vec512, k),
		Scales:    make([]float32, k),
		Counts:    make([]int, k),
	}
}

// parallelRange splits [0,n) into one contiguous chunk per worker.
func parallelRange(n int, fn func(worker, start, end int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		if n > 0 {
			fn(0, 0, n)
		}
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start, end := w*chunk, min((w+1)*chunk, n)
		if start >= end {
			break
		}
		wg.Add(1)
		go func(w, start, end int) {
			defer wg.Done()
			fn(w, start, end)
		}(w, start, end)
	}
	wg.Wait()
}

func selfDot(v *simd.Vec512) int32 { return simd.Dot512(v, v) }

// SetCentroids installs trained centroids (e.g. restored from disk) and
// prepares the routing tables. It does not retrain.
func (km *KMeans) SetCentroids(centroids []simd.Vec512, scales []float32) {
	km.K = len(centroids)
	km.Centroids = centroids
	km.Scales = scales
	if len(km.Counts) != km.K {
		km.Counts = make([]int, km.K)
	}
	km.refreshNorms()
}

func (km *KMeans) refreshNorms() {
	norms := make([]int32, len(km.Centroids))
	for k := range km.Centroids {
		norms[k] = selfDot(&km.Centroids[k])
	}
	km.norms = norms
}

// Fit trains k-means on the given vectors: k-means++ seeding, then Lloyd
// iterations until fewer than 0.1% of points change cluster or MaxIters.
// Both the seeding distance updates and the assignment step run across
// GOMAXPROCS workers using batched SIMD dot products.
func (km *KMeans) Fit(vectors []simd.Vec512, scales []float32) {
	n := len(vectors)
	if n < km.K {
		panic("not enough vectors for k-means")
	}

	km.initializeCentroidsPlusPlus(vectors, scales)
	km.refreshNorms()

	assignments := make([]int, n)
	for i := range assignments {
		assignments[i] = -1
	}
	rng := rand.New(rand.NewSource(int64(n)*31 + int64(km.K)))
	tol := n / 1000

	for iter := 0; iter < km.MaxIters; iter++ {
		changed := km.assign(vectors, assignments)
		if iter > 0 && changed <= tol {
			break
		}
		km.updateCentroids(vectors, scales, assignments, rng)
		km.refreshNorms()
	}

	for i := range km.Counts {
		km.Counts[i] = 0
	}
	for _, a := range assignments {
		km.Counts[a]++
	}
}

// initializeCentroidsPlusPlus uses k-means++ seeding. Each step only folds
// the newest centroid into the per-point nearest distance, so the cost is
// O(n*K) dot products rather than O(n*K^2).
func (km *KMeans) initializeCentroidsPlusPlus(vectors []simd.Vec512, scales []float32) {
	n := len(vectors)
	rng := rand.New(rand.NewSource(int64(n)*17 + int64(km.K)))

	vnorm := make([]int32, n)
	parallelRange(n, func(_, s, e int) {
		for i := s; i < e; i++ {
			vnorm[i] = selfDot(&vectors[i])
		}
	})

	first := rng.Intn(n)
	km.Centroids[0] = vectors[first]
	km.Scales[0] = scales[first]

	dist := make([]float64, n)
	for i := range dist {
		dist[i] = math.MaxFloat64
	}
	workers := runtime.GOMAXPROCS(0)
	partial := make([]float64, workers+1)

	for k := 1; k < km.K; k++ {
		c := &km.Centroids[k-1]
		cn := selfDot(c)
		for i := range partial {
			partial[i] = 0
		}
		parallelRange(n, func(w, s, e int) {
			sum := 0.0
			for i := s; i < e; i++ {
				d := float64(vnorm[i] + cn - 2*simd.Dot512(&vectors[i], c))
				if d < dist[i] {
					dist[i] = d
				}
				sum += dist[i]
			}
			partial[w] = sum
		})
		total := 0.0
		for _, p := range partial {
			total += p
		}

		chosen := -1
		if total > 0 && !math.IsInf(total, 0) && !math.IsNaN(total) {
			threshold := rng.Float64() * total
			cum := 0.0
			for i := 0; i < n; i++ {
				cum += dist[i]
				if cum >= threshold {
					chosen = i
					break
				}
			}
			if chosen < 0 {
				chosen = n - 1
			}
		} else {
			chosen = rng.Intn(n)
		}
		km.Centroids[k] = vectors[chosen]
		km.Scales[k] = scales[chosen]
	}
}

// nearest returns the closest centroid using buf (len >= K) as scratch.
func (km *KMeans) nearest(vec *simd.Vec512, buf []int32) int {
	dots := simd.Dot512Batch(vec, km.Centroids, buf)
	best, bestD := 0, int64(math.MaxInt64)
	for k, d := range dots {
		v := int64(km.norms[k]) - 2*int64(d)
		if v < bestD {
			bestD, best = v, k
		}
	}
	return best
}

// assign sets each point to its nearest centroid and returns how many moved.
func (km *KMeans) assign(vectors []simd.Vec512, assignments []int) int {
	var changed atomic.Int64
	parallelRange(len(vectors), func(_, s, e int) {
		buf := make([]int32, km.K)
		local := 0
		for i := s; i < e; i++ {
			b := km.nearest(&vectors[i], buf)
			if assignments[i] != b {
				assignments[i] = b
				local++
			}
		}
		changed.Add(int64(local))
	})
	return int(changed.Load())
}

// assignToCentroids is kept for callers of the old API.
func (km *KMeans) assignToCentroids(vectors []simd.Vec512, assignments []int) bool {
	if len(km.norms) != len(km.Centroids) {
		km.refreshNorms()
	}
	return km.assign(vectors, assignments) > 0
}

// updateCentroids recomputes each centroid as the mean of its dequantized
// members. Points are bucketed by cluster first so workers own disjoint
// clusters and no per-worker K x 512 accumulators are needed. An empty
// cluster is reseeded with a random member of the largest cluster.
func (km *KMeans) updateCentroids(vectors []simd.Vec512, scales []float32, assignments []int, rng *rand.Rand) {
	K := km.K
	offsets := make([]int, K+1)
	for _, a := range assignments {
		offsets[a+1]++
	}
	for k := 0; k < K; k++ {
		offsets[k+1] += offsets[k]
	}
	order := make([]int, len(assignments))
	fill := append([]int(nil), offsets[:K]...)
	for i, a := range assignments {
		order[fill[a]] = i
		fill[a]++
	}

	parallelRange(K, func(_, s, e int) {
		var sum [512]float32
		for k := s; k < e; k++ {
			members := order[offsets[k]:offsets[k+1]]
			if len(members) == 0 {
				continue
			}
			sum = [512]float32{}
			for _, i := range members {
				v, sc := &vectors[i], scales[i]
				for d := 0; d < 512; d++ {
					sum[d] += float32(v[d]) * sc
				}
			}
			inv := 1 / float32(len(members))
			for d := range sum {
				sum[d] *= inv
			}
			km.Centroids[k], km.Scales[k] = quantizeVector(sum[:])
		}
	})

	largest := 0
	for k := 1; k < K; k++ {
		if offsets[k+1]-offsets[k] > offsets[largest+1]-offsets[largest] {
			largest = k
		}
	}
	for k := 0; k < K; k++ {
		if offsets[k+1] == offsets[k] {
			members := order[offsets[largest]:offsets[largest+1]]
			i := members[rng.Intn(len(members))]
			km.Centroids[k] = vectors[i]
			km.Scales[k] = scales[i]
		}
	}
}

// squaredDistance computes squared L2 distance between int8 vectors
func (km *KMeans) squaredDistance(a, b *simd.Vec512) float32 {
	return float32(simd.L2Squared512(a, b))
}

func (km *KMeans) getBuf() []int32 {
	if v := km.bufPool.Get(); v != nil {
		if b := v.([]int32); len(b) >= km.K {
			return b
		}
	}
	return make([]int32, km.K)
}

// Predict returns the cluster assignment for a query vector
func (km *KMeans) Predict(vec *simd.Vec512) int {
	if len(km.norms) != len(km.Centroids) {
		best, bestD := 0, float32(math.MaxFloat32)
		for k := range km.Centroids {
			if d := km.squaredDistance(vec, &km.Centroids[k]); d < bestD {
				bestD, best = d, k
			}
		}
		return best
	}
	buf := km.getBuf()
	best := km.nearest(vec, buf)
	km.bufPool.Put(buf)
	return best
}

// PredictMultiple returns the nprobe nearest clusters for a query, nearest first.
func (km *KMeans) PredictMultiple(vec *simd.Vec512, nprobe int) []int {
	K := len(km.Centroids)
	if nprobe > K {
		nprobe = K
	}
	if nprobe <= 0 {
		return nil
	}
	type distPair struct {
		cluster int
		dist    int64
	}
	pairs := make([]distPair, K)
	if len(km.norms) == K {
		buf := km.getBuf()
		dots := simd.Dot512Batch(vec, km.Centroids, buf)
		for k, d := range dots {
			pairs[k] = distPair{k, int64(km.norms[k]) - 2*int64(d)}
		}
		km.bufPool.Put(buf)
	} else {
		for k := range km.Centroids {
			pairs[k] = distPair{k, int64(simd.L2Squared512(vec, &km.Centroids[k]))}
		}
	}
	for i := 0; i < nprobe; i++ {
		m := i
		for j := i + 1; j < K; j++ {
			if pairs[j].dist < pairs[m].dist {
				m = j
			}
		}
		pairs[i], pairs[m] = pairs[m], pairs[i]
	}
	out := make([]int, nprobe)
	for i := range out {
		out[i] = pairs[i].cluster
	}
	return out
}

// quantizeVector converts float32 vector to int8 with scaling
func quantizeVector(vec []float32) (simd.Vec512, float32) {
	// Find min and max for quantization range
	minVal := vec[0]
	maxVal := vec[0]
	for _, v := range vec[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute scale for symmetric quantization
	absMax := maxVal
	if -minVal > absMax {
		absMax = -minVal
	}

	scale := absMax / 127.0
	if scale == 0 {
		scale = 1.0
	}

	var result simd.Vec512
	for i := 0; i < 512; i++ {
		quantized := int32(vec[i] / scale)
		if quantized < -128 {
			quantized = -128
		} else if quantized > 127 {
			quantized = 127
		}
		result[i] = int8(quantized)
	}

	return result, scale
}
