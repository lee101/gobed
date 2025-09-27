package ivf

import (
	"math"
	"math/rand"
	"sync"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// KMeans performs k-means clustering on int8 vectors
type KMeans struct {
	K         int           // Number of clusters
	MaxIters  int           // Maximum iterations
	Centroids []simd.Vec512 // Cluster centroids (quantized)
	Scales    []float32     // Centroid scales for dequantization
	Counts    []int         // Number of points per cluster
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

// Fit trains k-means on the given vectors
func (km *KMeans) Fit(vectors []simd.Vec512, scales []float32) {
	n := len(vectors)
	if n < km.K {
		panic("not enough vectors for k-means")
	}

	// Initialize centroids with k-means++
	km.initializeCentroidsPlusPlus(vectors, scales)

	assignments := make([]int, n)
	prevAssignments := make([]int, n)

	// Iterate until convergence or max iterations
	for iter := 0; iter < km.MaxIters; iter++ {
		// Assignment step: assign each point to nearest centroid
		changed := km.assignToCentroids(vectors, assignments)

		// Check for convergence
		if !changed && iter > 0 {
			break
		}

		// Update step: recompute centroids
		km.updateCentroids(vectors, scales, assignments)

		copy(prevAssignments, assignments)
	}

	// Count final assignments
	for i := range km.Counts {
		km.Counts[i] = 0
	}
	for _, assign := range assignments {
		km.Counts[assign]++
	}
}

// initializeCentroidsPlusPlus uses k-means++ initialization
func (km *KMeans) initializeCentroidsPlusPlus(vectors []simd.Vec512, scales []float32) {
	n := len(vectors)

	// Choose first centroid randomly
	first := rand.Intn(n)
	km.Centroids[0] = vectors[first]
	km.Scales[0] = scales[first]

	// Distance from each point to nearest centroid
	distances := make([]float32, n)

	for k := 1; k < km.K; k++ {
		// Compute distances to nearest centroid
		totalDist := float32(0)
		for i := range vectors {
			minDist := float32(math.MaxFloat32)
			for j := 0; j < k; j++ {
				dist := km.squaredDistance(&vectors[i], &km.Centroids[j])
				if dist < minDist {
					minDist = dist
				}
			}
			distances[i] = minDist
			totalDist += minDist
		}

		// Choose next centroid with probability proportional to squared distance
		threshold := rand.Float32() * totalDist
		cumSum := float32(0)
		for i := range vectors {
			cumSum += distances[i]
			if cumSum >= threshold {
				km.Centroids[k] = vectors[i]
				km.Scales[k] = scales[i]
				break
			}
		}
	}
}

// assignToCentroids assigns each vector to its nearest centroid
func (km *KMeans) assignToCentroids(vectors []simd.Vec512, assignments []int) bool {
	changed := false

	// Parallel assignment for better performance
	var wg sync.WaitGroup
	var mu sync.Mutex

	chunkSize := (len(vectors) + 7) / 8 // Process in 8 chunks

	for start := 0; start < len(vectors); start += chunkSize {
		end := start + chunkSize
		if end > len(vectors) {
			end = len(vectors)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			localChanged := false
			for i := start; i < end; i++ {
				minDist := float32(math.MaxFloat32)
				bestCluster := 0

				// Find nearest centroid
				for k := 0; k < km.K; k++ {
					dist := km.squaredDistance(&vectors[i], &km.Centroids[k])
					if dist < minDist {
						minDist = dist
						bestCluster = k
					}
				}

				if assignments[i] != bestCluster {
					assignments[i] = bestCluster
					localChanged = true
				}
			}

			if localChanged {
				mu.Lock()
				changed = true
				mu.Unlock()
			}
		}(start, end)
	}

	wg.Wait()
	return changed
}

// updateCentroids recomputes centroids based on assignments
func (km *KMeans) updateCentroids(vectors []simd.Vec512, scales []float32, assignments []int) {
	// Accumulate in float32 for precision
	type accumulator struct {
		sum   [512]float32
		count int
	}

	accumulators := make([]accumulator, km.K)

	// Accumulate vectors for each cluster
	for i, vec := range vectors {
		cluster := assignments[i]
		scale := scales[i]

		for d := 0; d < 512; d++ {
			// Dequantize and accumulate
			accumulators[cluster].sum[d] += float32(vec[d]) * scale
		}
		accumulators[cluster].count++
	}

	// Compute new centroids
	for k := 0; k < km.K; k++ {
		if accumulators[k].count == 0 {
			// Empty cluster - reinitialize randomly
			idx := rand.Intn(len(vectors))
			km.Centroids[k] = vectors[idx]
			km.Scales[k] = scales[idx]
			continue
		}

		// Compute mean
		count := float32(accumulators[k].count)
		var mean [512]float32
		for d := 0; d < 512; d++ {
			mean[d] = accumulators[k].sum[d] / count
		}

		// Quantize back to int8
		km.Centroids[k], km.Scales[k] = quantizeVector(mean[:])
	}
}

// squaredDistance computes squared L2 distance between int8 vectors
func (km *KMeans) squaredDistance(a, b *simd.Vec512) float32 {
	var sum int32
	for i := 0; i < 512; i++ {
		diff := int32(a[i]) - int32(b[i])
		sum += diff * diff
	}
	return float32(sum)
}

// Predict returns the cluster assignment for a query vector
func (km *KMeans) Predict(vec *simd.Vec512) int {
	minDist := float32(math.MaxFloat32)
	bestCluster := 0

	for k := 0; k < km.K; k++ {
		dist := km.squaredDistance(vec, &km.Centroids[k])
		if dist < minDist {
			minDist = dist
			bestCluster = k
		}
	}

	return bestCluster
}

// PredictMultiple returns the nprobe nearest clusters for a query
func (km *KMeans) PredictMultiple(vec *simd.Vec512, nprobe int) []int {
	if nprobe > km.K {
		nprobe = km.K
	}

	type distPair struct {
		cluster int
		dist    float32
	}

	distances := make([]distPair, km.K)
	for k := 0; k < km.K; k++ {
		distances[k] = distPair{
			cluster: k,
			dist:    km.squaredDistance(vec, &km.Centroids[k]),
		}
	}

	// Partial sort to get top nprobe
	for i := 0; i < nprobe; i++ {
		minIdx := i
		for j := i + 1; j < km.K; j++ {
			if distances[j].dist < distances[minIdx].dist {
				minIdx = j
			}
		}
		distances[i], distances[minIdx] = distances[minIdx], distances[i]
	}

	result := make([]int, nprobe)
	for i := 0; i < nprobe; i++ {
		result[i] = distances[i].cluster
	}

	return result
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
