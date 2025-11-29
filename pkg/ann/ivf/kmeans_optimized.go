package ivf

import (
	"math"
	"math/rand"
	"sync"

	"github.com/lee101/gobed/pkg/ann/simd"
)

// distPair represents a cluster-distance pair for sorting
type distPair struct {
	cluster int
	dist    float32
}

// KMeansOptimized is an optimized k-means implementation with better memory management
// and SIMD-accelerated distance computations
type KMeansOptimized struct {
	K         int           // Number of clusters
	MaxIters  int           // Maximum iterations
	Centroids []simd.Vec512 // Cluster centroids (quantized)
	Scales    []float32     // Centroid scales for dequantization
	Counts    []int         // Number of points per cluster

	// Pre-allocated buffers to reduce allocations
	assignments     []int
	prevAssignments []int
	distances       []float32
	changed         []bool
}

// NewKMeansOptimized creates a new optimized k-means clusterer
func NewKMeansOptimized(k int, maxIters int) *KMeansOptimized {
	if maxIters <= 0 {
		maxIters = 25
	}
	return &KMeansOptimized{
		K:         k,
		MaxIters:  maxIters,
		Centroids: make([]simd.Vec512, k),
		Scales:    make([]float32, k),
		Counts:    make([]int, k),
	}
}

// Fit trains k-means on the given vectors with optimizations
func (km *KMeansOptimized) Fit(vectors []simd.Vec512, scales []float32) {
	n := len(vectors)
	if n < km.K {
		panic("not enough vectors for k-means")
	}

	// Pre-allocate buffers to reduce allocations during iterations
	if cap(km.assignments) < n {
		km.assignments = make([]int, n)
		km.prevAssignments = make([]int, n)
		km.distances = make([]float32, km.K)
		km.changed = make([]bool, km.K)
	} else {
		km.assignments = km.assignments[:n]
		km.prevAssignments = km.prevAssignments[:n]
	}

	// Initialize centroids with optimized k-means++
	km.initializeCentroidsOptimized(vectors, scales)

	// Iterate until convergence or max iterations
	converged := false
	for iter := 0; iter < km.MaxIters && !converged; iter++ {
		// Assignment step: assign each point to nearest centroid
		changed := km.assignToCentroidsOptimized(vectors)

		// Early convergence check
		if !changed && iter > 0 {
			converged = true
			break
		}

		// Update step: recompute centroids
		km.updateCentroidsOptimized(vectors, scales, km.assignments)

		copy(km.prevAssignments, km.assignments)
	}

	// Count final assignments
	for i := range km.Counts {
		km.Counts[i] = 0
	}
	for _, assign := range km.assignments {
		km.Counts[assign]++
	}
}

// initializeCentroidsOptimized uses optimized k-means++ initialization
func (km *KMeansOptimized) initializeCentroidsOptimized(vectors []simd.Vec512, scales []float32) {
	n := len(vectors)

	// Choose first centroid randomly
	first := rand.Intn(n)
	km.Centroids[0] = vectors[first]
	km.Scales[0] = scales[first]

	// Pre-allocate distance buffer
	if cap(km.distances) < n {
		km.distances = make([]float32, n)
	} else {
		km.distances = km.distances[:n]
	}

	for k := 1; k < km.K; k++ {
		// Compute distances to nearest centroid using SIMD
		totalDist := float32(0)
		for i := range vectors {
			minDist := float32(math.MaxFloat32)
			for j := 0; j < k; j++ {
				// Use SIMD-optimized squared distance
				dist := float32(simd.L2Squared512(&vectors[i], &km.Centroids[j]))
				if dist < minDist {
					minDist = dist
				}
			}
			km.distances[i] = minDist
			totalDist += minDist
		}

		// Choose next centroid with probability proportional to squared distance
		if totalDist == 0 {
			// All remaining points are identical to existing centroids
			km.Centroids[k] = vectors[rand.Intn(n)]
			km.Scales[k] = scales[rand.Intn(n)]
			continue
		}

		threshold := rand.Float32() * totalDist
		cumSum := float32(0)
		chosen := false
		for i := range vectors {
			cumSum += km.distances[i]
			if cumSum >= threshold {
				km.Centroids[k] = vectors[i]
				km.Scales[k] = scales[i]
				chosen = true
				break
			}
		}

		if !chosen {
			// Fallback: choose last vector
			km.Centroids[k] = vectors[n-1]
			km.Scales[k] = scales[n-1]
		}
	}
}

// assignToCentroidsOptimized assigns each vector to its nearest centroid with optimizations
func (km *KMeansOptimized) assignToCentroidsOptimized(vectors []simd.Vec512) bool {
	n := len(vectors)
	changed := false

	// Use parallel assignment for better performance on large datasets
	if n > 1000 {
		return km.assignToCentroidsParallel(vectors)
	}

	// Sequential assignment for smaller datasets (better cache locality)
	for i := 0; i < n; i++ {
		minDist := float32(math.MaxFloat32)
		bestCluster := 0

		// Find nearest centroid using SIMD
		for k := 0; k < km.K; k++ {
			dist := float32(simd.L2Squared512(&vectors[i], &km.Centroids[k]))
			if dist < minDist {
				minDist = dist
				bestCluster = k
			}
		}

		if km.assignments[i] != bestCluster {
			km.assignments[i] = bestCluster
			changed = true
		}
	}

	return changed
}

// assignToCentroidsParallel uses parallel assignment for large datasets
func (km *KMeansOptimized) assignToCentroidsParallel(vectors []simd.Vec512) bool {
	n := len(vectors)
	var globalChanged bool
	var mu sync.Mutex

	// Process in chunks for better performance
	numWorkers := 8
	chunkSize := (n + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			localChanged := false
			for i := start; i < end; i++ {
				minDist := float32(math.MaxFloat32)
				bestCluster := 0

				// Find nearest centroid using SIMD
				for k := 0; k < km.K; k++ {
					dist := float32(simd.L2Squared512(&vectors[i], &km.Centroids[k]))
					if dist < minDist {
						minDist = dist
						bestCluster = k
					}
				}

				if km.assignments[i] != bestCluster {
					km.assignments[i] = bestCluster
					localChanged = true
				}
			}

			if localChanged {
				mu.Lock()
				globalChanged = true
				mu.Unlock()
			}
		}(start, end)
	}

	wg.Wait()
	return globalChanged
}

// updateCentroidsOptimized recomputes centroids with optimized allocation patterns
func (km *KMeansOptimized) updateCentroidsOptimized(vectors []simd.Vec512, scales []float32, assignments []int) {
	// Pre-allocate accumulator to avoid repeated allocations
	type accumulator struct {
		sum   [512]float32
		count int
	}

	// Reuse accumulator slice to reduce allocations
	accumulators := make([]accumulator, km.K)

	// Accumulate vectors for each cluster
	for i, vec := range vectors {
		cluster := assignments[i]
		scale := scales[i]

		// Vectorized accumulation (unrolled for better performance)
		acc := &accumulators[cluster]
		for d := 0; d < 512; d += 8 {
			// Dequantize and accumulate in chunks
			acc.sum[d] += float32(vec[d]) * scale
			acc.sum[d+1] += float32(vec[d+1]) * scale
			acc.sum[d+2] += float32(vec[d+2]) * scale
			acc.sum[d+3] += float32(vec[d+3]) * scale
			acc.sum[d+4] += float32(vec[d+4]) * scale
			acc.sum[d+5] += float32(vec[d+5]) * scale
			acc.sum[d+6] += float32(vec[d+6]) * scale
			acc.sum[d+7] += float32(vec[d+7]) * scale
		}
		acc.count++
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

		// Compute mean and quantize
		count := float32(accumulators[k].count)
		var mean [512]float32
		for d := 0; d < 512; d++ {
			mean[d] = accumulators[k].sum[d] / count
		}

		// Quantize back to int8
		km.Centroids[k], km.Scales[k] = quantizeVectorOptimized(mean[:])
	}
}

// Predict returns the cluster assignment for a query vector (SIMD optimized)
func (km *KMeansOptimized) Predict(vec *simd.Vec512) int {
	minDist := float32(math.MaxFloat32)
	bestCluster := 0

	for k := 0; k < km.K; k++ {
		dist := float32(simd.L2Squared512(vec, &km.Centroids[k]))
		if dist < minDist {
			minDist = dist
			bestCluster = k
		}
	}

	return bestCluster
}

// PredictMultiple returns the nprobe nearest clusters for a query (optimized)
func (km *KMeansOptimized) PredictMultiple(vec *simd.Vec512, nprobe int) []int {
	if nprobe > km.K {
		nprobe = km.K
	}

	// Reuse buffer if possible
	distances := make([]distPair, km.K)
	for k := 0; k < km.K; k++ {
		distances[k] = distPair{
			cluster: k,
			dist:    float32(simd.L2Squared512(vec, &km.Centroids[k])),
		}
	}

	// Use quickselect for better performance than full sort
	quickSelect(distances, 0, km.K-1, nprobe-1)

	result := make([]int, nprobe)
	for i := 0; i < nprobe; i++ {
		result[i] = distances[i].cluster
	}

	return result
}

// quickSelect implements quickselect algorithm for partial sorting
func quickSelect(arr []distPair, left, right, k int) {
	if left >= right {
		return
	}

	pivotIndex := partition(arr, left, right)

	if pivotIndex == k {
		return
	} else if pivotIndex > k {
		quickSelect(arr, left, pivotIndex-1, k)
	} else {
		quickSelect(arr, pivotIndex+1, right, k)
	}
}

// partition function for quickselect
func partition(arr []distPair, left, right int) int {
	pivot := arr[right].dist
	i := left

	for j := left; j < right; j++ {
		if arr[j].dist <= pivot {
			arr[i], arr[j] = arr[j], arr[i]
			i++
		}
	}
	arr[i], arr[right] = arr[right], arr[i]
	return i
}

// quantizeVectorOptimized is an optimized version of vector quantization
func quantizeVectorOptimized(vec []float32) (simd.Vec512, float32) {
	// Find min and max using vectorized approach
	minVal, maxVal := vec[0], vec[0]

	// Unroll loop for better performance
	for i := 0; i < len(vec); i += 8 {
		end := i + 8
		if end > len(vec) {
			end = len(vec)
		}

		for j := i; j < end; j++ {
			if vec[j] < minVal {
				minVal = vec[j]
			}
			if vec[j] > maxVal {
				maxVal = vec[j]
			}
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
	invScale := 1.0 / scale

	// Vectorized quantization
	for i := 0; i < 512; i += 8 {
		// Process 8 elements at once
		result[i] = clampToInt8(vec[i] * invScale)
		result[i+1] = clampToInt8(vec[i+1] * invScale)
		result[i+2] = clampToInt8(vec[i+2] * invScale)
		result[i+3] = clampToInt8(vec[i+3] * invScale)
		result[i+4] = clampToInt8(vec[i+4] * invScale)
		result[i+5] = clampToInt8(vec[i+5] * invScale)
		result[i+6] = clampToInt8(vec[i+6] * invScale)
		result[i+7] = clampToInt8(vec[i+7] * invScale)
	}

	return result, scale
}

// clampToInt8 clamps a float32 value to int8 range
func clampToInt8(val float32) int8 {
	if val < -128 {
		return -128
	}
	if val > 127 {
		return 127
	}
	return int8(val)
}
