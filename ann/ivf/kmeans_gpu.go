//go:build legacy && gpu

package ivf

import (
	"log"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// KMeans implementation with GPU acceleration support
// Falls back to CPU if GPU is not available

// FitWithGPU attempts to use GPU K-means if available
func (km *KMeans) FitWithGPU(vectors []simd.Vec512, scales []float32) {
	// Check if GPU is available
	if !gobed.IsCUDAAvailable() {
		log.Println("GPU not available, falling back to CPU K-means")
		km.Fit(vectors, scales)
		return
	}

	n := len(vectors)

	// Use GPU for large datasets
	if n >= 5000 {
		log.Printf("Using GPU K-means for %d vectors", n)

		// Create GPU K-means instance
		gpuKM := gobed.NewGPUKMeans(km.K, km.MaxIters)

		// Run GPU K-means
		err := gpuKM.Fit(vectors, scales)
		if err != nil {
			log.Printf("GPU K-means failed: %v, falling back to CPU", err)
			km.Fit(vectors, scales)
			return
		}

		// Copy results
		km.Centroids, km.Scales = gpuKM.GetCentroids()

		// Count final assignments (optional, for statistics)
		for i := range km.Counts {
			km.Counts[i] = 0
		}
		// Could compute counts if needed

		return
	}

	// For small datasets, CPU is fine
	km.Fit(vectors, scales)
}

// PredictWithGPU uses GPU for batch prediction if available
func (km *KMeans) PredictBatchGPU(vectors []simd.Vec512) ([]int, error) {
	if !gobed.IsCUDAAvailable() {
		// Fall back to CPU
		results := make([]int, len(vectors))
		for i, vec := range vectors {
			results[i] = km.Predict(&vec)
		}
		return results, nil
	}

	gpuKM := gobed.NewGPUKMeans(km.K, km.MaxIters)
	// Set trained centroids
	gpuKM.centroids = km.Centroids
	gpuKM.scales = km.Scales
	gpuKM.trained = true

	return gpuKM.PredictBatch(vectors)
}

// PredictMultipleWithGPU uses GPU for multi-probe prediction
func (km *KMeans) PredictMultipleWithGPU(vec *simd.Vec512, nprobe int) []int {
	// For single query with small nprobe, CPU is usually faster (GPU overhead)
	if nprobe <= 8 || km.K <= 32 {
		return km.PredictMultiple(vec, nprobe)
	}

	if !gobed.IsCUDAAvailable() {
		return km.PredictMultiple(vec, nprobe)
	}

	// Use GPU for large K with many probes
	gpuKM := gobed.NewGPUKMeans(km.K, km.MaxIters)
	gpuKM.centroids = km.Centroids
	gpuKM.scales = km.Scales
	gpuKM.trained = true

	return gpuKM.PredictMultiple(vec, nprobe)
}
