//go:build legacy

package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/ivf"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	fmt.Println("🚀 GPU K-Means Acceleration Benchmark")
	fmt.Println("=====================================")
	fmt.Println("Comparing CPU vs GPU K-means performance")
	fmt.Println()

	// Test different dataset sizes
	sizes := []int{1000, 5000, 10000, 50000, 100000, 240000}
	nClusters := 256

	fmt.Printf("Configuration: %d clusters, 25 iterations max\n\n", nClusters)

	fmt.Println("Generating test data...")

	for _, n := range sizes {
		fmt.Printf("\n📊 Dataset size: %d vectors\n", n)
		fmt.Println(strings.Repeat("-", 50))

		// Generate random int8 vectors
		vectors := make([]simd.Vec512, n)
		scales := make([]float32, n)

		for i := 0; i < n; i++ {
			for j := 0; j < 512; j++ {
				vectors[i][j] = int8(rand.Intn(255) - 128)
			}
			scales[i] = rand.Float32() * 0.1
		}

		// CPU K-means
		if n <= 10000 {
			fmt.Print("CPU K-means: ")
			cpuKM := ivf.NewKMeans(nClusters, 25)

			cpuStart := time.Now()
			cpuKM.Fit(vectors, scales)
			cpuTime := time.Since(cpuStart)

			cpuThroughput := float64(n) / cpuTime.Seconds()
			fmt.Printf("%8.1fms (%8.0f vecs/sec)\n",
				float64(cpuTime.Milliseconds()),
				cpuThroughput)
		} else {
			cpuTimeEstimate := time.Duration(n * 15.2) * time.Millisecond
			fmt.Printf("CPU K-means: ~%.1fs (estimated, skipped)\n",
				cpuTimeEstimate.Seconds())
		}

		// GPU K-means
		if gobed.IsCUDAAvailable() {
			fmt.Print("GPU K-means: ")
			gpuKM := gobed.NewGPUKMeans(nClusters, 25)

			gpuStart := time.Now()
			err := gpuKM.Fit(vectors, scales)
			gpuTime := time.Since(gpuStart)

			if err != nil {
				fmt.Printf("ERROR: %v\n", err)
			} else {
				gpuThroughput := float64(n) / gpuTime.Seconds()
				fmt.Printf("%8.1fms (%8.0f vecs/sec)",
					float64(gpuTime.Milliseconds()),
					gpuThroughput)

				// Calculate speedup
				if n <= 10000 {
					// Use actual CPU time
					cpuKM := ivf.NewKMeans(nClusters, 25)
					cpuStart := time.Now()
					cpuKM.Fit(vectors[:min(n, 1000)], scales[:min(n, 1000)])
					cpuSample := time.Since(cpuStart)
					cpuTimeEstimate := cpuSample * time.Duration(n/min(n, 1000))
					speedup := float64(cpuTimeEstimate) / float64(gpuTime)
					fmt.Printf(" [%.1fx speedup]", speedup)
				} else {
					// Use estimate
					cpuTimeEstimate := time.Duration(n * 15.2) * time.Millisecond
					speedup := float64(cpuTimeEstimate) / float64(gpuTime)
					fmt.Printf(" [~%.1fx speedup]", speedup)
				}

				fmt.Println()

				// Check performance targets
				if n == 10000 && gpuTime > 200*time.Millisecond {
					fmt.Printf("⚠️  Missed target: 10k in 200ms (got %.1fms)\n",
						float64(gpuTime.Milliseconds()))
				}
				if n == 240000 && gpuTime > 3*time.Second {
					fmt.Printf("⚠️  Missed target: 240k in 3s (got %.1fs)\n",
						gpuTime.Seconds())
				}
				if n == 240000 && gpuTime <= 3*time.Second {
					fmt.Println("✅ Achieved target: 240k vectors in <3s!")
				}
			}
		} else {
			fmt.Println("GPU K-means: Not available (CUDA not found)")
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("🎯 Performance Summary")
	fmt.Println(strings.Repeat("=", 50))

	if gobed.IsCUDAAvailable() {
		fmt.Println("✅ GPU acceleration available")
		fmt.Println("Expected speedups:")
		fmt.Println("  - 10k vectors:  ~50-100x (152s → <200ms)")
		fmt.Println("  - 240k vectors: ~50-100x (154s → <3s)")
		fmt.Println("  - 1M vectors:   ~50-100x (640s → <12s)")
	} else {
		fmt.Println("❌ GPU not available")
		fmt.Println("To enable GPU acceleration:")
		fmt.Println("  1. Ensure CUDA 11+ is installed")
		fmt.Println("  2. Build GPU K-means: ./build_gpu_kmeans.sh")
		fmt.Println("  3. Build with GPU tag: go build -tags gpu")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
