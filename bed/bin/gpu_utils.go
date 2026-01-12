package main

// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
// #include <cuda_runtime.h>
import "C"

import (
	"log"

	"github.com/lee101/gobed/metrics/perfdata"
)

// CheckGPUAvailable checks if CUDA GPU is available
func CheckGPUAvailable() bool {
	var count C.int
	result := C.cudaGetDeviceCount(&count)
	if result != 0 || count == 0 {
		return false
	}
	return true
}

// GetGPUMemory returns available GPU memory in MB
func GetGPUMemory() int64 {
	if !CheckGPUAvailable() {
		return 0
	}

	var free, total C.size_t
	result := C.cudaMemGetInfo(&free, &total)
	if result != 0 {
		return 0
	}
	return int64(free) / (1024 * 1024)
}

// ShouldUseGPU determines if GPU should be used based on data size
func ShouldUseGPU(numDocs int, debug bool) bool {
	if !CheckGPUAvailable() {
		if debug {
			log.Println("GPU not available, using CPU")
		}
		return false
	}

	if preferGPU, ok := perfdata.ShouldPreferGPU(numDocs); ok {
		if debug {
			if preferGPU {
				log.Printf("Using GPU based on empirical perfdata (size=%d)", numDocs)
			} else {
				log.Printf("Using CPU based on empirical perfdata (size=%d)", numDocs)
			}
		}
		return preferGPU
	}

	cpuLatency := perfdata.CPULatencyNs(numDocs)
	const cpuFavorCutoff = 40000.0 // 40µs -> 25K QPS
	if cpuLatency <= cpuFavorCutoff {
		if debug {
			log.Printf("Using CPU (latency %.0f ns) due to fast CPU measurements", cpuLatency)
		}
		return false
	}

	// Use GPU for larger datasets (>1000 docs)
	if numDocs > 1000 {
		if debug {
			log.Printf("Using GPU for %d documents", numDocs)
		}
		return true
	}

	// Check available memory
	memMB := GetGPUMemory()
	estimatedMemMB := int64(numDocs * 384 * 4 / (1024 * 1024)) // rough estimate

	if memMB > estimatedMemMB*2 { // Need at least 2x estimated memory
		if debug {
			log.Printf("Using GPU (memory: %d MB available, ~%d MB needed)", memMB, estimatedMemMB)
		}
		return true
	}

	if debug {
		log.Println("Using CPU - insufficient GPU memory or small dataset")
	}
	return false
}

// ShouldBuildIndex determines if an index should be built based on data size
func ShouldBuildIndex(numDocs int, debug bool) bool {
	// Build index for datasets larger than 10000 documents
	if numDocs > 10000 {
		if debug {
			log.Printf("Building index for %d documents", numDocs)
		}
		return true
	}

	if debug {
		log.Printf("Skipping index build for %d documents (linear search)", numDocs)
	}
	return false
}
