package src

import (
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// SearchBackend represents available search backends
type SearchBackend int

const (
	BackendCPU SearchBackend = iota
	BackendSIMD
	BackendGPU
	BackendHybrid
)

func (b SearchBackend) String() string {
	switch b {
	case BackendCPU:
		return "CPU"
	case BackendSIMD:
		return "SIMD"
	case BackendGPU:
		return "GPU"
	case BackendHybrid:
		return "Hybrid"
	default:
		return "Unknown"
	}
}

// SystemCapabilities describes the system's compute resources
type SystemCapabilities struct {
	NumCPUs       int
	HasAVX2       bool
	HasAVX512     bool
	GPUAvailable  bool
	GPUName       string
	GPUMemoryMB   int
	TotalMemoryMB int
}

// SearchProfile captures workload characteristics
type SearchProfile struct {
	IndexSize      int
	AvgQueryLength int
	QueryRate      float64
	BatchSize      int
}

// OptimizationResult contains the recommended configuration
type OptimizationResult struct {
	Backend           SearchBackend
	NumWorkers        int
	BatchSize         int
	UsePrecomputedNorms bool
	UseInt8           bool
	Reason            string
}

// AutoOptimizer automatically selects the best search configuration
type AutoOptimizer struct {
	capabilities SystemCapabilities
	profile      SearchProfile

	mu           sync.RWMutex
	queryTimes   []time.Duration
	lastOptimize time.Time
}

var (
	globalOptimizer *AutoOptimizer
	optimizerOnce   sync.Once
)

// GetOptimizer returns the global auto optimizer instance
func GetOptimizer() *AutoOptimizer {
	optimizerOnce.Do(func() {
		globalOptimizer = &AutoOptimizer{
			capabilities: detectCapabilities(),
			queryTimes:   make([]time.Duration, 0, 100),
		}
	})
	return globalOptimizer
}

// detectCapabilities probes the system for compute resources
func detectCapabilities() SystemCapabilities {
	caps := SystemCapabilities{
		NumCPUs: runtime.NumCPU(),
	}

	caps.HasAVX2 = checkAVX2()
	caps.HasAVX512 = checkAVX512()
	caps.GPUAvailable, caps.GPUName, caps.GPUMemoryMB = detectGPU()

	return caps
}

// checkAVX2 checks for AVX2 support
func checkAVX2() bool {
	if runtime.GOARCH != "amd64" {
		return false
	}
	out, err := exec.Command("grep", "-c", "avx2", "/proc/cpuinfo").Output()
	if err != nil {
		return false
	}
	count, _ := strconv.Atoi(strings.TrimSpace(string(out)))
	return count > 0
}

// checkAVX512 checks for AVX-512 support
func checkAVX512() bool {
	if runtime.GOARCH != "amd64" {
		return false
	}
	out, err := exec.Command("grep", "-c", "avx512", "/proc/cpuinfo").Output()
	if err != nil {
		return false
	}
	count, _ := strconv.Atoi(strings.TrimSpace(string(out)))
	return count > 0
}

// detectGPU checks for CUDA GPU availability
func detectGPU() (available bool, name string, memMB int) {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return false, "", 0
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 {
		return false, "", 0
	}

	parts := strings.Split(lines[0], ", ")
	if len(parts) >= 2 {
		name = strings.TrimSpace(parts[0])
		memMB, _ = strconv.Atoi(strings.TrimSpace(parts[1]))
		return true, name, memMB
	}

	return false, "", 0
}

// UpdateProfile updates the workload profile
func (o *AutoOptimizer) UpdateProfile(indexSize, avgQueryLen int, queryRate float64) {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.profile = SearchProfile{
		IndexSize:      indexSize,
		AvgQueryLength: avgQueryLen,
		QueryRate:      queryRate,
	}
}

// RecordQueryTime records a query execution time for adaptive tuning
func (o *AutoOptimizer) RecordQueryTime(d time.Duration) {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.queryTimes = append(o.queryTimes, d)
	if len(o.queryTimes) > 100 {
		o.queryTimes = o.queryTimes[1:]
	}
}

// Optimize returns the recommended search configuration
func (o *AutoOptimizer) Optimize() OptimizationResult {
	o.mu.RLock()
	defer o.mu.RUnlock()

	result := OptimizationResult{
		NumWorkers:        o.capabilities.NumCPUs,
		UsePrecomputedNorms: true,
		UseInt8:           true,
	}

	indexSize := o.profile.IndexSize
	if indexSize == 0 {
		indexSize = 10000
	}

	// Decision logic based on index size and capabilities
	switch {
	case indexSize < 1000:
		// Small index: CPU is fine, SIMD helps
		result.Backend = BackendSIMD
		result.BatchSize = 256
		result.Reason = "small index, SIMD optimal"

	case indexSize < 50000:
		// Medium index: SIMD with parallel workers
		if o.capabilities.HasAVX2 {
			result.Backend = BackendSIMD
			result.NumWorkers = min(o.capabilities.NumCPUs, 8)
			result.BatchSize = 512
			result.Reason = "medium index, parallel SIMD"
		} else {
			result.Backend = BackendCPU
			result.NumWorkers = o.capabilities.NumCPUs
			result.BatchSize = 256
			result.Reason = "medium index, parallel CPU"
		}

	case indexSize < 500000:
		// Large index: GPU if available, else parallel SIMD
		if o.capabilities.GPUAvailable && o.capabilities.GPUMemoryMB >= 4000 {
			result.Backend = BackendGPU
			result.BatchSize = 1024
			result.Reason = "large index, GPU acceleration"
		} else if o.capabilities.HasAVX2 {
			result.Backend = BackendSIMD
			result.NumWorkers = o.capabilities.NumCPUs
			result.BatchSize = 1024
			result.Reason = "large index, full parallel SIMD"
		} else {
			result.Backend = BackendCPU
			result.NumWorkers = o.capabilities.NumCPUs
			result.BatchSize = 512
			result.Reason = "large index, parallel CPU"
		}

	default:
		// Very large index: GPU strongly preferred
		if o.capabilities.GPUAvailable {
			if o.capabilities.GPUMemoryMB >= 8000 {
				result.Backend = BackendGPU
				result.BatchSize = 2048
				result.Reason = "very large index, GPU with large memory"
			} else {
				result.Backend = BackendHybrid
				result.BatchSize = 1024
				result.Reason = "very large index, hybrid GPU+CPU"
			}
		} else {
			result.Backend = BackendSIMD
			result.NumWorkers = o.capabilities.NumCPUs
			result.BatchSize = 2048
			result.UseInt8 = true
			result.Reason = "very large index, optimized CPU"
		}
	}

	// Adjust for high query rate
	if o.profile.QueryRate > 100 {
		result.BatchSize = max(result.BatchSize/2, 128)
		result.Reason += ", low-latency tuning"
	}

	return result
}

// GetCapabilities returns the detected system capabilities
func (o *AutoOptimizer) GetCapabilities() SystemCapabilities {
	return o.capabilities
}

// ShouldUseGPU returns whether GPU should be used for given index size
func ShouldUseGPU(indexSize int) bool {
	opt := GetOptimizer()
	caps := opt.GetCapabilities()

	if !caps.GPUAvailable {
		return false
	}

	// GPU worthwhile for medium+ indexes with sufficient VRAM
	if indexSize >= 10000 && caps.GPUMemoryMB >= 2000 {
		return true
	}

	// For very large indexes, GPU is strongly preferred
	if indexSize >= 100000 {
		return true
	}

	return false
}

// GetOptimalWorkerCount returns optimal worker count for given workload
func GetOptimalWorkerCount(indexSize int) int {
	opt := GetOptimizer()
	caps := opt.GetCapabilities()

	if indexSize < 1000 {
		return min(caps.NumCPUs, 4)
	}
	if indexSize < 10000 {
		return min(caps.NumCPUs, 8)
	}
	return caps.NumCPUs
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
