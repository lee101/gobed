package perfdata

import (
	"math"
	"sort"
)

// Sample captures empirical latency measurements for a given collection size.
type Sample struct {
	Size           int
	CPULatencyNano float64
	GPULatencyNano *float64
	Source         string
}

func floatPtr(v float64) *float64 {
	value := v
	return &value
}

var cpuGPUSamples = []Sample{
	{Size: 128, CPULatencyNano: 8411, Source: "cpu_flat_search_baseline.json"},
	{Size: 512, CPULatencyNano: 21347, Source: "cpu_flat_search_baseline.json"},
	{Size: 1024, CPULatencyNano: 38006, GPULatencyNano: floatPtr(413775), Source: "cpu_flat_search_baseline.json|cagra_gpu_bench"},
	{Size: 2048, CPULatencyNano: 69779, Source: "cpu_flat_search_baseline.json"},
	{Size: 5000, CPULatencyNano: 164465, GPULatencyNano: floatPtr(452047), Source: "cpu_flat_search_baseline.json|cagra_gpu_bench"},
	{Size: 8192, CPULatencyNano: 266564, Source: "cpu_flat_search_baseline.json"},
	{Size: 10000, CPULatencyNano: 321793, GPULatencyNano: floatPtr(470930), Source: "cpu_flat_search_baseline.json|cagra_gpu_bench"},
	{Size: 20000, CPULatencyNano: 677646, GPULatencyNano: floatPtr(506659), Source: "cpu_flat_search_baseline.json|cagra_gpu_bench"},
	{Size: 32768, CPULatencyNano: 1310297, Source: "cpu_flat_search_baseline.json"},
	{Size: 50000, CPULatencyNano: 2460441, GPULatencyNano: floatPtr(531594), Source: "cpu_flat_search_baseline.json|cagra_gpu_bench"},
	{Size: 100000, CPULatencyNano: 4827139, GPULatencyNano: floatPtr(519908), Source: "cpu_flat_search_baseline.json|cagra_gpu_bench"},
}

// cpuLatencyEstimate returns an interpolated CPU latency for the requested size.
func cpuLatencyEstimate(size int) float64 {
	if size <= cpuGPUSamples[0].Size {
		return cpuGPUSamples[0].CPULatencyNano
	}
	if size >= cpuGPUSamples[len(cpuGPUSamples)-1].Size {
		return cpuGPUSamples[len(cpuGPUSamples)-1].CPULatencyNano
	}

	idx := sort.Search(len(cpuGPUSamples), func(i int) bool {
		return cpuGPUSamples[i].Size >= size
	})
	if idx == 0 {
		return cpuGPUSamples[0].CPULatencyNano
	}
	lo := cpuGPUSamples[idx-1]
	hi := cpuGPUSamples[idx]

	if hi.Size == lo.Size {
		return hi.CPULatencyNano
	}

	// Log-linear interpolation better matches search scaling behaviour.
	lnSize := math.Log(float64(size))
	lnLo := math.Log(float64(lo.Size))
	lnHi := math.Log(float64(hi.Size))
	ratio := (lnSize - lnLo) / (lnHi - lnLo)
	return lo.CPULatencyNano + ratio*(hi.CPULatencyNano-lo.CPULatencyNano)
}

// gpuLatencyEstimate returns the interpolated GPU latency if datapoints exist.
func gpuLatencyEstimate(size int) (float64, bool) {
	hasGPU := false
	filtered := make([]Sample, 0, len(cpuGPUSamples))
	for _, s := range cpuGPUSamples {
		if s.GPULatencyNano != nil {
			filtered = append(filtered, s)
			hasGPU = true
		}
	}
	if !hasGPU || len(filtered) == 0 {
		return 0, false
	}
	if size <= filtered[0].Size {
		return *filtered[0].GPULatencyNano, true
	}
	if size >= filtered[len(filtered)-1].Size {
		return *filtered[len(filtered)-1].GPULatencyNano, true
	}
	idx := sort.Search(len(filtered), func(i int) bool {
		return filtered[i].Size >= size
	})
	if idx == 0 {
		return *filtered[0].GPULatencyNano, true
	}
	lo := filtered[idx-1]
	hi := filtered[idx]
	if hi.Size == lo.Size {
		return *hi.GPULatencyNano, true
	}
	lnSize := math.Log(float64(size))
	lnLo := math.Log(float64(lo.Size))
	lnHi := math.Log(float64(hi.Size))
	ratio := (lnSize - lnLo) / (lnHi - lnLo)
	return *lo.GPULatencyNano + ratio*(*hi.GPULatencyNano-*lo.GPULatencyNano), true
}

// ShouldPreferGPU returns whether GPU latency is expected to beat CPU latency.
// ok indicates whether the decision is backed by empirical GPU data.
func ShouldPreferGPU(size int) (preferGPU bool, ok bool) {
	cpuLatency := cpuLatencyEstimate(size)
	gpuLatency, hasGPU := gpuLatencyEstimate(size)
	if !hasGPU {
		return false, false
	}
	return gpuLatency < cpuLatency, true
}

// CPULatencyNs exposes the CPU latency curve for reporting.
func CPULatencyNs(size int) float64 {
	return cpuLatencyEstimate(size)
}
