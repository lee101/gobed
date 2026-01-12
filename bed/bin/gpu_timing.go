package main

// GPU timing utilities for CUDA performance analysis

// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
// #include <cuda_runtime.h>
// #include <stdio.h>
//
// typedef struct {
//     cudaEvent_t start;
//     cudaEvent_t stop;
// } CudaTimer;
//
// CudaTimer* cuda_timer_create() {
//     CudaTimer* timer = (CudaTimer*)malloc(sizeof(CudaTimer));
//     cudaEventCreate(&timer->start);
//     cudaEventCreate(&timer->stop);
//     return timer;
// }
//
// void cuda_timer_start(CudaTimer* timer) {
//     cudaEventRecord(timer->start, 0);
// }
//
// float cuda_timer_stop(CudaTimer* timer) {
//     cudaEventRecord(timer->stop, 0);
//     cudaEventSynchronize(timer->stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, timer->start, timer->stop);
//     return milliseconds;
// }
//
// void cuda_timer_destroy(CudaTimer* timer) {
//     cudaEventDestroy(timer->start);
//     cudaEventDestroy(timer->stop);
//     free(timer);
// }
//
// void cuda_get_memory_info(size_t* free, size_t* total) {
//     cudaMemGetInfo(free, total);
// }
//
// int cuda_get_device_count() {
//     int count;
//     cudaGetDeviceCount(&count);
//     return count;
// }
//
// void cuda_get_device_properties(int device, char* name, int* major, int* minor, size_t* global_mem) {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, device);
//     strcpy(name, prop.name);
//     *major = prop.major;
//     *minor = prop.minor;
//     *global_mem = prop.totalGlobalMem;
// }
import "C"

import (
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

// GPUTimer for precise CUDA timing
type GPUTimer struct {
	timer unsafe.Pointer
}

// NewGPUTimer creates a CUDA event-based timer
func NewGPUTimer() *GPUTimer {
	return &GPUTimer{
		timer: C.cuda_timer_create(),
	}
}

// Start begins timing
func (t *GPUTimer) Start() {
	C.cuda_timer_start((*C.CudaTimer)(t.timer))
}

// Stop ends timing and returns milliseconds
func (t *GPUTimer) Stop() float32 {
	return float32(C.cuda_timer_stop((*C.CudaTimer)(t.timer)))
}

// Destroy cleans up CUDA events
func (t *GPUTimer) Destroy() {
	C.cuda_timer_destroy((*C.CudaTimer)(t.timer))
}

// GPUInfo holds device information
type GPUInfo struct {
	Name       string
	Major      int
	Minor      int
	GlobalMem  uint64
	FreeMem    uint64
	TotalMem   uint64
}

// GetGPUInfo returns current GPU information
func GetGPUInfo() *GPUInfo {
	if C.cuda_get_device_count() == 0 {
		return nil
	}

	name := make([]byte, 256)
	var major, minor C.int
	var globalMem C.size_t

	C.cuda_get_device_properties(0, (*C.char)(unsafe.Pointer(&name[0])), &major, &minor, &globalMem)

	var freeMem, totalMem C.size_t
	C.cuda_get_memory_info(&freeMem, &totalMem)

	// Find null terminator in name
	nameStr := ""
	for i, b := range name {
		if b == 0 {
			nameStr = string(name[:i])
			break
		}
	}

	return &GPUInfo{
		Name:      nameStr,
		Major:     int(major),
		Minor:     int(minor),
		GlobalMem: uint64(globalMem),
		FreeMem:   uint64(freeMem),
		TotalMem:  uint64(totalMem),
	}
}

// PerfStats holds comprehensive performance metrics
type PerfStats struct {
	TokenizeTime   time.Duration
	EmbedTime      time.Duration
	GPUTransferTime float32 // milliseconds
	KernelTime     float32 // milliseconds
	ResultTime     float32 // milliseconds
	TotalTime      time.Duration
	AllocBytes     uint64
	AllocCount     uint64
}

// BenchmarkFullPipeline runs comprehensive performance analysis
func BenchmarkFullPipeline(query string, numDocs int) *PerfStats {
	stats := &PerfStats{}

	// Memory stats before
	var m1, m2 runtime.MemStats
	runtime.ReadMemStats(&m1)

	totalStart := time.Now()

	// 1. Tokenization benchmark
	tokenStart := time.Now()
	tokens := benchModel.tokenizer.Tokenize(query)
	stats.TokenizeTime = time.Since(tokenStart)

	// 2. Embedding benchmark
	embedStart := time.Now()
	queryEmb, err := benchModel.EmbedInt8(query)
	if err != nil {
		return stats
	}
	stats.EmbedTime = time.Since(embedStart)

	// 3. GPU setup and transfer benchmark
	gpuTimer := NewGPUTimer()
	defer gpuTimer.Destroy()

	handle := C.cuda_fast_search_create(C.int(numDocs+100), C.int(512))
	defer C.cuda_fast_search_destroy(handle)

	// Create test embeddings
	embeddings := make([]int8, numDocs*512)
	for i := range embeddings {
		embeddings[i] = int8(i % 256 - 128)
	}

	// Time GPU transfer
	gpuTimer.Start()
	C.cuda_fast_search_add_vectors(
		handle,
		(*C.schar)(unsafe.Pointer(&embeddings[0])),
		C.int(numDocs),
	)
	stats.GPUTransferTime = gpuTimer.Stop()

	// 4. Kernel execution benchmark
	indices := make([]int32, 10)
	scores := make([]float32, 10)

	gpuTimer.Start()
	C.cuda_fast_search_query(
		handle,
		(*C.schar)(unsafe.Pointer(&queryEmb[0])),
		C.int(10),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)
	stats.KernelTime = gpuTimer.Stop()

	stats.TotalTime = time.Since(totalStart)

	// Memory stats after
	runtime.ReadMemStats(&m2)
	stats.AllocBytes = m2.TotalAlloc - m1.TotalAlloc
	stats.AllocCount = m2.Mallocs - m1.Mallocs

	fmt.Printf("🔍 Query: '%s' | Docs: %d\n", query, numDocs)
	fmt.Printf("📊 Tokenize: %v | Embed: %v\n", stats.TokenizeTime, stats.EmbedTime)
	fmt.Printf("🚀 GPU Transfer: %.3fms | Kernel: %.3fms\n", stats.GPUTransferTime, stats.KernelTime)
	fmt.Printf("💾 Allocs: %d bytes, %d objects\n", stats.AllocBytes, stats.AllocCount)
	fmt.Printf("⏱️  Total: %v (%.2f QPS)\n\n", stats.TotalTime, 1000.0/float64(stats.TotalTime.Milliseconds()))

	_ = tokens // Use tokens variable
	return stats
}

// RunPerformanceAnalysis runs comprehensive benchmarks
func RunPerformanceAnalysis() {
	fmt.Println("🚀 GPU Performance Analysis")
	fmt.Println("=" * 50)

	// GPU Info
	if info := GetGPUInfo(); info != nil {
		fmt.Printf("🔧 GPU: %s (SM %d.%d)\n", info.Name, info.Major, info.Minor)
		fmt.Printf("💾 Memory: %.1f GB total, %.1f GB free\n",
			float64(info.TotalMem)/1e9, float64(info.FreeMem)/1e9)
		fmt.Println()
	}

	// Test different scales
	testCases := []struct {
		query   string
		numDocs int
	}{
		{"anime", 1000},
		{"Studio Ghibli", 10000},
		{"neural networks", 50000},
		{"CUDA optimization", 100000},
	}

	for _, tc := range testCases {
		BenchmarkFullPipeline(tc.query, tc.numDocs)
	}
}