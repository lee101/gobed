// test_gpu_benchmark.go - RTX 3090 performance test
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

func main() {
	fmt.Println("\n⚡ RTX 3090 PERFORMANCE BENCHMARKS")
	fmt.Println(strings.Repeat("=", 60))

	// Test 1: GPU vs CPU comparison
	fmt.Println("\n1. GPU vs CPU Performance Comparison")
	fmt.Println(strings.Repeat("-", 40))

	queries := []string{"anime", "Studio Ghibli", "Dragon Ball", "BERT", "GPU"}

	for _, query := range queries {
		// Test CPU version
		startCPU := time.Now()
		cpuCmd := exec.Command("./bed", "-cpu", "-dir", "testdata", "-k", "5", query)
		cpuOutput, cpuErr := cpuCmd.Output()
		cpuTime := time.Since(startCPU)

		// Test GPU version
		startGPU := time.Now()
		gpuCmd := exec.Command("./bed", "-gpu", "-dir", "testdata", "-k", "5", query)
		gpuCmd.Env = append(os.Environ(), "LD_LIBRARY_PATH=.")
		gpuOutput, gpuErr := gpuCmd.Output()
		gpuTime := time.Since(startGPU)

		fmt.Printf("Query: %s\n", query)

		if cpuErr == nil {
			fmt.Printf("  CPU:  %.2fms (%d chars output)\n",
				float64(cpuTime.Milliseconds()), len(cpuOutput))
		} else {
			fmt.Printf("  CPU:  ERROR\n")
		}

		if gpuErr == nil {
			fmt.Printf("  GPU:  %.2fms (%d chars output)\n",
				float64(gpuTime.Milliseconds()), len(gpuOutput))
			if cpuErr == nil && cpuTime > 0 && gpuTime > 0 {
				speedup := float64(cpuTime) / float64(gpuTime)
				fmt.Printf("  Speedup: %.2fx\n", speedup)
			}
		} else {
			fmt.Printf("  GPU:  ERROR - %s\n", gpuErr.Error())
		}
		fmt.Println()
	}

	// Test 2: GPU utilization test
	fmt.Println("\n2. GPU Utilization Test")
	fmt.Println(strings.Repeat("-", 40))

	// Check GPU usage before
	beforeCmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits")
	beforeOutput, _ := beforeCmd.Output()
	fmt.Printf("Before: %s", beforeOutput)

	// Run intensive GPU workload
	batchCmd := exec.Command("./bed", "-gpu", "-dir", "testdata", "-k", "10", "test")
	batchCmd.Env = append(os.Environ(), "LD_LIBRARY_PATH=.")
	batchCmd.Run()

	// Check GPU usage after
	afterCmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits")
	afterOutput, _ := afterCmd.Output()
	fmt.Printf("After:  %s", afterOutput)

	// Test 3: Memory usage analysis
	fmt.Println("\n3. Memory Usage Analysis")
	fmt.Println(strings.Repeat("-", 40))

	memCmd := exec.Command("nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader")
	memOutput, err := memCmd.Output()
	if err == nil {
		fmt.Printf("GPU Memory: %s", memOutput)
	}

	fmt.Println("\n✅ Benchmark complete")
}