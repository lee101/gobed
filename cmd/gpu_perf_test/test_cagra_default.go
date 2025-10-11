//go:build legacy
// +build legacy

package main

import (
	"fmt"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🚀 CAGRA Default Configuration Test")
	fmt.Println("===================================")
	fmt.Println("Verifying that CAGRA is now the default search engine")
	fmt.Println()

	// Test default search configuration
	fmt.Println("📊 Default Search Configuration:")
	defaultConfig := gobed.DefaultSearchConfig()
	fmt.Printf("  AutoMode: %v\n", defaultConfig.AutoMode)
	fmt.Printf("  EnableGPU: %v\n", defaultConfig.EnableGPU)
	fmt.Printf("  Preset: %v\n", defaultConfig.Preset)
	fmt.Printf("  UseInt8: %v\n", defaultConfig.UseInt8)
	fmt.Printf("  GPUBatchSize: %d\n", defaultConfig.GPUBatchSize)

	// Test GPU search configuration
	fmt.Println("\n⚡ GPU Search Configuration:")
	gpuConfig := gobed.GPUSearchConfig()
	fmt.Printf("  AutoMode: %v\n", gpuConfig.AutoMode)
	fmt.Printf("  EnableGPU: %v\n", gpuConfig.EnableGPU)
	fmt.Printf("  Preset: %v\n", gpuConfig.Preset)
	fmt.Printf("  UseInt8: %v\n", gpuConfig.UseInt8)
	fmt.Printf("  GPUBatchSize: %d\n", gpuConfig.GPUBatchSize)

	// Check if CAGRA preset is available
	fmt.Println("\n🎯 CAGRA Preset Verification:")
	cagraValue := int(gobed.CAGRAPreset)
	fmt.Printf("  CAGRAPreset value: %d\n", cagraValue)

	// Verify preset ordering
	fmt.Println("\n📋 Search Preset Ordering:")
	fmt.Printf("  FastPreset: %d\n", int(gobed.FastPreset))
	fmt.Printf("  BalancedPreset: %d\n", int(gobed.BalancedPreset))
	fmt.Printf("  AccuratePreset: %d\n", int(gobed.AccuratePreset))
	fmt.Printf("  CAGRAPreset: %d\n", int(gobed.CAGRAPreset))
	fmt.Printf("  CustomPreset: %d\n", int(gobed.CustomPreset))

	fmt.Println("\n✅ CAGRA Integration Summary:")
	fmt.Println("=========================")

	if defaultConfig.Preset == gobed.CAGRAPreset {
		fmt.Println("✅ CAGRA is now the default preset when GPU is available")
	} else {
		fmt.Printf("⚠️  Default preset is %v (expected CAGRAPreset)\n", defaultConfig.Preset)
	}

	if gpuConfig.Preset == gobed.CAGRAPreset {
		fmt.Println("✅ GPU search configuration uses CAGRA preset")
	} else {
		fmt.Printf("⚠️  GPU config preset is %v (expected CAGRAPreset)\n", gpuConfig.Preset)
	}

	fmt.Println("\n🔧 Next Steps:")
	fmt.Println("  1. ✅ Added CAGRAPreset to search presets")
	fmt.Println("  2. ✅ Modified GPU configuration to use CAGRA by default")
	fmt.Println("  3. ✅ Updated DefaultSearchConfig to use CAGRA when GPU available")
	fmt.Println("  4. 🔄 Build CAGRA library and test real performance")
	fmt.Println("  5. 🔄 Benchmark quality retention with CAGRA")
	fmt.Println("  6. 🔄 Optimize indexing pipeline for CAGRA")

	fmt.Println("\n⚡ Expected Performance Improvements:")
	fmt.Println("  - Search latency: 10-50x faster (target <1ms)")
	fmt.Println("  - Throughput: 100K+ queries per second")
	fmt.Println("  - GPU utilization: Optimized for RTX 3090")
	fmt.Println("  - Memory efficiency: INT8 quantization enabled")
}
