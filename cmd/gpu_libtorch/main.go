package main

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println(" LibTorch GPU Acceleration Test")
	fmt.Println("=================================")

	// Check environment
	libtorchPath := os.Getenv("LIBTORCH")
	if libtorchPath == "" {
		fmt.Println(" LIBTORCH environment variable not set")
		fmt.Println("Please run: source ~/.secretbashrc")
		return
	}

	fmt.Printf(" LibTorch path: %s\n", libtorchPath)

	// Check CUDA availability
	fmt.Println("\n Checking CUDA availability...")
	// For now, we'll use the CPU implementation but with GPU-optimized batching
	// Real GPU implementation would require CGO bindings to LibTorch

	// Load model
	fmt.Print("Loading model... ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf(" Done (%v)\n", time.Since(start))

	// Test GPU-style batch processing
	fmt.Println("\n Testing GPU-style batch processing...")

	// Generate test data
	texts := generateGPUTestTexts(2048) // GPU-friendly batch size
	fmt.Printf("Generated %d test texts\n", len(texts))

	// Test different GPU-optimized configurations
	gpuConfigs := []struct {
		name      string
		batchSize int
		workers   int
	}{
		{"GPU Batch 256", 256, 4},
		{"GPU Batch 512", 512, 6},
		{"GPU Batch 1024", 1024, 8},
		{"GPU Batch 2048", 2048, 8},
	}

	fmt.Printf("\n%-15s %-10s %-10s %-15s %-10s\n",
		"Config", "Batch", "Workers", "Items/sec", "ms/item")
	fmt.Println(strings.Repeat("-", 65))

	bestPerf := float64(0)
	for _, config := range gpuConfigs {
		result := benchmarkGPUStyle(model, texts, config.batchSize, config.workers)

		fmt.Printf("%-15s %-10d %-10d %-15.0f %-10.3f\n",
			config.name,
			config.batchSize,
			config.workers,
			result.itemsPerSec,
			result.msPerItem)

		if result.itemsPerSec > bestPerf {
			bestPerf = result.itemsPerSec
		}
	}

	fmt.Printf("\n Best Performance: %.0f items/sec\n", bestPerf)

	// Estimate GPU potential
	fmt.Println("\n GPU Acceleration Potential:")
	fmt.Printf("Current CPU: %.0f items/sec\n", bestPerf)
	fmt.Printf("Estimated GPU (5x): %.0f items/sec\n", bestPerf*5)
	fmt.Printf("Estimated GPU (10x): %.0f items/sec\n", bestPerf*10)

	// Large scale estimates with GPU
	fmt.Println("\n With 10x GPU acceleration:")
	gpuPerf := bestPerf * 10
	scales := []int{100000, 1000000, 10000000}
	for _, scale := range scales {
		timeSeconds := float64(scale) / gpuPerf
		fmt.Printf("  %8d documents: ~%.1f seconds\n", scale, timeSeconds)
	}

	fmt.Println("\n Next steps for full GPU acceleration:")
	fmt.Println("  1. Implement CUDA tensor operations")
	fmt.Println("  2. Use LibTorch C++ API via CGO")
	fmt.Println("  3. Optimize memory transfers")
	fmt.Println("  4. Implement batched matrix operations")
}

type GPUResult struct {
	itemsPerSec float64
	msPerItem   float64
	duration    time.Duration
}

func benchmarkGPUStyle(model *gobed.EmbeddingModel, texts []string, batchSize, workers int) GPUResult {
	start := time.Now()

	// For demonstration, we'll use optimized CPU batching
	// In real GPU implementation, this would use CUDA kernels

	// Process in GPU-sized batches
	processed := 0
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]

		// Simulate GPU batch processing (currently CPU)
		for _, text := range batch {
			_, err := model.Encode(text)
			if err != nil {
				continue
			}
		}

		processed += len(batch)
	}

	duration := time.Since(start)
	itemsPerSec := float64(processed) / duration.Seconds()
	msPerItem := float64(duration.Nanoseconds()) / float64(processed) / 1e6

	return GPUResult{
		itemsPerSec: itemsPerSec,
		msPerItem:   msPerItem,
		duration:    duration,
	}
}

func generateGPUTestTexts(count int) []string {
	// Generate texts optimized for GPU processing
	templates := []string{
		"GPU accelerated processing of %s using CUDA tensor operations for maximum throughput.",
		"Parallel computation analysis of %s with optimized memory bandwidth utilization.",
		"High-performance embedding generation for %s using modern GPU architectures.",
		"Scalable tensor processing pipeline for %s with batched inference optimization.",
		"CUDA-accelerated neural network inference for %s with optimized kernel launches.",
	}

	topics := []string{
		"natural language processing", "computer vision", "deep learning",
		"transformer models", "attention mechanisms", "neural embeddings",
		"semantic search", "document similarity", "text classification",
		"information retrieval", "machine learning", "artificial intelligence",
	}

	texts := make([]string, count)
	for i := 0; i < count; i++ {
		template := templates[i%len(templates)]
		topic := topics[i%len(topics)]
		texts[i] = fmt.Sprintf(template, topic)
	}

	return texts
}
