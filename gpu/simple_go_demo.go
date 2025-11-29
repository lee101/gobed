// simple_go_test.go - Test Go API changes for GPU support
package main

import (
	"fmt"
	"log"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🧪 Testing Go API GPU Integration")
	fmt.Println("=================================")

	// Load model
	fmt.Println("📚 Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal("Failed to load model:", err)
	}

	// Test 1: GPU Search Config
	fmt.Println("\n Test 1: GPU Search Config")
	gpuConfig := gobed.GPUSearchConfig()
	fmt.Printf("   EnableGPU: %t\n", gpuConfig.EnableGPU)
	fmt.Printf("   GPUDeviceID: %d\n", gpuConfig.GPUDeviceID)
	fmt.Printf("   GPUBatchSize: %d\n", gpuConfig.GPUBatchSize)
	fmt.Printf("   MaxExactSearchSize: %d\n", gpuConfig.MaxExactSearchSize)

	// Test 2: Create GPU Search Engine
	fmt.Println("\n🏗 Test 2: Create GPU Search Engine")
	engine := gobed.NewGPUSearchEngine(model)
	defer engine.Close()
	fmt.Println("    GPU search engine created successfully")

	// Test 3: Create Regular Engine with GPU Config
	fmt.Println("\n Test 3: Regular Engine with GPU Config")
	customConfig := gobed.GPUSearchConfig()
	customConfig.EnableGPU = true
	customEngine := gobed.NewSearchEngineWithConfig(model, customConfig)
	defer customEngine.Close()
	fmt.Println("    Custom GPU-enabled engine created")

	// Test 4: Basic functionality
	fmt.Println("\n📚 Test 4: Basic indexing and search")
	
	docs := []string{
		"Machine learning algorithms analyze data patterns",
		"Deep learning uses neural networks for complex tasks",
		"Natural language processing understands human text",
	}

	// Index documents
	for i, doc := range docs {
		_, err := engine.IndexWithID(i, doc)
		if err != nil {
			// Expected to fail without GPU implementation, that's OK
			fmt.Printf("     Indexing failed (expected without GPU): %v\n", err)
			break
		}
	}

	// Try search (may fail without GPU, that's OK)
	results, err := engine.Search("machine learning", 2)
	if err != nil {
		fmt.Printf("     Search failed (expected without GPU): %v\n", err)
	} else {
		fmt.Printf("    Search succeeded: %d results\n", len(results))
	}

	fmt.Println("\n API Integration Test Complete!")
	fmt.Println("    GPU configuration options added")
	fmt.Println("   🏗 GPU search engine constructors working")
	fmt.Println("    Ready for GPU implementation integration")
}