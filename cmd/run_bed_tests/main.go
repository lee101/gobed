//go:build legacy

package main

import (
	"fmt"
	"log"
	"os"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run run_bed_tests.go <directory>")
		os.Exit(1)
	}

	dir := os.Args[1]

	// Create test suite
	suite := NewTestSuite()

	// Run comprehensive tests
	fmt.Printf("Testing directory: %s\n", dir)
	startTime := time.Now()

	suite.RunComprehensiveTests(dir)

	fmt.Printf("\n=== TOTAL TEST TIME: %.2fs ===\n", time.Since(startTime).Seconds())

	// Run profiling if needed
	if os.Getenv("PROFILE") == "1" {
		fmt.Println("\nRunning CPU profiling...")
		stopProfile := ProfileCPU("bed_cpu.prof")
		defer stopProfile()

		// Run intensive workload for profiling
		for i := 0; i < 100; i++ {
			query := suite.createEmbedding("test query for profiling")
			suite.performCPUSearch(query, 10)
		}
	}
}
