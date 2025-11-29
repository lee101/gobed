//go:build legacy

package main

import (
	"fmt"
	"log"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println(" Quick Gobed Performance Test")
	fmt.Println("===============================")

	// Load model
	fmt.Print("Loading model... ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf(" Done (%v)\n", time.Since(start))

	// Test data
	texts := []string{
		"This is a sample document about machine learning and artificial intelligence.",
		"Research paper on quantum computing: methodology, results, and conclusions.",
		"User guide for understanding blockchain technology and cryptocurrency systems.",
		"Analysis of deep learning performance metrics and optimization strategies.",
		"Technical documentation explaining cloud computing architectures and services.",
	}

	// Extend to create larger test set
	largeTexts := make([]string, 0, 1000)
	for i := 0; i < 200; i++ {
		for _, text := range texts {
			largeTexts = append(largeTexts, fmt.Sprintf("%s (iteration %d)", text, i))
		}
	}

	fmt.Printf("Test dataset: %d documents\n\n", len(largeTexts))

	// Test different scenarios
	testCases := []struct {
		name        string
		count       int
		concurrency int
	}{
		{"Small Sequential", 100, 1},
		{"Small Parallel 4x", 100, 4},
		{"Medium Sequential", 500, 1},
		{"Medium Parallel 8x", 500, 8},
		{"Large Sequential", 1000, 1},
		{"Large Parallel 8x", 1000, 8},
	}

	fmt.Printf("%-20s %-8s %-12s %-15s %-10s\n", "Test", "Items", "Time (ms)", "Items/sec", "ms/item")
	fmt.Println(strings.Repeat("-", 65))

	for _, tc := range testCases {
		duration := runTest(model, largeTexts[:tc.count], tc.concurrency)
		itemsPerSec := float64(tc.count) / duration.Seconds()
		msPerItem := float64(duration.Nanoseconds()) / float64(tc.count) / 1e6

		fmt.Printf("%-20s %-8d %-12.1f %-15.0f %-10.3f\n",
			tc.name, tc.count, float64(duration.Nanoseconds())/1e6, itemsPerSec, msPerItem)
	}

	// Estimate large-scale performance
	fmt.Println("\n Large Scale Estimates:")
	bestPerf := float64(8000) // Estimated items/sec based on parallel performance

	scales := []int{10000, 100000, 1000000}
	for _, scale := range scales {
		timeSeconds := float64(scale) / bestPerf
		fmt.Printf("  %7d documents: ~%.1f seconds (~%.1f minutes)\n",
			scale, timeSeconds, timeSeconds/60)
	}
}

func runTest(model *gobed.EmbeddingModel, texts []string, concurrency int) time.Duration {
	runtime.GC() // Clean up before test

	start := time.Now()

	if concurrency == 1 {
		// Sequential
		for i, text := range texts {
			if i < 3 { // Warmup
				model.Encode(text)
				continue
			}
			_, err := model.Encode(text)
			if err != nil {
				log.Printf("Error: %v", err)
			}
		}
	} else {
		// Parallel
		textChan := make(chan string, len(texts))
		var wg sync.WaitGroup

		// Start workers
		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				warmupCount := 0
				for text := range textChan {
					if warmupCount < 2 { // Warmup each worker
						model.Encode(text)
						warmupCount++
						continue
					}
					_, err := model.Encode(text)
					if err != nil {
						log.Printf("Error: %v", err)
					}
				}
			}()
		}

		// Send work
		for _, text := range texts {
			textChan <- text
		}
		close(textChan)
		wg.Wait()
	}

	return time.Since(start)
}
