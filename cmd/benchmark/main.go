//go:build legacy

package main

import (
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	// Load model once (not timed)
	fmt.Println("Loading model (not timed)...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("Model loaded successfully!\n")

	// Test sentences for benchmarking
	testSentences := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning models can process natural language",
		"Go is a statically typed programming language",
		"Embeddings represent text as numerical vectors",
		"This is a performance benchmark test",
		"Natural language processing is fascinating",
		"Vector databases store embeddings efficiently",
		"Semantic search uses similarity between embeddings",
		"The model converts text to dense vectors",
		"Benchmarking helps measure system performance",
	}

	// Warmup runs (not counted)
	fmt.Println("Warming up...")
	for i := 0; i < 10; i++ {
		_, _ = model.Encode(testSentences[0])
	}

	// Performance test
	fmt.Println("\n PERFORMANCE TEST - Embeddings Per Second")
	fmt.Println(strings.Repeat("=", 50))

	numIterations := 1000
	totalEmbeddings := numIterations * len(testSentences)

	fmt.Printf("Processing %d embeddings (%d iterations × %d sentences)\n\n",
		totalEmbeddings, numIterations, len(testSentences))

	// Start timing ONLY the inference
	start := time.Now()

	successfulEmbeddings := 0
	for i := 0; i < numIterations; i++ {
		for _, sentence := range testSentences {
			_, err := model.Encode(sentence)
			if err == nil {
				successfulEmbeddings++
			}
		}
	}

	elapsed := time.Since(start)

	// Calculate metrics
	embeddingsPerSecond := float64(successfulEmbeddings) / elapsed.Seconds()
	timePerEmbedding := elapsed / time.Duration(successfulEmbeddings)

	fmt.Printf(" Results:\n")
	fmt.Printf("   Total time:            %v\n", elapsed)
	fmt.Printf("   Successful embeddings: %d\n", successfulEmbeddings)
	fmt.Printf("   Embeddings/second:     %.0f\n", embeddingsPerSecond)
	fmt.Printf("   Time per embedding:    %v\n", timePerEmbedding)
	fmt.Printf("   Latency (microseconds): %.2f μs\n", float64(timePerEmbedding.Nanoseconds())/1000)

	// Test with single embeddings for more accurate per-embedding timing
	fmt.Println("\n Individual embedding timings:")
	for i, sentence := range testSentences[:5] {
		start := time.Now()
		_, err := model.Encode(sentence)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("   %d. Error: %v\n", i+1, err)
		} else {
			displayText := sentence
			if len(displayText) > 40 {
				displayText = displayText[:37] + "..."
			}
			fmt.Printf("   %d. %8.2f μs - \"%s\"\n", i+1,
				float64(elapsed.Nanoseconds())/1000, displayText)
		}
	}
}
