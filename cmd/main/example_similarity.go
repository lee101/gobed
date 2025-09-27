package main

import (
	"fmt"
	"strings"
	"time"
)

func runSimilarityExample() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println(" Text Embedding Similarity & Distance Examples")
	fmt.Println(strings.Repeat("=", 80))

	// Load model
	model, err := LoadModel()
	if err != nil {
		fmt.Printf(" Error: %v\n", err)
		return
	}

	// Get available texts
	texts := model.GetAvailableTexts()
	fmt.Printf("\n📚 Loaded %d pre-tokenized texts\n", len(texts))

	// Show sample texts
	fmt.Println("\nSample texts available:")
	for i, text := range texts {
		if i >= 5 {
			fmt.Printf("  ... and %d more\n", len(texts)-5)
			break
		}
		fmt.Printf("  %d. %s\n", i+1, text)
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" SIMILARITY BETWEEN RELATED TEXTS")
	fmt.Println(strings.Repeat("=", 80))

	// Test similar texts
	similarPairs := [][]string{
		{"Hello world", "Hi there friend"},
		{"Python is a programming language.", "JavaScript runs in browsers."},
		{"Machine learning is fascinating.", "Deep learning models are powerful."},
		{"Trees grow tall in the forest.", "Birds are singing beautifully."},
	}

	fmt.Println("\nExpected: High similarity (> 0.3) for semantically related texts")
	fmt.Println(strings.Repeat("-", 70))

	var relatedSims []float32
	for _, pair := range similarPairs {
		sim, err := model.Similarity(pair[0], pair[1])
		if err != nil {
			continue
		}
		relatedSims = append(relatedSims, sim)
		distance := 1.0 - sim

		fmt.Printf("\n\"%s\"\n↔ \"%s\"\n", pair[0], pair[1])
		fmt.Printf("  → Similarity: %.4f | Distance: %.4f\n", sim, distance)
	}

	if len(relatedSims) > 0 {
		avg := average(relatedSims)
		fmt.Printf("\n Average similarity for related texts: %.4f\n", avg)
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" SIMILARITY BETWEEN UNRELATED TEXTS")
	fmt.Println(strings.Repeat("=", 80))

	// Test unrelated texts
	unrelatedPairs := [][]string{
		{"Hello world", "Pizza tastes delicious."},
		{"Python is a programming language.", "The weather is nice today."},
		{"Mathematics requires practice.", "Birds are singing beautifully."},
		{"The cat sits on the mat", "JavaScript runs in browsers."},
	}

	fmt.Println("\nExpected: Low similarity (< 0.2) for semantically unrelated texts")
	fmt.Println(strings.Repeat("-", 70))

	var unrelatedSims []float32
	for _, pair := range unrelatedPairs {
		sim, err := model.Similarity(pair[0], pair[1])
		if err != nil {
			continue
		}
		unrelatedSims = append(unrelatedSims, sim)
		distance := 1.0 - sim

		fmt.Printf("\n\"%s\"\n↔ \"%s\"\n", pair[0], pair[1])
		fmt.Printf("  → Similarity: %.4f | Distance: %.4f\n", sim, distance)
	}

	if len(unrelatedSims) > 0 {
		avg := average(unrelatedSims)
		fmt.Printf("\n Average similarity for unrelated texts: %.4f\n", avg)
	}

	// Find most similar text
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" FINDING MOST SIMILAR TEXTS")
	fmt.Println(strings.Repeat("=", 80))

	queryText := "Hello world"
	candidates := texts[1:] // Exclude the query itself

	if len(candidates) > 0 {
		fmt.Printf("\nQuery: \"%s\"\n", queryText)
		fmt.Println("Finding top 5 most similar texts...")

		similar, err := model.FindMostSimilar(queryText, candidates, 5)
		if err == nil {
			fmt.Println("\nMost similar texts:")
			for i, result := range similar {
				distance := 1.0 - result.Similarity
				fmt.Printf("  %d. \"%s\"\n     Similarity: %.4f | Distance: %.4f\n",
					i+1, result.Text2, result.Similarity, distance)
			}
		}
	}

	// Summary statistics
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" SUMMARY STATISTICS")
	fmt.Println(strings.Repeat("=", 80))

	if len(relatedSims) > 0 && len(unrelatedSims) > 0 {
		avgRelated := average(relatedSims)
		avgUnrelated := average(unrelatedSims)
		separation := avgRelated - avgUnrelated

		fmt.Printf("\n Related texts average similarity:   %.4f\n", avgRelated)
		fmt.Printf(" Unrelated texts average similarity: %.4f\n", avgUnrelated)
		fmt.Printf(" Separation (difference):            %.4f\n", separation)

		if separation > 0.2 {
			fmt.Println("\n Good separation! The model clearly distinguishes related vs unrelated texts.")
		} else if separation > 0.1 {
			fmt.Println("\n  Moderate separation. The model somewhat distinguishes related vs unrelated.")
		} else {
			fmt.Println("\n Poor separation. The model struggles to distinguish related vs unrelated.")
		}
	}

	// Performance benchmark
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" PERFORMANCE BENCHMARK")
	fmt.Println(strings.Repeat("=", 80))

	if len(texts) > 0 {
		testText := texts[0]

		// Warm up
		for i := 0; i < 100; i++ {
			_, _ = model.Encode(testText)
		}

		// Benchmark
		iterations := 10000
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, _ = model.Encode(testText)
		}
		elapsed := time.Since(start)

		avgLatency := elapsed / time.Duration(iterations)
		throughput := float64(iterations) / elapsed.Seconds()

		fmt.Printf("\nIterations: %d\n", iterations)
		fmt.Printf("Total time: %v\n", elapsed)
		fmt.Printf("Average latency: %v\n", avgLatency)
		fmt.Printf("Throughput: %.0f embeddings/sec\n", throughput)
	}

	fmt.Println("\n Example completed!")
}

func average(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	sum := float32(0)
	for _, v := range values {
		sum += v
	}
	return sum / float32(len(values))
}
