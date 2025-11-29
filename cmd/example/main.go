//go:build legacy

package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("GOBED: Text Embedding Similarity Examples")
	fmt.Println(strings.Repeat("=", 80))

	fmt.Println("\nLoading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		fmt.Printf(" Error loading model: %v\n", err)
		return
	}
	fmt.Printf(" Model loaded: %d vocab × %d dims\n", model.VocabSize, model.EmbedDim)

	// Get available texts for testing
	availableTexts := model.GetAvailableTexts()
	fmt.Printf("📚 Found %d pre-tokenized texts\n", len(availableTexts))

	// Define test groups with semantic relationships
	testGroups := map[string][]string{
		"greetings": {
			"Hello world",
			"Hi there friend",
			"Good morning everyone",
		},
		"tech": {
			"Python is a programming language.",
			"JavaScript runs in browsers.",
			"Machine learning is fascinating.",
			"Deep learning models are powerful.",
		},
		"nature": {
			"Trees grow tall in the forest.",
			"Birds are singing beautifully.",
			"The weather is nice today.",
		},
		"misc": {
			"Pizza tastes delicious.",
			"Mathematics requires practice.",
			"The cat sits on the mat",
		},
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" SIMILARITY WITHIN GROUPS (Related Texts)")
	fmt.Println(strings.Repeat("=", 80))

	// Test similarity within groups (should be higher)
	for groupName, texts := range testGroups {
		fmt.Printf("\n🏷  Group: %s\n", groupName)
		fmt.Println(strings.Repeat("-", 50))

		validTexts := []string{}
		for _, text := range texts {
			// Check if text is available
			for _, available := range availableTexts {
				if available == text {
					validTexts = append(validTexts, text)
					break
				}
			}
		}

		if len(validTexts) < 2 {
			fmt.Printf("  Not enough texts available in this group\n")
			continue
		}

		// Calculate all pairwise similarities within group
		similarities := []float32{}
		for i := 0; i < len(validTexts); i++ {
			for j := i + 1; j < len(validTexts); j++ {
				emb1, err1 := model.Encode(validTexts[i])
				emb2, err2 := model.Encode(validTexts[j])

				if err1 != nil || err2 != nil {
					continue
				}

				sim := gobed.CosineSimilarity(emb1, emb2)
				similarities = append(similarities, sim)

				fmt.Printf("  • \"%s\"\n    ↔ \"%s\"\n    → Similarity: %.4f\n\n",
					truncateText(validTexts[i], 40),
					truncateText(validTexts[j], 40),
					sim)
			}
		}

		if len(similarities) > 0 {
			avg := average(similarities)
			fmt.Printf("   Average similarity within %s: %.4f\n", groupName, avg)
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" SIMILARITY ACROSS GROUPS (Unrelated Texts)")
	fmt.Println(strings.Repeat("=", 80))

	// Test similarity across groups (should be lower)
	crossGroupPairs := []struct {
		group1, text1 string
		group2, text2 string
	}{
		{"greetings", "Hello world", "nature", "Trees grow tall in the forest."},
		{"tech", "Python is a programming language.", "misc", "Pizza tastes delicious."},
		{"nature", "Birds are singing beautifully.", "tech", "JavaScript runs in browsers."},
		{"greetings", "Good morning everyone", "misc", "Mathematics requires practice."},
		{"tech", "Machine learning is fascinating.", "nature", "The weather is nice today."},
	}

	crossSimilarities := []float32{}
	for _, pair := range crossGroupPairs {
		// Check if texts are available
		text1Available := false
		text2Available := false

		for _, available := range availableTexts {
			if available == pair.text1 {
				text1Available = true
			}
			if available == pair.text2 {
				text2Available = true
			}
		}

		if !text1Available || !text2Available {
			continue
		}

		emb1, err1 := model.Encode(pair.text1)
		emb2, err2 := model.Encode(pair.text2)

		if err1 != nil || err2 != nil {
			continue
		}

		sim := gobed.CosineSimilarity(emb1, emb2)
		crossSimilarities = append(crossSimilarities, sim)

		fmt.Printf("\n[%s ↔ %s]\n", pair.group1, pair.group2)
		fmt.Printf("  • \"%s\"\n    ↔ \"%s\"\n    → Similarity: %.4f\n",
			truncateText(pair.text1, 40),
			truncateText(pair.text2, 40),
			sim)
	}

	if len(crossSimilarities) > 0 {
		avgCross := average(crossSimilarities)
		fmt.Printf("\n Average cross-group similarity: %.4f\n", avgCross)
	}

	// Distance metrics
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("📏 DISTANCE METRICS (1 - Similarity)")
	fmt.Println(strings.Repeat("=", 80))

	// Show some example distances
	distanceExamples := []struct {
		text1, text2 string
		expected     string
	}{
		{"Hello world", "Hi there friend", "small (related greetings)"},
		{"Python is a programming language.", "JavaScript runs in browsers.", "medium (both tech)"},
		{"Hello world", "Pizza tastes delicious.", "large (unrelated)"},
		{"Machine learning is fascinating.", "Deep learning models are powerful.", "small (ML concepts)"},
		{"The weather is nice today.", "Mathematics requires practice.", "large (unrelated)"},
	}

	fmt.Println("\nDistance = 1 - CosineSimilarity (0 = identical, 2 = opposite)")
	fmt.Println(strings.Repeat("-", 70))

	for _, example := range distanceExamples {
		// Check availability
		text1Available := false
		text2Available := false

		for _, available := range availableTexts {
			if available == example.text1 {
				text1Available = true
			}
			if available == example.text2 {
				text2Available = true
			}
		}

		if !text1Available || !text2Available {
			continue
		}

		emb1, err1 := model.Encode(example.text1)
		emb2, err2 := model.Encode(example.text2)

		if err1 != nil || err2 != nil {
			continue
		}

		similarity := gobed.CosineSimilarity(emb1, emb2)
		distance := 1.0 - similarity

		fmt.Printf("\n\"%s\"\n↔ \"%s\"\n",
			truncateText(example.text1, 50),
			truncateText(example.text2, 50))
		fmt.Printf("  Similarity: %.4f | Distance: %.4f | Expected: %s\n",
			similarity, distance, example.expected)
	}

	// Performance test
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" PERFORMANCE TEST")
	fmt.Println(strings.Repeat("=", 80))

	if len(availableTexts) > 0 {
		testText := availableTexts[0]

		// Single encoding
		start := time.Now()
		_, _ = model.Encode(testText)
		singleTime := time.Since(start)

		// Batch timing (sequential)
		iterations := 1000
		start = time.Now()
		for i := 0; i < iterations; i++ {
			_, _ = model.Encode(testText)
		}
		batchTime := time.Since(start)

		fmt.Printf("\nSingle encoding: %v\n", singleTime)
		fmt.Printf("Batch %d encodings: %v\n", iterations, batchTime)
		fmt.Printf("Average per encoding: %v\n", batchTime/time.Duration(iterations))
		fmt.Printf("Throughput: %.0f encodings/sec\n", float64(iterations)/batchTime.Seconds())
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println(" Example completed!")
}

func truncateText(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen-3] + "..."
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
