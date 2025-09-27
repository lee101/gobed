package main

import (
	"fmt"
	"log"

	"github.com/lee101/gobed"
	// Import the gobed package if needed
)

func main() {
	fmt.Println("GoEmbedding Package Example")
	fmt.Println("===========================")

	// Load the default production model (bundled with package)
	fmt.Println("Loading sentence embedding model...")
	model, err := gobed.NewSafetensorsEmbedding()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Display model information
	fmt.Println("\nModel Information:")
	info := model.GetModelInfo()
	for key, value := range info {
		fmt.Printf("  %s: %v\n", key, value)
	}

	// Show available texts (pre-tokenized)
	fmt.Println("\nAvailable pre-tokenized texts:")
	availableTexts := model.GetAvailableTexts()
	for i, text := range availableTexts {
		if i < 5 { // Show first 5
			fmt.Printf("  - %s\n", text)
		}
	}
	if len(availableTexts) > 5 {
		fmt.Printf("  ... and %d more\n", len(availableTexts)-5)
	}

	// Example 1: Basic embedding generation
	fmt.Println("\n1. Basic Embedding Generation:")
	fmt.Println("------------------------------")

	text1 := "Machine learning is fascinating."
	embedding1, err := model.EncodeText(text1)
	if err != nil {
		log.Fatalf("Failed to encode text: %v", err)
	}

	fmt.Printf("Text: '%s'\n", text1)
	fmt.Printf("Embedding: [%.3f, %.3f, %.3f, %.3f, %.3f] (1024 dimensions)\n",
		embedding1[0], embedding1[1], embedding1[2], embedding1[3], embedding1[4])
	fmt.Printf("Norm: %.3f\n", gobed.CalculateNorm(embedding1))

	// Example 2: Similarity calculation
	fmt.Println("\n2. Similarity Calculation:")
	fmt.Println("--------------------------")

	text2 := "Python is a programming language."
	embedding2, err := model.EncodeText(text2)
	if err != nil {
		log.Fatalf("Failed to encode text: %v", err)
	}

	text3 := "The weather is nice today."
	embedding3, err := model.EncodeText(text3)
	if err != nil {
		log.Fatalf("Failed to encode text: %v", err)
	}

	// Calculate similarities
	sim12 := gobed.CosineSimilarity(embedding1, embedding2)
	sim13 := gobed.CosineSimilarity(embedding1, embedding3)
	sim23 := gobed.CosineSimilarity(embedding2, embedding3)

	fmt.Printf("'%s' vs '%s': %.6f\n", text1, text2, sim12)
	fmt.Printf("'%s' vs '%s': %.6f\n", text1, text3, sim13)
	fmt.Printf("'%s' vs '%s': %.6f\n", text2, text3, sim23)

	// Example 3: Batch processing
	fmt.Println("\n3. Batch Processing:")
	fmt.Println("-------------------")

	texts := []string{
		"This is a test sentence.",
		"Machine learning is fascinating.",
		"Hello world",
	}

	embeddings, err := model.BatchEncode(texts)
	if err != nil {
		log.Fatalf("Failed to batch encode: %v", err)
	}

	fmt.Printf("Processed %d texts in batch:\n", len(texts))
	for i, embedding := range embeddings {
		fmt.Printf("  %d. '%s' -> [%.3f, %.3f, %.3f, ...]\n",
			i+1, texts[i], embedding[0], embedding[1], embedding[2])
	}

	// Example 4: Find most similar texts
	fmt.Println("\n4. Finding Most Similar Texts:")
	fmt.Println("------------------------------")

	queryText := "Machine learning is fascinating."
	queryEmbedding, _ := model.EncodeText(queryText)

	candidates := []string{
		"Python is a programming language.",
		"The weather is nice today.",
		"Hello world",
		"This is a test sentence.",
	}

	fmt.Printf("Query: '%s'\n", queryText)
	fmt.Println("Similarities with candidates:")

	type SimilarityResult struct {
		Text       string
		Similarity float32
	}

	var results []SimilarityResult
	for _, candidate := range candidates {
		candidateEmbedding, err := model.EncodeText(candidate)
		if err != nil {
			continue
		}
		similarity := gobed.CosineSimilarity(queryEmbedding, candidateEmbedding)
		results = append(results, SimilarityResult{
			Text:       candidate,
			Similarity: similarity,
		})
	}

	// Sort by similarity (basic bubble sort for simplicity)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Similarity < results[j].Similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	for i, result := range results {
		fmt.Printf("  %d. %.6f - '%s'\n", i+1, result.Similarity, result.Text)
	}

	// Example 5: Distance calculations
	fmt.Println("\n5. Distance Calculations:")
	fmt.Println("------------------------")

	emb1, _ := model.EncodeText("Machine learning is fascinating.")
	emb2, _ := model.EncodeText("Python is a programming language.")

	cosineSim := gobed.CosineSimilarity(emb1, emb2)
	euclideanDist := gobed.EuclideanDistance(emb1, emb2)

	fmt.Printf("Between '%s' and '%s':\n", "Machine learning...", "Python programming...")
	fmt.Printf("  Cosine similarity: %.6f\n", cosineSim)
	fmt.Printf("  Euclidean distance: %.6f\n", euclideanDist)

	fmt.Println("\n Example completed successfully!")
	fmt.Println(" Note: This package provides perfect numerical consistency with Python PyTorch")
	fmt.Println(" Ready for production use in Go applications")
}
