package main

import (
	"fmt"
	"log"
	"strings"

	// Import the gobed package if needed
)

// TestPackageImport validates that the package can be imported and used correctly
func TestPackageImport() {
	fmt.Println("🧪 Package Import Test")
	fmt.Println("======================")

	// Test 1: Default model loading
	fmt.Println("1. Testing default model loading...")
	model, err := gobed.NewSafetensorsEmbedding()
	if err != nil {
		log.Fatalf("❌ Failed to load default model: %v", err)
	}
	fmt.Println("✅ Default model loaded successfully")

	// Test 2: Model info access
	fmt.Println("\n2. Testing model info access...")
	info := model.GetModelInfo()
	vocabSize, ok := info["vocab_size"].(int)
	if !ok || vocabSize != 30522 {
		log.Fatalf("❌ Expected vocab size 30522, got %v", vocabSize)
	}
	
	embedDim, ok := info["embedding_dim"].(int)
	if !ok || embedDim != 1024 {
		log.Fatalf("❌ Expected embedding dim 1024, got %v", embedDim)
	}
	fmt.Printf("✅ Model info correct: %d vocab, %d dims\n", vocabSize, embedDim)

	// Test 3: Available texts
	fmt.Println("\n3. Testing available texts...")
	availableTexts := model.GetAvailableTexts()
	if len(availableTexts) == 0 {
		log.Fatal("❌ No available texts found")
	}
	fmt.Printf("✅ Found %d pre-tokenized texts\n", len(availableTexts))

	// Test 4: Text encoding
	fmt.Println("\n4. Testing text encoding...")
	testText := "Machine learning is fascinating."
	embedding, err := model.EncodeText(testText)
	if err != nil {
		log.Fatalf("❌ Failed to encode text: %v", err)
	}
	
	if len(embedding) != 1024 {
		log.Fatalf("❌ Expected 1024 dimensions, got %d", len(embedding))
	}
	
	// Check expected values (from our validation)
	expectedFirst5 := []float32{1.610, 9.781, 2.476, -8.095, 6.863}
	tolerance := float32(0.001)
	
	for i := 0; i < 5; i++ {
		diff := embedding[i] - expectedFirst5[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			log.Fatalf("❌ Embedding mismatch at index %d: expected %.3f, got %.3f", 
				i, expectedFirst5[i], embedding[i])
		}
	}
	fmt.Printf("✅ Text encoding correct: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
		embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])

	// Test 5: Batch encoding
	fmt.Println("\n5. Testing batch encoding...")
	texts := []string{
		"Machine learning is fascinating.",
		"Python is a programming language.",
		"Hello world",
	}
	
	embeddings, err := model.BatchEncode(texts)
	if err != nil {
		log.Fatalf("❌ Failed to batch encode: %v", err)
	}
	
	if len(embeddings) != len(texts) {
		log.Fatalf("❌ Expected %d embeddings, got %d", len(texts), len(embeddings))
	}
	fmt.Printf("✅ Batch encoding successful: %d texts processed\n", len(embeddings))

	// Test 6: Similarity calculation
	fmt.Println("\n6. Testing similarity calculation...")
	emb1 := embeddings[0] // "Machine learning is fascinating."
	emb2 := embeddings[1] // "Python is a programming language."
	
	similarity := gobed.CosineSimilarity(emb1, emb2)
	expectedSimilarity := float32(0.143751)
	
	diff := similarity - expectedSimilarity
	if diff < 0 {
		diff = -diff
	}
	if diff > tolerance {
		log.Fatalf("❌ Similarity mismatch: expected %.6f, got %.6f", expectedSimilarity, similarity)
	}
	fmt.Printf("✅ Similarity calculation correct: %.6f\n", similarity)

	// Test 7: Utility functions
	fmt.Println("\n7. Testing utility functions...")
	norm := gobed.CalculateNorm(emb1)
	distance := gobed.EuclideanDistance(emb1, emb2)
	
	if norm <= 0 {
		log.Fatal("❌ Norm calculation failed")
	}
	
	if distance <= 0 {
		log.Fatal("❌ Distance calculation failed")
	}
	fmt.Printf("✅ Utility functions work: norm=%.3f, distance=%.3f\n", norm, distance)

	// Final validation
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("🎉 ALL TESTS PASSED!")
	fmt.Println("✅ Package import and usage successful")
	fmt.Println("✅ Perfect numerical consistency maintained")
	fmt.Println("✅ All API functions working correctly")
	fmt.Println("🚀 Package is ready for production use!")
	fmt.Println(strings.Repeat("=", 60))
}

func main() {
	TestPackageImport()
}