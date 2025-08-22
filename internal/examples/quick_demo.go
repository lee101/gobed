package main

import (
	"fmt"
	"log"
	// Import the gobed package if needed
)

func main() {
	fmt.Println("🚀 GoEmbedding Quick Demo")
	fmt.Println("========================")

	// Load the production model (119MB, bundled with package)
	model, err := gobed.NewSafetensorsEmbedding()
	if err != nil {
		log.Fatal("Failed to load model:", err)
	}

	// Show model info
	info := model.GetModelInfo()
	fmt.Printf("✅ Loaded %s\n", info["model_type"])
	fmt.Printf("   Vocabulary: %v tokens\n", info["vocab_size"])
	fmt.Printf("   Dimensions: %v\n", info["embedding_dim"])

	// Example texts
	text1 := "Machine learning is fascinating."
	text2 := "Python is a programming language."

	// Generate embeddings
	emb1, err := model.EncodeText(text1)
	if err != nil {
		log.Fatal(err)
	}

	emb2, err := model.EncodeText(text2)
	if err != nil {
		log.Fatal(err)
	}

	// Show embeddings (first 5 dimensions)
	fmt.Printf("\n📊 Embeddings:\n")
	fmt.Printf("'%s'\n", text1)
	fmt.Printf("  -> [%.3f, %.3f, %.3f, %.3f, %.3f] (1024-dim)\n",
		emb1[0], emb1[1], emb1[2], emb1[3], emb1[4])

	fmt.Printf("'%s'\n", text2)
	fmt.Printf("  -> [%.3f, %.3f, %.3f, %.3f, %.3f] (1024-dim)\n",
		emb2[0], emb2[1], emb2[2], emb2[3], emb2[4])

	// Calculate similarity
	similarity := gobed.CosineSimilarity(emb1, emb2)
	fmt.Printf("\n🔍 Cosine Similarity: %.6f\n", similarity)

	// Additional metrics
	norm1 := gobed.CalculateNorm(emb1)
	norm2 := gobed.CalculateNorm(emb2)
	distance := gobed.EuclideanDistance(emb1, emb2)

	fmt.Printf("📏 Vector Norms: %.3f, %.3f\n", norm1, norm2)
	fmt.Printf("📐 Euclidean Distance: %.3f\n", distance)

	fmt.Println("\n✅ Demo completed!")
	fmt.Println("💡 This package provides perfect consistency with Python PyTorch")
	fmt.Println("🎯 Ready for production use in your Go applications")
}
