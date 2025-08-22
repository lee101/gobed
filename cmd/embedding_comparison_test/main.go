package main

import (
	"fmt"
	"log"
	"math"
	"time"

	"github.com/lee101/gobed"
)

// compareEmbeddings compares two embedding vectors for similarity
func compareEmbeddings(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}

	return dotProduct / (normA * normB)
}

// testEmbeddingConsistency tests that embeddings are consistent
func testEmbeddingConsistency(model *gobed.EmbeddingModel) {
	fmt.Println("🧪 Testing Embedding Consistency")
	fmt.Println("=================================")

	testTexts := []string{
		"This is a simple test sentence.",
		"Machine learning and artificial intelligence are fascinating fields.",
		"Natural language processing enables computers to understand human language.",
		"Vector databases provide efficient similarity search capabilities.",
		"GPU acceleration significantly improves deep learning performance.",
	}

	fmt.Printf("Testing %d texts for embedding consistency...\n\n", len(testTexts))

	for i, text := range testTexts {
		fmt.Printf("Text %d: \"%s\"\n", i+1, text)

		// Generate embedding multiple times to test consistency
		var embeddings [][]float32
		times := make([]time.Duration, 5)

		for j := 0; j < 5; j++ {
			start := time.Now()
			embedding, err := model.Encode(text)
			elapsed := time.Since(start)
			times[j] = elapsed

			if err != nil {
				log.Printf("❌ Failed to generate embedding %d: %v", j+1, err)
				continue
			}

			embeddings = append(embeddings, embedding)
		}

		if len(embeddings) < 2 {
			fmt.Printf("❌ Not enough embeddings generated for comparison\n\n")
			continue
		}

		// Compare all embeddings for consistency
		allSame := true
		similarities := make([]float64, len(embeddings)-1)

		for j := 1; j < len(embeddings); j++ {
			similarity := compareEmbeddings(embeddings[0], embeddings[j])
			similarities[j-1] = similarity

			if similarity < 0.9999 { // Should be nearly identical
				allSame = false
			}
		}

		// Calculate timing statistics
		var totalTime time.Duration
		for _, t := range times {
			totalTime += t
		}
		avgTime := totalTime / time.Duration(len(times))

		// Results
		fmt.Printf("   Embeddings: %d generated successfully\n", len(embeddings))
		fmt.Printf("   Dimensions: %d\n", len(embeddings[0]))
		fmt.Printf("   Avg time: %.2fms\n", float64(avgTime.Nanoseconds())/1e6)

		if allSame {
			fmt.Printf("   ✅ All embeddings identical (similarity > 0.9999)\n")
		} else {
			fmt.Printf("   ⚠️  Embeddings vary (similarities: ")
			for j, sim := range similarities {
				fmt.Printf("%.6f", sim)
				if j < len(similarities)-1 {
					fmt.Printf(", ")
				}
			}
			fmt.Printf(")\n")
		}

		// Show first few dimensions
		fmt.Printf("   Sample dims: [%.6f, %.6f, %.6f, ...]\n",
			embeddings[0][0], embeddings[0][1], embeddings[0][2])

		fmt.Println()
	}
}

// testInt8Quantization tests the int8 quantization process
func testInt8Quantization(model *gobed.EmbeddingModel) {
	fmt.Println("🔢 Testing Int8 Quantization")
	fmt.Println("=============================")

	testText := "This is a test for quantization accuracy."
	fmt.Printf("Test text: \"%s\"\n\n", testText)

	// Generate float32 embedding
	start := time.Now()
	float32Embedding, err := model.Encode(testText)
	float32Time := time.Since(start)

	if err != nil {
		log.Fatalf("Failed to generate float32 embedding: %v", err)
	}

	// Generate int8 embedding
	start = time.Now()
	int8Result, err := model.EmbedInt8(testText)
	int8Time := time.Since(start)

	if err != nil {
		log.Fatalf("Failed to generate int8 embedding: %v", err)
	}

	// Convert int8 back to float32 for comparison
	reconstructed := make([]float32, len(int8Result.Vector))
	for i, val := range int8Result.Vector {
		reconstructed[i] = float32(val) * int8Result.Scale / 127.0
	}

	// Compare original and reconstructed
	similarity := compareEmbeddings(float32Embedding, reconstructed)

	fmt.Printf("Results:\n")
	fmt.Printf("   Float32 time: %.2fms\n", float64(float32Time.Nanoseconds())/1e6)
	fmt.Printf("   Int8 time: %.2fms\n", float64(int8Time.Nanoseconds())/1e6)
	fmt.Printf("   Dimensions: %d\n", len(float32Embedding))
	fmt.Printf("   Scale factor: %.6f\n", int8Result.Scale)
	fmt.Printf("   Similarity: %.6f\n", similarity)

	if similarity > 0.95 {
		fmt.Printf("   ✅ High fidelity quantization (> 0.95)\n")
	} else if similarity > 0.90 {
		fmt.Printf("   ✅ Good quantization (> 0.90)\n")
	} else {
		fmt.Printf("   ⚠️  Low fidelity quantization (< 0.90)\n")
	}

	// Show sample values
	fmt.Printf("\n   Sample comparison:\n")
	for i := 0; i < 5 && i < len(float32Embedding); i++ {
		fmt.Printf("     [%d] Float32: %8.6f, Int8: %4d, Reconstructed: %8.6f\n",
			i, float32Embedding[i], int8Result.Vector[i], reconstructed[i])
	}

	fmt.Println()
}

// benchmarkEmbeddingGeneration benchmarks embedding generation performance
func benchmarkEmbeddingGeneration(model *gobed.EmbeddingModel) {
	fmt.Println("⚡ Benchmarking Embedding Generation")
	fmt.Println("====================================")

	// Create test documents
	docs := make([]string, 1000)
	for i := 0; i < len(docs); i++ {
		docs[i] = fmt.Sprintf("This is test document number %d with some varied content about topic %d.", i, i%10)
	}

	fmt.Printf("Benchmarking with %d documents...\n\n", len(docs))

	// Benchmark float32 embeddings
	fmt.Println("Float32 Embeddings:")
	start := time.Now()
	float32Count := 0
	for _, doc := range docs {
		_, err := model.Encode(doc)
		if err == nil {
			float32Count++
		}
	}
	float32Time := time.Since(start)
	float32Throughput := float64(float32Count) / float32Time.Seconds()

	fmt.Printf("   Processed: %d/%d documents\n", float32Count, len(docs))
	fmt.Printf("   Time: %.2fs\n", float32Time.Seconds())
	fmt.Printf("   Throughput: %.1f docs/sec\n", float32Throughput)

	// Benchmark int8 embeddings
	fmt.Println("\nInt8 Embeddings:")
	start = time.Now()
	int8Count := 0
	for _, doc := range docs {
		_, err := model.EmbedInt8(doc)
		if err == nil {
			int8Count++
		}
	}
	int8Time := time.Since(start)
	int8Throughput := float64(int8Count) / int8Time.Seconds()

	fmt.Printf("   Processed: %d/%d documents\n", int8Count, len(docs))
	fmt.Printf("   Time: %.2fs\n", int8Time.Seconds())
	fmt.Printf("   Throughput: %.1f docs/sec\n", int8Throughput)

	// Performance comparison
	fmt.Printf("\nPerformance Comparison:\n")
	if int8Throughput > float32Throughput {
		ratio := int8Throughput / float32Throughput
		fmt.Printf("   Int8 is %.2fx faster than Float32\n", ratio)
	} else {
		ratio := float32Throughput / int8Throughput
		fmt.Printf("   Float32 is %.2fx faster than Int8\n", ratio)
	}

	fmt.Println()
}

// testIndexingWorkflow tests the complete indexing workflow
func testIndexingWorkflow(model *gobed.EmbeddingModel) {
	fmt.Println("📚 Testing Indexing Workflow")
	fmt.Println("=============================")

	// Create test documents
	docs := []gobed.Document{
		{ID: 1, Text: "Machine learning algorithms process data to make predictions."},
		{ID: 2, Text: "Natural language processing enables computers to understand text."},
		{ID: 3, Text: "Vector databases store high-dimensional embedding vectors efficiently."},
		{ID: 4, Text: "GPU acceleration provides massive parallel processing power."},
		{ID: 5, Text: "Deep learning models learn complex patterns from data."},
	}

	fmt.Printf("Testing with %d documents...\n\n", len(docs))

	// Create vector index (CPU only for now)
	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = false // Disable GPU for this test
	index := gobed.NewVectorIndex(model, config)

	// Add documents
	fmt.Println("Adding documents to index...")
	start := time.Now()
	err := index.AddDocuments(docs)
	indexTime := time.Since(start)

	if err != nil {
		log.Fatalf("Failed to add documents: %v", err)
	}

	fmt.Printf("✅ Indexed %d documents in %.2fms\n", len(docs), float64(indexTime.Nanoseconds())/1e6)
	fmt.Printf("   Index size: %d\n", index.Size())

	// Test search
	fmt.Println("\nTesting search functionality...")
	queries := []string{
		"machine learning prediction",
		"natural language text",
		"vector database storage",
		"GPU parallel processing",
		"deep learning patterns",
	}

	for i, query := range queries {
		start := time.Now()
		results, err := index.Search(query, 3)
		searchTime := time.Since(start)

		if err != nil {
			log.Printf("❌ Search failed for query %d: %v", i+1, err)
			continue
		}

		fmt.Printf("   Q%d: \"%.40s...\" - %.2fms, %d results\n",
			i+1, query, float64(searchTime.Nanoseconds())/1e6, len(results))

		if len(results) > 0 {
			fmt.Printf("        Top result: Doc %d (similarity: %.4f)\n",
				results[0].ID, results[0].Similarity)
		}
	}

	fmt.Println()
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🔍 GOBED EMBEDDING COMPARISON AND VALIDATION TEST")
	fmt.Println("================================================================================")
	fmt.Println("Testing embedding consistency, quantization accuracy, and indexing workflow")
	fmt.Println()

	// Load model
	fmt.Printf("📦 Loading embedding model...\n")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	fmt.Printf("✅ Model loaded successfully\n\n")

	// Run tests
	testEmbeddingConsistency(model)
	testInt8Quantization(model)
	benchmarkEmbeddingGeneration(model)
	testIndexingWorkflow(model)

	// Summary
	fmt.Println("================================================================================")
	fmt.Println("✅ EMBEDDING VALIDATION COMPLETED")
	fmt.Println("================================================================================")
	fmt.Println("Key validations performed:")
	fmt.Println("  • Embedding consistency across multiple generations")
	fmt.Println("  • Int8 quantization accuracy and performance")
	fmt.Println("  • CPU-based indexing workflow verification")
	fmt.Println("  • Search functionality validation")
	fmt.Println()
	fmt.Println("Next steps:")
	fmt.Println("  • Install libtorch for GPU acceleration testing")
	fmt.Println("  • Compare GPU vs CPU embedding generation")
	fmt.Println("  • Validate bulk GPU indexing performance")
}
