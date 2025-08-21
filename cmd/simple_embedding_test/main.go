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

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🔍 SIMPLE EMBEDDING VALIDATION TEST")
	fmt.Println("================================================================================")
	
	// Load model
	fmt.Printf("📦 Loading embedding model...\n")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	// model.Close() not needed for current implementation

	fmt.Printf("✅ Model loaded successfully\n\n")

	// Test basic embedding generation
	testTexts := []string{
		"This is a simple test sentence.",
		"Machine learning and artificial intelligence.",
		"Natural language processing technology.",
	}

	fmt.Println("🧪 Testing basic embedding generation:")
	for i, text := range testTexts {
		start := time.Now()
		embedding, err := model.Encode(text)
		elapsed := time.Since(start)

		if err != nil {
			log.Printf("❌ Failed to encode text %d: %v", i+1, err)
			continue
		}

		fmt.Printf("   Text %d: \"%.50s...\" - %dD in %.2fms\n", 
			i+1, text, len(embedding), float64(elapsed.Nanoseconds())/1e6)
	}

	// Test int8 quantization
	fmt.Println("\n🔢 Testing int8 quantization:")
	testText := "This is a test for quantization accuracy."
	
	// Float32 embedding
	start := time.Now()
	float32Embedding, err := model.Encode(testText)
	float32Time := time.Since(start)
	if err != nil {
		log.Fatalf("Failed to generate float32 embedding: %v", err)
	}

	// Int8 embedding
	start = time.Now()
	int8Result, err := model.EmbedInt8(testText)
	int8Time := time.Since(start)
	if err != nil {
		log.Fatalf("Failed to generate int8 embedding: %v", err)
	}

	// Reconstruct for comparison
	reconstructed := make([]float32, len(int8Result.Vector))
	for i, val := range int8Result.Vector {
		reconstructed[i] = float32(val) * int8Result.Scale / 127.0
	}

	similarity := compareEmbeddings(float32Embedding, reconstructed)

	fmt.Printf("   Float32 time: %.2fms\n", float64(float32Time.Nanoseconds())/1e6)
	fmt.Printf("   Int8 time: %.2fms\n", float64(int8Time.Nanoseconds())/1e6)
	fmt.Printf("   Similarity: %.6f\n", similarity)
	fmt.Printf("   Scale: %.6f\n", int8Result.Scale)

	if similarity > 0.95 {
		fmt.Printf("   ✅ High fidelity quantization\n")
	} else {
		fmt.Printf("   ⚠️  Quantization may need tuning\n")
	}

	// Test vector indexing (CPU only)
	fmt.Println("\n📚 Testing vector indexing (CPU):")
	
	docs := []gobed.Document{
		{ID: 1, Text: "Machine learning algorithms for data analysis."},
		{ID: 2, Text: "Natural language processing and text understanding."},
		{ID: 3, Text: "Computer vision and image recognition systems."},
		{ID: 4, Text: "Database systems for efficient data storage."},
		{ID: 5, Text: "Web development with modern frameworks."},
	}

	// Create CPU-only index
	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = false // Force CPU only
	index := gobed.NewVectorIndex(model, config)

	// Index documents
	start = time.Now()
	err = index.AddDocuments(docs)
	indexTime := time.Since(start)

	if err != nil {
		log.Fatalf("Failed to index documents: %v", err)
	}

	fmt.Printf("   Indexed %d docs in %.2fms\n", len(docs), float64(indexTime.Nanoseconds())/1e6)
	fmt.Printf("   Index size: %d\n", index.Size())

	// Test search
	query := "machine learning data processing"
	start = time.Now()
	results, err := index.Search(query, 3)
	searchTime := time.Since(start)

	if err != nil {
		log.Printf("❌ Search failed: %v", err)
	} else {
		fmt.Printf("   Search \"%s\" in %.2fms, %d results\n", 
			query, float64(searchTime.Nanoseconds())/1e6, len(results))
		
		for i, result := range results {
			fmt.Printf("     %d. Doc %d (similarity: %.4f)\n", 
				i+1, result.ID, result.Similarity)
		}
	}

	fmt.Println("\n✅ Basic functionality validated!")
	fmt.Println("💡 To test GPU acceleration:")
	fmt.Println("   1. Install libtorch for your system")
	fmt.Println("   2. Run: go run cmd/bulk_gpu_demo/main.go")
}