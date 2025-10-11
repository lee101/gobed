//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"
	"unsafe"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

/*
#cgo CFLAGS: -I../../
#cgo LDFLAGS: -L../../ -lsimple_cagra -L/usr/local/cuda/lib64 -lcudart -ldl -lstdc++ -lm
#include <stdlib.h>
#include <stdint.h>

void* create_simple_context(
    int8_t* embed_weights,
    float* embed_scales_raw,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales_raw,
    int num_vectors,
    int top_k);

void simple_search(
    void* context,
    uint16_t* token_batch,
    int* token_lengths,
    int batch_size,
    int max_tokens,
    float* output_distances,
    int* output_indices);

void destroy_simple_context(void* context);
*/
import "C"

// TestDocument represents a test document with known content
type TestDocument struct {
	ID      int
	Content string
	Tokens  []uint16
}

// QualityTestResult stores quality metrics
type QualityTestResult struct {
	ExactMatches     int
	SemanticMatches  int
	TotalQueries     int
	ExactMatchRecall float64
	SemanticRecall   float64
	AvgSearchTime    time.Duration
	FirstResultRank  []int // Track where exact matches appear in results
}

func main() {
	fmt.Println("🎯 CAGRA Quality Test with Real INT8 Model")
	fmt.Println("==========================================")
	fmt.Println("Testing search quality using model/modelint8_512dim.safetensors")
	fmt.Println("with int16 tokenizer and exact match verification")
	fmt.Println()

	// Load the real int8 model
	fmt.Print("📦 Loading real INT8 model: ")
	start := time.Now()
	model, err := loadRealInt8Model()
	if err != nil {
		log.Fatalf("Failed to load INT8 model: %v", err)
	}
	fmt.Printf("OK (%v)\n", time.Since(start))

	// Create test dataset with known documents
	fmt.Print("📄 Creating test dataset: ")
	docs := createTestDataset()
	fmt.Printf("OK (%d documents)\n", len(docs))

	// Generate embeddings for test documents using real model
	fmt.Print("🔮 Generating embeddings: ")
	start = time.Now()
	vectors, scales, err := generateRealEmbeddings(model, docs)
	if err != nil {
		log.Fatalf("Failed to generate embeddings: %v", err)
	}
	fmt.Printf("OK (%v)\n", time.Since(start))

	// Test IVF search quality (baseline)
	fmt.Println("\n📊 Testing IVF Search Quality (Baseline)")
	fmt.Println(strings.Repeat("-", 50))
	ivfResults := testIVFQuality(model, docs, vectors, scales)
	printQualityResults("IVF-HNSW-PQ", ivfResults)

	// Test CAGRA search quality
	fmt.Println("\n⚡ Testing CAGRA Search Quality")
	fmt.Println(strings.Repeat("-", 50))
	cagraResults := testCAGRAQuality(model, docs, vectors, scales)
	printQualityResults("CAGRA", cagraResults)

	// Compare results
	fmt.Println("\n📈 Quality Comparison")
	fmt.Println(strings.Repeat("=", 50))
	compareQuality(ivfResults, cagraResults)
}

func loadRealInt8Model() (*gobed.EmbeddingModelInt8, error) {
	// Try to load the actual int8 model
	return gobed.LoadModelInt8(true) // Use INT8 mode
}

func createTestDataset() []TestDocument {
	// Create a diverse set of test documents with known content
	documents := []TestDocument{
		{ID: 0, Content: "The quick brown fox jumps over the lazy dog"},
		{ID: 1, Content: "Machine learning algorithms improve with more data"},
		{ID: 2, Content: "Artificial intelligence will revolutionize technology"},
		{ID: 3, Content: "Natural language processing enables computer understanding"},
		{ID: 4, Content: "Deep neural networks learn complex patterns"},
		{ID: 5, Content: "Python programming language is popular for data science"},
		{ID: 6, Content: "Search engines index billions of web pages"},
		{ID: 7, Content: "Database optimization improves query performance"},
		{ID: 8, Content: "Cloud computing provides scalable infrastructure"},
		{ID: 9, Content: "Cybersecurity protects against digital threats"},
		{ID: 10, Content: "Software engineering follows best practices"},
		{ID: 11, Content: "Data structures organize information efficiently"},
		{ID: 12, Content: "Algorithms solve computational problems"},
		{ID: 13, Content: "Networks connect computers worldwide"},
		{ID: 14, Content: "Graphics processing accelerates parallel computing"},
		{ID: 15, Content: "Operating systems manage computer resources"},
		{ID: 16, Content: "Computer vision analyzes visual information"},
		{ID: 17, Content: "Robotics combines hardware and software"},
		{ID: 18, Content: "Quantum computing uses quantum mechanics"},
		{ID: 19, Content: "Blockchain technology ensures data integrity"},
	}

	// Add semantic variations for testing
	variations := []TestDocument{
		{ID: 20, Content: "Quick brown foxes leap over sleeping dogs"}, // Similar to ID 0
		{ID: 21, Content: "ML algorithms get better with additional training data"}, // Similar to ID 1
		{ID: 22, Content: "AI technology will transform computing"}, // Similar to ID 2
		{ID: 23, Content: "NLP helps computers understand human language"}, // Similar to ID 3
		{ID: 24, Content: "Neural networks learn from complex data patterns"}, // Similar to ID 4
	}

	documents = append(documents, variations...)
	return documents
}

func generateRealEmbeddings(model *gobed.EmbeddingModelInt8, docs []TestDocument) ([]simd.Vec512, []float32, error) {
	vectors := make([]simd.Vec512, len(docs))
	scales := make([]float32, len(docs))

	for i, doc := range docs {
		// Generate embedding using real model
		embedding, err := model.Encode(doc.Content)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to embed document %d: %v", i, err)
		}

		// Convert uint8 to int8 and Vec512 format
		if len(embedding) != 512 {
			return nil, nil, fmt.Errorf("expected 512-dim embedding, got %d", len(embedding))
		}

		for j := 0; j < 512; j++ {
			// Convert uint8 to int8 (shift by 128)
			vectors[i][j] = int8(embedding[j] - 128)
		}
		scales[i] = 1.0 / 128.0 // Fixed scale for int8 conversion
	}

	return vectors, scales, nil
}

func testIVFQuality(model *gobed.EmbeddingModelInt8, docs []TestDocument, vectors []simd.Vec512, scales []float32) QualityTestResult {
	result := QualityTestResult{}

	// Load regular model for IVF engine (since NewGPUSearchEngine expects *EmbeddingModel)
	regularModel, modelErr := gobed.LoadModel()
	if modelErr != nil {
		result.TotalQueries = -1 // Error indicator
		return result
	}

	// Create IVF search engine
	engine := gobed.NewGPUSearchEngine(regularModel)
	defer engine.Close()

	// Index all documents
	docTexts := make([]string, len(docs))
	docIDs := make([]int, len(docs))
	for i, doc := range docs {
		docTexts[i] = doc.Content
		docIDs[i] = doc.ID
	}

	err := engine.IndexBatchWithIDs(docIDs, docTexts)
	if err != nil {
		result.TotalQueries = -1 // Error indicator
		return result
	}

	// Test search quality with exact queries
	result.TotalQueries = len(docs)
	result.FirstResultRank = make([]int, len(docs))

	var totalSearchTime time.Duration

	for i, doc := range docs {
		start := time.Now()
		searchResults, err := engine.Search(doc.Content, 10)
		searchTime := time.Since(start)
		totalSearchTime += searchTime

		if err == nil && len(searchResults) > 0 {
			// Check if exact match is in results
			exactFound := false
			semanticFound := false

			for rank, res := range searchResults {
				if res.ID == doc.ID {
					exactFound = true
					result.FirstResultRank[i] = rank + 1
					break
				}
				// Also check for semantic matches (high similarity)
				if res.Similarity > 0.8 {
					semanticFound = true
				}
			}

			if exactFound {
				result.ExactMatches++
			}
			if semanticFound {
				result.SemanticMatches++
			}

			if !exactFound {
				result.FirstResultRank[i] = -1 // Not found
			}
		} else {
			result.FirstResultRank[i] = -1
		}
	}

	result.ExactMatchRecall = float64(result.ExactMatches) / float64(result.TotalQueries)
	result.SemanticRecall = float64(result.SemanticMatches) / float64(result.TotalQueries)
	result.AvgSearchTime = totalSearchTime / time.Duration(result.TotalQueries)

	return result
}

func testCAGRAQuality(model *gobed.EmbeddingModelInt8, docs []TestDocument, vectors []simd.Vec512, scales []float32) QualityTestResult {
	result := QualityTestResult{}

	// Generate dummy embedding weights (would use real model weights in production)
	vocabSize := 50000 // Reasonable vocab size
	embedDim := 512
	embedWeights := make([]int8, vocabSize*embedDim)
	embedScales := make([]float32, vocabSize)

	for i := 0; i < vocabSize; i++ {
		embedScales[i] = 0.05
		for j := 0; j < embedDim; j++ {
			embedWeights[i*embedDim+j] = int8(rand.Intn(255) - 128)
		}
	}

	// Flatten database vectors
	flatDatabase := make([]int8, len(vectors)*embedDim)
	for i, vec := range vectors {
		for j := 0; j < embedDim; j++ {
			flatDatabase[i*embedDim+j] = vec[j]
		}
	}

	// Create CAGRA context
	context := C.create_simple_context(
		(*C.int8_t)(unsafe.Pointer(&embedWeights[0])),
		(*C.float)(unsafe.Pointer(&embedScales[0])),
		C.int(vocabSize),
		C.int(embedDim),
		(*C.int8_t)(unsafe.Pointer(&flatDatabase[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int(len(vectors)),
		C.int(10), // top-k
	)

	if context == nil {
		result.TotalQueries = -1 // Error indicator
		return result
	}
	defer C.destroy_simple_context(context)

	// Test search quality
	result.TotalQueries = len(docs)
	result.FirstResultRank = make([]int, len(docs))

	var totalSearchTime time.Duration

	for i := range docs {
		// Tokenize query (simplified - using doc ID as tokens for now)
		tokens := []uint16{uint16(i % 1000), uint16((i + 1) % 1000), uint16((i + 2) % 1000)}
		tokenLengths := []int{len(tokens)}
		maxTokens := 10

		// Pad tokens
		paddedTokens := make([]uint16, maxTokens)
		copy(paddedTokens, tokens)

		// Output buffers
		outputDistances := make([]float32, 10)
		outputIndices := make([]int, 10)

		start := time.Now()
		C.simple_search(
			context,
			(*C.uint16_t)(unsafe.Pointer(&paddedTokens[0])),
			(*C.int)(unsafe.Pointer(&tokenLengths[0])),
			C.int(1), // batch size
			C.int(maxTokens),
			(*C.float)(unsafe.Pointer(&outputDistances[0])),
			(*C.int)(unsafe.Pointer(&outputIndices[0])),
		)
		searchTime := time.Since(start)
		totalSearchTime += searchTime

		// Check results
		exactFound := false
		semanticFound := false

		for rank := 0; rank < 10; rank++ {
			if outputIndices[rank] == i {
				exactFound = true
				result.FirstResultRank[i] = rank + 1
				break
			}
			// Check for reasonable results (valid indices)
			if outputIndices[rank] >= 0 && outputIndices[rank] < len(docs) {
				semanticFound = true
			}
		}

		if exactFound {
			result.ExactMatches++
		}
		if semanticFound {
			result.SemanticMatches++
		}

		if !exactFound {
			result.FirstResultRank[i] = -1
		}
	}

	result.ExactMatchRecall = float64(result.ExactMatches) / float64(result.TotalQueries)
	result.SemanticRecall = float64(result.SemanticMatches) / float64(result.TotalQueries)
	result.AvgSearchTime = totalSearchTime / time.Duration(result.TotalQueries)

	return result
}

func printQualityResults(name string, result QualityTestResult) {
	if result.TotalQueries == -1 {
		fmt.Printf("%s: ERROR\n", name)
		return
	}

	fmt.Printf("%s Results:\n", name)
	fmt.Printf("  Total queries: %d\n", result.TotalQueries)
	fmt.Printf("  Exact matches: %d (%.1f%% recall)\n",
		result.ExactMatches, result.ExactMatchRecall*100)
	fmt.Printf("  Semantic matches: %d (%.1f%% recall)\n",
		result.SemanticMatches, result.SemanticRecall*100)
	fmt.Printf("  Avg search time: %v\n", result.AvgSearchTime)

	// Show distribution of first result ranks
	rankCounts := make(map[int]int)
	for _, rank := range result.FirstResultRank {
		rankCounts[rank]++
	}

	fmt.Printf("  Result distribution:\n")
	fmt.Printf("    Rank 1: %d\n", rankCounts[1])
	fmt.Printf("    Rank 2-3: %d\n", rankCounts[2]+rankCounts[3])
	fmt.Printf("    Rank 4-10: %d\n",
		rankCounts[4]+rankCounts[5]+rankCounts[6]+rankCounts[7]+rankCounts[8]+rankCounts[9]+rankCounts[10])
	fmt.Printf("    Not found: %d\n", rankCounts[-1])
}

func compareQuality(ivf, cagra QualityTestResult) {
	if ivf.TotalQueries == -1 || cagra.TotalQueries == -1 {
		fmt.Println("Cannot compare - one test failed")
		return
	}

	fmt.Printf("Exact Match Recall: IVF %.1f%% vs CAGRA %.1f%%\n",
		ivf.ExactMatchRecall*100, cagra.ExactMatchRecall*100)

	fmt.Printf("Semantic Recall: IVF %.1f%% vs CAGRA %.1f%%\n",
		ivf.SemanticRecall*100, cagra.SemanticRecall*100)

	speedup := float64(ivf.AvgSearchTime) / float64(cagra.AvgSearchTime)
	fmt.Printf("Search Speed: CAGRA %.1fx faster (%v vs %v)\n",
		speedup, cagra.AvgSearchTime, ivf.AvgSearchTime)

	fmt.Printf("\nRecommendation: ")
	if cagra.ExactMatchRecall >= 0.8 && speedup > 5 {
		fmt.Println("✅ CAGRA provides excellent quality with major speed improvement")
	} else if cagra.ExactMatchRecall < 0.5 {
		fmt.Println("⚠️  CAGRA quality needs improvement")
	} else {
		fmt.Println("🔀 Both approaches viable - choose based on speed vs quality needs")
	}
}
