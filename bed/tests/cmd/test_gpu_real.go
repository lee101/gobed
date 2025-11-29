package main

/*
#cgo LDFLAGS: -L. -lcuda_simple -L/usr/local/cuda/lib64 -lcudart -lstdc++
#include <stdlib.h>
extern void* simple_search_create(int max_vectors, int dim);
extern void simple_search_destroy(void* handle);
extern int simple_search_add_vectors(void* handle, const signed char* vectors, int num_vectors);
extern int simple_search_query(void* handle, const signed char* query, int k, int* indices, float* scores);
*/
import "C"

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
	"unsafe"
)

type Document struct {
	Index     int
	Content   string
	Embedding []int8
}

// GPU + Real Content Test - Proves both speed and quality
func main() {
	fmt.Println("GPU Real Content Test - Speed + Quality")

	// Load model
	model, err := LoadFastModel("../model/modelint8_512dim.safetensors", "../model/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Load real ai.txt content (limited for speed)
	file, err := os.Open("testdata/ai.txt")
	if err != nil {
		log.Fatalf("Failed to open ai.txt: %v", err)
	}
	defer file.Close()

	var documents []*Document
	scanner := bufio.NewScanner(file)
	maxDocs := 500 // Reasonable size for GPU test

	fmt.Printf("Loading and embedding first %d lines...\n", maxDocs)
	start := time.Now()

	for scanner.Scan() && len(documents) < maxDocs {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Generate embedding
		embedding, err := model.EmbedInt8(line)
		if err != nil {
			continue
		}

		documents = append(documents, &Document{
			Index:     len(documents),
			Content:   line,
			Embedding: embedding,
		})
	}

	indexTime := time.Since(start)
	fmt.Printf("Indexed %d documents in %.2fs (%.0f docs/sec)\n",
		len(documents), indexTime.Seconds(), float64(len(documents))/indexTime.Seconds())

	// Create GPU search
	gpuSearch := C.simple_search_create(C.int(len(documents)+10), C.int(512))
	if gpuSearch == nil {
		log.Fatal("Failed to create GPU search")
	}
	defer C.simple_search_destroy(gpuSearch)

	// Add embeddings to GPU
	flatEmbeddings := make([]int8, len(documents)*512)
	for i, doc := range documents {
		copy(flatEmbeddings[i*512:(i+1)*512], doc.Embedding)
	}

	C.simple_search_add_vectors(
		gpuSearch,
		(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int(len(documents)),
	)

	// Test queries with exact word expectations
	testQueries := []struct {
		query    string
		expected string
	}{
		{"art", "art"},
		{"anime", "anime"},
		{"dragon", "dragon"},
		{"friend", "friend"},
	}

	fmt.Printf("\nGPU Search Tests (%d documents):\n", len(documents))
	fmt.Println(strings.Repeat("=", 50))

	var totalSearchTime time.Duration
	passCount := 0

	for _, test := range testQueries {
		// Generate query embedding
		queryEmbedding, err := model.EmbedInt8(test.query)
		if err != nil {
			log.Printf("Failed to encode query: %v", err)
			continue
		}

		// GPU search
		start := time.Now()
		indices := make([]int32, 3)
		scores := make([]float32, 3)

		C.simple_search_query(
			gpuSearch,
			(*C.schar)(unsafe.Pointer(&queryEmbedding[0])),
			C.int(3),
			(*C.int)(unsafe.Pointer(&indices[0])),
			(*C.float)(unsafe.Pointer(&scores[0])),
		)

		searchTime := time.Since(start)
		totalSearchTime += searchTime

		// Check results
		topIdx := indices[0]
		if topIdx >= 0 && int(topIdx) < len(documents) {
			topDoc := documents[topIdx]
			containsExpected := strings.Contains(strings.ToLower(topDoc.Content), test.expected)

			fmt.Printf("\nQuery: \"%s\" (%.3fms)\n", test.query, float64(searchTime.Microseconds())/1000.0)
			fmt.Printf("Top result (score: %.0f):\n", scores[0])
			fmt.Printf("  %s\n", truncate(topDoc.Content, 70))

			if containsExpected {
				fmt.Printf("✓ PASS: Found '%s' in top result\n", test.expected)
				passCount++
			} else {
				fmt.Printf("✗ FAIL: Expected '%s' not found\n", test.expected)
			}

			// Show top 3
			fmt.Printf("Top 3:\n")
			for i := 0; i < 3; i++ {
				idx := indices[i]
				if idx >= 0 && int(idx) < len(documents) {
					doc := documents[idx]
					hasWord := strings.Contains(strings.ToLower(doc.Content), test.expected)
					marker := "  "
					if hasWord {
						marker = "→ "
					}
					fmt.Printf("%s%d. %.0f - %s\n", marker, i+1, scores[i], truncate(doc.Content, 60))
				}
			}
		}
	}

	// Performance summary
	avgSearchTime := totalSearchTime / time.Duration(len(testQueries))
	qps := 1000.0 / (float64(avgSearchTime.Microseconds()) / 1000.0)

	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Printf("GPU Performance Summary:\n")
	fmt.Printf("  Documents: %d\n", len(documents))
	fmt.Printf("  Model: 15MB int8 (vs 119MB float32)\n")
	fmt.Printf("  Avg search: %.3fms\n", float64(avgSearchTime.Microseconds())/1000.0)
	fmt.Printf("  Throughput: %.0f QPS\n", qps)
	fmt.Printf("  Quality: %d/%d tests passed (%.1f%%)\n",
		passCount, len(testQueries), float64(passCount)/float64(len(testQueries))*100)

	if passCount == len(testQueries) {
		fmt.Printf("🎯 Perfect: GPU delivers both speed AND quality!\n")
	} else {
		fmt.Printf("⚠ Partial success: %d quality tests passed\n", passCount)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}