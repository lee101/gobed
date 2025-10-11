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
	"os"
	"strings"
	"time"
	"unsafe"
)

type Document struct {
	Index   int
	Content string
	Lines   int
}

// Chunk lines into max 1500 character blocks
func chunkLines(lines []string) []Document {
	var documents []Document
	var currentChunk strings.Builder
	currentLines := 0

	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Check if adding this line would exceed 1500 chars
		if currentChunk.Len() > 0 && currentChunk.Len()+len(line)+1 > 1500 {
			// Save current chunk
			documents = append(documents, Document{
				Index:   len(documents),
				Content: currentChunk.String(),
				Lines:   currentLines,
			})

			// Start new chunk
			currentChunk.Reset()
			currentLines = 0
		}

		// Add line to current chunk
		if currentChunk.Len() > 0 {
			currentChunk.WriteString(" ")
		}
		currentChunk.WriteString(line)
		currentLines++

		// If this is the last line, save the chunk
		if i == len(lines)-1 && currentChunk.Len() > 0 {
			documents = append(documents, Document{
				Index:   len(documents),
				Content: currentChunk.String(),
				Lines:   currentLines,
			})
		}
	}

	return documents
}

// Full scale test - 200k lines with chunking
func main() {
	fmt.Printf("Full Scale Test - 200k Lines from ai.txt\n")

	// Load model silently
	model, err := LoadFastModel("../model/modelint8_512dim.safetensors", "../model/tokenizer.json")
	if err != nil {
		fmt.Printf("Model load failed: %v\n", err)
		return
	}

	// Load all lines from ai.txt
	file, err := os.Open("testdata/ai.txt")
	if err != nil {
		fmt.Printf("File open failed: %v\n", err)
		return
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	fmt.Printf("Loaded %d lines\n", len(lines))

	// Chunk into documents
	start := time.Now()
	documents := chunkLines(lines)
	chunkTime := time.Since(start)

	fmt.Printf("Created %d chunks (avg %.1f lines/chunk) in %.3fs\n",
		len(documents), float64(len(lines))/float64(len(documents)), chunkTime.Seconds())

	// Generate embeddings with progress
	fmt.Printf("Generating embeddings...")
	start = time.Now()
	flatEmbeddings := make([]int8, len(documents)*512)
	validDocs := 0

	for i, doc := range documents {
		if i%1000 == 0 && i > 0 {
			fmt.Printf(".")
		}

		embedding, err := model.EmbedInt8(doc.Content)
		if err != nil {
			continue
		}

		copy(flatEmbeddings[validDocs*512:(validDocs+1)*512], embedding)
		if validDocs != i {
			documents[validDocs] = documents[i]
		}
		validDocs++
	}
	documents = documents[:validDocs]

	embeddingTime := time.Since(start)
	fmt.Printf("\nEmbedded %d documents in %.2fs (%.0f docs/sec)\n",
		validDocs, embeddingTime.Seconds(), float64(validDocs)/embeddingTime.Seconds())

	// Create GPU search
	gpuSearch := C.simple_search_create(C.int(validDocs+100), C.int(512))
	if gpuSearch == nil {
		fmt.Printf("GPU init failed\n")
		return
	}
	defer C.simple_search_destroy(gpuSearch)

	// Upload to GPU
	start = time.Now()
	C.simple_search_add_vectors(
		gpuSearch,
		(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int(validDocs),
	)
	uploadTime := time.Since(start)

	fmt.Printf("GPU upload: %.3fs\n", uploadTime.Seconds())

	// Test queries
	testQueries := []string{
		"art", "anime", "dragon", "friend", "warrior", "magic", "robot", "space",
	}

	fmt.Printf("\nSearching %d documents:\n", validDocs)

	var totalSearchTime time.Duration
	for _, query := range testQueries {
		queryEmbedding, err := model.EmbedInt8(query)
		if err != nil {
			continue
		}

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

		// Show result
		topIdx := indices[0]
		if topIdx >= 0 && int(topIdx) < len(documents) {
			doc := documents[topIdx]
			hasMatch := strings.Contains(strings.ToLower(doc.Content), query)
			marker := "✓"
			if !hasMatch {
				marker = "~"
			}

			fmt.Printf("%s \"%s\" %.3fms → %.0f: %s\n",
				marker, query, float64(searchTime.Microseconds())/1000.0,
				scores[0], truncate(doc.Content, 60))
		}
	}

	// Performance summary
	avgSearchTime := totalSearchTime / time.Duration(len(testQueries))
	qps := 1000.0 / (float64(avgSearchTime.Microseconds()) / 1000.0)

	fmt.Printf("\n" + strings.Repeat("=", 50))
	fmt.Printf("\nFull Scale Results:\n")
	fmt.Printf("  Original lines: %d\n", len(lines))
	fmt.Printf("  Chunked docs: %d\n", validDocs)
	fmt.Printf("  Embedding time: %.2fs (%.0f docs/sec)\n", embeddingTime.Seconds(),
		float64(validDocs)/embeddingTime.Seconds())
	fmt.Printf("  Avg search: %.3fms\n", float64(avgSearchTime.Microseconds())/1000.0)
	fmt.Printf("  Throughput: %.0f QPS\n", qps)
	fmt.Printf("  Model: 15MB int8 (auto-truncates, silent errors)\n")
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}