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
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
	"unsafe"
)

type Document struct {
	FilePath  string
	LineNum   int
	Content   string
	Embedding []int8
}

func main() {
	var (
		topK  = flag.Int("k", 5, "Number of results")
		debug = flag.Bool("debug", false, "Debug output")
	)
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		fmt.Fprintf(os.Stderr, "Usage: bed_simple [options] <query>\n")
		os.Exit(1)
	}

	if *debug {
		fmt.Fprintf(os.Stderr, "Loading simple CUDA model...\n")
	}

	// Load model
	model, err := LoadFastModel("../model/modelint8_512dim.safetensors", "../model/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Index a few test documents only for debugging
	testDocs := []string{
		"machine learning algorithms are powerful",
		"neural networks and deep learning",
		"anime and manga culture in Japan",
		"cooking recipes for dinner",
		"programming languages like Go and Python",
		"artificial intelligence research",
		"optimization techniques in software",
	}

	var documents []*Document
	for i, doc := range testDocs {
		embedding, err := model.EmbedInt8(doc)
		if err != nil {
			log.Printf("Failed to embed doc %d: %v", i, err)
			continue
		}

		documents = append(documents, &Document{
			FilePath:  fmt.Sprintf("test%d.txt", i),
			LineNum:   1,
			Content:   doc,
			Embedding: embedding,
		})
	}

	if *debug {
		fmt.Fprintf(os.Stderr, "Indexed %d test documents\n", len(documents))
	}

	// Create simple GPU search
	gpuSearch := C.simple_search_create(C.int(len(documents)+10), C.int(512))
	if gpuSearch == nil {
		log.Fatal("Failed to create simple GPU search")
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

	// Encode query
	queryEmbedding, err := model.EmbedInt8(query)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}

	// GPU search
	startSearch := time.Now()
	indices := make([]int32, *topK)
	scores := make([]float32, *topK)

	C.simple_search_query(
		gpuSearch,
		(*C.schar)(unsafe.Pointer(&queryEmbedding[0])),
		C.int(*topK),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	searchTime := time.Since(startSearch)

	// Display results
	fmt.Printf("Simple CUDA Search: \"%s\"\n", query)
	fmt.Printf("Time: %.3fms\n", float64(searchTime.Microseconds())/1000.0)
	fmt.Printf("Top %d results:\n", *topK)

	for i := 0; i < *topK; i++ {
		idx := indices[i]
		if idx < 0 || int(idx) >= len(documents) {
			break
		}

		doc := documents[idx]
		fmt.Printf("  %d. %.1f - %s\n", i+1, scores[i], doc.Content)
	}
}