package main

// Ultra-fast semantic search with vectorized CUDA kernels and zero-alloc tokenizer

// #cgo LDFLAGS: -L. -lcuda_similarity -L/usr/local/cuda/lib64 -lcudart -lcublas
// #include <stdlib.h>
// extern void* cuda_fast_search_create(int max_vectors, int dim);
// extern void cuda_fast_search_destroy(void* handle);
// extern int cuda_fast_search_add_vectors(void* handle, const signed char* vectors, int num_vectors);
// extern int cuda_fast_search_query(void* handle, const signed char* query, int k, int* indices, float* scores);
import "C"

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unsafe"

	"github.com/fatih/color"
)

var (
	cyan = color.New(color.FgCyan).SprintFunc()
)

type Document struct {
	FilePath  string
	LineNum   int
	Content   string
	Embedding []int8
}

func main() {
	var (
		directory = flag.String("dir", ".", "Directory to search")
		topK      = flag.Int("k", 10, "Number of results")
		debug     = flag.Bool("debug", false, "Debug output")
	)
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		fmt.Fprintf(os.Stderr, "Usage: bed_ultra [options] <query>\n")
		os.Exit(1)
	}

	if *debug {
		fmt.Fprintf(os.Stderr, " Loading ultra-fast model...\n")
	}

	// Load optimized model and tokenizer
	model, err := LoadFastModel("../model/modelint8_512dim.safetensors", "../model/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Index documents
	startIndex := time.Now()
	documents := indexDirectory(*directory, model, *debug)

	if len(documents) == 0 {
		fmt.Println("No documents found")
		os.Exit(0)
	}

	indexTime := time.Since(startIndex)
	if *debug {
		rate := float64(len(documents)) / indexTime.Seconds()
		fmt.Fprintf(os.Stderr, " Indexed %d lines in %.2fs (%.0f lines/sec)\n",
			len(documents), indexTime.Seconds(), rate)
	}

	// Create optimized GPU search with vectorized kernels
	gpuSearch := C.cuda_fast_search_create(C.int(len(documents)+100), C.int(512))
	if gpuSearch == nil {
		log.Fatal("Failed to create GPU search")
	}
	defer C.cuda_fast_search_destroy(gpuSearch)

	// Add embeddings to GPU with coalesced memory access
	flatEmbeddings := make([]int8, len(documents)*512)
	for i, doc := range documents {
		copy(flatEmbeddings[i*512:(i+1)*512], doc.Embedding)
	}

	C.cuda_fast_search_add_vectors(
		gpuSearch,
		(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int(len(documents)),
	)

	// Encode query with zero-alloc tokenizer
	queryEmbedding, err := model.EmbedInt8(query)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}

	// Ultra-fast GPU search with vectorized similarity
	startSearch := time.Now()
	indices := make([]int32, *topK)
	scores := make([]float32, *topK)

	C.cuda_fast_search_query(
		gpuSearch,
		(*C.schar)(unsafe.Pointer(&queryEmbedding[0])),
		C.int(*topK),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	searchTime := time.Since(startSearch)

	// Display results
	fmt.Printf("\n Search: \"%s\"\n", query)
	fmt.Printf(" Time: %.3fms\n", float64(searchTime.Microseconds())/1000.0)
	fmt.Printf(" Top %d results:\n\n", *topK)

	for i := 0; i < *topK; i++ {
		idx := indices[i]
		if idx < 0 || int(idx) >= len(documents) {
			break
		}

		doc := documents[idx]
		contains := strings.Contains(strings.ToLower(doc.Content), strings.ToLower(query))
		marker := "  "
		if contains {
			marker = ""
		}

		relPath, _ := filepath.Rel(*directory, doc.FilePath)
		fmt.Printf("%s %2d. %s:%d (%.4f)\n", marker, i+1, relPath, doc.LineNum, scores[i])
		fmt.Printf("     %s\n\n", doc.Content)
	}

	if *debug {
		qps := 1000.0 / (float64(searchTime.Microseconds()) / 1000.0)
		fmt.Printf(" Performance: %.2f QPS\n", qps)
	}
}

func indexDirectory(dir string, model *FastModel, debug bool) []*Document {
	var documents []*Document

	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !isTextFile(path) {
			return nil
		}

		if info.Size() > 50*1024*1024 { // 50MB limit for large files
			return nil
		}

		docs := indexFile(path, model)
		documents = append(documents, docs...)
		return nil
	})

	return documents
}

func indexFile(path string, model *FastModel) []*Document {
	file, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer file.Close()

	var documents []*Document
	scanner := bufio.NewScanner(file)
	lineNum := 1

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			lineNum++
			continue
		}

		embedding, err := model.EmbedInt8(line)
		if err != nil {
			lineNum++
			continue
		}

		documents = append(documents, &Document{
			FilePath:  path,
			LineNum:   lineNum,
			Content:   line,
			Embedding: embedding,
		})
		lineNum++
	}

	return documents
}

func isTextFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return map[string]bool{
		".txt": true, ".md": true, ".go": true, ".py": true,
		".js": true, ".ts": true, ".c": true, ".cpp": true,
		".h": true, ".java": true, ".rs": true, ".yaml": true,
		".json": true, ".xml": true, ".html": true, ".css": true,
		".sh": true, ".bash": true, ".zsh": true,
	}[ext]
}