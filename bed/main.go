package main

// GPU-accelerated semantic search for the bed tool
// This is the main entry point that properly returns unique top-k matches

// #cgo LDFLAGS: -L. -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda_unique_topk
// #include <stdlib.h>
// extern void* create_unique_topk_search(int max_docs, int dim, int max_k);
// extern void destroy_unique_topk_search(void* handle);
// extern void add_documents_topk(void* handle, const signed char* docs, int num_docs, int dim);
// extern int search_topk_unique(void* handle, const signed char* query, int dim, int k,
//                              int* out_indices, float* out_scores);
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
	"github.com/lee101/gobed"
)

var (
	green  = color.New(color.FgGreen).SprintFunc()
	yellow = color.New(color.FgYellow).SprintFunc()
	cyan   = color.New(color.FgCyan).SprintFunc()
	bold   = color.New(color.Bold).SprintFunc()
)

type Document struct {
	FilePath  string
	LineNum   int
	Content   string
	Embedding []int8
}

type SearchResult struct {
	Document *Document
	Score    float32
}

func main() {
	// Parse command line flags
	var (
		searchQuery = flag.String("q", "", "Search query")
		directory   = flag.String("dir", ".", "Directory to search")
		topK        = flag.Int("k", 12, "Number of results to show")
		debug       = flag.Bool("debug", false, "Enable debug output")
		maxFiles    = flag.Int("max-files", 0, "Maximum files to index (0=unlimited)")
		filePattern = flag.String("pattern", "", "File pattern to match (e.g., '*.go')")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "bed - GPU-accelerated semantic search\n\n")
		fmt.Fprintf(os.Stderr, "Usage: bed [options] <query>\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  bed \"search for anime\"              # Search current directory\n")
		fmt.Fprintf(os.Stderr, "  bed -dir /path/to/dir \"query\"       # Search specific directory\n")
		fmt.Fprintf(os.Stderr, "  bed -k 20 \"find matches\"            # Show top 20 results\n")
		fmt.Fprintf(os.Stderr, "  bed -pattern \"*.txt\" \"search text\"  # Search only .txt files\n")
	}

	flag.Parse()

	// Get search query from args if not specified with -q
	if *searchQuery == "" && flag.NArg() > 0 {
		*searchQuery = strings.Join(flag.Args(), " ")
	}

	if *searchQuery == "" {
		fmt.Fprintf(os.Stderr, "Error: No search query provided\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Initialize embedding model
	if *debug {
		log.Println("Loading embedding model...")
	}

	model, err := gobed.LoadModel()
	if err != nil {
		// Fallback to mock embeddings for testing
		if *debug {
			log.Printf("Warning: Using mock embeddings: %v", err)
		}
	}

	// Index documents
	documents, err := indexDirectory(*directory, *filePattern, *maxFiles, model, *debug)
	if err != nil {
		log.Fatalf("Failed to index directory: %v", err)
	}

	if len(documents) == 0 {
		fmt.Println("No documents found to index")
		os.Exit(0)
	}

	if *debug {
		log.Printf("Indexed %d documents", len(documents))
	}

	// Create GPU search index
	const dim = 384
	gpuSearch := C.create_unique_topk_search(
		C.int(len(documents)+100),
		C.int(dim),
		C.int(100), // max_k
	)
	if gpuSearch == nil {
		log.Fatal("Failed to create GPU search index")
	}
	defer C.destroy_unique_topk_search(gpuSearch)

	// Add documents to GPU index
	if *debug {
		log.Println("Adding documents to GPU index...")
	}

	// Flatten embeddings
	flatEmbeddings := make([]int8, len(documents)*dim)
	for i, doc := range documents {
		copy(flatEmbeddings[i*dim:], doc.Embedding)
	}

	C.add_documents_topk(
		gpuSearch,
		(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int(len(documents)),
		C.int(dim),
	)

	// Encode search query
	queryEmbedding, err := encodeQuery(*searchQuery, model)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}

	// Perform GPU search
	indices := make([]int32, *topK)
	scores := make([]float32, *topK)

	startSearch := time.Now()

	numResults := int(C.search_topk_unique(
		gpuSearch,
		(*C.schar)(unsafe.Pointer(&queryEmbedding[0])),
		C.int(dim),
		C.int(*topK),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	))

	searchTime := time.Since(startSearch)

	// Display results
	fmt.Printf("\n🔍 Search: \"%s\"\n", *searchQuery)
	fmt.Printf("⏱️  Time: %.2fms\n", float64(searchTime.Microseconds())/1000.0)
	fmt.Printf("📊 Results: %d matches\n\n", numResults)

	for i := 0; i < numResults; i++ {
		idx := indices[i]
		if idx < 0 || int(idx) >= len(documents) {
			continue
		}

		doc := documents[idx]
		score := scores[i]

		// Check if query appears in content
		contains := strings.Contains(
			strings.ToLower(doc.Content),
			strings.ToLower(*searchQuery),
		)

		marker := "  "
		if contains {
			marker = "✅"
		}

		// Format output
		relPath := doc.FilePath
		if absDir, err := filepath.Abs(*directory); err == nil {
			if rel, err := filepath.Rel(absDir, doc.FilePath); err == nil {
				relPath = rel
			}
		}

		fmt.Printf("%s %2d. %s:%d (%.4f)\n",
			marker, i+1, relPath, doc.LineNum, score)

		// Show content preview
		content := strings.TrimSpace(doc.Content)
		if len(content) > 100 {
			content = content[:97] + "..."
		}

		// Highlight query in content
		if contains {
			content = highlightQuery(content, *searchQuery)
		}

		fmt.Printf("     %s\n\n", content)
	}

	if *debug {
		fmt.Printf("Search completed in %.2fms\n", float64(searchTime.Microseconds())/1000.0)
	}
}

func indexDirectory(dir, pattern string, maxFiles int, model *gobed.EmbeddingModel, debug bool) ([]*Document, error) {
	var documents []*Document
	fileCount := 0

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip errors
		}

		// Check file limit
		if maxFiles > 0 && fileCount >= maxFiles {
			return filepath.SkipDir
		}

		// Skip directories and hidden files
		if info.IsDir() || strings.HasPrefix(info.Name(), ".") {
			return nil
		}

		// Check pattern if specified
		if pattern != "" {
			matched, _ := filepath.Match(pattern, info.Name())
			if !matched {
				return nil
			}
		}

		// Skip large files
		if info.Size() > 10*1024*1024 {
			return nil
		}

		// Check if text file
		if !isTextFile(path) {
			return nil
		}

		// Index file
		docs, err := indexFile(path, model)
		if err != nil {
			if debug {
				log.Printf("Failed to index %s: %v", path, err)
			}
			return nil
		}

		documents = append(documents, docs...)
		fileCount++

		if debug && fileCount%100 == 0 {
			log.Printf("Indexed %d files...", fileCount)
		}

		return nil
	})

	return documents, err
}

func indexFile(path string, model *gobed.EmbeddingModel) ([]*Document, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var documents []*Document
	scanner := bufio.NewScanner(file)
	lineNum := 1

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			lineNum++
			continue
		}

		// Create embedding for line
		embedding, err := createEmbedding(line, model)
		if err != nil {
			continue
		}

		doc := &Document{
			FilePath:  path,
			LineNum:   lineNum,
			Content:   line,
			Embedding: embedding,
		}

		documents = append(documents, doc)
		lineNum++
	}

	return documents, scanner.Err()
}

func createEmbedding(text string, model *gobed.EmbeddingModel) ([]int8, error) {
	if model == nil {
		// Mock embedding for testing
		return mockEmbedding(text), nil
	}

	// Use real model
	floatEmbed, err := model.Encode(text)
	if err != nil {
		return nil, err
	}

	// Quantize to int8
	return quantizeEmbedding(floatEmbed), nil
}

func encodeQuery(query string, model *gobed.EmbeddingModel) ([]int8, error) {
	return createEmbedding(query, model)
}

func quantizeEmbedding(embedding []float32) []int8 {
	// Find min/max for quantization
	minVal, maxVal := embedding[0], embedding[0]
	for _, v := range embedding {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	scale := (maxVal - minVal) / 255.0
	if scale == 0 {
		scale = 1.0
	}

	quantized := make([]int8, len(embedding))
	for i, v := range embedding {
		q := int((v - minVal) / scale)
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		quantized[i] = int8(q - 128)
	}

	return quantized
}

func mockEmbedding(text string) []int8 {
	const dim = 384
	embedding := make([]int8, dim)

	// Simple hash-based embedding
	textLower := strings.ToLower(text)
	hash := 0
	for _, c := range textLower {
		hash = (hash*31 + int(c)) % 256
	}

	for i := 0; i < dim; i++ {
		embedding[i] = int8((hash + i) % 256 - 128)
	}

	return embedding
}

func isTextFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	textExts := map[string]bool{
		".txt": true, ".md": true, ".go": true, ".py": true,
		".js": true, ".ts": true, ".c": true, ".cpp": true,
		".h": true, ".java": true, ".rs": true, ".yaml": true,
		".json": true, ".xml": true, ".html": true, ".css": true,
	}
	return textExts[ext] || ext == ""
}

func highlightQuery(content, query string) string {
	lower := strings.ToLower(content)
	queryLower := strings.ToLower(query)

	idx := strings.Index(lower, queryLower)
	if idx < 0 {
		return content
	}

	before := content[:idx]
	match := content[idx : idx+len(query)]
	after := content[idx+len(query):]

	return before + yellow(match) + after
}