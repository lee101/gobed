package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

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

func main() {
	// Parse command line flags
	var (
		searchQuery = flag.String("q", "", "Search query")
		directory   = flag.String("dir", ".", "Directory to search")
		topK        = flag.Int("k", 12, "Number of results to show")
		debug       = flag.Bool("debug", false, "Enable debug output")
		maxFiles    = flag.Int("max-files", 0, "Maximum files to index (0=unlimited)")
		filePattern = flag.String("pattern", "", "File pattern to match (e.g., '*.go')")
		forceGPU    = flag.Bool("gpu", false, "Force GPU usage")
		forceCPU    = flag.Bool("cpu", false, "Force CPU usage")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "bed - High-performance semantic search\n\n")
		fmt.Fprintf(os.Stderr, "Usage: bed [options] <query>\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  bed \"search for anime\"              # Search current directory\n")
		fmt.Fprintf(os.Stderr, "  bed -dir /path/to/dir \"query\"       # Search specific directory\n")
		fmt.Fprintf(os.Stderr, "  bed -k 20 \"find matches\"            # Show top 20 results\n")
		fmt.Fprintf(os.Stderr, "  bed -pattern \"*.txt\" \"search text\"  # Search only .txt files\n")
		fmt.Fprintf(os.Stderr, "  bed -gpu \"search with GPU\"          # Force GPU usage\n")
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

	// Create search engine (auto-detects GPU)
	const dim = 384
	engine := CreateSearchEngine(len(documents), dim, *forceGPU, *forceCPU, *debug)
	defer engine.Destroy()

	// Flatten embeddings
	flatEmbeddings := make([]int8, len(documents)*dim)
	for i, doc := range documents {
		copy(flatEmbeddings[i*dim:], doc.Embedding)
	}

	// Add documents to search engine
	if *debug {
		log.Printf("Adding documents to %s engine...", engine.GetMode())
	}
	engine.AddDocuments(flatEmbeddings, len(documents), dim)

	// Encode search query
	queryEmbedding, err := encodeQuery(*searchQuery, model)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}

	// Perform search
	startSearch := time.Now()
	indices, scores, numResults := engine.Search(queryEmbedding, *topK)
	searchTime := time.Since(startSearch)

	// Display results
	fmt.Printf("\n🔍 %s [%s]\n", *searchQuery, engine.GetMode())
	fmt.Printf("⚡ %.3fms | %d results\n\n", float64(searchTime.Microseconds())/1000.0, numResults)

	for i := 0; i < numResults; i++ {
		idx := indices[i]
		if idx < 0 || int(idx) >= len(documents) {
			continue
		}

		doc := documents[idx]
		score := scores[i]

		// Format output
		relPath := doc.FilePath
		if absDir, err := filepath.Abs(*directory); err == nil {
			if rel, err := filepath.Rel(absDir, doc.FilePath); err == nil {
				relPath = rel
			}
		}

		// Concise output format
		content := strings.TrimSpace(doc.Content)
		if len(content) > 80 {
			content = content[:77] + "..."
		}

		// Check if query appears in content
		if strings.Contains(strings.ToLower(doc.Content), strings.ToLower(*searchQuery)) {
			content = highlightQuery(content, *searchQuery)
			fmt.Printf("✅ %s:%d (%.3f)\n  %s\n\n",
				relPath, doc.LineNum, score, content)
		} else {
			fmt.Printf("%s:%d (%.3f)\n  %s\n\n",
				relPath, doc.LineNum, score, content)
		}
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

		// Skip extremely large files (>100MB)
		if info.Size() > 100*1024*1024 {
			if debug {
				log.Printf("Skipping large file %s (%d MB)", path, info.Size()/(1024*1024))
			}
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