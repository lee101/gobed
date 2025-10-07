package main

import (
    "bufio"
    "flag"
    "fmt"
    "log"
    "os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
    "unsafe"
)

// #cgo LDFLAGS: -L. -lcuda_simple -L/usr/local/cuda/lib64 -lcudart
// #include <stdlib.h>
// #include <stdbool.h>
// extern void* simple_search_create(int max_docs, int dim);
// extern void simple_search_destroy(void* handle);
// extern int simple_search_add_vectors(void* handle, const signed char* docs, int num_docs);
// extern int simple_search_query(void* handle, const signed char* query, int k, int* indices, float* scores);
import "C"
import (
    simdpkg "github.com/lee101/gobed/pkg/ann/simd"
)

// ChunkedDocument represents a semantically meaningful text chunk with line tracking
type ChunkedDocument struct {
	FilePath   string
	StartLine  int
	EndLine    int
	Content    string
	Embedding  []int8
	LineCount  int
}

// ChunkingConfig defines intelligent chunking parameters
type ChunkingConfig struct {
	MaxChunkSize      int  // Maximum characters per chunk
	MinChunkSize      int  // Minimum characters before considering a new chunk
	MaxLinesPerChunk  int  // Maximum lines to combine
	RespectParagraphs bool // Split on empty lines
	SmartBoundaries   bool // Try to split at sentence/paragraph boundaries
}

// Global model instance
var globalModel *FastModel

// GPU availability flag
var gpuAvailable bool
var gpuCheckDone bool

// Default chunking configuration optimized for code and text
func DefaultChunkingConfig() ChunkingConfig {
	return ChunkingConfig{
		MaxChunkSize:      400,  // Good balance for semantic context
		MinChunkSize:      50,   // Avoid tiny chunks
		MaxLinesPerChunk:  8,    // Reasonable multi-line context
		RespectParagraphs: true, // Split on empty lines
		SmartBoundaries:   true, // Smart splitting at boundaries
	}
}

// ChunkFile intelligently chunks a file into semantically meaningful segments
func ChunkFile(path string, model *FastModel, config ChunkingConfig) ([]*ChunkedDocument, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var chunks []*ChunkedDocument
	scanner := bufio.NewScanner(file)

	currentChunk := &strings.Builder{}
	startLine := 1
	currentLine := 0
	linesInChunk := 0

	flushChunk := func() {
		if currentChunk.Len() > 0 {
			content := strings.TrimSpace(currentChunk.String())
			if content != "" {
				embedding, err := model.EmbedInt8(content)
				if err == nil {
					chunks = append(chunks, &ChunkedDocument{
						FilePath:  path,
						StartLine: startLine,
						EndLine:   currentLine,
						Content:   content,
						Embedding: embedding,
						LineCount: linesInChunk,
					})
				}
			}
			currentChunk.Reset()
			startLine = currentLine + 1
			linesInChunk = 0
		}
	}

	for scanner.Scan() {
		currentLine++
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		// Check if we should start a new chunk
		shouldSplit := false

		// Empty line - respect paragraph boundaries
		if config.RespectParagraphs && trimmed == "" && currentChunk.Len() > config.MinChunkSize {
			shouldSplit = true
		}

		// Exceeded max chunk size
		if currentChunk.Len()+len(line) > config.MaxChunkSize {
			shouldSplit = true
		}

		// Exceeded max lines per chunk
		if linesInChunk >= config.MaxLinesPerChunk {
			shouldSplit = true
		}

		// Smart boundary detection
		if config.SmartBoundaries && currentChunk.Len() > config.MinChunkSize {
			if isNaturalBoundary(trimmed) {
				shouldSplit = true
			}
		}

		if shouldSplit && currentChunk.Len() > 0 {
			flushChunk()
		}

		// Add line to current chunk
		if !(config.RespectParagraphs && trimmed == "" && currentChunk.Len() == 0) {
			if currentChunk.Len() > 0 {
				currentChunk.WriteString(" ")
			}
			currentChunk.WriteString(trimmed)
			linesInChunk++

			if linesInChunk == 1 {
				startLine = currentLine
			}
		}
	}

	// Flush any remaining chunk
	flushChunk()

	return chunks, nil
}

// isNaturalBoundary detects natural splitting points in text/code
func isNaturalBoundary(line string) bool {
	if line == "" {
		return true
	}

	// Common code/config boundaries
	if strings.HasPrefix(line, "---") || strings.HasPrefix(line, "===") {
		return true
	}

	// Function/class definitions
	keywords := []string{"func ", "def ", "class ", "public ", "private ", "package ", "import "}
	for _, keyword := range keywords {
		if strings.HasPrefix(line, keyword) {
			return true
		}
	}

	// Markdown headers
	if strings.HasPrefix(line, "#") {
		return true
	}

	return false
}

// CheckGPUAvailable checks if CUDA GPU is available
func CheckGPUAvailable() bool {
	if gpuCheckDone {
		return gpuAvailable
	}

	gpuCheckDone = true

	// Try to create a small test context
	handle := C.simple_search_create(C.int(10), C.int(512))
	if handle != nil {
		C.simple_search_destroy(handle)
		gpuAvailable = true
		return true
	}

	gpuAvailable = false
	return false
}

// CPUSearch performs brute-force search on CPU
func CPUSearch(chunks []*ChunkedDocument, query string, k int) ([]*ChunkedDocument, []float32, error) {
    if len(chunks) == 0 {
        return nil, nil, fmt.Errorf("no documents to search")
    }

	// Generate query embedding
	queryEmb, err := globalModel.EmbedInt8(query)
	if err != nil {
		return nil, nil, err
	}

	// Calculate similarities
	type scoreIndex struct {
		score float32
		index int
	}

	scores := make([]scoreIndex, len(chunks))

    // Compute dot products in parallel (SIMD-accelerated int8 dot products)
    numWorkers := runtime.NumCPU()
    chunkSize := (len(chunks) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(chunks) {
			end = len(chunks)
		}

		wg.Add(1)
        go func(start, end int) {
            defer wg.Done()
            for i := start; i < end; i++ {
                emb := chunks[i].Embedding
                // Guard against malformed embeddings
                if len(emb) != 512 || len(queryEmb) != 512 {
                    // Fallback to safe loop if dimensions don't match
                    var dot int32
                    max := len(emb)
                    if len(queryEmb) < max {
                        max = len(queryEmb)
                    }
                    for j := 0; j < max; j++ {
                        dot += int32(emb[j]) * int32(queryEmb[j])
                    }
                    scores[i] = scoreIndex{float32(dot), i}
                    continue
                }

                dot := simdpkg.DotN(emb, queryEmb)
                scores[i] = scoreIndex{float32(dot), i}
            }
        }(start, end)
    }
	wg.Wait()

	// Sort scores
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Collect top-k results
	if k > len(scores) {
		k = len(scores)
	}

	results := make([]*ChunkedDocument, k)
	resultScores := make([]float32, k)

	for i := 0; i < k; i++ {
		results[i] = chunks[scores[i].index]
		resultScores[i] = scores[i].score
	}

	return results, resultScores, nil
}

// ScalableGPUSearch performs search using our optimized CUDA library
func ScalableGPUSearch(chunks []*ChunkedDocument, query string, k int) ([]*ChunkedDocument, []float32, error) {
	if len(chunks) == 0 {
		return nil, nil, fmt.Errorf("no documents to search")
	}

	// Check GPU availability
	if !CheckGPUAvailable() {
		return CPUSearch(chunks, query, k)
	}

	numChunks := len(chunks)

	// Create GPU context sized exactly for our data
	handle := C.simple_search_create(C.int(numChunks), C.int(512))
	if handle == nil {
		// Fallback to CPU
		fmt.Fprintf(os.Stderr, "⚠️ GPU initialization failed, falling back to CPU\n")
		return CPUSearch(chunks, query, k)
	}
	defer C.simple_search_destroy(handle)

	// Prepare flat embeddings
	flatEmbeddings := make([]int8, numChunks*512)
	for i, chunk := range chunks {
		copy(flatEmbeddings[i*512:(i+1)*512], chunk.Embedding)
	}

	// Add documents to GPU
	ret := C.simple_search_add_vectors(
		handle,
		(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int(numChunks),
	)

	if ret != 0 {
		fmt.Fprintf(os.Stderr, "⚠️ GPU add vectors failed, falling back to CPU\n")
		return CPUSearch(chunks, query, k)
	}

	// Generate query embedding
	queryEmb, err := globalModel.EmbedInt8(query)
	if err != nil {
		return nil, nil, err
	}

	// Perform search
	indices := make([]int32, k)
	scores := make([]float32, k)

	ret = C.simple_search_query(
		handle,
		(*C.schar)(unsafe.Pointer(&queryEmb[0])),
		C.int(k),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	if ret != 0 {
		fmt.Fprintf(os.Stderr, "⚠️ GPU query failed, falling back to CPU\n")
		return CPUSearch(chunks, query, k)
	}

	// Collect results
	results := make([]*ChunkedDocument, 0, k)
	resultScores := make([]float32, 0, k)

	for i := 0; i < k; i++ {
		idx := indices[i]
		if idx >= 0 && int(idx) < len(chunks) {
			results = append(results, chunks[idx])
			resultScores = append(resultScores, scores[i])
		}
	}

	return results, resultScores, nil
}

// FormatChunkResult formats a search result with line ranges
func FormatChunkResult(chunk *ChunkedDocument, score float32, query string) string {
	var result strings.Builder

	relPath, _ := filepath.Rel(".", chunk.FilePath)

	// Format line range
	lineRange := fmt.Sprintf("L%d", chunk.StartLine)
	if chunk.EndLine > chunk.StartLine {
		lineRange = fmt.Sprintf("L%d-%d", chunk.StartLine, chunk.EndLine)
	}

	// Check if query appears in content
	contains := strings.Contains(strings.ToLower(chunk.Content), strings.ToLower(query))
	marker := "  "
	if contains {
		marker = "✓ "
	}

	result.WriteString(fmt.Sprintf("%s%s:%s (%.3f)\n", marker, relPath, lineRange, score))

	// Show content preview
	content := chunk.Content
	if len(content) > 200 {
		content = content[:197] + "..."
	}

	// Indent content
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		result.WriteString(fmt.Sprintf("     %s\n", line))
	}

	return result.String()
}

// IndexDirectory processes all text files in a directory with intelligent chunking
func IndexDirectory(dir string, model *FastModel, config ChunkingConfig, extensions []string, maxFiles int, debug bool) ([]*ChunkedDocument, error) {
	var allChunks []*ChunkedDocument
	fileCount := 0

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}

		// Check file extension
		ext := strings.ToLower(filepath.Ext(path))
		validExt := false
		for _, allowedExt := range extensions {
			if ext == allowedExt {
				validExt = true
				break
			}
		}

		if !validExt {
			return nil
		}

		// Check file count limit
		if maxFiles > 0 && fileCount >= maxFiles {
			return nil
		}

		// Skip very large files
		if info.Size() > 50*1024*1024 { // 50MB
			return nil
		}

		// Chunk the file
		chunks, err := ChunkFile(path, model, config)
		if err == nil && len(chunks) > 0 {
			allChunks = append(allChunks, chunks...)
			fileCount++

			if debug {
				fmt.Fprintf(os.Stderr, "  Indexed %s: %d chunks\n", path, len(chunks))
			}
		}

		return nil
	})

	return allChunks, err
}

// Main function with comprehensive command-line interface
func main() {
	var (
		directory   = flag.String("dir", ".", "Directory to search")
		topK        = flag.Int("k", 10, "Number of results")
		debug       = flag.Bool("debug", false, "Debug output")
		maxFiles    = flag.Int("max-files", 0, "Max files to index (0=unlimited)")
		chunkSize   = flag.Int("chunk", 400, "Max chunk size in characters")
		minChunk    = flag.Int("min", 50, "Min chunk size in characters")
		maxLines    = flag.Int("lines", 8, "Max lines per chunk")
		showStats   = flag.Bool("stats", false, "Show performance statistics")
		modelPath   = flag.String("model", "model/modelint8_512dim.safetensors", "Path to model file")
		tokenizerPath = flag.String("tokenizer", "model/tokenizer.json", "Path to tokenizer file")
		forceGPU    = flag.Bool("gpu", false, "Force GPU usage (fail if not available)")
		forceCPU    = flag.Bool("cpu", false, "Force CPU usage")
	)
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		fmt.Fprintf(os.Stderr, "Usage: bed [options] <query>\n")
		fmt.Fprintf(os.Stderr, "\nOptions:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Handle GPU/CPU mode
	if *forceCPU {
		gpuAvailable = false
		gpuCheckDone = true
		if *debug {
			fmt.Fprintf(os.Stderr, "💻 CPU mode forced\n")
		}
	} else if *forceGPU {
		if !CheckGPUAvailable() {
			log.Fatalf("GPU requested but not available")
		}
		if *debug {
			fmt.Fprintf(os.Stderr, "🎮 GPU mode forced\n")
		}
	} else {
		// Auto-detect
		if CheckGPUAvailable() {
			if *debug {
				fmt.Fprintf(os.Stderr, "🎮 GPU detected and enabled\n")
			}
		} else {
			if *debug {
				fmt.Fprintf(os.Stderr, "💻 No GPU detected, using CPU\n")
			}
		}
	}

	// Load model
	if *debug {
		fmt.Fprintf(os.Stderr, "🔄 Loading optimized model...\n")
	}

	startLoad := time.Now()
	var err error
	globalModel, err = LoadFastModel(*modelPath, *tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	loadTime := time.Since(startLoad)

	if *debug {
		fmt.Fprintf(os.Stderr, "✅ Model loaded in %.2fs\n", loadTime.Seconds())
	}

	// Configure chunking
	config := ChunkingConfig{
		MaxChunkSize:      *chunkSize,
		MinChunkSize:      *minChunk,
		MaxLinesPerChunk:  *maxLines,
		RespectParagraphs: true,
		SmartBoundaries:   true,
	}

	// Index directory with intelligent chunking
	startIndex := time.Now()

	extensions := []string{
		".txt", ".md", ".go", ".py", ".js", ".ts", ".c", ".cpp",
		".h", ".java", ".rs", ".yaml", ".yml", ".json", ".xml",
		".html", ".css", ".sh", ".bash", ".zsh", ".toml", ".cfg",
		".ini", ".conf", ".config",
	}

	chunks, err := IndexDirectory(*directory, globalModel, config, extensions, *maxFiles, *debug)
	if err != nil {
		log.Fatalf("Failed to index directory: %v", err)
	}

	indexTime := time.Since(startIndex)

	if len(chunks) == 0 {
		fmt.Println("No documents found")
		os.Exit(0)
	}

	// Calculate chunking statistics
	totalLines := 0
	maxChunkLines := 0
	minChunkLines := 999999

	for _, chunk := range chunks {
		lines := chunk.LineCount
		totalLines += lines
		if lines > maxChunkLines {
			maxChunkLines = lines
		}
		if lines < minChunkLines {
			minChunkLines = lines
		}
	}

	avgLinesPerChunk := 0
	if len(chunks) > 0 {
		avgLinesPerChunk = totalLines / len(chunks)
	}

	if *debug || *showStats {
		fmt.Fprintf(os.Stderr, "\n📊 Indexing Statistics:\n")
		fmt.Fprintf(os.Stderr, "  Chunks: %d from %d total lines\n", len(chunks), totalLines)
		fmt.Fprintf(os.Stderr, "  Lines per chunk: avg=%d, min=%d, max=%d\n", avgLinesPerChunk, minChunkLines, maxChunkLines)
		fmt.Fprintf(os.Stderr, "  Index time: %.2fs (%.0f chunks/sec)\n",
			indexTime.Seconds(), float64(len(chunks))/indexTime.Seconds())
		fmt.Fprintf(os.Stderr, "  Memory: ~%.1f MB\n", float64(len(chunks)*600)/1024/1024) // Estimate
	}

	// Perform GPU search
	startSearch := time.Now()

	results, scores, err := ScalableGPUSearch(chunks, query, *topK)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	searchTime := time.Since(startSearch)

	// Display results
	fmt.Printf("\n🔍 Search: \"%s\"\n", query)
	if gpuAvailable {
		fmt.Printf("⚡ Time: %.3fms (GPU)\n", float64(searchTime.Microseconds())/1000.0)
	} else {
		fmt.Printf("⚡ Time: %.3fms (CPU)\n", float64(searchTime.Microseconds())/1000.0)
	}
	fmt.Printf("📊 Top %d results from %d chunks:\n\n", len(results), len(chunks))

	for i, chunk := range results {
		fmt.Printf("%d. %s", i+1, FormatChunkResult(chunk, scores[i], query))
		fmt.Println()
	}

	// Performance summary
	if *debug || *showStats {
		qps := 1000.0 / (float64(searchTime.Microseconds()) / 1000.0)
		totalTime := loadTime + indexTime + searchTime

		fmt.Printf("\n🚀 Performance Summary:\n")
		fmt.Printf("  Load: %.2fs | Index: %.2fs | Search: %.3fms\n",
			loadTime.Seconds(), indexTime.Seconds(), float64(searchTime.Microseconds())/1000.0)
		fmt.Printf("  Total: %.2fs | QPS: %.2f\n", totalTime.Seconds(), qps)
		fmt.Printf("  Efficiency: %.1f chunks/MB\n", float64(len(chunks))*1024*1024/float64(len(chunks)*600))

		// Memory info
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		fmt.Printf("  Memory: %.1f MB allocated\n", float64(m.Alloc)/1024/1024)
	}
}
