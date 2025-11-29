package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

var benchModel *FastModel

func main() {
	var (
		directory = flag.String("dir", ".", "Directory to search")
		topK      = flag.Int("k", 10, "Number of results")
		debug     = flag.Bool("debug", false, "Debug output")
		chunkSize = flag.Int("chunk", 300, "Max chunk size in characters")
		minChunk  = flag.Int("min", 50, "Min chunk size in characters")
		maxLines  = flag.Int("lines", 5, "Max lines per chunk")
	)
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		fmt.Fprintf(os.Stderr, "Usage: bed_chunked [options] <query>\n")
		os.Exit(1)
	}

	// Load model
	if *debug {
		fmt.Fprintf(os.Stderr, "🔄 Loading model...\n")
	}

	var err error
	benchModel, err = LoadFastModel("../../model/modelint8_512dim.safetensors", "../../model/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Configure chunking
	config := ChunkingConfig{
		MaxChunkSize:      *chunkSize,
		MinChunkSize:      *minChunk,
		MaxLinesPerChunk:  *maxLines,
		RespectParagraphs: true,
		SmartBoundaries:   true,
	}

	// Create chunker
	chunker := NewIntelligentChunker(benchModel, config)

	// Index directory with chunking
	startIndex := time.Now()

	extensions := []string{
		".txt", ".md", ".go", ".py", ".js", ".ts", ".c", ".cpp",
		".h", ".java", ".rs", ".yaml", ".json", ".xml", ".html",
		".css", ".sh", ".bash", ".zsh", ".toml", ".cfg", ".ini",
	}

	chunks, err := chunker.ChunkDirectory(*directory, extensions)
	if err != nil {
		log.Fatalf("Failed to chunk directory: %v", err)
	}

	indexTime := time.Since(startIndex)

	if len(chunks) == 0 {
		fmt.Println("No documents found")
		os.Exit(0)
	}

	// Calculate statistics
	totalLines := 0
	avgLinesPerChunk := 0
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

	if len(chunks) > 0 {
		avgLinesPerChunk = totalLines / len(chunks)
	}

	if *debug {
		fmt.Fprintf(os.Stderr, "\n📊 Chunking Statistics:\n")
		fmt.Fprintf(os.Stderr, "  Documents: %d chunks from %d total lines\n", len(chunks), totalLines)
		fmt.Fprintf(os.Stderr, "  Chunk lines: avg=%d, min=%d, max=%d\n", avgLinesPerChunk, minChunkLines, maxChunkLines)
		fmt.Fprintf(os.Stderr, "  Index time: %.2fs (%.0f chunks/sec)\n",
			indexTime.Seconds(), float64(len(chunks))/indexTime.Seconds())
	}

	// Perform search
	startSearch := time.Now()

	results, scores, err := ChunkedSearch(chunks, query, *topK)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	searchTime := time.Since(startSearch)

	// Display results
	fmt.Printf("\n🔍 Search: \"%s\"\n", query)
	fmt.Printf("⚡ Time: %.3fms\n", float64(searchTime.Microseconds())/1000.0)
	fmt.Printf("📊 Top %d results from %d chunks:\n\n", len(results), len(chunks))

	for i, chunk := range results {
		fmt.Printf("%d. %s", i+1, FormatChunkResult(chunk, scores[i], query))
		fmt.Println()
	}

	if *debug {
		qps := 1000.0 / (float64(searchTime.Microseconds()) / 1000.0)
		fmt.Printf("\n🚀 Performance: %.2f QPS over %d chunks\n", qps, len(chunks))

		// Memory efficiency
		avgBytesPerChunk := 0
		for _, chunk := range chunks {
			avgBytesPerChunk += len(chunk.Content)
		}
		if len(chunks) > 0 {
			avgBytesPerChunk /= len(chunks)
			fmt.Printf("💾 Memory: ~%d bytes/chunk average\n", avgBytesPerChunk)
		}
	}
}

// Example of test data generation for chunking demonstration
func generateTestFile(filename string) error {
	content := `# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence
that enables systems to learn from data.

## Neural Networks

Neural networks consist of layers of interconnected nodes.
Each node processes information and passes it forward.
Deep learning uses many layers to extract features.

## Training Process

The training process involves:
1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Weight updates

This iterative process continues until convergence.

## Applications

Machine learning is used in:
- Computer vision for image recognition
- Natural language processing for text analysis
- Recommendation systems for personalized content
- Autonomous vehicles for decision making

## Code Example

func trainModel(data [][]float32, labels []int) {
    model := NewNeuralNetwork(784, 128, 10)
    optimizer := NewAdam(0.001)

    for epoch := 0; epoch < 100; epoch++ {
        loss := model.Forward(data)
        model.Backward(loss)
        optimizer.Update(model)
    }
}

## Performance Optimization

To improve performance:
- Use GPU acceleration
- Implement batch processing
- Apply quantization techniques
- Optimize memory access patterns

These techniques can speed up training significantly.

## Conclusion

Machine learning continues to evolve rapidly.
New architectures and techniques emerge regularly.
The field offers exciting opportunities for innovation.`

	return os.WriteFile(filename, []byte(content), 0644)
}

func init() {
	// Create test file if running in test mode
	if _, err := os.Stat("test_chunking.txt"); os.IsNotExist(err) {
		generateTestFile("test_chunking.txt")
	}
}