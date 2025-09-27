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

	"github.com/lee101/gobed/pkg/gobed"
	"github.com/lee101/gobed/pkg/search"
)

const version = "1.0.0"

func main() {
	var (
		query       = flag.String("q", "", "Search query")
		queryFile   = flag.String("qf", "", "File containing search query")
		searchPath  = flag.String("path", ".", "Path to search")
		topK        = flag.Int("k", 10, "Number of results")
		extensions  = flag.String("ext", "", "File extensions to search (comma-separated)")
		useGPU      = flag.Bool("gpu", false, "Use GPU acceleration if available")
		useInt8     = flag.Bool("int8", false, "Use INT8 quantized model for faster search")
		showVersion = flag.Bool("version", false, "Show version")
		verbose     = flag.Bool("v", false, "Verbose output")
		benchmark   = flag.Bool("bench", false, "Run in benchmark mode")
	)

	flag.Parse()

	if *showVersion {
		fmt.Printf("bed v%s\n", version)
		return
	}

	// Determine query
	var searchQuery string
	if *queryFile != "" {
		content, err := os.ReadFile(*queryFile)
		if err != nil {
			log.Fatalf("Failed to read query file: %v", err)
		}
		searchQuery = strings.TrimSpace(string(content))
	} else if *query != "" {
		searchQuery = *query
	} else if len(flag.Args()) > 0 {
		searchQuery = strings.Join(flag.Args(), " ")
	} else if stat, _ := os.Stdin.Stat(); (stat.Mode() & os.ModeCharDevice) == 0 {
		// Read from stdin if not a terminal
		scanner := bufio.NewScanner(os.Stdin)
		var lines []string
		for scanner.Scan() {
			lines = append(lines, scanner.Text())
		}
		searchQuery = strings.Join(lines, "\n")
	}

	if searchQuery == "" {
		fmt.Fprintln(os.Stderr, "Usage: bed [-q query | -qf file] [-path dir] [-k num] [-ext exts] [-gpu] [-int8]")
		fmt.Fprintln(os.Stderr, "       bed [query ...]")
		fmt.Fprintln(os.Stderr, "       echo query | bed")
		os.Exit(1)
	}

	// Initialize model
	start := time.Now()

	var model interface{}
	var err error

	if *useInt8 {
		if *verbose {
			fmt.Fprintf(os.Stderr, "Loading INT8 quantized model...\n")
		}
		// TODO: Load INT8 model
		log.Fatal("INT8 model not yet integrated")
	} else {
		if *verbose {
			fmt.Fprintf(os.Stderr, "Loading embedding model...\n")
		}
		model, err = gobed.LoadModel()
		if err != nil {
			log.Fatalf("Failed to load model: %v", err)
		}
	}

	loadTime := time.Since(start)
	if *verbose {
		fmt.Fprintf(os.Stderr, "Model loaded in %v\n", loadTime)
	}

	// Configure search
	config := search.Config{
		Path:       *searchPath,
		Query:      searchQuery,
		TopK:       *topK,
		UseGPU:     *useGPU,
		Extensions: parseExtensions(*extensions),
	}

	// Run search
	searchStart := time.Now()
	results, err := performSearch(model, config)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	searchTime := time.Since(searchStart)

	// Output results
	if *benchmark {
		fmt.Printf("Load time: %v\n", loadTime)
		fmt.Printf("Search time: %v\n", searchTime)
		fmt.Printf("Total time: %v\n", time.Since(start))
		fmt.Printf("Files searched: %d\n", results.FilesSearched)
		fmt.Printf("Results found: %d\n", len(results.Matches))
	} else {
		for i, match := range results.Matches {
			fmt.Printf("%d. %s (%.4f)\n", i+1, match.Path, match.Score)
			if *verbose && match.Preview != "" {
				fmt.Printf("   %s\n", match.Preview)
			}
		}
	}

	if *verbose {
		fmt.Fprintf(os.Stderr, "\nSearched %d files in %v\n", results.FilesSearched, searchTime)
	}
}

func parseExtensions(ext string) []string {
	if ext == "" {
		return nil
	}
	exts := strings.Split(ext, ",")
	for i := range exts {
		exts[i] = strings.TrimSpace(exts[i])
		if !strings.HasPrefix(exts[i], ".") {
			exts[i] = "." + exts[i]
		}
	}
	return exts
}

type SearchResult struct {
	Matches       []Match
	FilesSearched int
}

type Match struct {
	Path    string
	Score   float32
	Preview string
}

func performSearch(model interface{}, config search.Config) (*SearchResult, error) {
	// Walk directory and collect files
	var files []string
	err := filepath.Walk(config.Path, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip inaccessible files
		}

		// Skip directories and hidden files
		if info.IsDir() || strings.HasPrefix(info.Name(), ".") {
			return nil
		}

		// Check extension filter
		if len(config.Extensions) > 0 {
			ext := filepath.Ext(path)
			found := false
			for _, allowed := range config.Extensions {
				if ext == allowed {
					found = true
					break
				}
			}
			if !found {
				return nil
			}
		}

		// Skip binary files and very large files
		if info.Size() > 10*1024*1024 { // 10MB limit
			return nil
		}

		files = append(files, path)
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}

	// TODO: Implement actual search using embeddings
	// This is a placeholder that would integrate with the gobed library

	result := &SearchResult{
		FilesSearched: len(files),
		Matches:       []Match{},
	}

	// Placeholder results
	for i, file := range files {
		if i >= config.TopK {
			break
		}
		result.Matches = append(result.Matches, Match{
			Path:  file,
			Score: float32(1.0 - float32(i)*0.1),
		})
	}

	return result, nil
}