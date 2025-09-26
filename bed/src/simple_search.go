package src

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

// BedSearchOptions configures search behavior
type BedSearchOptions struct {
	Query          string
	Limit          int
	Context        int
	Threshold      float32
	NoIndex        bool
	ForceIndex     bool
	SearchBinaries bool
	Progressive    bool
	UseGPU         bool
	Verbose        bool
}

// SimpleSearchEngine provides semantic search with gobed
type SimpleSearchEngine struct {
	model        *gobed.EmbeddingModel
	documents    []Document
	embeddings   [][]float32
	mu           sync.RWMutex
	ignoreFilter *EnhancedIgnoreFilter
}

// Document represents an indexed document
type Document struct {
	ID         int
	Path       string
	LineNumber int
	Content    string
	IsBinary   bool
}

// NewSimpleSearchEngine creates a new search engine
func NewSimpleSearchEngine() (*SimpleSearchEngine, error) {
	model, err := gobed.LoadModel()
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}
	
	ignoreFilter, err := NewEnhancedIgnoreFilter(".",
		WithMaxFileSize(10*1024*1024), // 10MB max
		WithBinarySearch(false),
	)
	if err != nil {
		return nil, err
	}
	
	return &SimpleSearchEngine{
		model:        model,
		documents:    make([]Document, 0),
		embeddings:   make([][]float32, 0),
		ignoreFilter: ignoreFilter,
	}, nil
}

// IndexDirectory indexes all files in a directory
func (se *SimpleSearchEngine) IndexDirectory(path string, options BedSearchOptions) error {
	if options.Verbose {
		fmt.Printf("Indexing directory: %s\n", path)
	}
	
	startTime := time.Now()
	fileCount := 0
	lineCount := 0
	
	// Use goroutines for parallel processing
	type workItem struct {
		path    string
		content string
		lines   []string
	}
	
	workChan := make(chan workItem, 100)
	resultChan := make(chan []Document, 100)
	embeddingChan := make(chan struct {
		doc      Document
		embedding []float32
	}, 1000)
	
	// Start workers
	numWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	
	// File reader goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(workChan)
		
		filepath.WalkDir(path, func(filePath string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			
			shouldProcess, fileType := se.ignoreFilter.ShouldProcess(filePath)
			if !shouldProcess {
				if options.Verbose && fileType == FileTypeTooLarge {
					fmt.Printf("Skipping large file: %s\n", filePath)
				}
				return nil
			}
			
			// Handle binary files
			if fileType == FileTypeBinary {
				if options.SearchBinaries {
					workChan <- workItem{
						path:    filePath,
						content: "binary",
						lines:   []string{"binary"},
					}
				}
				return nil
			}
			
			// Read text file
			content, err := os.ReadFile(filePath)
			if err != nil {
				return nil
			}
			
			lines := strings.Split(string(content), "\n")
			workChan <- workItem{
				path:    filePath,
				content: string(content),
				lines:   lines,
			}
			
			fileCount++
			return nil
		})
	}()
	
	// Document processor workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for item := range workChan {
				docs := make([]Document, 0)
				
				if item.content == "binary" {
					doc := Document{
						Path:     item.path,
						Content:  fmt.Sprintf("%s binary file", filepath.Base(item.path)),
						IsBinary: true,
					}
					docs = append(docs, doc)
				} else {
					for lineNum, line := range item.lines {
						if strings.TrimSpace(line) == "" {
							continue
						}
						
						doc := Document{
							Path:       item.path,
							LineNumber: lineNum + 1,
							Content:    line,
						}
						docs = append(docs, doc)
						lineCount++
					}
				}
				
				if len(docs) > 0 {
					resultChan <- docs
				}
			}
		}()
	}
	
	// Result collector
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(embeddingChan)
		
		for docs := range resultChan {
			for _, doc := range docs {
				// Generate embedding
				embedding, err := se.model.Encode(doc.Content)
				if err != nil {
					continue
				}
				
				embeddingChan <- struct {
					doc      Document
					embedding []float32
				}{doc, embedding}
			}
		}
	}()
	
	// Embedding collector
	go func() {
		for item := range embeddingChan {
			se.mu.Lock()
			item.doc.ID = len(se.documents)
			se.documents = append(se.documents, item.doc)
			se.embeddings = append(se.embeddings, item.embedding)
			se.mu.Unlock()
		}
	}()
	
	// Start result collector in background
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	// Wait for everything to complete
	wg.Wait()
	time.Sleep(100 * time.Millisecond) // Give embedding collector time to finish
	
	elapsed := time.Since(startTime)
	if options.Verbose {
		fmt.Printf("Indexed %d files with %d lines in %.2fs\n", 
			fileCount, len(se.documents), elapsed.Seconds())
	}
	
	return nil
}

// Search performs semantic search
func (se *SimpleSearchEngine) Search(query string, limit int, threshold float32) ([]SearchMatch, error) {
	// Generate query embedding
	queryEmbedding, err := se.model.Encode(query)
	if err != nil {
		return nil, fmt.Errorf("failed to encode query: %w", err)
	}
	
	se.mu.RLock()
	defer se.mu.RUnlock()
	
	// Calculate similarities
	type scoredDoc struct {
		doc        Document
		similarity float32
	}
	
	scores := make([]scoredDoc, 0, len(se.documents))
	
	for i, doc := range se.documents {
		if i >= len(se.embeddings) {
			continue
		}
		
		similarity := cosineSimilarity(queryEmbedding, se.embeddings[i])
		if similarity >= threshold {
			scores = append(scores, scoredDoc{
				doc:        doc,
				similarity: similarity,
			})
		}
	}
	
	// Sort by similarity
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].similarity > scores[j].similarity
	})
	
	// Limit results
	if limit > 0 && len(scores) > limit {
		scores = scores[:limit]
	}
	
	// Convert to search matches
	matches := make([]SearchMatch, len(scores))
	for i, score := range scores {
		matches[i] = SearchMatch{
			Document:   score.doc,
			Similarity: score.similarity,
		}
	}
	
	return matches, nil
}

// SearchMatch represents a search result
type SearchMatch struct {
	Document   Document
	Similarity float32
}

// SimpleBedSearcher uses SimpleSearchEngine for bed
type SimpleBedSearcher struct {
	engine         *SimpleSearchEngine
	queryProcessor *QueryProcessor
	verbose        bool
}

// NewSimpleBedSearcher creates a bed searcher
func NewSimpleBedSearcher() (*SimpleBedSearcher, error) {
	engine, err := NewSimpleSearchEngine()
	if err != nil {
		return nil, err
	}
	
	return &SimpleBedSearcher{
		engine:         engine,
		queryProcessor: NewQueryProcessor(),
	}, nil
}

// Search performs natural language search
func (sbs *SimpleBedSearcher) Search(options BedSearchOptions) error {
	sbs.verbose = options.Verbose
	
	// Process query
	processed := sbs.queryProcessor.Process(options.Query)
	
	if options.Verbose {
		fmt.Printf("Query: %s\n", processed.Original)
		fmt.Printf("Enhanced: %s\n", processed.Enhanced)
		fmt.Printf("Type: %s\n\n", queryTypeString(processed.QueryType))
	}
	
	// Update binary search setting
	if options.SearchBinaries {
		sbs.engine.ignoreFilter.searchBinaries = true
	}
	
	// Build index
	if !options.NoIndex {
		if err := sbs.engine.IndexDirectory(".", options); err != nil {
			return fmt.Errorf("indexing failed: %w", err)
		}
	}
	
	// Search
	matches, err := sbs.engine.Search(processed.Enhanced, options.Limit, options.Threshold)
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}
	
	// Display results
	sbs.displayResults(matches, options)
	
	return nil
}

// displayResults shows search results
func (sbs *SimpleBedSearcher) displayResults(matches []SearchMatch, options BedSearchOptions) {
	if len(matches) == 0 {
		fmt.Println("No results found")
		return
	}
	
	fmt.Printf("Found %d result(s)\n\n", len(matches))
	
	for i, match := range matches {
		doc := match.Document
		
		// Handle binary files
		if doc.IsBinary {
			fmt.Printf("%s: Binary file (similarity: %.3f)\n", doc.Path, match.Similarity)
			continue
		}
		
		// Format like ripgrep
		if shouldUseColor("auto") {
			fmt.Printf("\033[35m%s\033[0m:\033[32m%d\033[0m: %s\n",
				doc.Path, doc.LineNumber, doc.Content)
		} else {
			fmt.Printf("%s:%d: %s\n", doc.Path, doc.LineNumber, doc.Content)
		}
		
		if options.Verbose {
			fmt.Printf("  [Similarity: %.3f]\n", match.Similarity)
		}
		
		if i < len(matches)-1 {
			fmt.Println()
		}
	}
}

// Close releases resources
func (sbs *SimpleBedSearcher) Close() error {
	// Cleanup if needed
	return nil
}