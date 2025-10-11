//go:build cagra

package src

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/lee101/gobed"
	simdpkg "github.com/lee101/gobed/pkg/ann/simd"
)

// CAGRADoc holds minimal metadata for a line-level document.
type CAGRADoc struct {
	Path       string
	LineNumber int
	Content    string
	IsBinary   bool
}

// CAGRABedSearcher builds an int8 embedding corpus and searches via cuVS CAGRA.
type CAGRABedSearcher struct {
	model   *gobed.Int8EmbeddingModel512
	index   *gobed.CAGRAIndex
	docs    []CAGRADoc
	verbose bool
}

func NewCAGRABedSearcher() (*CAGRABedSearcher, error) {
	model, err := gobed.LoadInt8Model512()
	if err != nil {
		return nil, fmt.Errorf("failed to load int8 model: %w", err)
	}
	// Create index with default config; will be built after collection
	cfg := gobed.DefaultCAGRAConfig()
	cfg.VectorDim = gobed.Int8EmbeddingDim
	// Disable cache by default for CLI runs; can be wired to config later
	cfg.CachePath = ""
	idx, err := gobed.NewCAGRAIndex(cfg)
	if err != nil {
		return nil, err
	}
	return &CAGRABedSearcher{model: model, index: idx, docs: make([]CAGRADoc, 0)}, nil
}

// Search mirrors the SimpleBedSearcher API but uses CAGRA on GPU.
func (c *CAGRABedSearcher) Search(options BedSearchOptions) error {
	c.verbose = options.Verbose
	// Index if needed
	if !options.NoIndex {
		if err := c.indexDirectory(".", options); err != nil {
			return fmt.Errorf("indexing failed: %w", err)
		}
	}

	// Tokenize+embed query to int8
	qr, err := c.model.EmbedInt8(options.Query)
	if err != nil {
		return fmt.Errorf("failed to embed query: %w", err)
	}
	var qv simdpkg.Vec512
	copy(qv[:], qr.Vector)

	// Run CAGRA search
	k := options.Limit
	if k <= 0 {
		k = 10
	}
	results, err := c.index.Search(qv, qr.Scale, k)
	if err != nil {
		return fmt.Errorf("cagra search failed: %w", err)
	}

	// Convert and display
	matches := make([]SearchMatch, 0, len(results))
	for _, r := range results {
		if r.ID < 0 || r.ID >= len(c.docs) {
			continue
		}
		d := c.docs[r.ID]
		matches = append(matches, SearchMatch{
			Document: Document{
				ID:         r.ID,
				Path:       d.Path,
				LineNumber: d.LineNumber,
				Content:    d.Content,
				IsBinary:   d.IsBinary,
			},
			Similarity: r.Similarity,
		})
	}

	// Keep same display as SimpleBedSearcher for consistency
	s := &SimpleBedSearcher{verbose: c.verbose}
	s.displayResults(matches, options)
	return nil
}

// indexDirectory builds the CAGRA index from the current working tree.
func (c *CAGRABedSearcher) indexDirectory(path string, options BedSearchOptions) error {
	// Reuse the enhanced ignore filter for IO hygiene
	filter, err := NewEnhancedIgnoreFilter(path,
		WithMaxFileSize(10*1024*1024), // 10MB
		WithBinarySearch(options.SearchBinaries),
	)
	if err != nil {
		return err
	}

	type workItem struct {
		path     string
		lines    []string
		isBinary bool
	}
	work := make(chan workItem, 256)
	var wg sync.WaitGroup

	// Collector of embeddings
	var mu sync.Mutex
	vecs := make([]simdpkg.Vec512, 0, 8192)
	scales := make([]float32, 0, 8192)
	docs := make([]CAGRADoc, 0, 8192)

	// Workers
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range work {
				for i, line := range item.lines {
					if strings.TrimSpace(line) == "" {
						continue
					}
					// Int8 embed
					r, err := c.model.EmbedInt8(line)
					if err != nil {
						continue
					}
					var v simdpkg.Vec512
					copy(v[:], r.Vector)
					// Capture metadata
					if len(line) > 240 {
						line = line[:240] + "..."
					}
					lineNumber := i + 1
					if item.isBinary {
						lineNumber = 0
					}
					mu.Lock()
					vecs = append(vecs, v)
					scales = append(scales, r.Scale)
					docs = append(docs, CAGRADoc{
						Path:       item.path,
						LineNumber: lineNumber,
						Content:    line,
						IsBinary:   item.isBinary,
					})
					mu.Unlock()
				}
			}
		}()
	}

	// Producer: walk files
	filepath.WalkDir(path, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		ok, fileType := filter.ShouldProcess(p)
		if !ok {
			return nil
		}
		if fileType == FileTypeBinary {
			if !options.SearchBinaries {
				return nil
			}
			summary := fmt.Sprintf("%s binary file", filepath.Base(p))
			work <- workItem{path: p, lines: []string{summary}, isBinary: true}
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		lines := strings.Split(string(b), "\n")
		work <- workItem{path: p, lines: lines}
		return nil
	})

	close(work)
	wg.Wait()

	// Build CAGRA index
	if len(vecs) == 0 {
		return fmt.Errorf("no data to index")
	}
	if err := c.index.BuildIndex(vecs, scales); err != nil {
		return err
	}

	// Save docs
	c.docs = docs

	if c.verbose {
		// Light summary
		// Sort top files by frequency just for signal
		type fcount struct {
			path string
			n    int
		}
		counts := map[string]int{}
		for _, d := range docs {
			counts[d.Path]++
		}
		top := make([]fcount, 0, len(counts))
		for k, v := range counts {
			top = append(top, fcount{k, v})
		}
		sort.Slice(top, func(i, j int) bool { return top[i].n > top[j].n })
		if len(top) > 5 {
			top = top[:5]
		}
		fmt.Printf("Indexed %d lines across %d files. Top files: \n", len(docs), len(counts))
		for _, t := range top {
			fmt.Printf("  %s (%d lines)\n", t.path, t.n)
		}
	}
	return nil
}

func (c *CAGRABedSearcher) Close() error { return nil }
