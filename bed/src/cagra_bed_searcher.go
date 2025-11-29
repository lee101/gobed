//go:build cagra

package src

import (
	"crypto/sha1"
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
	model     *gobed.Int8EmbeddingModel512
	index     *gobed.CAGRAIndex
	docs      []CAGRADoc
	verbose   bool
	cache     *cagraCacheManager
	config    *Config
	basePath  string
	cagraConf gobed.CAGRAConfig
}

func NewCAGRABedSearcher() (*CAGRABedSearcher, error) {
	model, err := gobed.LoadInt8Model512()
	if err != nil {
		return nil, fmt.Errorf("failed to load int8 model: %w", err)
	}

	cfg, cfgErr := LoadConfig()
	if cfgErr != nil {
		cfg = DefaultConfig()
	}

	basePath, err := os.Getwd()
	if err != nil {
		basePath = "."
	}
	basePath, _ = filepath.Abs(basePath)

	projectHash := fmt.Sprintf("%x", sha1.Sum([]byte(basePath)))
	cacheDir := filepath.Join(cfg.CacheDir, "cagra", projectHash)
	cagraConf := gobed.DefaultCAGRAConfig()
	cagraConf.VectorDim = gobed.Int8EmbeddingDim
	cacheMgr, err := newCAGRACacheManager(cacheDir)
	if err != nil {
		gobed.Debugf("Failed to initialize CAGRA cache manager: %v", err)
		cacheMgr = nil
		cagraConf.CachePath = ""
	} else {
		cagraConf.CachePath = filepath.Join(cacheDir, "graph.cagrabin")
	}

	idx, err := gobed.NewCAGRAIndex(cagraConf)
	if err != nil {
		return nil, err
	}
	return &CAGRABedSearcher{
		model:     model,
		index:     idx,
		docs:      make([]CAGRADoc, 0),
		cache:     cacheMgr,
		config:    cfg,
		basePath:  basePath,
		cagraConf: cagraConf,
	}, nil
}

// Search mirrors the SimpleBedSearcher API but uses CAGRA on GPU.
func (c *CAGRABedSearcher) Search(options BedSearchOptions) error {
	c.verbose = options.Verbose
	if err := c.ensureIndex(".", options); err != nil {
		return fmt.Errorf("indexing failed: %w", err)
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

func (c *CAGRABedSearcher) ensureIndex(path string, options BedSearchOptions) error {
	if err := c.ensureBasePath(path); err != nil {
		return err
	}

	if options.NoIndex {
		return c.loadCachedIndex(options)
	}
	return c.buildIndex(c.basePath, options)
}

func (c *CAGRABedSearcher) ensureBasePath(path string) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("failed to resolve path %s: %w", path, err)
	}

	if c.basePath == absPath {
		return nil
	}

	projectHash := fmt.Sprintf("%x", sha1.Sum([]byte(absPath)))
	cacheDir := filepath.Join(c.config.CacheDir, "cagra", projectHash)
	cacheMgr, err := newCAGRACacheManager(cacheDir)
	if err != nil {
		gobed.Debugf("Failed to reinitialize CAGRA cache manager: %v", err)
		c.cache = nil
		c.cagraConf.CachePath = ""
	} else {
		c.cache = cacheMgr
		c.cagraConf.CachePath = filepath.Join(cacheDir, "graph.cagrabin")
	}

	c.basePath = absPath

	if c.index != nil {
		c.index.Close()
	}
	idx, err := gobed.NewCAGRAIndex(c.cagraConf)
	if err != nil {
		return fmt.Errorf("failed to create CAGRA index: %w", err)
	}
	c.index = idx
	return nil
}

func (c *CAGRABedSearcher) loadCachedIndex(options BedSearchOptions) error {
	if c.cache == nil {
		return fmt.Errorf("GPU cache is not available")
	}

	entries, err := c.cache.LoadAll()
	if err != nil {
		return fmt.Errorf("failed to load GPU cache: %w", err)
	}
	if len(entries) == 0 {
		return fmt.Errorf("no cached GPU index found; run without --no-index first")
	}

	vecs := make([]simdpkg.Vec512, 0)
	scales := make([]float32, 0)
	docs := make([]CAGRADoc, 0)

	for _, entry := range entries {
		entryDocs, entryVecs, entryScales := entry.ToDocuments(c.basePath)
		if !options.SearchBinaries {
			entryDocs, entryVecs, entryScales = filterBinaryDocs(entryDocs, entryVecs, entryScales)
		}
		if len(entryVecs) == 0 {
			continue
		}
		vecs = append(vecs, entryVecs...)
		scales = append(scales, entryScales...)
		docs = append(docs, entryDocs...)
	}

	if len(vecs) == 0 {
		return fmt.Errorf("cached GPU index contains no usable data; rebuild the index")
	}

	if err := c.rebuildIndex(vecs, scales); err != nil {
		return err
	}
	c.docs = docs
	return nil
}

// indexDirectory builds the CAGRA index from the current working tree.
func (c *CAGRABedSearcher) buildIndex(path string, options BedSearchOptions) error {
	// Reuse the enhanced ignore filter for IO hygiene
	maxFileSize := int64(10 * 1024 * 1024)
	if c.config != nil && c.config.MaxFileSize > 0 {
		maxFileSize = c.config.MaxFileSize
	}
	filter, err := NewEnhancedIgnoreFilter(path,
		WithMaxFileSize(maxFileSize),
		WithBinarySearch(options.SearchBinaries),
	)
	if err != nil {
		return err
	}

	type workItem struct {
		absPath  string
		relPath  string
		lines    []string
		isBinary bool
		info     os.FileInfo
	}
	type workResult struct {
		absPath   string
		relPath   string
		docs      []CAGRADoc
		vecs      []simdpkg.Vec512
		scales    []float32
		modTime   int64
		size      int64
		fromCache bool
	}

	workChan := make(chan workItem, 256)
	resultChan := make(chan workResult, 256)

	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}
	var workerWG sync.WaitGroup
	for w := 0; w < workers; w++ {
		workerWG.Add(1)
		go func() {
			defer workerWG.Done()
			for item := range workChan {
				lineDocs := make([]CAGRADoc, 0, len(item.lines))
				lineVecs := make([]simdpkg.Vec512, 0, len(item.lines))
				lineScales := make([]float32, 0, len(item.lines))

				for i, rawLine := range item.lines {
					if !item.isBinary && strings.TrimSpace(rawLine) == "" {
						continue
					}
					r, err := c.model.EmbedInt8(rawLine)
					if err != nil {
						continue
					}
					var v simdpkg.Vec512
					copy(v[:], r.Vector)
					display := rawLine
					if len(display) > 240 {
						display = display[:240] + "..."
					}
					lineNumber := i + 1
					if item.isBinary {
						lineNumber = 0
					}
					lineDocs = append(lineDocs, CAGRADoc{
						Path:       item.absPath,
						LineNumber: lineNumber,
						Content:    display,
						IsBinary:   item.isBinary,
					})
					lineVecs = append(lineVecs, v)
					lineScales = append(lineScales, r.Scale)
				}

				if len(lineVecs) == 0 {
					continue
				}

				resultChan <- workResult{
					absPath:   item.absPath,
					relPath:   item.relPath,
					docs:      lineDocs,
					vecs:      lineVecs,
					scales:    lineScales,
					modTime:   item.info.ModTime().UnixNano(),
					size:      item.info.Size(),
					fromCache: false,
				}
			}
		}()
	}

	go func() {
		filepath.WalkDir(path, func(p string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}

			ok, fileType := filter.ShouldProcess(p)
			if !ok {
				if c.verbose && fileType == FileTypeTooLarge {
					fmt.Printf("Skipping large file: %s\n", p)
				}
				return nil
			}

			info, err := d.Info()
			if err != nil {
				return nil
			}

			relPath, err := filepath.Rel(path, p)
			if err != nil {
				relPath = p
			}

			if c.cache != nil {
				if entry, ok := c.cache.TryLoad(relPath, info); ok {
					entryDocs, entryVecs, entryScales := entry.ToDocuments(path)
					if !options.SearchBinaries {
						entryDocs, entryVecs, entryScales = filterBinaryDocs(entryDocs, entryVecs, entryScales)
					}
					if len(entryVecs) > 0 {
						resultChan <- workResult{
							absPath:   filepath.Join(path, relPath),
							relPath:   relPath,
							docs:      entryDocs,
							vecs:      entryVecs,
							scales:    entryScales,
							modTime:   entry.ModTime,
							size:      entry.Size,
							fromCache: true,
						}
					}
					return nil
				}
			}

			if fileType == FileTypeBinary {
				if !options.SearchBinaries {
					return nil
				}
				summary := fmt.Sprintf("%s binary file", filepath.Base(p))
				workChan <- workItem{
					absPath:  p,
					relPath:  relPath,
					lines:    []string{summary},
					isBinary: true,
					info:     info,
				}
				return nil
			}

			data, err := os.ReadFile(p)
			if err != nil {
				return nil
			}
			lines := strings.Split(string(data), "\n")
			workChan <- workItem{
				absPath:  p,
				relPath:  relPath,
				lines:    lines,
				isBinary: false,
				info:     info,
			}
			return nil
		})
		close(workChan)
		workerWG.Wait()
		close(resultChan)
	}()

	vecs := make([]simdpkg.Vec512, 0, 8192)
	scales := make([]float32, 0, 8192)
	docs := make([]CAGRADoc, 0, 8192)

	type fcount struct {
		path string
		n    int
	}
	counts := map[string]int{}

	var collectErr error
	for res := range resultChan {
		if len(res.vecs) == 0 {
			continue
		}
		vecs = append(vecs, res.vecs...)
		scales = append(scales, res.scales...)
		docs = append(docs, res.docs...)

		for _, doc := range res.docs {
			counts[doc.Path]++
		}

		if !res.fromCache && c.cache != nil {
			cacheDocs := make([]cagraCachedDoc, len(res.docs))
			for i, doc := range res.docs {
				cacheDocs[i] = cagraCachedDoc{
					LineNumber: doc.LineNumber,
					Content:    doc.Content,
					IsBinary:   doc.IsBinary,
					Vector:     copyVec(res.vecs[i]),
					Scale:      res.scales[i],
				}
			}
			entry := &cagraFileCache{
				RelPath: res.relPath,
				ModTime: res.modTime,
				Size:    res.size,
				Docs:    cacheDocs,
			}
			if err := c.cache.Save(entry); err != nil && collectErr == nil {
				collectErr = err
			}
		}
	}

	if collectErr != nil {
		return collectErr
	}

	if len(vecs) == 0 {
		return fmt.Errorf("no data to index")
	}

	if err := c.rebuildIndex(vecs, scales); err != nil {
		return err
	}

	c.docs = docs

	if c.cache != nil {
		_ = c.cache.Cleanup()
	}

	if c.verbose {
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

func (c *CAGRABedSearcher) rebuildIndex(vecs []simdpkg.Vec512, scales []float32) error {
	if c.index != nil {
		c.index.Close()
	}
	idx, err := gobed.NewCAGRAIndex(c.cagraConf)
	if err != nil {
		return err
	}
	if err := idx.BuildIndex(vecs, scales); err != nil {
		idx.Close()
		return err
	}
	c.index = idx
	return nil
}

func filterBinaryDocs(docs []CAGRADoc, vecs []simdpkg.Vec512, scales []float32) ([]CAGRADoc, []simdpkg.Vec512, []float32) {
	if len(docs) == 0 {
		return docs, vecs, scales
	}
	filteredDocs := make([]CAGRADoc, 0, len(docs))
	filteredVecs := make([]simdpkg.Vec512, 0, len(vecs))
	filteredScales := make([]float32, 0, len(scales))
	for i, doc := range docs {
		if doc.IsBinary {
			continue
		}
		filteredDocs = append(filteredDocs, doc)
		filteredVecs = append(filteredVecs, vecs[i])
		filteredScales = append(filteredScales, scales[i])
	}
	return filteredDocs, filteredVecs, filteredScales
}
