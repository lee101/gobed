package src

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lee101/gobed"
)

const (
	fastIndexMagic   = 0xFA57BED1
	fastIndexVersion = 2 // v2: separate metadata/embeddings/norms sections
	cacheDir         = ".bed"
	cacheFile        = "fast_index.bin"
)

// FastBedSearcher uses the optimized int8 model for fast semantic search
type FastBedSearcher struct {
	model            fastEmbeddingModel
	documents        []Document
	embeddings       [][]float32
	precomputedNorms []float32
	precomputedInv   []float32
	mu               sync.RWMutex
	ignoreFilter     *EnhancedIgnoreFilter
	queryProcessor   *QueryProcessor
	verbose          bool
	indexed          int64

	minLineLength  int
	maxLineLength  int
	ignoreLongLine bool
}

type fastEmbeddingModel interface {
	EmbedFast(string) ([]float32, func())
	Close() error
}

type fastSearcherOptions struct {
	maxFileSize     int64
	searchBinaries  bool
	minLineLength   int
	maxLineLength   int
	ignoreLongLines bool
}

func defaultFastSearcherOptions() fastSearcherOptions {
	return fastSearcherOptions{
		maxFileSize:     10 * 1024 * 1024,
		searchBinaries:  false,
		minLineLength:   3,
		maxLineLength:   1200,
		ignoreLongLines: true,
	}
}

func fastSearcherOptionsFromConfig(cfg *Config) fastSearcherOptions {
	opts := defaultFastSearcherOptions()
	if cfg == nil {
		return opts
	}

	if cfg.MaxFileSize > 0 {
		opts.maxFileSize = cfg.MaxFileSize
	}
	opts.searchBinaries = cfg.SearchBinaries
	if cfg.MinLineLength > 0 {
		opts.minLineLength = cfg.MinLineLength
	}
	if cfg.MaxLineLength > 0 {
		opts.maxLineLength = cfg.MaxLineLength
	}
	opts.ignoreLongLines = cfg.IgnoreLongLines
	return opts
}

// NewFastBedSearcher creates a new fast searcher using int8 model
func NewFastBedSearcher() (*FastBedSearcher, error) {
	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		return nil, fmt.Errorf("failed to load int8 model: %w", err)
	}

	baseDir, err := os.Getwd()
	if err != nil {
		baseDir = "."
	}
	cfg, _ := LoadConfig()
	return newFastBedSearcherWithModel(model, baseDir, fastSearcherOptionsFromConfig(cfg))
}

func newFastBedSearcherWithModel(model fastEmbeddingModel, baseDir string, opts fastSearcherOptions) (*FastBedSearcher, error) {
	ignoreFilter, err := NewEnhancedIgnoreFilter(baseDir,
		WithMaxFileSize(opts.maxFileSize),
		WithBinarySearch(opts.searchBinaries),
	)
	if err != nil {
		return nil, err
	}

	return &FastBedSearcher{
		model:            model,
		documents:        make([]Document, 0, 10000),
		embeddings:       make([][]float32, 0, 10000),
		precomputedNorms: make([]float32, 0, 10000),
		precomputedInv:   make([]float32, 0, 10000),
		ignoreFilter:     ignoreFilter,
		queryProcessor:   NewQueryProcessor(),
		minLineLength:    opts.minLineLength,
		maxLineLength:    opts.maxLineLength,
		ignoreLongLine:   opts.ignoreLongLines,
	}, nil
}

type lineInfo struct {
	num     int
	content string
}

type workItem struct {
	path  string
	lines []lineInfo
}

type scored struct {
	idx   int
	score float32
}

// IndexDirectory indexes all files using the fast int8 model.
// Existing in-memory entries are replaced.
func (fs *FastBedSearcher) IndexDirectory(path string, options BedSearchOptions) error {
	return fs.indexDirectory(path, options, false)
}

// IndexDirectoryAppend indexes an additional directory and appends it to the current in-memory index.
func (fs *FastBedSearcher) IndexDirectoryAppend(path string, options BedSearchOptions) error {
	return fs.indexDirectory(path, options, true)
}

func (fs *FastBedSearcher) indexDirectory(path string, options BedSearchOptions, appendMode bool) error {
	startTime := time.Now()

	// Try loading cached index first for single-path full index builds.
	if !appendMode && !options.ForceIndex && fs.tryLoadCachedIndex(path, options.Verbose) {
		return nil
	}

	if !appendMode {
		fs.resetIndex()
	}

	workChan := make(chan workItem, 256)
	numWorkers := runtime.NumCPU()

	var wg sync.WaitGroup

	type embedResult struct {
		docs  []Document
		embs  [][]float32
		norms []float32
	}
	resultChan := make(chan embedResult, 1000)

	// File walker
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(workChan)

		_ = filepath.WalkDir(path, func(filePath string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}

			normalizedPath := normalizeIndexPath(filePath)
			shouldProcess, fileType := fs.ignoreFilter.ShouldProcess(normalizedPath)
			if !shouldProcess {
				if fileType == FileTypeBinary && options.SearchBinaries {
					workChan <- workItem{
						path: normalizedPath,
						lines: []lineInfo{
							{
								num:     1,
								content: fmt.Sprintf("%s binary file", filepath.Base(normalizedPath)),
							},
						},
					}
				}
				return nil
			}
			if fileType == FileTypeBinary && !options.SearchBinaries {
				return nil
			}

			content, err := os.ReadFile(normalizedPath)
			if err != nil {
				return nil
			}

			lineInfos := fs.buildLineInfos(content)

			if len(lineInfos) > 0 {
				workChan <- workItem{path: normalizedPath, lines: lineInfos}
			}
			return nil
		})
	}()

	// Embedding workers using zero-alloc path
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range workChan {
				docs := make([]Document, 0, len(item.lines))
				embs := make([][]float32, 0, len(item.lines))
				norms := make([]float32, 0, len(item.lines))

				for _, line := range item.lines {
					emb, release := fs.model.EmbedFast(line.content)

					embCopy := make([]float32, len(emb))
					var normSq float32
					for j, v := range emb {
						embCopy[j] = v
						normSq += v * v
					}
					release()

					docs = append(docs, Document{
						Path:       item.path,
						LineNumber: line.num,
						Content:    line.content,
					})
					embs = append(embs, embCopy)
					norms = append(norms, normSq)
				}

				if len(docs) > 0 {
					resultChan <- embedResult{
						docs:  docs,
						embs:  embs,
						norms: norms,
					}
					atomic.AddInt64(&fs.indexed, int64(len(docs)))
				}
			}
		}()
	}

	// Result collector
	done := make(chan struct{})
	go func() {
		for result := range resultChan {
			fs.mu.Lock()
			fs.appendDocumentsLocked(result.docs, result.embs, result.norms)
			fs.mu.Unlock()
		}
		close(done)
	}()

	wg.Wait()
	close(resultChan)
	<-done

	elapsed := time.Since(startTime)
	if options.Verbose {
		fs.mu.RLock()
		docCount := len(fs.documents)
		fs.mu.RUnlock()
		fmt.Printf("Indexed %d lines in %.2fs (%.0f lines/sec)\n",
			docCount, elapsed.Seconds(), float64(docCount)/elapsed.Seconds())
	}

	// Save cache only for non-append mode to keep a stable single-root cache file.
	if !appendMode {
		saveStart := time.Now()
		if err := fs.saveCachedIndex(path); err != nil {
			if options.Verbose {
				fmt.Printf("Warning: failed to save index cache: %v\n", err)
			}
		} else if options.Verbose {
			fmt.Printf("Saved index cache in %.2fs\n", time.Since(saveStart).Seconds())
		}
	}

	return nil
}

// Search performs fast semantic search with pre-computed norms
func (fs *FastBedSearcher) Search(options BedSearchOptions) error {
	matches, err := fs.SearchMatches(options)
	if err != nil {
		return err
	}

	fs.mu.RLock()
	hasDocs := len(fs.documents) > 0
	fs.mu.RUnlock()
	if !hasDocs {
		fmt.Println("No documents indexed")
		return nil
	}

	fs.displayResults(matches, options)
	return nil
}

// SearchMatches performs fast semantic search and returns matches for programmatic use.
func (fs *FastBedSearcher) SearchMatches(options BedSearchOptions) ([]SearchMatch, error) {
	fs.verbose = options.Verbose
	if options.Limit <= 0 {
		options.Limit = 10
	}

	// Process query - use original query, not enhanced (enhanced hurts embedding quality)
	queryText := options.Query
	if options.Verbose {
		fmt.Printf("Query: %s\n", queryText)
	}

	// Index if needed
	if !options.NoIndex {
		if err := fs.IndexDirectory(".", options); err != nil {
			return nil, err
		}
	}

	// Embed query using zero-alloc path
	qEmb, release := fs.model.EmbedFast(queryText)
	defer release()

	// Pre-compute query norm
	var queryNormSq float32
	for _, v := range qEmb {
		queryNormSq += v * v
	}
	if queryNormSq <= 0 {
		return nil, nil
	}
	invQueryNorm := fastInvSqrtSafe(queryNormSq)

	fs.mu.RLock()
	defer fs.mu.RUnlock()

	// Parallel search with pre-computed norms
	numDocs := len(fs.documents)
	if numDocs == 0 {
		return nil, nil
	}

	// Use parallel workers for large corpora
	numWorkers := runtime.NumCPU()
	chunkSize := (numDocs + numWorkers - 1) / numWorkers

	resultChans := make([]chan scored, numWorkers)
	for i := range resultChans {
		// Buffer large enough for worst case (all matches above threshold)
		// In practice, most workers find few matches
		resultChans[i] = make(chan scored, options.Limit*3+10)
	}

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			defer close(resultChans[workerID])

			start := workerID * chunkSize
			end := start + chunkSize
			if end > numDocs {
				end = numDocs
			}

			// Keep top-k per worker
			topK := make([]scored, 0, options.Limit+1)

			for i := start; i < end; i++ {
				docEmb := fs.embeddings[i]
				docInv := fs.precomputedInv[i]
				if docInv <= 0 {
					continue
				}

				// Compute dot product (unrolled for 512 dims)
				var dot float32
				for j := 0; j < 512; j += 8 {
					dot += qEmb[j]*docEmb[j] + qEmb[j+1]*docEmb[j+1] +
						qEmb[j+2]*docEmb[j+2] + qEmb[j+3]*docEmb[j+3] +
						qEmb[j+4]*docEmb[j+4] + qEmb[j+5]*docEmb[j+5] +
						qEmb[j+6]*docEmb[j+6] + qEmb[j+7]*docEmb[j+7]
				}

				// Cosine similarity using pre-computed inverse norms.
				sim := dot * invQueryNorm * docInv

				if sim >= options.Threshold {
					topK = append(topK, scored{idx: i, score: sim})
					// Keep only top-k
					if len(topK) > options.Limit*2 {
						partialSort(topK, options.Limit)
						topK = topK[:options.Limit]
					}
				}
			}

			// Final trim to limit
			if len(topK) > options.Limit {
				partialSort(topK, options.Limit)
				topK = topK[:options.Limit]
			}

			for _, s := range topK {
				resultChans[workerID] <- s
			}
		}(w)
	}

	wg.Wait()

	// Merge results from all workers
	allResults := make([]scored, 0, numWorkers*options.Limit)
	for _, ch := range resultChans {
		for s := range ch {
			allResults = append(allResults, s)
		}
	}

	// Final sort
	partialSort(allResults, options.Limit)
	if len(allResults) > options.Limit {
		allResults = allResults[:options.Limit]
	}

	matches := make([]SearchMatch, len(allResults))
	for i, r := range allResults {
		matches[i] = SearchMatch{
			Document:   fs.documents[r.idx],
			Similarity: r.score,
		}
	}

	return matches, nil
}

// NumDocuments returns number of indexed documents.
func (fs *FastBedSearcher) NumDocuments() int {
	fs.mu.RLock()
	defer fs.mu.RUnlock()
	return len(fs.documents)
}

// IndexPaths builds a single in-memory index for multiple roots.
func (fs *FastBedSearcher) IndexPaths(paths []string, options BedSearchOptions) error {
	if len(paths) == 0 {
		return nil
	}

	for i, p := range paths {
		if i == 0 {
			if err := fs.IndexDirectory(p, options); err != nil {
				return err
			}
			continue
		}

		appendOpts := options
		appendOpts.ForceIndex = true // append mode never uses cache.
		if err := fs.IndexDirectoryAppend(p, appendOpts); err != nil {
			return err
		}
	}

	return nil
}

// UpsertFile removes any existing entries for path, then re-indexes it if eligible.
func (fs *FastBedSearcher) UpsertFile(path string, options BedSearchOptions) error {
	normalizedPath := normalizeIndexPath(path)

	lines, err := fs.collectFileLines(normalizedPath, options.SearchBinaries)
	if err != nil {
		if os.IsNotExist(err) {
			fs.RemoveFile(normalizedPath)
			return nil
		}
		return err
	}

	// No content to index for this file type/state: remove stale entries.
	if len(lines) == 0 {
		fs.RemoveFile(normalizedPath)
		return nil
	}

	docs := make([]Document, 0, len(lines))
	embeddings := make([][]float32, 0, len(lines))
	norms := make([]float32, 0, len(lines))

	for _, line := range lines {
		emb, release := fs.model.EmbedFast(line.content)
		embCopy := make([]float32, len(emb))
		var normSq float32
		for j, v := range emb {
			embCopy[j] = v
			normSq += v * v
		}
		release()

		docs = append(docs, Document{
			Path:       normalizedPath,
			LineNumber: line.num,
			Content:    line.content,
		})
		embeddings = append(embeddings, embCopy)
		norms = append(norms, normSq)
	}

	fs.mu.Lock()
	defer fs.mu.Unlock()

	fs.removeFileLocked(normalizedPath)
	fs.appendDocumentsLocked(docs, embeddings, norms)
	return nil
}

// RemoveFile removes all indexed entries for the file.
func (fs *FastBedSearcher) RemoveFile(path string) {
	normalizedPath := normalizeIndexPath(path)
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.removeFileLocked(normalizedPath)
}

func (fs *FastBedSearcher) appendDocumentsLocked(docs []Document, embeddings [][]float32, norms []float32) {
	for i := range docs {
		docs[i].ID = len(fs.documents)
		fs.documents = append(fs.documents, docs[i])
		fs.embeddings = append(fs.embeddings, embeddings[i])
		fs.precomputedNorms = append(fs.precomputedNorms, norms[i])
		fs.precomputedInv = append(fs.precomputedInv, fastInvSqrtSafe(norms[i]))
	}
}

func (fs *FastBedSearcher) removeFileLocked(path string) int {
	if len(fs.documents) == 0 {
		return 0
	}

	writeIdx := 0
	for readIdx := range fs.documents {
		if fs.documents[readIdx].Path == path {
			continue
		}

		if writeIdx != readIdx {
			fs.documents[writeIdx] = fs.documents[readIdx]
			fs.embeddings[writeIdx] = fs.embeddings[readIdx]
			fs.precomputedNorms[writeIdx] = fs.precomputedNorms[readIdx]
			fs.precomputedInv[writeIdx] = fs.precomputedInv[readIdx]
		}
		fs.documents[writeIdx].ID = writeIdx
		writeIdx++
	}

	removed := len(fs.documents) - writeIdx
	if removed <= 0 {
		return 0
	}

	fs.documents = fs.documents[:writeIdx]
	fs.embeddings = fs.embeddings[:writeIdx]
	fs.precomputedNorms = fs.precomputedNorms[:writeIdx]
	fs.precomputedInv = fs.precomputedInv[:writeIdx]
	return removed
}

func (fs *FastBedSearcher) resetIndex() {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.documents = fs.documents[:0]
	fs.embeddings = fs.embeddings[:0]
	fs.precomputedNorms = fs.precomputedNorms[:0]
	fs.precomputedInv = fs.precomputedInv[:0]
	atomic.StoreInt64(&fs.indexed, 0)
}

func normalizeIndexPath(path string) string {
	if abs, err := filepath.Abs(path); err == nil {
		return filepath.Clean(abs)
	}
	return filepath.Clean(path)
}

func (fs *FastBedSearcher) collectFileLines(path string, searchBinaries bool) ([]lineInfo, error) {
	shouldProcess, fileType := fs.ignoreFilter.ShouldProcess(path)
	if !shouldProcess {
		if fileType == FileTypeBinary && searchBinaries {
			return []lineInfo{
				{
					num:     1,
					content: fmt.Sprintf("%s binary file", filepath.Base(path)),
				},
			}, nil
		}
		return nil, nil
	}

	// Keep binary handling deterministic: include a synthetic line when enabled.
	if fileType == FileTypeBinary {
		if !searchBinaries {
			return nil, nil
		}
		return []lineInfo{
			{
				num:     1,
				content: fmt.Sprintf("%s binary file", filepath.Base(path)),
			},
		}, nil
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	return fs.buildLineInfos(content), nil
}

func (fs *FastBedSearcher) buildLineInfos(content []byte) []lineInfo {
	if len(content) == 0 {
		return nil
	}

	estimatedLines := bytes.Count(content, []byte{'\n'}) + 1
	lines := make([]lineInfo, 0, estimatedLines)

	lineStart := 0
	lineNum := 1
	for i := 0; i <= len(content); i++ {
		if i < len(content) && content[i] != '\n' {
			continue
		}

		rawLine := content[lineStart:i]
		lineStart = i + 1

		trimmed := bytes.TrimSpace(rawLine)
		if len(trimmed) < fs.minLineLength {
			lineNum++
			continue
		}

		if fs.maxLineLength > 0 && len(trimmed) > fs.maxLineLength {
			if fs.ignoreLongLine {
				lineNum++
				continue
			}
			trimmed = trimmed[:fs.maxLineLength]
		}

		lines = append(lines, lineInfo{
			num:     lineNum,
			content: string(trimmed),
		})
		lineNum++
	}

	return lines
}

func (fs *FastBedSearcher) displayResults(results []SearchMatch, options BedSearchOptions) {
	if len(results) == 0 {
		fmt.Println("No results found")
		return
	}

	fmt.Printf("Found %d result(s)\n\n", len(results))

	for i, r := range results {
		doc := r.Document

		if shouldUseColor("auto") {
			fmt.Printf("\033[35m%s\033[0m:\033[32m%d\033[0m: %s\n",
				doc.Path, doc.LineNumber, truncate(doc.Content, 200))
		} else {
			fmt.Printf("%s:%d: %s\n", doc.Path, doc.LineNumber, truncate(doc.Content, 200))
		}

		if options.Verbose {
			fmt.Printf("  [Similarity: %.3f]\n", r.Similarity)
		}

		if i < len(results)-1 && options.Context > 0 {
			fmt.Println()
		}
	}
}

func (fs *FastBedSearcher) Close() error {
	return fs.model.Close()
}

// Helper functions

func fastSqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func fastInvSqrtSafe(x float32) float32 {
	if x <= 0 {
		return 0
	}
	return 1.0 / fastSqrt(x)
}

func partialSort(s []scored, k int) {
	// Simple selection sort for top-k (efficient when k is small)
	n := len(s)
	if k > n {
		k = n
	}
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < n; j++ {
			if s[j].score > s[maxIdx].score {
				maxIdx = j
			}
		}
		s[i], s[maxIdx] = s[maxIdx], s[i]
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// saveIndex saves the index to disk for fast loading
func (fs *FastBedSearcher) saveIndex(basePath string) error {
	if err := os.MkdirAll(filepath.Join(basePath, cacheDir), 0755); err != nil {
		return err
	}

	path := filepath.Join(basePath, cacheDir, cacheFile)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 4<<20) // 4MB buffer

	// Header
	header := make([]byte, 16)
	binary.LittleEndian.PutUint32(header[0:], fastIndexMagic)
	binary.LittleEndian.PutUint32(header[4:], fastIndexVersion)
	binary.LittleEndian.PutUint32(header[8:], uint32(len(fs.documents)))
	binary.LittleEndian.PutUint32(header[12:], 512) // embedding dim
	w.Write(header)

	// Write all document metadata first
	for _, doc := range fs.documents {
		// Path (len + bytes)
		pathBytes := []byte(doc.Path)
		binary.Write(w, binary.LittleEndian, uint16(len(pathBytes)))
		w.Write(pathBytes)

		// Line number
		binary.Write(w, binary.LittleEndian, uint32(doc.LineNumber))

		// Content (len + bytes)
		contentBytes := []byte(doc.Content)
		binary.Write(w, binary.LittleEndian, uint16(len(contentBytes)))
		w.Write(contentBytes)
	}

	// Write all embeddings as contiguous binary blob
	embBuf := make([]byte, 4)
	for i := range fs.embeddings {
		emb := fs.embeddings[i]
		for j := 0; j < 512; j++ {
			binary.LittleEndian.PutUint32(embBuf, math.Float32bits(emb[j]))
			w.Write(embBuf)
		}
	}

	// Write all norms
	for _, norm := range fs.precomputedNorms {
		binary.LittleEndian.PutUint32(embBuf, math.Float32bits(norm))
		w.Write(embBuf)
	}

	return w.Flush()
}

// loadIndex loads the index from disk
func (fs *FastBedSearcher) loadIndex(basePath string) error {
	path := filepath.Join(basePath, cacheDir, cacheFile)
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 4<<20) // 4MB buffer

	// Header
	header := make([]byte, 16)
	if _, err := io.ReadFull(r, header); err != nil {
		return err
	}
	magic := binary.LittleEndian.Uint32(header[0:])
	version := binary.LittleEndian.Uint32(header[4:])
	numDocs := binary.LittleEndian.Uint32(header[8:])
	embDim := binary.LittleEndian.Uint32(header[12:])

	if magic != fastIndexMagic {
		return fmt.Errorf("invalid index magic")
	}
	if version != fastIndexVersion {
		return fmt.Errorf("index version mismatch")
	}

	fs.documents = make([]Document, numDocs)
	fs.embeddings = make([][]float32, numDocs)
	fs.precomputedNorms = make([]float32, numDocs)
	fs.precomputedInv = make([]float32, numDocs)

	pathBuf := make([]byte, 1024)
	contentBuf := make([]byte, 4096)

	// Read all document metadata
	for i := uint32(0); i < numDocs; i++ {
		fs.documents[i].ID = int(i)

		// Path
		var pathLen uint16
		binary.Read(r, binary.LittleEndian, &pathLen)
		if int(pathLen) > len(pathBuf) {
			pathBuf = make([]byte, pathLen)
		}
		io.ReadFull(r, pathBuf[:pathLen])
		fs.documents[i].Path = string(pathBuf[:pathLen])

		// Line number
		var lineNum uint32
		binary.Read(r, binary.LittleEndian, &lineNum)
		fs.documents[i].LineNumber = int(lineNum)

		// Content
		var contentLen uint16
		binary.Read(r, binary.LittleEndian, &contentLen)
		if int(contentLen) > len(contentBuf) {
			contentBuf = make([]byte, contentLen)
		}
		io.ReadFull(r, contentBuf[:contentLen])
		fs.documents[i].Content = string(contentBuf[:contentLen])
	}

	// Read all embeddings as contiguous block
	embDataSize := int(numDocs) * int(embDim) * 4
	embData := make([]byte, embDataSize)
	if _, err := io.ReadFull(r, embData); err != nil {
		return err
	}

	// Create float32 slices pointing into the data
	allFloats := make([]float32, int(numDocs)*int(embDim))
	for i := range allFloats {
		allFloats[i] = math.Float32frombits(binary.LittleEndian.Uint32(embData[i*4:]))
	}
	for i := uint32(0); i < numDocs; i++ {
		start := int(i) * int(embDim)
		fs.embeddings[i] = allFloats[start : start+int(embDim)]
	}

	// Read all norms
	normData := make([]byte, numDocs*4)
	if _, err := io.ReadFull(r, normData); err != nil {
		return err
	}
	for i := uint32(0); i < numDocs; i++ {
		norm := math.Float32frombits(binary.LittleEndian.Uint32(normData[i*4:]))
		fs.precomputedNorms[i] = norm
		fs.precomputedInv[i] = fastInvSqrtSafe(norm)
	}

	return nil
}

// computeDirHash computes a quick hash of directory state (file paths + mod times)
func computeDirHash(basePath string, filter *EnhancedIgnoreFilter) (string, error) {
	h := sha256.New()
	root := normalizeIndexPath(basePath)
	cachePath := filepath.Join(root, cacheDir)
	filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		// Skip our own cache directory
		if strings.HasPrefix(p, cachePath) {
			return nil
		}
		shouldProcess, fileType := filter.ShouldProcess(p)
		if !shouldProcess || fileType == FileTypeBinary {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		h.Write([]byte(p))
		binary.Write(h, binary.LittleEndian, info.ModTime().Unix())
		return nil
	})
	return fmt.Sprintf("%x", h.Sum(nil)[:16]), nil
}

// tryLoadCachedIndex attempts to load cached index if valid
func (fs *FastBedSearcher) tryLoadCachedIndex(basePath string, verbose bool) bool {
	// Quick check: does cache file exist and have reasonable size?
	cachePath := filepath.Join(basePath, cacheDir, cacheFile)
	cacheInfo, err := os.Stat(cachePath)
	if err != nil || cacheInfo.Size() < 100 {
		return false
	}

	// Skip hash validation - just try to load. If directory changed significantly,
	// search quality will degrade and user can use --force-index
	loadStart := time.Now()
	if err := fs.loadIndex(basePath); err != nil {
		return false
	}

	if verbose {
		fmt.Printf("Loaded cached index: %d documents in %.2fs\n", len(fs.documents), time.Since(loadStart).Seconds())
	}
	return true
}

// saveCachedIndex saves index with directory hash
func (fs *FastBedSearcher) saveCachedIndex(basePath string) error {
	if err := fs.saveIndex(basePath); err != nil {
		return err
	}

	hash, err := computeDirHash(basePath, fs.ignoreFilter)
	if err != nil {
		return err
	}

	hashPath := filepath.Join(basePath, cacheDir, "hash")
	return os.WriteFile(hashPath, []byte(hash), 0644)
}

// use standard sqrt for better precision in final score
func sqrtf32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
