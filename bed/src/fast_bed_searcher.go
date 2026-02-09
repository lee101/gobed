package src

import (
	"bufio"
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
	model            *gobed.SimpleInt8Model512
	documents        []Document
	embeddings       [][]float32
	precomputedNorms []float32
	mu               sync.RWMutex
	ignoreFilter     *EnhancedIgnoreFilter
	queryProcessor   *QueryProcessor
	verbose          bool
	indexed          int64
}

// NewFastBedSearcher creates a new fast searcher using int8 model
func NewFastBedSearcher() (*FastBedSearcher, error) {
	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		return nil, fmt.Errorf("failed to load int8 model: %w", err)
	}

	ignoreFilter, err := NewEnhancedIgnoreFilter(".",
		WithMaxFileSize(10*1024*1024),
		WithBinarySearch(false),
	)
	if err != nil {
		return nil, err
	}

	return &FastBedSearcher{
		model:            model,
		documents:        make([]Document, 0, 10000),
		embeddings:       make([][]float32, 0, 10000),
		precomputedNorms: make([]float32, 0, 10000),
		ignoreFilter:     ignoreFilter,
		queryProcessor:   NewQueryProcessor(),
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

// IndexDirectory indexes all files using the fast int8 model
func (fs *FastBedSearcher) IndexDirectory(path string, options BedSearchOptions) error {
	startTime := time.Now()

	// Try loading cached index first (unless force reindex)
	if !options.ForceIndex && fs.tryLoadCachedIndex(path, options.Verbose) {
		return nil
	}

	workChan := make(chan workItem, 256)
	numWorkers := runtime.NumCPU()

	var wg sync.WaitGroup

	// Collect results in batches for efficiency
	type embedResult struct {
		doc  Document
		emb  []float32
		norm float32
	}
	resultChan := make(chan embedResult, 1000)

	// File walker
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(workChan)

		filepath.WalkDir(path, func(filePath string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}

			shouldProcess, fileType := fs.ignoreFilter.ShouldProcess(filePath)
			if !shouldProcess || fileType == FileTypeBinary {
				return nil
			}

			content, err := os.ReadFile(filePath)
			if err != nil {
				return nil
			}

			lines := strings.Split(string(content), "\n")
			lineInfos := make([]lineInfo, 0, len(lines))
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if trimmed == "" || len(trimmed) < 3 {
					continue
				}
				lineInfos = append(lineInfos, lineInfo{num: i + 1, content: line})
			}

			if len(lineInfos) > 0 {
				workChan <- workItem{path: filePath, lines: lineInfos}
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
				for _, line := range item.lines {
					emb, release := fs.model.EmbedFast(line.content)

					// Copy embedding and compute norm
					embCopy := make([]float32, len(emb))
					var normSq float32
					for j, v := range emb {
						embCopy[j] = v
						normSq += v * v
					}
					release()

					resultChan <- embedResult{
						doc: Document{
							Path:       item.path,
							LineNumber: line.num,
							Content:    line.content,
						},
						emb:  embCopy,
						norm: normSq,
					}
					atomic.AddInt64(&fs.indexed, 1)
				}
			}
		}()
	}

	// Result collector
	done := make(chan struct{})
	go func() {
		for result := range resultChan {
			fs.mu.Lock()
			result.doc.ID = len(fs.documents)
			fs.documents = append(fs.documents, result.doc)
			fs.embeddings = append(fs.embeddings, result.emb)
			fs.precomputedNorms = append(fs.precomputedNorms, result.norm)
			fs.mu.Unlock()
		}
		close(done)
	}()

	wg.Wait()
	close(resultChan)
	<-done

	elapsed := time.Since(startTime)
	if options.Verbose {
		fmt.Printf("Indexed %d lines in %.2fs (%.0f lines/sec)\n",
			len(fs.documents), elapsed.Seconds(),
			float64(len(fs.documents))/elapsed.Seconds())
	}

	// Save cache (synchronous to ensure it completes before program exits)
	saveStart := time.Now()
	if err := fs.saveCachedIndex(path); err != nil {
		if options.Verbose {
			fmt.Printf("Warning: failed to save index cache: %v\n", err)
		}
	} else if options.Verbose {
		fmt.Printf("Saved index cache in %.2fs\n", time.Since(saveStart).Seconds())
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
				docNormSq := fs.precomputedNorms[i]

				// Compute dot product (unrolled for 512 dims)
				var dot float32
				for j := 0; j < 512; j += 8 {
					dot += qEmb[j]*docEmb[j] + qEmb[j+1]*docEmb[j+1] +
						qEmb[j+2]*docEmb[j+2] + qEmb[j+3]*docEmb[j+3] +
						qEmb[j+4]*docEmb[j+4] + qEmb[j+5]*docEmb[j+5] +
						qEmb[j+6]*docEmb[j+6] + qEmb[j+7]*docEmb[j+7]
				}

				// Cosine similarity with pre-computed norms
				normProduct := queryNormSq * docNormSq
				if normProduct <= 0 {
					continue
				}
				sim := dot / fastSqrt(normProduct)

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
		fs.precomputedNorms[i] = math.Float32frombits(binary.LittleEndian.Uint32(normData[i*4:]))
	}

	return nil
}

// computeDirHash computes a quick hash of directory state (file paths + mod times)
func computeDirHash(basePath string, filter *EnhancedIgnoreFilter) (string, error) {
	h := sha256.New()
	cachePath := filepath.Join(basePath, cacheDir)
	filepath.WalkDir(basePath, func(p string, d os.DirEntry, err error) error {
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
