package src

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// AdaptiveSearcher automatically selects the best search strategy
type AdaptiveSearcher struct {
	simdEngine  *OptimizedSearchEngine
	cpuSearcher *SimpleBedSearcher
	gpuSearcher interface {
		Search(BedSearchOptions) error
		Close() error
	}

	optimizer   *AutoOptimizer
	config      OptimizationResult
	indexSize   int
	initialized bool

	mu sync.RWMutex
}

// NewAdaptiveSearcher creates a searcher that auto-selects backend
func NewAdaptiveSearcher() (*AdaptiveSearcher, error) {
	optimizer := GetOptimizer()

	as := &AdaptiveSearcher{
		optimizer: optimizer,
	}

	// Try to initialize SIMD engine (preferred for most cases)
	if engine, err := NewOptimizedSearchEngine(); err == nil {
		as.simdEngine = engine
	}

	// Try GPU if available
	if caps := optimizer.GetCapabilities(); caps.GPUAvailable {
		if gpu, err := NewCAGRABedSearcher(); err == nil {
			as.gpuSearcher = gpu
		}
	}

	// Fallback to simple CPU
	if as.simdEngine == nil && as.gpuSearcher == nil {
		cpu, err := NewSimpleBedSearcher()
		if err != nil {
			return nil, fmt.Errorf("no search backend available: %w", err)
		}
		as.cpuSearcher = cpu
	}

	return as, nil
}

// Search performs search using the optimal backend
func (as *AdaptiveSearcher) Search(options BedSearchOptions) error {
	start := time.Now()

	// Get optimal configuration
	as.mu.RLock()
	config := as.config
	as.mu.RUnlock()

	if !as.initialized {
		config = as.optimizer.Optimize()
		as.mu.Lock()
		as.config = config
		as.initialized = true
		as.mu.Unlock()
	}

	if options.Verbose {
		fmt.Printf("Backend: %s (%s)\n", config.Backend, config.Reason)
	}

	var err error
	switch config.Backend {
	case BackendGPU:
		if as.gpuSearcher != nil {
			err = as.gpuSearcher.Search(options)
		} else {
			err = as.searchWithSIMD(options)
		}
	case BackendSIMD:
		err = as.searchWithSIMD(options)
	case BackendHybrid:
		err = as.searchHybrid(options)
	default:
		if as.cpuSearcher != nil {
			err = as.cpuSearcher.Search(options)
		} else {
			err = as.searchWithSIMD(options)
		}
	}

	// Record timing for adaptive tuning
	as.optimizer.RecordQueryTime(time.Since(start))

	return err
}

// searchWithSIMD uses the SIMD-optimized engine
func (as *AdaptiveSearcher) searchWithSIMD(options BedSearchOptions) error {
	if as.simdEngine == nil {
		return fmt.Errorf("SIMD engine not available")
	}

	// Use fast indexer for directory indexing
	if !options.NoIndex {
		indexer, err := NewFastIndexer(DefaultFastIndexerConfig())
		if err != nil {
			return err
		}

		if err := indexer.IndexDirectory(".", options.Verbose); err != nil {
			return fmt.Errorf("indexing failed: %w", err)
		}

		// Update optimizer with index size
		as.mu.Lock()
		as.indexSize = indexer.GetEngine().NumDocuments()
		as.optimizer.UpdateProfile(as.indexSize, len(options.Query), 0)
		as.config = as.optimizer.Optimize()
		as.mu.Unlock()

		// If we should switch to GPU, do it
		if as.config.Backend == BackendGPU && as.gpuSearcher != nil {
			return as.gpuSearcher.Search(options)
		}

		// Search with the fast indexer's engine
		matches, err := indexer.Search(options.Query, options.Limit, options.Threshold)
		if err != nil {
			return err
		}

		displayMatches(matches, options)
		return nil
	}

	// Search with pre-loaded engine
	matches, err := as.simdEngine.Search(options.Query, options.Limit, options.Threshold)
	if err != nil {
		return err
	}

	displayMatches(matches, options)
	return nil
}

// searchHybrid uses both GPU and CPU in parallel
func (as *AdaptiveSearcher) searchHybrid(options BedSearchOptions) error {
	if as.gpuSearcher == nil {
		return as.searchWithSIMD(options)
	}

	type result struct {
		matches []SearchMatch
		err     error
		source  string
	}

	results := make(chan result, 2)
	var wg sync.WaitGroup

	// GPU search
	wg.Add(1)
	go func() {
		defer wg.Done()
		gpuOpts := options
		gpuOpts.Limit = options.Limit / 2
		err := as.gpuSearcher.Search(gpuOpts)
		results <- result{err: err, source: "GPU"}
	}()

	// SIMD search in parallel
	wg.Add(1)
	go func() {
		defer wg.Done()
		if as.simdEngine != nil {
			matches, err := as.simdEngine.Search(options.Query, options.Limit/2, options.Threshold)
			results <- result{matches: matches, err: err, source: "SIMD"}
		}
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	var allMatches []SearchMatch
	for r := range results {
		if r.err == nil && r.matches != nil {
			allMatches = append(allMatches, r.matches...)
		}
	}

	// Sort and limit
	if len(allMatches) > options.Limit {
		allMatches = allMatches[:options.Limit]
	}

	displayMatches(allMatches, options)
	return nil
}

// Close releases resources
func (as *AdaptiveSearcher) Close() error {
	if as.gpuSearcher != nil {
		as.gpuSearcher.Close()
	}
	if as.cpuSearcher != nil {
		as.cpuSearcher.Close()
	}
	return nil
}

// GetStats returns current search statistics
func (as *AdaptiveSearcher) GetStats() map[string]interface{} {
	as.mu.RLock()
	defer as.mu.RUnlock()

	caps := as.optimizer.GetCapabilities()

	return map[string]interface{}{
		"backend":       as.config.Backend.String(),
		"reason":        as.config.Reason,
		"num_workers":   as.config.NumWorkers,
		"batch_size":    as.config.BatchSize,
		"use_int8":      as.config.UseInt8,
		"index_size":    as.indexSize,
		"cpu_cores":     caps.NumCPUs,
		"has_avx2":      caps.HasAVX2,
		"gpu_available": caps.GPUAvailable,
		"gpu_name":      caps.GPUName,
		"gpu_memory_mb": caps.GPUMemoryMB,
	}
}

// displayMatches formats and prints search results
func displayMatches(matches []SearchMatch, options BedSearchOptions) {
	if len(matches) == 0 {
		fmt.Println("No results found")
		return
	}

	fmt.Printf("Found %d result(s)\n\n", len(matches))

	for i, match := range matches {
		doc := match.Document

		if doc.IsBinary {
			fmt.Printf("%s: Binary file (%.3f)\n", doc.Path, match.Similarity)
			continue
		}

		if shouldUseColor("auto") {
			fmt.Printf("\033[35m%s\033[0m:\033[32m%d\033[0m: %s\n",
				doc.Path, doc.LineNumber, doc.Content)
		} else {
			fmt.Printf("%s:%d: %s\n", doc.Path, doc.LineNumber, doc.Content)
		}

		if options.Verbose {
			fmt.Printf("  [%.3f]\n", match.Similarity)
		}

		if i < len(matches)-1 && i < options.Limit-1 {
			fmt.Println()
		}
	}
}

// BenchmarkBackends runs a quick benchmark of available backends
func BenchmarkBackends(indexSize int) map[string]time.Duration {
	results := make(map[string]time.Duration)
	numCPU := runtime.NumCPU()

	// Generate test data
	testDocs := make([]Document, indexSize)
	for i := range testDocs {
		testDocs[i] = Document{
			ID:      i,
			Path:    fmt.Sprintf("test%d.go", i),
			Content: fmt.Sprintf("test content line %d with some meaningful text", i),
		}
	}

	// Benchmark SIMD
	if engine, err := NewOptimizedSearchEngine(); err == nil {
		start := time.Now()
		engine.BatchAddDocuments(testDocs, numCPU)
		engine.Search("test content", 10, 0.5)
		results["simd"] = time.Since(start)
	}

	// Benchmark CPU
	if searcher, err := NewSimpleBedSearcher(); err == nil {
		start := time.Now()
		searcher.engine.IndexDirectory(".", BedSearchOptions{NoIndex: true})
		results["cpu"] = time.Since(start)
		searcher.Close()
	}

	return results
}
