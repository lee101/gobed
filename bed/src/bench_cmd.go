package src

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"

	"github.com/lee101/gobed/metrics"
	"github.com/spf13/cobra"
)

var (
	flagBenchQueries int
	flagBenchWarmup  int
	flagBenchGPU     bool
	flagBenchNDCG    bool
	flagBenchNDCGK   int
	flagBenchQuality string
)

type benchRunStats struct {
	Name        string
	IndexTime   time.Duration
	Docs        int
	QueryCount  int
	TotalSearch time.Duration
	AvgSearch   time.Duration
	P50Search   time.Duration
	P95Search   time.Duration
	QueriesPerS float64
}

type benchQualityCase struct {
	Query     string             `json:"query"`
	Relevance map[string]float64 `json:"relevance"`
}

type qualityBenchStats struct {
	Name      string
	Queries   int
	K         int
	AvgNDCG   float64
	AvgRecall float64
}

var reindexCmd = &cobra.Command{
	Use:   "reindex [path]",
	Short: "Force a full semantic index rebuild",
	Long: `Force a complete index rebuild. This is equivalent to:
  bed index --force [path]

Useful when you've changed a lot of files and want to discard stale index data.`,
	Args: cobra.MaximumNArgs(1),
	RunE: runReindex,
}

var benchCmd = &cobra.Command{
	Use:   "bench [path]",
	Short: "Benchmark indexing and semantic search performance",
	Long: `Benchmark indexing and query latency on CPU fast-index mode and optional GPU mode.

Examples:
  bed bench .                    # CPU benchmark only
  bed bench . --compare-gpu      # CPU + GPU benchmark (if GPU backend is built)
  bed bench . --queries 300`,
	Args: cobra.MaximumNArgs(1),
	RunE: runBench,
}

func init() {
	reindexCmd.Flags().IntVar(&flagIndexBatchSize, "batch-size", 1000, "Indexing batch size")
	reindexCmd.Flags().BoolVar(&flagIndexWatch, "watch", false, "Keep watching filesystem changes and live-update the index")

	benchCmd.Flags().IntVar(&flagBenchQueries, "queries", 100, "Number of timed queries")
	benchCmd.Flags().IntVar(&flagBenchWarmup, "warmup", 10, "Number of warmup queries")
	benchCmd.Flags().BoolVar(&flagBenchGPU, "compare-gpu", false, "Also run GPU benchmark when CAGRA backend is available")
	benchCmd.Flags().BoolVar(&flagBenchNDCG, "ndcg", false, "Run NDCG quality benchmark on the quality dataset")
	benchCmd.Flags().IntVar(&flagBenchNDCGK, "ndcg-k", 10, "K value for NDCG/Recall quality benchmark")
	benchCmd.Flags().StringVar(&flagBenchQuality, "quality-dir", "", "Quality dataset root containing docs/ and queries.json")

	rootCmd.AddCommand(reindexCmd)
	rootCmd.AddCommand(benchCmd)
}

func runReindex(cmd *cobra.Command, args []string) error {
	prev := flagForceIndex
	flagForceIndex = true
	defer func() { flagForceIndex = prev }()
	return runIndex(cmd, args)
}

func runBench(cmd *cobra.Command, args []string) error {
	path := "."
	if len(args) > 0 {
		path = args[0]
	}

	if flagBenchQueries <= 0 {
		return fmt.Errorf("--queries must be > 0")
	}
	if flagBenchWarmup < 0 {
		return fmt.Errorf("--warmup must be >= 0")
	}

	fmt.Printf("Benchmark path: %s\n", path)
	fmt.Printf("Queries: %d (warmup: %d)\n\n", flagBenchQueries, flagBenchWarmup)

	cpuStats, err := runCPUBench(path)
	if err != nil {
		return err
	}
	printBenchStats(cpuStats)

	if flagBenchNDCG {
		cpuQuality, err := runCPUQualityBench()
		if err != nil {
			fmt.Printf("\nCPU quality benchmark skipped: %v\n", err)
		} else {
			fmt.Println()
			printQualityStats(cpuQuality)
		}
	}

	if !flagBenchGPU {
		return nil
	}

	gpuStats, err := runGPUBench(path)
	if err != nil {
		fmt.Printf("\nGPU benchmark skipped: %v\n", err)
		return nil
	}

	fmt.Println()
	printBenchStats(gpuStats)

	if flagBenchNDCG {
		gpuQuality, err := runGPUQualityBench()
		if err != nil {
			fmt.Printf("\nGPU quality benchmark skipped: %v\n", err)
		} else {
			fmt.Println()
			printQualityStats(gpuQuality)
		}
	}

	return nil
}

func runCPUBench(path string) (*benchRunStats, error) {
	searcher, err := NewFastBedSearcher()
	if err != nil {
		return nil, fmt.Errorf("cpu benchmark unavailable: %w", err)
	}
	defer searcher.Close()

	indexStart := time.Now()
	if err := searcher.IndexDirectory(path, BedSearchOptions{
		ForceIndex:     true,
		SearchBinaries: flagSearchBinaries,
		Verbose:        flagVerbose,
	}); err != nil {
		return nil, fmt.Errorf("cpu index failed: %w", err)
	}
	indexDuration := time.Since(indexStart)

	searchLatencies, err := benchmarkSearchLoop(searcher.SearchMatches)
	if err != nil {
		return nil, fmt.Errorf("cpu search failed: %w", err)
	}

	return buildBenchStats("CPU Fast Int8", indexDuration, searcher.NumDocuments(), searchLatencies), nil
}

func runGPUBench(path string) (*benchRunStats, error) {
	searcher, err := NewCAGRABedSearcher()
	if err != nil {
		return nil, err
	}
	defer searcher.Close()

	indexStart := time.Now()
	if err := searcher.BuildIndex(path, BedSearchOptions{
		ForceIndex:     true,
		SearchBinaries: flagSearchBinaries,
		Verbose:        flagVerbose,
	}); err != nil {
		return nil, fmt.Errorf("gpu index failed: %w", err)
	}
	indexDuration := time.Since(indexStart)

	searchLatencies, err := benchmarkSearchLoop(searcher.SearchMatches)
	if err != nil {
		return nil, fmt.Errorf("gpu search failed: %w", err)
	}

	return buildBenchStats("GPU CAGRA", indexDuration, searcher.NumDocuments(), searchLatencies), nil
}

func benchmarkSearchLoop(searchFn func(BedSearchOptions) ([]SearchMatch, error)) ([]time.Duration, error) {
	queries := benchmarkQueries()
	totalRuns := flagBenchWarmup + flagBenchQueries
	latencies := make([]time.Duration, 0, flagBenchQueries)

	for i := 0; i < totalRuns; i++ {
		q := queries[i%len(queries)]
		start := time.Now()
		_, err := searchFn(BedSearchOptions{
			Query:          q,
			Limit:          flagLimit,
			Threshold:      float32(flagThreshold),
			NoIndex:        true,
			SearchBinaries: flagSearchBinaries,
		})
		if err != nil {
			return nil, err
		}
		if i >= flagBenchWarmup {
			latencies = append(latencies, time.Since(start))
		}
	}

	return latencies, nil
}

func benchmarkQueries() []string {
	return []string{
		"authentication handler",
		"database query execution",
		"error handling path",
		"vector similarity search",
		"http request middleware",
		"file indexing pipeline",
		"gpu acceleration kernel",
		"tokenization and embeddings",
		"cache invalidation logic",
		"concurrency and goroutines",
	}
}

func runCPUQualityBench() (*qualityBenchStats, error) {
	docsDir, cases, err := loadQualityDataset()
	if err != nil {
		return nil, err
	}

	searcher, err := NewFastBedSearcher()
	if err != nil {
		return nil, fmt.Errorf("cpu quality benchmark unavailable: %w", err)
	}
	defer searcher.Close()

	if err := searcher.IndexDirectory(docsDir, BedSearchOptions{
		ForceIndex: true,
		Verbose:    flagVerbose,
	}); err != nil {
		return nil, fmt.Errorf("cpu quality index failed: %w", err)
	}

	avgNDCG, avgRecall, err := evaluateQualityCases(cases, searcher.SearchMatches, flagBenchNDCGK)
	if err != nil {
		return nil, err
	}

	return &qualityBenchStats{
		Name:      "CPU Fast Int8 Quality",
		Queries:   len(cases),
		K:         flagBenchNDCGK,
		AvgNDCG:   avgNDCG,
		AvgRecall: avgRecall,
	}, nil
}

func runGPUQualityBench() (*qualityBenchStats, error) {
	docsDir, cases, err := loadQualityDataset()
	if err != nil {
		return nil, err
	}

	searcher, err := NewCAGRABedSearcher()
	if err != nil {
		return nil, err
	}
	defer searcher.Close()

	if err := searcher.BuildIndex(docsDir, BedSearchOptions{
		ForceIndex: true,
		Verbose:    flagVerbose,
	}); err != nil {
		return nil, fmt.Errorf("gpu quality index failed: %w", err)
	}

	avgNDCG, avgRecall, err := evaluateQualityCases(cases, searcher.SearchMatches, flagBenchNDCGK)
	if err != nil {
		return nil, err
	}

	return &qualityBenchStats{
		Name:      "GPU CAGRA Quality",
		Queries:   len(cases),
		K:         flagBenchNDCGK,
		AvgNDCG:   avgNDCG,
		AvgRecall: avgRecall,
	}, nil
}

func evaluateQualityCases(cases []benchQualityCase, searchFn func(BedSearchOptions) ([]SearchMatch, error), k int) (float64, float64, error) {
	if len(cases) == 0 {
		return 0, 0, fmt.Errorf("no quality cases found")
	}
	if k <= 0 {
		k = 10
	}

	totalNDCG := 0.0
	totalRecall := 0.0

	for _, qc := range cases {
		matches, err := searchFn(BedSearchOptions{
			Query:     qc.Query,
			Limit:     k,
			Threshold: 0.0,
			NoIndex:   true,
		})
		if err != nil {
			return 0, 0, fmt.Errorf("quality search failed for %q: %w", qc.Query, err)
		}

		ranking := topUniqueFileNames(matches, k)
		totalNDCG += metrics.NDCGAtK(ranking, qc.Relevance, k)

		relevantFound := 0
		for _, docID := range ranking {
			if qc.Relevance[docID] > 0 {
				relevantFound++
			}
		}
		if len(qc.Relevance) > 0 {
			totalRecall += float64(relevantFound) / float64(len(qc.Relevance))
		}
	}

	n := float64(len(cases))
	return totalNDCG / n, totalRecall / n, nil
}

func topUniqueFileNames(matches []SearchMatch, k int) []string {
	ranking := make([]string, 0, k)
	seen := make(map[string]struct{}, k)

	for _, match := range matches {
		base := filepath.Base(match.Document.Path)
		if _, ok := seen[base]; ok {
			continue
		}
		seen[base] = struct{}{}
		ranking = append(ranking, base)
		if len(ranking) >= k {
			break
		}
	}

	return ranking
}

func loadQualityDataset() (string, []benchQualityCase, error) {
	root := flagBenchQuality
	if root == "" {
		root = detectQualityRoot()
	}
	if root == "" {
		return "", nil, fmt.Errorf("quality dataset not found; set --quality-dir to a directory with docs/ and queries.json")
	}

	queriesPath := filepath.Join(root, "queries.json")
	docsDir := filepath.Join(root, "docs")

	if _, err := os.Stat(queriesPath); err != nil {
		return "", nil, fmt.Errorf("missing queries file: %s", queriesPath)
	}
	if _, err := os.Stat(docsDir); err != nil {
		return "", nil, fmt.Errorf("missing docs directory: %s", docsDir)
	}

	data, err := os.ReadFile(queriesPath)
	if err != nil {
		return "", nil, fmt.Errorf("failed to read quality queries: %w", err)
	}

	var cases []benchQualityCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return "", nil, fmt.Errorf("failed to parse quality queries: %w", err)
	}
	if len(cases) == 0 {
		return "", nil, fmt.Errorf("no quality queries found in %s", queriesPath)
	}

	return docsDir, cases, nil
}

func detectQualityRoot() string {
	candidates := []string{
		"testsearches",
		filepath.Join("bed", "testsearches"),
	}

	if _, file, _, ok := runtime.Caller(0); ok {
		candidates = append(candidates, filepath.Join(filepath.Dir(file), "..", "testsearches"))
	}

	for _, candidate := range candidates {
		queriesPath := filepath.Join(candidate, "queries.json")
		docsDir := filepath.Join(candidate, "docs")
		if _, err := os.Stat(queriesPath); err != nil {
			continue
		}
		if _, err := os.Stat(docsDir); err != nil {
			continue
		}
		return candidate
	}

	return ""
}

func buildBenchStats(name string, indexTime time.Duration, docs int, latencies []time.Duration) *benchRunStats {
	sorted := append([]time.Duration(nil), latencies...)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i] < sorted[j]
	})

	total := time.Duration(0)
	for _, l := range sorted {
		total += l
	}

	avg := time.Duration(0)
	p50 := time.Duration(0)
	p95 := time.Duration(0)
	qps := 0.0

	if len(sorted) > 0 {
		avg = total / time.Duration(len(sorted))
		p50 = sorted[len(sorted)/2]
		p95Idx := (len(sorted)*95 + 99) / 100
		if p95Idx >= len(sorted) {
			p95Idx = len(sorted) - 1
		}
		p95 = sorted[p95Idx]
		if total > 0 {
			qps = float64(len(sorted)) / total.Seconds()
		}
	}

	return &benchRunStats{
		Name:        name,
		IndexTime:   indexTime,
		Docs:        docs,
		QueryCount:  len(sorted),
		TotalSearch: total,
		AvgSearch:   avg,
		P50Search:   p50,
		P95Search:   p95,
		QueriesPerS: qps,
	}
}

func printBenchStats(stats *benchRunStats) {
	fmt.Printf("%s\n", stats.Name)
	fmt.Printf("  Indexed docs:   %d\n", stats.Docs)
	fmt.Printf("  Index time:     %s\n", stats.IndexTime.Round(time.Millisecond))
	fmt.Printf("  Search queries: %d\n", stats.QueryCount)
	fmt.Printf("  Avg latency:    %s\n", stats.AvgSearch.Round(time.Microsecond))
	fmt.Printf("  P50 latency:    %s\n", stats.P50Search.Round(time.Microsecond))
	fmt.Printf("  P95 latency:    %s\n", stats.P95Search.Round(time.Microsecond))
	fmt.Printf("  Throughput:     %.2f qps\n", stats.QueriesPerS)
}

func printQualityStats(stats *qualityBenchStats) {
	fmt.Printf("%s\n", stats.Name)
	fmt.Printf("  Queries:        %d\n", stats.Queries)
	fmt.Printf("  NDCG@%d:         %.4f\n", stats.K, stats.AvgNDCG)
	fmt.Printf("  Recall@%d:       %.4f\n", stats.K, stats.AvgRecall)
}
