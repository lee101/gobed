package src

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/lee101/gobed/metrics"
	"github.com/spf13/cobra"
)

var (
	flagBenchQueries              int
	flagBenchWarmup               int
	flagBenchGPU                  bool
	flagBenchNDCG                 bool
	flagBenchNDCGK                int
	flagBenchQuality              string
	flagBenchJSON                 bool
	flagBenchQueryFile            string
	flagBenchExact                bool
	flagBenchExactQueries         int
	flagBenchCAGRAGraphDegree     int
	flagBenchCAGRAIntermediateDeg int
	flagBenchCAGRAMaxIters        int
	flagBenchCAGRAItopk           int
	flagBenchCAGRASearchWidth     int
	flagBenchCAGRATeamSize        int
	flagBenchCAGRAAlgo            string
)

type benchRunStats struct {
	Name          string
	IndexTime     time.Duration
	Docs          int
	Files         int
	QueryCount    int
	QuerySource   string
	TotalSearch   time.Duration
	AvgSearch     time.Duration
	P50Search     time.Duration
	P95Search     time.Duration
	P99Search     time.Duration
	QueriesPerS   float64
	IndexDocsPerS float64
}

type benchRunReport struct {
	Name          string  `json:"name"`
	IndexedDocs   int     `json:"indexed_docs"`
	IndexedFiles  int     `json:"indexed_files"`
	QueryCount    int     `json:"query_count"`
	QuerySource   string  `json:"query_source"`
	IndexTimeMs   float64 `json:"index_time_ms"`
	IndexDocsPerS float64 `json:"index_docs_per_s"`
	AvgLatencyMs  float64 `json:"avg_latency_ms"`
	P50LatencyMs  float64 `json:"p50_latency_ms"`
	P95LatencyMs  float64 `json:"p95_latency_ms"`
	P99LatencyMs  float64 `json:"p99_latency_ms"`
	QPS           float64 `json:"qps"`
}

type benchQualityCase struct {
	Query     string             `json:"query"`
	Relevance map[string]float64 `json:"relevance"`
}

type qualityDataset struct {
	Root    string
	DocsDir string
	Cases   []benchQualityCase
}

type qualityBenchStats struct {
	Name        string  `json:"name"`
	DatasetRoot string  `json:"dataset_root"`
	Queries     int     `json:"queries"`
	K           int     `json:"k"`
	AvgNDCG     float64 `json:"avg_ndcg"`
	AvgRecall   float64 `json:"avg_recall"`
}

type exactANNStats struct {
	Name        string  `json:"name"`
	Queries     int     `json:"queries"`
	K           int     `json:"k"`
	AvgNDCG     float64 `json:"avg_ndcg"`
	AvgRecall   float64 `json:"avg_recall"`
	Top1HitRate float64 `json:"top1_hit_rate"`
}

type cagraBenchSettings struct {
	GraphDegree             int    `json:"graph_degree,omitempty"`
	IntermediateGraphDegree int    `json:"intermediate_graph_degree,omitempty"`
	MaxIterations           int    `json:"max_iterations,omitempty"`
	ItopkSize               int    `json:"itopk_size,omitempty"`
	SearchWidth             int    `json:"search_width,omitempty"`
	TeamSize                int    `json:"team_size,omitempty"`
	Algo                    string `json:"algo,omitempty"`
}

type benchJSONReport struct {
	Path          string             `json:"path"`
	Threshold     float64            `json:"threshold"`
	Limit         int                `json:"limit"`
	TimedQueries  int                `json:"timed_queries"`
	WarmupQueries int                `json:"warmup_queries"`
	QuerySource   string             `json:"query_source"`
	CompareGPU    bool               `json:"compare_gpu"`
	CPU           *benchRunReport    `json:"cpu,omitempty"`
	GPU           *benchRunReport    `json:"gpu,omitempty"`
	CPUQuality    *qualityBenchStats `json:"cpu_quality,omitempty"`
	GPUQuality    *qualityBenchStats `json:"gpu_quality,omitempty"`
	GPUExactANN   *exactANNStats     `json:"gpu_exact_ann,omitempty"`
	CAGRA         cagraBenchSettings `json:"cagra,omitempty"`
	Warnings      []string           `json:"warnings,omitempty"`
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
  bed bench .                               # CPU benchmark only
  bed bench . --compare-gpu                 # CPU + GPU benchmark (if CAGRA is available)
  bed bench . --query-file queries.txt      # Use a production-like query set
  bed bench . --compare-gpu --exact         # Add exact ANN quality on the built GPU index
  bed bench . --compare-gpu --json`,
	Args: cobra.MaximumNArgs(1),
	RunE: runBench,
}

func init() {
	reindexCmd.Flags().IntVar(&flagIndexBatchSize, "batch-size", 1000, "Indexing batch size")
	reindexCmd.Flags().BoolVar(&flagIndexWatch, "watch", false, "Keep watching filesystem changes and live-update the index")

	benchCmd.Flags().IntVar(&flagBenchQueries, "queries", 100, "Number of timed queries")
	benchCmd.Flags().IntVar(&flagBenchWarmup, "warmup", 10, "Number of warmup queries")
	benchCmd.Flags().BoolVar(&flagBenchGPU, "compare-gpu", false, "Also run GPU benchmark when CAGRA backend is available")
	benchCmd.Flags().BoolVar(&flagBenchNDCG, "ndcg", false, "Run curated NDCG/Recall evaluation from a docs/ + queries.json dataset")
	benchCmd.Flags().IntVar(&flagBenchNDCGK, "ndcg-k", 10, "K value for NDCG/Recall quality benchmarks")
	benchCmd.Flags().StringVar(&flagBenchQuality, "quality-dir", "", "Quality dataset root containing docs/ and queries.json")
	benchCmd.Flags().StringVar(&flagBenchQueryFile, "query-file", "", "Path to a newline-delimited query workload for latency benchmarking")
	benchCmd.Flags().BoolVar(&flagBenchJSON, "json", false, "Print machine-readable benchmark output as JSON")
	benchCmd.Flags().BoolVar(&flagBenchExact, "exact", false, "Run exact ANN quality evaluation against the built GPU corpus")
	benchCmd.Flags().IntVar(&flagBenchExactQueries, "exact-queries", 100, "Number of retained corpus vectors to use for exact ANN evaluation")

	benchCmd.Flags().IntVar(&flagBenchCAGRAGraphDegree, "cagra-graph-degree", 0, "Override GPU CAGRA graph degree")
	benchCmd.Flags().IntVar(&flagBenchCAGRAIntermediateDeg, "cagra-intermediate-degree", 0, "Override GPU CAGRA intermediate graph degree")
	benchCmd.Flags().IntVar(&flagBenchCAGRAMaxIters, "cagra-max-iters", 0, "Override GPU CAGRA search max iterations")
	benchCmd.Flags().IntVar(&flagBenchCAGRAItopk, "cagra-itopk", 0, "Override GPU CAGRA intermediate top-k size")
	benchCmd.Flags().IntVar(&flagBenchCAGRASearchWidth, "cagra-search-width", 0, "Override GPU CAGRA search width")
	benchCmd.Flags().IntVar(&flagBenchCAGRATeamSize, "cagra-team-size", 0, "Override GPU CAGRA team size")
	benchCmd.Flags().StringVar(&flagBenchCAGRAAlgo, "cagra-algo", "", "Override GPU CAGRA search algo (single_cta|multi_cta|multi_kernel|auto)")

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
	if flagBenchExactQueries < 0 {
		return fmt.Errorf("--exact-queries must be >= 0")
	}

	queries, querySource, err := loadBenchmarkQueries()
	if err != nil {
		return err
	}
	if len(queries) == 0 {
		return fmt.Errorf("benchmark query set is empty")
	}

	report := benchJSONReport{
		Path:          path,
		Threshold:     flagThreshold,
		Limit:         flagLimit,
		TimedQueries:  flagBenchQueries,
		WarmupQueries: flagBenchWarmup,
		QuerySource:   querySource,
		CompareGPU:    flagBenchGPU,
		CAGRA:         currentCAGRABenchSettings(),
	}

	if !flagBenchJSON {
		fmt.Printf("Benchmark path: %s\n", path)
		fmt.Printf("Timed queries: %d (warmup: %d)\n", flagBenchQueries, flagBenchWarmup)
		fmt.Printf("Query source: %s\n\n", querySource)
	}

	cpuStats, err := runCPUBench(path, querySource, queries)
	if err != nil {
		return err
	}
	report.CPU = cpuStats.Report()
	if !flagBenchJSON {
		printBenchStats(cpuStats)
	}

	if flagBenchNDCG {
		dataset, err := loadQualityDataset()
		if err != nil {
			report.Warnings = append(report.Warnings, fmt.Sprintf("CPU quality benchmark skipped: %v", err))
		} else {
			cpuQuality, err := runCPUQualityBench(dataset)
			if err != nil {
				report.Warnings = append(report.Warnings, fmt.Sprintf("CPU quality benchmark skipped: %v", err))
			} else {
				report.CPUQuality = cpuQuality
				if !flagBenchJSON {
					fmt.Println()
					printQualityStats(cpuQuality)
				}
			}
		}
	}

	if flagBenchGPU {
		gpuStats, gpuExact, err := runGPUBench(path, querySource, queries)
		if gpuStats != nil {
			report.GPU = gpuStats.Report()
			if !flagBenchJSON {
				fmt.Println()
				printBenchStats(gpuStats)
			}
		}
		if gpuExact != nil {
			report.GPUExactANN = gpuExact
			if !flagBenchJSON {
				fmt.Println()
				printExactANNStats(gpuExact)
			}
		}
		if err != nil {
			report.Warnings = append(report.Warnings, fmt.Sprintf("GPU benchmark warning: %v", err))
		}

		if flagBenchNDCG {
			dataset, err := loadQualityDataset()
			if err != nil {
				report.Warnings = append(report.Warnings, fmt.Sprintf("GPU quality benchmark skipped: %v", err))
			} else {
				gpuQuality, err := runGPUQualityBench(dataset)
				if err != nil {
					report.Warnings = append(report.Warnings, fmt.Sprintf("GPU quality benchmark skipped: %v", err))
				} else {
					report.GPUQuality = gpuQuality
					if !flagBenchJSON {
						fmt.Println()
						printQualityStats(gpuQuality)
					}
				}
			}
		}
	}

	if flagBenchJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(report)
	}

	for _, warning := range report.Warnings {
		fmt.Printf("\n%s\n", warning)
	}
	return nil
}

func runCPUBench(path, querySource string, queries []string) (*benchRunStats, error) {
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

	searchLatencies, err := benchmarkSearchLoop(searcher.SearchMatches, queries)
	if err != nil {
		return nil, fmt.Errorf("cpu search failed: %w", err)
	}

	return buildBenchStats(
		"CPU Fast Int8",
		indexDuration,
		searcher.NumDocuments(),
		countUniqueDocumentPaths(searcher.documents),
		querySource,
		searchLatencies,
	), nil
}

func runGPUBench(path, querySource string, queries []string) (*benchRunStats, *exactANNStats, error) {
	restoreEnv, err := applyCAGRABenchEnv()
	if err != nil {
		return nil, nil, err
	}
	defer restoreEnv()

	searcher, err := NewCAGRABedSearcher()
	if err != nil {
		return nil, nil, err
	}
	defer searcher.Close()

	searcher.SetBenchmarkRetention(flagBenchExact)
	searcher.ApplyBenchConfig(currentCAGRABenchSettings())

	indexStart := time.Now()
	if err := searcher.BuildIndex(path, BedSearchOptions{
		ForceIndex:     true,
		SearchBinaries: flagSearchBinaries,
		Verbose:        flagVerbose,
	}); err != nil {
		return nil, nil, fmt.Errorf("gpu index failed: %w", err)
	}
	indexDuration := time.Since(indexStart)

	searchLatencies, err := benchmarkSearchLoop(searcher.SearchMatches, queries)
	if err != nil {
		return nil, nil, fmt.Errorf("gpu search failed: %w", err)
	}

	stats := buildBenchStats(
		"GPU CAGRA",
		indexDuration,
		searcher.NumDocuments(),
		searcher.IndexedFiles(),
		querySource,
		searchLatencies,
	)

	if !flagBenchExact {
		return stats, nil, nil
	}

	exactStats, err := searcher.BenchmarkExactANN(flagBenchNDCGK, flagBenchExactQueries)
	if err != nil {
		return stats, nil, err
	}
	return stats, exactStats, nil
}

func benchmarkSearchLoop(searchFn func(BedSearchOptions) ([]SearchMatch, error), queries []string) ([]time.Duration, error) {
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

func loadBenchmarkQueries() ([]string, string, error) {
	if flagBenchQueryFile == "" {
		return defaultBenchmarkQueries(), "built-in", nil
	}

	file, err := os.Open(flagBenchQueryFile)
	if err != nil {
		return nil, "", fmt.Errorf("failed to open query file %s: %w", flagBenchQueryFile, err)
	}
	defer file.Close()

	queries := make([]string, 0, 32)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		queries = append(queries, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, "", fmt.Errorf("failed to read query file %s: %w", flagBenchQueryFile, err)
	}
	if len(queries) == 0 {
		return nil, "", fmt.Errorf("query file %s did not contain any usable queries", flagBenchQueryFile)
	}
	return queries, flagBenchQueryFile, nil
}

func defaultBenchmarkQueries() []string {
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

func runCPUQualityBench(dataset *qualityDataset) (*qualityBenchStats, error) {
	searcher, err := NewFastBedSearcher()
	if err != nil {
		return nil, fmt.Errorf("cpu quality benchmark unavailable: %w", err)
	}
	defer searcher.Close()

	if err := searcher.IndexDirectory(dataset.DocsDir, BedSearchOptions{
		ForceIndex: true,
		Verbose:    flagVerbose,
	}); err != nil {
		return nil, fmt.Errorf("cpu quality index failed: %w", err)
	}

	avgNDCG, avgRecall, err := evaluateQualityCases(dataset.DocsDir, dataset.Cases, searcher.SearchMatches, flagBenchNDCGK)
	if err != nil {
		return nil, err
	}

	return &qualityBenchStats{
		Name:        "CPU Fast Int8 Quality",
		DatasetRoot: dataset.Root,
		Queries:     len(dataset.Cases),
		K:           flagBenchNDCGK,
		AvgNDCG:     avgNDCG,
		AvgRecall:   avgRecall,
	}, nil
}

func runGPUQualityBench(dataset *qualityDataset) (*qualityBenchStats, error) {
	searcher, err := NewCAGRABedSearcher()
	if err != nil {
		return nil, err
	}
	defer searcher.Close()

	if err := searcher.BuildIndex(dataset.DocsDir, BedSearchOptions{
		ForceIndex: true,
		Verbose:    flagVerbose,
	}); err != nil {
		return nil, fmt.Errorf("gpu quality index failed: %w", err)
	}

	avgNDCG, avgRecall, err := evaluateQualityCases(dataset.DocsDir, dataset.Cases, searcher.SearchMatches, flagBenchNDCGK)
	if err != nil {
		return nil, err
	}

	return &qualityBenchStats{
		Name:        "GPU CAGRA Quality",
		DatasetRoot: dataset.Root,
		Queries:     len(dataset.Cases),
		K:           flagBenchNDCGK,
		AvgNDCG:     avgNDCG,
		AvgRecall:   avgRecall,
	}, nil
}

func evaluateQualityCases(docsRoot string, cases []benchQualityCase, searchFn func(BedSearchOptions) ([]SearchMatch, error), k int) (float64, float64, error) {
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

		ranking := topUniqueDocKeys(matches, k, docsRoot)
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

func topUniqueDocKeys(matches []SearchMatch, k int, docsRoot string) []string {
	ranking := make([]string, 0, k)
	seen := make(map[string]struct{}, k)

	for _, match := range matches {
		key := qualityDocKey(match.Document.Path, docsRoot)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		ranking = append(ranking, key)
		if len(ranking) >= k {
			break
		}
	}

	return ranking
}

func qualityDocKey(path, docsRoot string) string {
	if docsRoot != "" {
		if rel, err := filepath.Rel(docsRoot, path); err == nil && rel != "." && !strings.HasPrefix(rel, "..") {
			return filepath.ToSlash(rel)
		}
	}
	return filepath.Base(path)
}

func loadQualityDataset() (*qualityDataset, error) {
	root := flagBenchQuality
	if root == "" {
		root = detectQualityRoot()
	}
	if root == "" {
		return nil, fmt.Errorf("quality dataset not found; set --quality-dir to a directory with docs/ and queries.json")
	}

	queriesPath := filepath.Join(root, "queries.json")
	docsDir := filepath.Join(root, "docs")

	if _, err := os.Stat(queriesPath); err != nil {
		return nil, fmt.Errorf("missing queries file: %s", queriesPath)
	}
	if _, err := os.Stat(docsDir); err != nil {
		return nil, fmt.Errorf("missing docs directory: %s", docsDir)
	}

	data, err := os.ReadFile(queriesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read quality queries: %w", err)
	}

	var cases []benchQualityCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse quality queries: %w", err)
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("no quality queries found in %s", queriesPath)
	}

	return &qualityDataset{
		Root:    root,
		DocsDir: docsDir,
		Cases:   cases,
	}, nil
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

func buildBenchStats(name string, indexTime time.Duration, docs, files int, querySource string, latencies []time.Duration) *benchRunStats {
	sorted := append([]time.Duration(nil), latencies...)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i] < sorted[j]
	})

	total := time.Duration(0)
	for _, latency := range sorted {
		total += latency
	}

	stats := &benchRunStats{
		Name:        name,
		IndexTime:   indexTime,
		Docs:        docs,
		Files:       files,
		QueryCount:  len(sorted),
		QuerySource: querySource,
		TotalSearch: total,
	}

	if indexTime > 0 && docs > 0 {
		stats.IndexDocsPerS = float64(docs) / indexTime.Seconds()
	}
	if len(sorted) == 0 {
		return stats
	}

	stats.AvgSearch = total / time.Duration(len(sorted))
	stats.P50Search = percentileDuration(sorted, 50)
	stats.P95Search = percentileDuration(sorted, 95)
	stats.P99Search = percentileDuration(sorted, 99)
	if total > 0 {
		stats.QueriesPerS = float64(len(sorted)) / total.Seconds()
	}
	return stats
}

func percentileDuration(sorted []time.Duration, percentile int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(float64(percentile)/100.0*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func (stats *benchRunStats) Report() *benchRunReport {
	if stats == nil {
		return nil
	}
	return &benchRunReport{
		Name:          stats.Name,
		IndexedDocs:   stats.Docs,
		IndexedFiles:  stats.Files,
		QueryCount:    stats.QueryCount,
		QuerySource:   stats.QuerySource,
		IndexTimeMs:   durationMs(stats.IndexTime),
		IndexDocsPerS: stats.IndexDocsPerS,
		AvgLatencyMs:  durationMs(stats.AvgSearch),
		P50LatencyMs:  durationMs(stats.P50Search),
		P95LatencyMs:  durationMs(stats.P95Search),
		P99LatencyMs:  durationMs(stats.P99Search),
		QPS:           stats.QueriesPerS,
	}
}

func printBenchStats(stats *benchRunStats) {
	fmt.Printf("%s\n", stats.Name)
	fmt.Printf("  Indexed docs:   %d\n", stats.Docs)
	fmt.Printf("  Indexed files:  %d\n", stats.Files)
	fmt.Printf("  Index time:     %s\n", stats.IndexTime.Round(time.Millisecond))
	if stats.IndexDocsPerS > 0 {
		fmt.Printf("  Index rate:     %.0f docs/sec\n", stats.IndexDocsPerS)
	}
	fmt.Printf("  Search queries: %d\n", stats.QueryCount)
	fmt.Printf("  Query source:   %s\n", stats.QuerySource)
	fmt.Printf("  Avg latency:    %s\n", stats.AvgSearch.Round(time.Microsecond))
	fmt.Printf("  P50 latency:    %s\n", stats.P50Search.Round(time.Microsecond))
	fmt.Printf("  P95 latency:    %s\n", stats.P95Search.Round(time.Microsecond))
	fmt.Printf("  P99 latency:    %s\n", stats.P99Search.Round(time.Microsecond))
	fmt.Printf("  Throughput:     %.2f qps\n", stats.QueriesPerS)
}

func printQualityStats(stats *qualityBenchStats) {
	fmt.Printf("%s\n", stats.Name)
	fmt.Printf("  Dataset root:   %s\n", stats.DatasetRoot)
	fmt.Printf("  Queries:        %d\n", stats.Queries)
	fmt.Printf("  NDCG@%d:         %.4f\n", stats.K, stats.AvgNDCG)
	fmt.Printf("  Recall@%d:       %.4f\n", stats.K, stats.AvgRecall)
}

func printExactANNStats(stats *exactANNStats) {
	fmt.Printf("%s\n", stats.Name)
	fmt.Printf("  Queries:        %d\n", stats.Queries)
	fmt.Printf("  NDCG@%d:         %.4f\n", stats.K, stats.AvgNDCG)
	fmt.Printf("  Recall@%d:       %.4f\n", stats.K, stats.AvgRecall)
	fmt.Printf("  Top1 self-hit:  %.4f\n", stats.Top1HitRate)
}

func currentCAGRABenchSettings() cagraBenchSettings {
	return cagraBenchSettings{
		GraphDegree:             flagBenchCAGRAGraphDegree,
		IntermediateGraphDegree: flagBenchCAGRAIntermediateDeg,
		MaxIterations:           flagBenchCAGRAMaxIters,
		ItopkSize:               flagBenchCAGRAItopk,
		SearchWidth:             flagBenchCAGRASearchWidth,
		TeamSize:                flagBenchCAGRATeamSize,
		Algo:                    flagBenchCAGRAAlgo,
	}
}

func applyCAGRABenchEnv() (func(), error) {
	overrides := map[string]string{}

	setIntOverride := func(name string, value int) {
		if value > 0 {
			overrides[name] = fmt.Sprintf("%d", value)
		}
	}

	setIntOverride("CAGRA_ITOPK_SIZE", flagBenchCAGRAItopk)
	setIntOverride("CAGRA_SEARCH_WIDTH", flagBenchCAGRASearchWidth)
	setIntOverride("CAGRA_SEARCH_MAX_ITERS", flagBenchCAGRAMaxIters)
	setIntOverride("CAGRA_TEAM_SIZE", flagBenchCAGRATeamSize)

	if flagBenchCAGRAAlgo != "" {
		algoMap := map[string]string{
			"single_cta":   "0",
			"multi_cta":    "1",
			"multi_kernel": "2",
			"auto":         "3",
		}
		key := strings.ToLower(strings.TrimSpace(flagBenchCAGRAAlgo))
		value, ok := algoMap[key]
		if !ok {
			return nil, fmt.Errorf("unsupported --cagra-algo %q", flagBenchCAGRAAlgo)
		}
		overrides["CAGRA_ALGO"] = value
	}

	previous := make(map[string]*string, len(overrides))
	for name, value := range overrides {
		if current, ok := os.LookupEnv(name); ok {
			currentCopy := current
			previous[name] = &currentCopy
		} else {
			previous[name] = nil
		}
		if err := os.Setenv(name, value); err != nil {
			return nil, fmt.Errorf("failed to set %s: %w", name, err)
		}
	}

	return func() {
		for name, value := range previous {
			if value == nil {
				_ = os.Unsetenv(name)
				continue
			}
			_ = os.Setenv(name, *value)
		}
	}, nil
}

func countUniqueDocumentPaths(docs []Document) int {
	seen := make(map[string]struct{}, len(docs))
	for _, doc := range docs {
		seen[doc.Path] = struct{}{}
	}
	return len(seen)
}

func durationMs(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}
