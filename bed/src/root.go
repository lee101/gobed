package src

import (
	"fmt"
	"runtime"
	"strings"

	"github.com/lee101/gobed"
	"github.com/spf13/cobra"
)

var (
	// Global flags
	flagLimit          int
	flagContext        int
	flagColorMode      string
	flagNoIndex        bool
	flagForceIndex     bool
	flagGPU            bool
	flagThreshold      float64
	flagIgnoreCase     bool
	flagVerbose        bool
	flagConfig         string
	flagProgressive    bool
	flagSearchBinaries bool
)

var rootCmd = &cobra.Command{
	Use:   "bed [flags] <query>",
	Short: "Natural language search for your filesystem",
	Long: `Bed is a blazing-fast natural language search tool that finds code and files by meaning,
not just text patterns. It uses advanced embeddings with GPU acceleration for semantic
similarity search, automatically respects .gitignore, and handles binary files intelligently.

Features:
  • Natural language understanding - search by meaning, not just keywords
  • Async indexing - search starts immediately while index builds in background  
  • Progressive results - see matches as they're found
  • Smart filtering - respects .gitignore, skips binaries and large files
  • GPU acceleration - lightning-fast search when GPU is available
  • Context-aware - shows surrounding lines for better understanding

Examples:
  bed "where does authentication happen"       # Natural language query
  bed "functions that handle errors" -p        # Progressive search
  bed "TODO comments" --context 5              # Show 5 lines of context
  bed "database connections" --search-binaries # Include binary files
  bed "memory leak" --force-index              # Force index rebuild`,
	Args: cobra.MinimumNArgs(1),
	RunE: runSearch,
}

var indexCmd = &cobra.Command{
	Use:   "index [path]",
	Short: "Build semantic index for directory",
	Long: `Build or update the semantic index for the specified directory.
The index contains embeddings for all files, allowing fast semantic search.

This command respects .bedignore and .gitignore files to exclude unwanted files.`,
	Args: cobra.MaximumNArgs(1),
	RunE: runIndex,
}

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show index status and statistics",
	Long: `Display information about the current index including:
- Number of indexed files and lines
- Index size and memory usage
- Last update time
- GPU availability and usage`,
	RunE: runStatus,
}

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Manage bed configuration",
	Long:  `View and manage bed configuration settings`,
	RunE:  runConfig,
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	// Global flags
	rootCmd.PersistentFlags().IntVarP(&flagLimit, "limit", "l", 10, "Maximum number of results")
	rootCmd.PersistentFlags().IntVarP(&flagContext, "context", "c", 2, "Lines of context around matches")
	rootCmd.PersistentFlags().StringVar(&flagColorMode, "color", "auto", "When to colorize output (auto|always|never)")
	rootCmd.PersistentFlags().BoolVar(&flagNoIndex, "no-index", false, "Skip indexing, use existing index only")
	rootCmd.PersistentFlags().BoolVar(&flagForceIndex, "force-index", false, "Force re-indexing even if index exists")
	rootCmd.PersistentFlags().BoolVar(&flagGPU, "gpu", false, "Force GPU acceleration (auto-detect by default)")
	rootCmd.PersistentFlags().Float64Var(&flagThreshold, "threshold", 0.7, "Minimum similarity threshold (0.0-1.0)")
	rootCmd.PersistentFlags().BoolVarP(&flagIgnoreCase, "ignore-case", "i", false, "Case-insensitive search")
	rootCmd.PersistentFlags().BoolVarP(&flagVerbose, "verbose", "v", false, "Verbose output")
	rootCmd.PersistentFlags().StringVar(&flagConfig, "config", "", "Config file path")
	rootCmd.PersistentFlags().BoolVarP(&flagProgressive, "progressive", "p", false, "Show results progressively as index builds")
	rootCmd.PersistentFlags().BoolVar(&flagSearchBinaries, "search-binaries", false, "Include binary files in search")

	// Search-specific flags
	rootCmd.Flags().BoolVar(&flagNoIndex, "no-index", false, "Skip indexing, use existing index only")

	// Index-specific flags
	indexCmd.Flags().BoolVar(&flagForceIndex, "force", false, "Force complete re-indexing")
	indexCmd.Flags().IntVar(&flagContext, "batch-size", 1000, "Indexing batch size")

	// Add subcommands
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(configCmd)
}

func runSearch(cmd *cobra.Command, args []string) error {
	query := strings.Join(args, " ")

	if flagVerbose {
		gobed.EnableDebugLogging()
	}

	useGPU := true
	if cmd.Flags().Changed("gpu") {
		useGPU = flagGPU
	}

	type bedSearcher interface {
		Search(BedSearchOptions) error
		Close() error
	}

	var (
		searcher bedSearcher
		err      error
		gpuUsed  bool
	)

	if useGPU {
		if s, e := NewCAGRABedSearcher(); e == nil {
			searcher = s
			gpuUsed = true
		} else {
			if cmd.Flags().Changed("gpu") && flagGPU {
				return fmt.Errorf("gpu requested but unavailable: %w", e)
			}
		}
	}

	if searcher == nil {
		searcher, err = NewSimpleBedSearcher()
		if err != nil {
			return fmt.Errorf("failed to initialize searcher: %w", err)
		}
	}
	defer searcher.Close()

	options := BedSearchOptions{
		Query:          query,
		Limit:          flagLimit,
		Context:        flagContext,
		Threshold:      float32(flagThreshold),
		NoIndex:        flagNoIndex,
		ForceIndex:     flagForceIndex,
		SearchBinaries: flagSearchBinaries,
		Progressive:    flagProgressive,
		UseGPU:         gpuUsed,
		Verbose:        flagVerbose,
	}

	return searcher.Search(options)
}

func runIndex(cmd *cobra.Command, args []string) error {
	path := "."
	if len(args) > 0 {
		path = args[0]
	}

	if flagVerbose {
		gobed.EnableDebugLogging()
	}

	useGPU := true
	if cmd.Flags().Changed("gpu") {
		useGPU = flagGPU
	}

	indexer, err := NewIndexer()
	if err != nil {
		return fmt.Errorf("failed to initialize indexer: %w", err)
	}
	defer indexer.Close()

	options := IndexOptions{
		Path:      path,
		Force:     flagForceIndex,
		BatchSize: flagContext, // reusing context flag as batch size
		UseGPU:    useGPU,
		Verbose:   flagVerbose,
	}

	return indexer.Index(options)
}

func runStatus(cmd *cobra.Command, args []string) error {
	fmt.Println("Bed Search Status")
	fmt.Println("=================")
	fmt.Println()

	fmt.Printf("Natural language search: Enabled\n")
	fmt.Printf("Query enhancement:       Enabled\n")
	fmt.Printf("Gitignore support:       Enabled\n")
	fmt.Printf("Binary detection:        Enabled\n")
	fmt.Printf("Parallel workers:        %d\n", runtime.NumCPU())

	mode := "Auto (try GPU when available)"
	if cmd.Flags().Changed("gpu") {
		if flagGPU {
			mode = "Forced On"
		} else {
			mode = "Disabled"
		}
	}
	fmt.Printf("GPU acceleration:        %s\n", mode)

	return nil
}

func runConfig(cmd *cobra.Command, args []string) error {
	config, err := LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	return config.Display()
}
