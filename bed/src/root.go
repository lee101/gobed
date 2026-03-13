package src

import (
	"fmt"
	"strings"
	"time"

	"github.com/lee101/gobed"
	"github.com/spf13/cobra"
)

var (
	buildVersion = "dev"
	buildCommit  = ""
	buildDate    = ""

	// Global flags
	flagLimit          int
	flagContext        int
	flagColorMode      string
	flagNoIndex        bool
	flagForceIndex     bool
	flagGPU            bool
	flagAuto           bool
	flagPreset         string
	flagThreshold      float64
	flagIgnoreCase     bool
	flagVerbose        bool
	flagConfig         string
	flagProgressive    bool
	flagSearchBinaries bool

	// Index flags
	flagIndexBatchSize int
	flagIndexWatch     bool
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

var daemonCmd = &cobra.Command{
	Use:   "daemon [paths...]",
	Short: "Run as persistent daemon with inotify watching",
	Long: `Run bed as a persistent daemon that maintains an in-memory index
and watches for file changes via inotify. Exposes search via HTTP and Unix socket.

Examples:
  bed daemon .                           # Watch current directory
  bed daemon /home/user/code --port 8080 # Watch with HTTP API
  bed daemon . --socket /tmp/bed.sock    # Unix socket for local clients`,
	Args: cobra.MinimumNArgs(1),
	RunE: runDaemon,
}

var (
	flagDaemonPort       int
	flagDaemonSocket     string
	flagDaemonBatchDelay time.Duration
)

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.Version = cliVersion()
	rootCmd.SetVersionTemplate("{{.Name}} {{.Version}}\n")

	// Global flags
	rootCmd.PersistentFlags().IntVarP(&flagLimit, "limit", "l", 10, "Maximum number of results")
	rootCmd.PersistentFlags().IntVarP(&flagContext, "context", "c", 2, "Lines of context around matches")
	rootCmd.PersistentFlags().StringVar(&flagColorMode, "color", "auto", "When to colorize output (auto|always|never)")
	rootCmd.PersistentFlags().BoolVar(&flagNoIndex, "no-index", false, "Skip indexing, use existing index only")
	rootCmd.PersistentFlags().BoolVar(&flagForceIndex, "force-index", false, "Force re-indexing even if index exists")
	rootCmd.PersistentFlags().BoolVar(&flagGPU, "gpu", false, "Force GPU acceleration")
	rootCmd.PersistentFlags().BoolVarP(&flagAuto, "auto", "a", true, "Auto-select optimal backend (default)")
	rootCmd.PersistentFlags().StringVar(&flagPreset, "preset", "", "Use preset (fast|accurate|balanced|large|interactive|deep)")
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
	indexCmd.Flags().IntVar(&flagIndexBatchSize, "batch-size", 1000, "Indexing batch size")
	indexCmd.Flags().BoolVar(&flagIndexWatch, "watch", false, "Keep watching filesystem changes and live-update the index")

	// Daemon-specific flags
	daemonCmd.Flags().IntVar(&flagDaemonPort, "port", 8765, "HTTP server port")
	daemonCmd.Flags().StringVar(&flagDaemonSocket, "socket", "", "Unix socket path")
	daemonCmd.Flags().DurationVar(&flagDaemonBatchDelay, "batch-delay", 100*time.Millisecond, "Debounce window for filesystem update batching")

	// Add subcommands
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(configCmd)
	rootCmd.AddCommand(daemonCmd)
}

func runDaemon(cmd *cobra.Command, args []string) error {
	config := DaemonConfig{
		WatchPaths: args,
		HTTPPort:   flagDaemonPort,
		SocketPath: flagDaemonSocket,
		Verbose:    flagVerbose,
		BatchDelay: flagDaemonBatchDelay,
	}

	daemon, err := NewBedDaemon(config)
	if err != nil {
		return err
	}

	return daemon.Run()
}

func runSearch(cmd *cobra.Command, args []string) error {
	query := strings.Join(args, " ")

	if flagVerbose {
		gobed.EnableDebugLogging()
	}

	type bedSearcher interface {
		Search(BedSearchOptions) error
		Close() error
	}

	var (
		srch    bedSearcher
		err     error
		gpuUsed bool
	)

	// Try fast int8 searcher with caching first (unless GPU requested)
	if !cmd.Flags().Changed("gpu") {
		if fastSearcher, e := NewFastBedSearcher(); e == nil {
			srch = fastSearcher
			if flagVerbose {
				fmt.Println("Using fast int8 searcher with caching")
			}
		} else if flagVerbose {
			fmt.Printf("Fast searcher unavailable: %v\n", e)
		}
	}

	if srch == nil && flagNoIndex {
		return fmt.Errorf("--no-index requires a reusable fast index cache in the current directory; run `bed index .` first or rerun without --no-index")
	}

	// Fallback to adaptive searcher
	if srch == nil && flagAuto {
		srch, err = NewAdaptiveSearcher()
		if err != nil {
			if flagVerbose {
				fmt.Printf("Adaptive searcher unavailable: %v\n", err)
			}
		}
	}

	// If --gpu flag explicitly set, try GPU
	if srch == nil && cmd.Flags().Changed("gpu") && flagGPU {
		if s, e := NewCAGRABedSearcher(); e == nil {
			srch = s
			gpuUsed = true
		} else {
			return fmt.Errorf("gpu requested but unavailable: %w", e)
		}
	}

	// Fallback to simple searcher
	if srch == nil {
		srch, err = NewSimpleBedSearcher()
		if err != nil {
			return fmt.Errorf("failed to initialize searcher: %w", err)
		}
	}
	defer srch.Close()

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

	// Apply preset if specified
	if flagPreset != "" {
		preset := GetPreset(flagPreset)
		if preset != nil {
			ApplyPreset(&options, preset)
			if flagVerbose {
				fmt.Printf("Using preset: %s (%s)\n", preset.Name, preset.Description)
			}
		} else {
			fmt.Printf("Unknown preset: %s\n", flagPreset)
			fmt.Println("Available presets: fast, accurate, balanced, large, interactive, deep")
		}
	}

	return srch.Search(options)
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

	if flagIndexWatch {
		config := DaemonConfig{
			WatchPaths: []string{path},
			HTTPPort:   0,
			SocketPath: "",
			Verbose:    flagVerbose,
			BatchDelay: flagDaemonBatchDelay,
		}

		daemon, err := NewBedDaemon(config)
		if err != nil {
			return fmt.Errorf("failed to start watch mode: %w", err)
		}
		return daemon.Run()
	}

	searcher, err := NewFastBedSearcher()
	if err == nil {
		defer searcher.Close()

		start := time.Now()
		if err := searcher.IndexDirectory(path, BedSearchOptions{
			ForceIndex:     flagForceIndex,
			SearchBinaries: flagSearchBinaries,
			UseGPU:         useGPU,
			Verbose:        flagVerbose,
		}); err != nil {
			return err
		}

		fmt.Printf("Indexed %d documents in %.2fs\n", searcher.NumDocuments(), time.Since(start).Seconds())
		return nil
	}

	if flagVerbose {
		fmt.Printf("Fast indexer unavailable (%v), falling back to full model indexer\n", err)
	}

	indexer, legacyErr := NewIndexer()
	if legacyErr != nil {
		return fmt.Errorf("failed to initialize indexer: %w", legacyErr)
	}
	defer indexer.Close()

	batchSize := flagIndexBatchSize
	if batchSize <= 0 {
		batchSize = 1000
	}

	legacyOpts := DefaultIndexOptions()
	legacyOpts.Path = path
	legacyOpts.Force = flagForceIndex
	legacyOpts.BatchSize = batchSize
	legacyOpts.UseGPU = useGPU
	legacyOpts.Verbose = flagVerbose

	return indexer.Index(legacyOpts)
}

func runStatus(cmd *cobra.Command, args []string) error {
	fmt.Println("Bed Search Status")
	fmt.Println("=================")
	fmt.Println()

	// Get optimizer capabilities
	opt := GetOptimizer()
	caps := opt.GetCapabilities()
	config := opt.Optimize()

	fmt.Printf("System Capabilities:\n")
	fmt.Printf("  CPU cores:     %d\n", caps.NumCPUs)
	fmt.Printf("  AVX2 support:  %v\n", caps.HasAVX2)
	fmt.Printf("  AVX-512:       %v\n", caps.HasAVX512)
	if caps.GPUAvailable {
		fmt.Printf("  GPU:           %s (%d MB)\n", caps.GPUName, caps.GPUMemoryMB)
	} else {
		fmt.Printf("  GPU:           Not available\n")
	}
	fmt.Println()

	fmt.Printf("Auto-Optimization:\n")
	fmt.Printf("  Backend:       %s\n", config.Backend)
	fmt.Printf("  Workers:       %d\n", config.NumWorkers)
	fmt.Printf("  Batch size:    %d\n", config.BatchSize)
	fmt.Printf("  Int8 mode:     %v\n", config.UseInt8)
	fmt.Printf("  Reason:        %s\n", config.Reason)
	fmt.Println()

	fmt.Printf("Features:\n")
	fmt.Printf("  Natural language:  Enabled\n")
	fmt.Printf("  Query enhancement: Enabled\n")
	fmt.Printf("  Gitignore:         Enabled\n")
	fmt.Printf("  Binary detection:  Enabled\n")

	return nil
}

func runConfig(cmd *cobra.Command, args []string) error {
	config, err := LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	return config.Display()
}

func cliVersion() string {
	version := buildVersion
	if version == "" {
		version = "dev"
	}
	if buildCommit != "" {
		version += "+" + buildCommit
	}
	if buildDate != "" {
		version += " (" + buildDate + ")"
	}
	return version
}
