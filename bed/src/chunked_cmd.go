package src

import (
	"fmt"

	"github.com/spf13/cobra"
)

var (
	chunkPattern      string
	chunkCaseSensitive bool
	chunkMaxFileSize  int64
	chunkMaxLineLen   int
	chunkContextChars int
	chunkNoColor      bool
	chunkDirectory    string
)

// chunkedCmd represents the chunked search command
var chunkedCmd = &cobra.Command{
	Use:   "chunk [pattern]",
	Short: "Token-friendly text search with chunked output",
	Long: `Performs fast text search with limited output per match.
Similar to ripgrep but chunks long lines to avoid token overflow.

Examples:
  bed chunk emoji                    # Search for 'emoji' in current directory
  bed chunk -d /path/to/dir emoji    # Search in specific directory
  bed chunk -i pattern               # Case insensitive search
  bed chunk --max-line 100 pattern   # Limit line length to 100 chars`,
	Args: cobra.ExactArgs(1),
	RunE: runChunkedSearch,
}

func init() {
	rootCmd.AddCommand(chunkedCmd)

	chunkedCmd.Flags().BoolVarP(&chunkCaseSensitive, "case-sensitive", "s", true, "Case sensitive search")
	chunkedCmd.Flags().BoolVarP(&chunkCaseSensitive, "ignore-case", "i", false, "Case insensitive search")
	chunkedCmd.Flags().Int64Var(&chunkMaxFileSize, "max-file-size", 10<<20, "Maximum file size to search (bytes)")
	chunkedCmd.Flags().IntVar(&chunkMaxLineLen, "max-line", 200, "Maximum line length to display")
	chunkedCmd.Flags().IntVar(&chunkContextChars, "context", 80, "Context characters around match")
	chunkedCmd.Flags().BoolVar(&chunkNoColor, "no-color", false, "Disable colored output")
	chunkedCmd.Flags().StringVarP(&chunkDirectory, "directory", "d", ".", "Directory to search")
}

func runChunkedSearch(cmd *cobra.Command, args []string) error {
	pattern := args[0]

	// Use GPU-accelerated semantic search with chunked output
	searcher, err := NewGPUChunkedSearcher()
	if err != nil {
		return fmt.Errorf("failed to create GPU searcher: %w", err)
	}
	defer searcher.Close()

	// Configure search options for GPU-accelerated semantic search
	options := BedSearchOptions{
		Query:          pattern,
		Limit:          20, // Show more results for semantic search
		Context:        chunkContextChars / 40, // Convert chars to lines
		Threshold:      0.5, // Lower threshold for broader semantic matches
		NoIndex:        false,
		ForceIndex:     false,
		SearchBinaries: false,
		Progressive:    false,
		UseGPU:         true, // Enable GPU acceleration
		Verbose:        !chunkNoColor,
		ColorMode:      "auto",
	}

	// Handle ignore-case flag (less relevant for semantic search)
	ignoreCase, _ := cmd.Flags().GetBool("ignore-case")
	if ignoreCase {
		// Semantic search is naturally case-insensitive
		options.Threshold = 0.4 // Lower threshold for more flexible matching
	}

	// Override with custom settings
	if chunkMaxLineLen > 0 {
		searcher.maxLineLen = chunkMaxLineLen
	}
	if chunkContextChars > 0 {
		searcher.contextChars = chunkContextChars
	}

	// Change directory if specified
	if chunkDirectory != "." {
		if err := searcher.engine.IndexDirectory(chunkDirectory, options); err != nil {
			return fmt.Errorf("failed to index directory %s: %w", chunkDirectory, err)
		}
	}

	// Perform GPU-accelerated search with chunked output
	return searcher.SearchWithChunkedOutput(options)
}

