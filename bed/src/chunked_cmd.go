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

	// Handle ignore-case flag
	ignoreCase, _ := cmd.Flags().GetBool("ignore-case")
	if ignoreCase {
		chunkCaseSensitive = false
	}

	// Create chunked searcher
	searcher, err := NewChunkedSearcher()
	if err != nil {
		return fmt.Errorf("failed to create searcher: %w", err)
	}

	// Configure searcher
	searcher.maxFileSize = chunkMaxFileSize
	searcher.maxLineLen = chunkMaxLineLen
	searcher.contextChars = chunkContextChars

	// Perform search
	results, err := searcher.SearchDirectory(chunkDirectory, pattern, chunkCaseSensitive)
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}

	// Determine color usage
	useColor := !chunkNoColor && isTerminal()

	// Print results
	searcher.PrintResults(results, useColor)

	return nil
}

