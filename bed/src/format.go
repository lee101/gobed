package src

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/fatih/color"
)

// OutputFormatter handles colorized output formatting like ripgrep
type OutputFormatter struct {
	colorMode     string
	showLineNums  bool
	showContext   bool
	contextLines  int
	highlightFunc func(string) string
	pathFunc      func(string) string
	lineNumFunc   func(string) string
	matchFunc     func(string) string
	contextFunc   func(string) string
}

// NewOutputFormatter creates a new formatter
func NewOutputFormatter(colorMode string) *OutputFormatter {
	formatter := &OutputFormatter{
		colorMode:    colorMode,
		showLineNums: true,
		showContext:  true,
		contextLines: 2,
	}
	
	// Setup color functions based on mode
	if formatter.shouldUseColor() {
		pathColor := color.New(color.FgMagenta, color.Bold)
		lineNumColor := color.New(color.FgGreen, color.Bold)
		matchColor := color.New(color.FgRed, color.Bold)
		contextColor := color.New(color.FgBlue)
		highlightColor := color.New(color.BgYellow, color.FgBlack)
		
		formatter.pathFunc = func(s string) string { return pathColor.Sprint(s) }
		formatter.lineNumFunc = func(s string) string { return lineNumColor.Sprint(s) }
		formatter.matchFunc = func(s string) string { return matchColor.Sprint(s) }
		formatter.contextFunc = func(s string) string { return contextColor.Sprint(s) }
		formatter.highlightFunc = func(s string) string { return highlightColor.Sprint(s) }
	} else {
		// No-color functions
		formatter.pathFunc = func(s string) string { return s }
		formatter.lineNumFunc = func(s string) string { return s }
		formatter.matchFunc = func(s string) string { return s }
		formatter.contextFunc = func(s string) string { return s }
		formatter.highlightFunc = func(s string) string { return s }
	}
	
	return formatter
}

// FormatSearchResults formats search results like ripgrep
func (f *OutputFormatter) FormatSearchResults(results []*SearchResult, query string) string {
	if len(results) == 0 {
		return fmt.Sprintf("No results found for: %s\n", query)
	}
	
	var output strings.Builder
	
	// Group results by file for cleaner output
	fileGroups := f.groupResultsByFile(results)
	
	for i, group := range fileGroups {
		if i > 0 {
			output.WriteString("\n") // Separator between files
		}
		
		// Write file header
		output.WriteString(f.formatFileHeader(group.FilePath, len(group.Results)))
		output.WriteString("\n")
		
		// Write results for this file
		for j, result := range group.Results {
			if j > 0 {
				output.WriteString("--\n") // Context separator
			}
			output.WriteString(f.formatSingleResult(result, query))
		}
	}
	
	return output.String()
}

// FileResultGroup groups results by file
type FileResultGroup struct {
	FilePath string
	Results  []*SearchResult
}

// groupResultsByFile groups search results by file path
func (f *OutputFormatter) groupResultsByFile(results []*SearchResult) []FileResultGroup {
	groups := make(map[string]*FileResultGroup)
	var orderedFiles []string
	
	for _, result := range results {
		if group, exists := groups[result.FilePath]; exists {
			group.Results = append(group.Results, result)
		} else {
			groups[result.FilePath] = &FileResultGroup{
				FilePath: result.FilePath,
				Results:  []*SearchResult{result},
			}
			orderedFiles = append(orderedFiles, result.FilePath)
		}
	}
	
	// Convert to ordered slice
	orderedGroups := make([]FileResultGroup, len(orderedFiles))
	for i, filePath := range orderedFiles {
		orderedGroups[i] = *groups[filePath]
	}
	
	return orderedGroups
}

// formatFileHeader formats the file path header
func (f *OutputFormatter) formatFileHeader(filePath string, resultCount int) string {
	if f.shouldUseColor() {
		return fmt.Sprintf("%s (%d matches)", 
			f.pathFunc(filePath), 
			resultCount)
	}
	return fmt.Sprintf("%s (%d matches)", filePath, resultCount)
}

// formatSingleResult formats a single search result with context
func (f *OutputFormatter) formatSingleResult(result *SearchResult, query string) string {
	var output strings.Builder
	
	// Format context before (if any)
	if result.ContextBefore != "" {
		contextLines := strings.Split(result.ContextBefore, "\n")
		for i, line := range contextLines {
			if line == "" {
				continue
			}
			lineNum := result.LineNumber - len(contextLines) + i
			output.WriteString(f.formatContextLine(lineNum, line))
			output.WriteString("\n")
		}
	}
	
	// Format the main matching line
	output.WriteString(f.formatMatchingLine(result, query))
	output.WriteString("\n")
	
	// Format context after (if any)
	if result.ContextAfter != "" {
		contextLines := strings.Split(result.ContextAfter, "\n")
		for i, line := range contextLines {
			if line == "" {
				continue
			}
			lineNum := result.LineNumber + 1 + i
			output.WriteString(f.formatContextLine(lineNum, line))
			output.WriteString("\n")
		}
	}
	
	return output.String()
}

// formatMatchingLine formats the main matching line with highlighting
func (f *OutputFormatter) formatMatchingLine(result *SearchResult, query string) string {
	lineNumStr := fmt.Sprintf("%d", result.LineNumber)
	content := result.Content
	
	// Highlight potential matches in the content
	// This is a simple approach - in practice, you might want more sophisticated highlighting
	highlighted := f.highlightMatches(content, query)
	
	// Format similarity score
	similarity := fmt.Sprintf("(%.3f)", result.Similarity)
	
	if f.shouldUseColor() {
		return fmt.Sprintf("%s:%s %s %s", 
			f.lineNumFunc(lineNumStr),
			f.matchFunc("→"),
			highlighted,
			f.contextFunc(similarity))
	}
	
	return fmt.Sprintf("%s:→ %s %s", lineNumStr, highlighted, similarity)
}

// formatContextLine formats a context line
func (f *OutputFormatter) formatContextLine(lineNum int, content string) string {
	lineNumStr := fmt.Sprintf("%d", lineNum)
	
	if f.shouldUseColor() {
		return fmt.Sprintf("%s%s %s", 
			f.lineNumFunc(lineNumStr),
			f.contextFunc("-"),
			content)
	}
	
	return fmt.Sprintf("%s- %s", lineNumStr, content)
}

// highlightMatches highlights potential matches in the content
func (f *OutputFormatter) highlightMatches(content, query string) string {
	if !f.shouldUseColor() {
		return content
	}
	
	// Simple keyword highlighting - split query into words and highlight each
	words := strings.Fields(strings.ToLower(query))
	result := content
	
	for _, word := range words {
		if len(word) < 3 { // Skip very short words
			continue
		}
		
		// Create case-insensitive regex
		pattern := fmt.Sprintf("(?i)\\b%s\\w*", regexp.QuoteMeta(word))
		re, err := regexp.Compile(pattern)
		if err != nil {
			continue
		}
		
		// Replace matches with highlighted version
		result = re.ReplaceAllStringFunc(result, func(match string) string {
			return f.highlightFunc(match)
		})
	}
	
	return result
}

// FormatProgressBar formats a progress bar for indexing
func (f *OutputFormatter) FormatProgressBar(current, total int64, rate float64, eta string) string {
	if !f.shouldUseColor() {
		return fmt.Sprintf("Progress: %d/%d (%.1f files/sec) ETA: %s", 
			current, total, rate, eta)
	}
	
	percentage := float64(current) / float64(total) * 100
	barLength := 40
	filled := int(percentage / 100 * float64(barLength))
	
	bar := fmt.Sprintf("[%s%s]", 
		strings.Repeat("█", filled),
		strings.Repeat("░", barLength-filled))
	
	return fmt.Sprintf("%s %s %.1f%% (%d/%d) %.1f files/sec ETA: %s",
		f.contextFunc("Indexing"),
		f.matchFunc(bar),
		percentage,
		current, total,
		rate,
		f.contextFunc(eta))
}

// FormatStatistics formats indexing/search statistics
func (f *OutputFormatter) FormatStatistics(stats map[string]interface{}) string {
	var output strings.Builder
	
	output.WriteString(f.formatHeader("Statistics"))
	output.WriteString("\n")
	
	// Define order and formatting for common statistics
	statOrder := []struct {
		key    string
		label  string
		format string
	}{
		{"total_files", "Files indexed", "%d"},
		{"total_lines", "Lines indexed", "%d"},
		{"processed_files", "Files processed", "%d"},
		{"error_count", "Errors", "%d"},
		{"processing_time", "Processing time", "%s"},
		{"processing_rate", "Processing rate", "%s"},
		{"memory_usage", "Memory usage", "%.2f MB"},
		{"using_gpu", "GPU acceleration", "%v"},
		{"gpu_device_id", "GPU device", "%d"},
	}
	
	for _, stat := range statOrder {
		if value, exists := stats[stat.key]; exists {
			formatted := f.formatStatistic(stat.label, value, stat.format)
			if formatted != "" {
				output.WriteString(formatted)
				output.WriteString("\n")
			}
		}
	}
	
	return output.String()
}

// formatHeader formats a section header
func (f *OutputFormatter) formatHeader(title string) string {
	if f.shouldUseColor() {
		return f.pathFunc(fmt.Sprintf("=== %s ===", title))
	}
	return fmt.Sprintf("=== %s ===", title)
}

// formatStatistic formats a single statistic
func (f *OutputFormatter) formatStatistic(label string, value interface{}, format string) string {
	var formatted string
	
	switch v := value.(type) {
	case int:
		formatted = fmt.Sprintf(format, v)
	case int64:
		if strings.Contains(format, "MB") {
			formatted = fmt.Sprintf(format, float64(v)/1024/1024)
		} else {
			formatted = fmt.Sprintf(format, v)
		}
	case float64:
		formatted = fmt.Sprintf(format, v)
	case string:
		formatted = fmt.Sprintf(format, v)
	case bool:
		if v {
			if f.shouldUseColor() {
				formatted = f.matchFunc("Yes ")
			} else {
				formatted = "Yes"
			}
		} else {
			formatted = "No"
		}
	default:
		formatted = fmt.Sprintf("%v", v)
	}
	
	if f.shouldUseColor() {
		return fmt.Sprintf("  %s: %s", f.lineNumFunc(label), formatted)
	}
	
	return fmt.Sprintf("  %s: %s", label, formatted)
}

// FormatError formats an error message
func (f *OutputFormatter) FormatError(err error) string {
	if f.shouldUseColor() {
		return f.matchFunc(fmt.Sprintf("Error: %v", err))
	}
	return fmt.Sprintf("Error: %v", err)
}

// FormatWarning formats a warning message
func (f *OutputFormatter) FormatWarning(message string) string {
	if f.shouldUseColor() {
		return color.New(color.FgYellow).Sprintf("Warning: %s", message)
	}
	return fmt.Sprintf("Warning: %s", message)
}

// FormatSuccess formats a success message
func (f *OutputFormatter) FormatSuccess(message string) string {
	if f.shouldUseColor() {
		return color.New(color.FgGreen).Sprintf("✓ %s", message)
	}
	return fmt.Sprintf("✓ %s", message)
}

// shouldUseColor determines if color should be used
func (f *OutputFormatter) shouldUseColor() bool {
	switch f.colorMode {
	case "always":
		return true
	case "never":
		return false
	case "auto":
		return isTerminal()
	default:
		return false
	}
}

// SetContextLines sets the number of context lines to show
func (f *OutputFormatter) SetContextLines(lines int) {
	f.contextLines = lines
}

// SetLineNumbers enables or disables line number display
func (f *OutputFormatter) SetLineNumbers(show bool) {
	f.showLineNums = show
}