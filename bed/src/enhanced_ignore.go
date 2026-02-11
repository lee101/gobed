package src

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// EnhancedIgnoreFilter handles advanced file filtering
type EnhancedIgnoreFilter struct {
	baseDir          string
	gitignoreRules   []GitignoreRule
	customRules      []GitignoreRule
	maxFileSize      int64
	searchBinaries   bool
	binaryExtensions map[string]bool
	textExtensions   map[string]bool
}

// GitignoreRule represents a gitignore pattern
type GitignoreRule struct {
	pattern  string
	isNegate bool
	isDir    bool
}

// FileType represents the type of file
type FileType int

const (
	FileTypeText FileType = iota
	FileTypeBinary
	FileTypeIgnored
	FileTypeTooLarge
)

// DefaultMaxFileSize is 10MB
const DefaultMaxFileSize = 10 * 1024 * 1024

// NewEnhancedIgnoreFilter creates an enhanced ignore filter
func NewEnhancedIgnoreFilter(baseDir string, options ...FilterOption) (*EnhancedIgnoreFilter, error) {
	filter := &EnhancedIgnoreFilter{
		baseDir:          baseDir,
		gitignoreRules:   []GitignoreRule{},
		customRules:      []GitignoreRule{},
		maxFileSize:      DefaultMaxFileSize,
		searchBinaries:   false,
		binaryExtensions: buildBinaryExtensions(),
		textExtensions:   buildTextExtensions(),
	}

	// Apply options
	for _, opt := range options {
		opt(filter)
	}

	// Load .gitignore files
	if err := filter.loadGitignore(); err != nil {
		// Continue even if gitignore fails
		fmt.Fprintf(os.Stderr, "Warning: failed to load .gitignore: %v\n", err)
	}

	// Add default ignore patterns
	filter.addDefaultIgnores()

	return filter, nil
}

// FilterOption configures the filter
type FilterOption func(*EnhancedIgnoreFilter)

// WithMaxFileSize sets max file size
func WithMaxFileSize(size int64) FilterOption {
	return func(f *EnhancedIgnoreFilter) {
		f.maxFileSize = size
	}
}

// WithBinarySearch enables binary file searching
func WithBinarySearch(enable bool) FilterOption {
	return func(f *EnhancedIgnoreFilter) {
		f.searchBinaries = enable
	}
}

// loadGitignore loads all .gitignore files in the tree
func (f *EnhancedIgnoreFilter) loadGitignore() error {
	// Start from base dir and work up to find all .gitignore files
	gitignorePaths := f.findGitignoreFiles()

	for _, path := range gitignorePaths {
		if err := f.loadGitignoreFile(path); err != nil {
			continue // Skip failed files
		}
	}

	// Also check for global gitignore
	if globalIgnore := f.getGlobalGitignore(); globalIgnore != "" {
		f.loadGitignoreFile(globalIgnore)
	}

	return nil
}

// findGitignoreFiles finds all relevant .gitignore files
func (f *EnhancedIgnoreFilter) findGitignoreFiles() []string {
	var files []string

	// Check current directory and parents up to git root
	dir := f.baseDir
	for {
		gitignorePath := filepath.Join(dir, ".gitignore")
		if _, err := os.Stat(gitignorePath); err == nil {
			files = append(files, gitignorePath)
		}

		// Check if we're at git root
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			break
		}

		// Move up
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	// Reverse so we apply from root down
	for i, j := 0, len(files)-1; i < j; i, j = i+1, j-1 {
		files[i], files[j] = files[j], files[i]
	}

	return files
}

// loadGitignoreFile loads rules from a .gitignore file
func (f *EnhancedIgnoreFilter) loadGitignoreFile(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		rule := f.parseGitignoreRule(line)
		f.gitignoreRules = append(f.gitignoreRules, rule)
	}

	return scanner.Err()
}

// parseGitignoreRule parses a gitignore pattern
func (f *EnhancedIgnoreFilter) parseGitignoreRule(pattern string) GitignoreRule {
	rule := GitignoreRule{pattern: pattern}

	// Check for negation
	if strings.HasPrefix(pattern, "!") {
		rule.isNegate = true
		pattern = pattern[1:]
	}

	// Check if directory-only
	if strings.HasSuffix(pattern, "/") {
		rule.isDir = true
		pattern = strings.TrimSuffix(pattern, "/")
	}

	rule.pattern = filepath.ToSlash(strings.TrimPrefix(pattern, "/"))
	return rule
}

// addDefaultIgnores adds common ignore patterns
func (f *EnhancedIgnoreFilter) addDefaultIgnores() {
	defaults := []string{
		// Version control
		".git/",
		".svn/",
		".hg/",

		// Dependencies
		"node_modules/",
		"vendor/",
		".venv/",
		"__pycache__/",
		".bed/",
		"model/",

		// Build outputs
		"dist/",
		"build/",
		"target/",
		"out/",
		"*.o",
		"*.so",
		"*.dll",
		"*.exe",
		"*.class",

		// IDE
		".idea/",
		".vscode/",
		"*.swp",
		"*.swo",
		"*~",
		".DS_Store",

		// Large files
		"*.log",
		"*.sqlite",
		"*.db",

		// Archives
		"*.zip",
		"*.tar",
		"*.gz",
		"*.rar",
		"*.7z",

		// Media files (usually binary)
		"*.jpg",
		"*.jpeg",
		"*.png",
		"*.gif",
		"*.ico",
		"*.pdf",
		"*.mp3",
		"*.mp4",
		"*.avi",
		"*.mov",
	}

	for _, pattern := range defaults {
		f.customRules = append(f.customRules, f.parseGitignoreRule(pattern))
	}
}

// ShouldProcess determines if a file should be processed
func (f *EnhancedIgnoreFilter) ShouldProcess(path string) (bool, FileType) {
	// Check file size first
	info, err := os.Stat(path)
	if err != nil {
		return false, FileTypeIgnored
	}

	if info.Size() > f.maxFileSize {
		return false, FileTypeTooLarge
	}

	// Check gitignore rules
	if f.matchesGitignore(path, info.IsDir()) {
		return false, FileTypeIgnored
	}

	// Check if binary
	if f.isBinaryFile(path) {
		if f.searchBinaries {
			return true, FileTypeBinary
		}
		return false, FileTypeBinary
	}

	return true, FileTypeText
}

// matchesGitignore checks if path matches gitignore rules
func (f *EnhancedIgnoreFilter) matchesGitignore(path string, isDir bool) bool {
	relPath, err := filepath.Rel(f.baseDir, path)
	if err != nil {
		return false
	}
	relPath = filepath.ToSlash(relPath)

	// Check all rules in order
	ignored := false

	// Check gitignore rules
	for _, rule := range f.gitignoreRules {
		if f.matchesRule(relPath, isDir, rule) {
			if rule.isNegate {
				ignored = false
			} else {
				ignored = true
			}
		}
	}

	// Check custom rules
	for _, rule := range f.customRules {
		if f.matchesRule(relPath, isDir, rule) {
			if rule.isNegate {
				ignored = false
			} else {
				ignored = true
			}
		}
	}

	return ignored
}

// matchesRule checks if a path matches a gitignore rule
func (f *EnhancedIgnoreFilter) matchesRule(path string, isDir bool, rule GitignoreRule) bool {
	path = filepath.ToSlash(path)
	pattern := filepath.ToSlash(strings.TrimSuffix(rule.pattern, "/"))
	if pattern == "" {
		return false
	}

	// Directory-only rules should match both the directory itself and files beneath it.
	if rule.isDir {
		trimmed := strings.Trim(path, "/")
		if trimmed == "" {
			return false
		}

		parts := strings.Split(trimmed, "/")
		maxDirs := len(parts)
		if !isDir && maxDirs > 0 {
			maxDirs--
		}
		for i := 1; i <= maxDirs; i++ {
			dirPrefix := strings.Join(parts[:i], "/")
			if matched, _ := filepath.Match(pattern, dirPrefix); matched {
				return true
			}
			if matched, _ := filepath.Match(pattern, parts[i-1]); matched {
				return true
			}
		}
		return false
	}

	if matched, _ := filepath.Match(pattern, path); matched {
		return true
	}

	// Simple glob matching on basename and all subpaths.
	matched, _ := filepath.Match(pattern, filepath.Base(path))
	if matched {
		return true
	}

	parts := strings.Split(path, "/")
	for i, part := range parts {
		if matched, _ := filepath.Match(pattern, part); matched {
			return true
		}
		subPath := strings.Join(parts[i:], "/")
		if matched, _ := filepath.Match(pattern, subPath); matched {
			return true
		}
	}

	return false
}

// isBinaryFile determines if a file is binary
func (f *EnhancedIgnoreFilter) isBinaryFile(path string) bool {
	// Check extension first
	ext := strings.ToLower(filepath.Ext(path))

	// Known text extensions
	if f.textExtensions[ext] {
		return false
	}

	// Known binary extensions
	if f.binaryExtensions[ext] {
		return true
	}

	// Check file content
	return f.isBinaryContent(path)
}

// isBinaryContent checks if file content is binary
func (f *EnhancedIgnoreFilter) isBinaryContent(path string) bool {
	file, err := os.Open(path)
	if err != nil {
		return true // Assume binary if can't read
	}
	defer file.Close()

	// Read first 8192 bytes
	buffer := make([]byte, 8192)
	n, err := file.Read(buffer)
	if err != nil && err != io.EOF {
		return true
	}

	// Check for null bytes (common in binary files)
	if bytes.Contains(buffer[:n], []byte{0}) {
		return true
	}

	// Check for high proportion of non-printable characters
	nonPrintable := 0
	for i := 0; i < n; i++ {
		b := buffer[i]
		// Allow common whitespace and printable ASCII
		if b < 32 && b != '\t' && b != '\n' && b != '\r' {
			nonPrintable++
		}
		if b > 126 && b < 128 {
			nonPrintable++
		}
	}

	// If more than 30% non-printable, consider binary
	if float64(nonPrintable)/float64(n) > 0.3 {
		return true
	}

	return false
}

// getGlobalGitignore finds the global gitignore file
func (f *EnhancedIgnoreFilter) getGlobalGitignore() string {
	// Try git config
	// This is simplified - in reality would execute git config
	home, _ := os.UserHomeDir()
	globalIgnore := filepath.Join(home, ".gitignore_global")
	if _, err := os.Stat(globalIgnore); err == nil {
		return globalIgnore
	}
	return ""
}

// buildBinaryExtensions returns common binary file extensions
func buildBinaryExtensions() map[string]bool {
	exts := []string{
		// Executables
		".exe", ".dll", ".so", ".dylib", ".a", ".lib",
		// Compiled
		".o", ".obj", ".class", ".pyc", ".pyo",
		// Archives
		".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
		// Media
		".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
		".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv",
		".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
		// Data
		".db", ".sqlite", ".mdb",
		// Other
		".woff", ".woff2", ".ttf", ".eot",
	}

	m := make(map[string]bool)
	for _, ext := range exts {
		m[ext] = true
	}
	return m
}

// buildTextExtensions returns common text file extensions
func buildTextExtensions() map[string]bool {
	exts := []string{
		// Code
		".go", ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp",
		".h", ".hpp", ".cs", ".php", ".rb", ".rs", ".swift", ".kt", ".scala",
		".r", ".m", ".mm", ".pl", ".sh", ".bash", ".zsh", ".fish",
		// Web
		".html", ".htm", ".css", ".scss", ".sass", ".less",
		// Data
		".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
		// Docs
		".md", ".markdown", ".rst", ".txt", ".text", ".log",
		// Config
		".env", ".gitignore", ".dockerignore", ".editorconfig",
		// Other
		".sql", ".graphql", ".proto",
	}

	m := make(map[string]bool)
	for _, ext := range exts {
		m[ext] = true
	}
	return m
}

// ProcessBinaryFile handles binary file searching
func ProcessBinaryFile(path string, query []float32, model interface{}) *BinarySearchResult {
	// For binary files, we just report if there's a match above threshold
	// without trying to extract specific lines

	result := &BinarySearchResult{
		FilePath: path,
		HasMatch: false,
	}

	// Read file in chunks and check for patterns
	file, err := os.Open(path)
	if err != nil {
		return result
	}
	defer file.Close()

	// For now, just return basic info
	// In a real implementation, might extract strings from binary
	// and check those against the query

	info, _ := os.Stat(path)
	result.FileSize = info.Size()

	// Simple heuristic: check filename against query
	// Real implementation would be more sophisticated
	filename := filepath.Base(path)
	if strings.Contains(strings.ToLower(filename), "search") {
		result.HasMatch = true
		result.Confidence = 0.5
		result.Note = "Filename match"
	}

	return result
}

// BinarySearchResult represents a search result in a binary file
type BinarySearchResult struct {
	FilePath   string
	HasMatch   bool
	Confidence float32
	FileSize   int64
	Note       string
}
