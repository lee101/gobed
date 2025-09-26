package src

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// IgnoreFilter handles .bedignore and .gitignore pattern matching
type IgnoreFilter struct {
	patterns    []*ignorePattern
	gitPatterns []*ignorePattern
	bedPatterns []*ignorePattern
	basePath    string
}

type ignorePattern struct {
	pattern    string
	regex      *regexp.Regexp
	isNegation bool
	isDir      bool
	isGlobal   bool
}

// NewIgnoreFilter creates a new ignore filter for the given directory
func NewIgnoreFilter(basePath string, respectGitignore bool) (*IgnoreFilter, error) {
	filter := &IgnoreFilter{
		basePath: basePath,
		patterns: make([]*ignorePattern, 0),
	}

	// Load .bedignore patterns
	bedignorePath := filepath.Join(basePath, ".bedignore")
	if bedPatterns, err := loadIgnoreFile(bedignorePath); err == nil {
		filter.bedPatterns = bedPatterns
		filter.patterns = append(filter.patterns, bedPatterns...)
	}

	// Load .gitignore patterns if requested
	if respectGitignore {
		gitignorePath := filepath.Join(basePath, ".gitignore")
		if gitPatterns, err := loadIgnoreFile(gitignorePath); err == nil {
			filter.gitPatterns = gitPatterns
			filter.patterns = append(filter.patterns, gitPatterns...)
		}

		// Also load global gitignore
		if globalGitignore := getGlobalGitignore(); globalGitignore != "" {
			if globalPatterns, err := loadIgnoreFile(globalGitignore); err == nil {
				filter.patterns = append(filter.patterns, globalPatterns...)
			}
		}
	}

	return filter, nil
}

// ShouldIgnore returns true if the given path should be ignored
func (f *IgnoreFilter) ShouldIgnore(path string) bool {
	// Make path relative to base
	relPath, err := filepath.Rel(f.basePath, path)
	if err != nil {
		relPath = path
	}

	// Normalize path separators
	relPath = filepath.ToSlash(relPath)

	// Check against all patterns
	ignored := false
	for _, pattern := range f.patterns {
		if pattern.matches(relPath) {
			if pattern.isNegation {
				ignored = false // Un-ignore
			} else {
				ignored = true // Ignore
			}
		}
	}

	return ignored
}

// IsTextFile returns true if the file appears to be text-based
func (f *IgnoreFilter) IsTextFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	
	// Known text extensions
	textExtensions := map[string]bool{
		".txt": true, ".md": true, ".rst": true, ".tex": true,
		".go": true, ".py": true, ".js": true, ".ts": true,
		".java": true, ".c": true, ".cpp": true, ".h": true, ".hpp": true,
		".rs": true, ".rb": true, ".php": true, ".swift": true, ".kt": true,
		".scala": true, ".clj": true, ".hs": true, ".r": true, ".m": true,
		".cs": true, ".vb": true, ".fs": true, ".ml": true, ".elm": true,
		".html": true, ".css": true, ".scss": true, ".sass": true, ".less": true,
		".vue": true, ".jsx": true, ".tsx": true, ".xml": true, ".json": true,
		".yaml": true, ".yml": true, ".toml": true, ".ini": true, ".conf": true,
		".sh": true, ".bash": true, ".zsh": true, ".fish": true, ".ps1": true,
		".bat": true, ".cmd": true, ".sql": true, ".proto": true, ".graphql": true,
		".dockerfile": true, ".makefile": true, ".cmake": true, ".gradle": true,
		".properties": true, ".env": true, ".gitignore": true, ".dockerignore": true,
	}

	if textExtensions[ext] {
		return true
	}

	// Files without extension might be text (like Makefile, README, etc.)
	if ext == "" {
		name := strings.ToLower(filepath.Base(path))
		commonTextFiles := map[string]bool{
			"readme": true, "license": true, "changelog": true, "makefile": true,
			"dockerfile": true, "rakefile": true, "gemfile": true, "podfile": true,
			"cmakelist": true, "authors": true, "contributors": true, "copying": true,
		}
		
		if commonTextFiles[name] {
			return true
		}
	}

	return false
}

// AddPattern adds a custom ignore pattern
func (f *IgnoreFilter) AddPattern(pattern string) error {
	p, err := parseIgnorePattern(pattern)
	if err != nil {
		return err
	}
	f.patterns = append(f.patterns, p)
	return nil
}

// loadIgnoreFile loads patterns from an ignore file
func loadIgnoreFile(path string) ([]*ignorePattern, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var patterns []*ignorePattern
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		
		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		if pattern, err := parseIgnorePattern(line); err == nil {
			patterns = append(patterns, pattern)
		}
	}

	return patterns, scanner.Err()
}

// parseIgnorePattern parses a gitignore-style pattern
func parseIgnorePattern(pattern string) (*ignorePattern, error) {
	p := &ignorePattern{
		pattern: pattern,
	}

	// Handle negation
	if strings.HasPrefix(pattern, "!") {
		p.isNegation = true
		pattern = pattern[1:]
	}

	// Handle directory-only patterns
	if strings.HasSuffix(pattern, "/") {
		p.isDir = true
		pattern = strings.TrimSuffix(pattern, "/")
	}

	// Handle global patterns (starting with /)
	if strings.HasPrefix(pattern, "/") {
		p.isGlobal = true
		pattern = pattern[1:]
	}

	// Convert gitignore pattern to regex
	regexPattern := gitignoreToRegex(pattern)
	
	regex, err := regexp.Compile(regexPattern)
	if err != nil {
		return nil, err
	}
	
	p.regex = regex
	return p, nil
}

// matches checks if a path matches this pattern
func (p *ignorePattern) matches(path string) bool {
	// For directory patterns, only match directories
	if p.isDir && !strings.HasSuffix(path, "/") {
		// Check if any parent directory matches
		parts := strings.Split(path, "/")
		for i := 1; i <= len(parts); i++ {
			dirPath := strings.Join(parts[:i], "/") + "/"
			if p.regex.MatchString(dirPath) {
				return true
			}
		}
		return false
	}

	// For global patterns, match from root
	if p.isGlobal {
		return p.regex.MatchString(path)
	}

	// For relative patterns, match any level
	parts := strings.Split(path, "/")
	for i := 0; i < len(parts); i++ {
		subPath := strings.Join(parts[i:], "/")
		if p.regex.MatchString(subPath) {
			return true
		}
	}

	return false
}

// gitignoreToRegex converts a gitignore pattern to a regex pattern
func gitignoreToRegex(pattern string) string {
	// Escape special regex characters except * and ?
	pattern = regexp.QuoteMeta(pattern)
	
	// Convert gitignore wildcards to regex
	pattern = strings.ReplaceAll(pattern, "\\*\\*", ".*")  // ** matches any number of directories
	pattern = strings.ReplaceAll(pattern, "\\*", "[^/]*") // * matches within directory
	pattern = strings.ReplaceAll(pattern, "\\?", "[^/]")  // ? matches single character
	
	// Anchor the pattern
	if !strings.HasPrefix(pattern, ".*") {
		pattern = "^" + pattern
	}
	if !strings.HasSuffix(pattern, ".*") {
		pattern = pattern + "$"
	}
	
	return pattern
}

// getGlobalGitignore returns the path to the global gitignore file
func getGlobalGitignore() string {
	// Check git config for global ignore file
	// This is a simplified version - in practice you'd run `git config --get core.excludesfile`
	
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	
	// Common locations for global gitignore
	candidates := []string{
		filepath.Join(homeDir, ".gitignore_global"),
		filepath.Join(homeDir, ".config", "git", "ignore"),
		filepath.Join(homeDir, ".gitignore"),
	}
	
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	
	return ""
}