package src

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// Config holds bed configuration
type Config struct {
	// Index settings
	IndexPath   string `json:"index_path"`
	CacheDir    string `json:"cache_dir"`
	MaxFileSize int64  `json:"max_file_size"` // bytes
	MaxFiles    int    `json:"max_files"`

	// Search settings
	DefaultLimit     int     `json:"default_limit"`
	DefaultContext   int     `json:"default_context"`
	DefaultThreshold float32 `json:"default_threshold"`
	ColorMode        string  `json:"color_mode"`

	// Performance settings
	UseGPU        bool `json:"use_gpu"`
	GPUDeviceID   int  `json:"gpu_device_id"`
	WorkerThreads int  `json:"worker_threads"`
	BatchSize     int  `json:"batch_size"`
	AsyncIndexing bool `json:"async_indexing"`

	// File filtering
	RespectGitignore  bool     `json:"respect_gitignore"`
	IgnorePatterns    []string `json:"ignore_patterns"`
	IncludeExtensions []string `json:"include_extensions"`
	ExcludeExtensions []string `json:"exclude_extensions"`
	SearchBinaries    bool     `json:"search_binaries"`
	MinLineLength     int      `json:"min_line_length"`
	MaxLineLength     int      `json:"max_line_length"`
	IgnoreLongLines   bool     `json:"ignore_long_lines"`

	// Advanced
	EmbeddingModel   string `json:"embedding_model"`
	IndexFormat      string `json:"index_format"`
	CompressionLevel int    `json:"compression_level"`
}

// DefaultConfig returns sensible defaults
func DefaultConfig() *Config {
	homeDir, _ := os.UserHomeDir()
	cacheDir := filepath.Join(homeDir, ".cache", "bed")

	return &Config{
		// Index settings
		IndexPath:   filepath.Join(cacheDir, "index"),
		CacheDir:    cacheDir,
		MaxFileSize: 10 * 1024 * 1024, // 10MB
		MaxFiles:    100000,

		// Search settings
		DefaultLimit:     5,
		DefaultContext:   2,
		DefaultThreshold: 0.7,
		ColorMode:        "auto",

		// Performance settings
		UseGPU:        false, // Auto-detect in practice
		GPUDeviceID:   0,
		WorkerThreads: runtime.NumCPU(),
		BatchSize:     1000,
		AsyncIndexing: true,

		// File filtering
		RespectGitignore: true,
		IgnorePatterns: []string{
			// Common build/cache directories
			"node_modules", ".git", ".svn", ".hg",
			"build", "dist", "target", "bin", "obj",
			".cache", ".tmp", "temp", "vendor", ".bed", "model",

			// Binary/media files
			"*.exe", "*.dll", "*.so", "*.dylib",
			"*.jpg", "*.jpeg", "*.png", "*.gif", "*.svg",
			"*.mp3", "*.mp4", "*.avi", "*.mov",
			"*.zip", "*.tar", "*.gz", "*.7z",

			// Generated files
			"*.min.js", "*.min.css", "*.map",
			"*.lock", "package-lock.json", "yarn.lock",
		},
		IncludeExtensions: []string{
			// Source code
			".go", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
			".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".clj", ".hs",
			".r", ".m", ".mm", ".cs", ".vb", ".fs", ".ml", ".elm",

			// Web/markup
			".html", ".css", ".scss", ".sass", ".less", ".vue", ".jsx", ".tsx",
			".xml", ".json", ".yaml", ".yml", ".toml", ".ini", ".conf",

			// Documentation/text
			".md", ".rst", ".txt", ".tex", ".org", ".adoc",

			// Shell/scripts
			".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",

			// Config/data
			".sql", ".proto", ".thrift", ".graphql", ".dockerfile",
		},
		ExcludeExtensions: []string{},
		SearchBinaries:    false,
		MinLineLength:     3,
		MaxLineLength:     1200,
		IgnoreLongLines:   true,

		// Advanced
		EmbeddingModel:   "sentence-transformers/static-retrieval-mrl-en-v1",
		IndexFormat:      "binary", // binary, json, msgpack
		CompressionLevel: 6,        // 0-9, 0=none, 9=max
	}
}

// LoadConfig loads configuration from ~/.bed/config.json
func LoadConfig() (*Config, error) {
	config := DefaultConfig()

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return config, nil // Use defaults if can't get home dir
	}

	configPath := filepath.Join(homeDir, ".bed", "config.json")

	// Create config directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(configPath), 0755); err != nil {
		return config, nil // Use defaults if can't create dir
	}

	// If config file doesn't exist, create it with defaults
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		if err := config.Save(); err != nil {
			return config, nil // Use defaults if can't save
		}
		return config, nil
	}

	// Load existing config
	data, err := os.ReadFile(configPath)
	if err != nil {
		return config, nil // Use defaults if can't read
	}

	if err := json.Unmarshal(data, config); err != nil {
		return config, nil // Use defaults if can't parse
	}

	// Ensure cache directory exists
	if err := os.MkdirAll(config.CacheDir, 0755); err != nil {
		// Fall back to temp directory
		config.CacheDir = os.TempDir()
		config.IndexPath = filepath.Join(config.CacheDir, "bed_index")
	}

	return config, nil
}

// Save saves configuration to ~/.bed/config.json
func (c *Config) Save() error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	configDir := filepath.Join(homeDir, ".bed")
	configPath := filepath.Join(configDir, "config.json")

	// Create config directory
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	// Create cache directory
	if err := os.MkdirAll(c.CacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Marshal config to JSON
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	// Write to file
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// Display prints configuration in a readable format
func (c *Config) Display() error {
	fmt.Printf("Bed Configuration\n")
	fmt.Printf("=================\n\n")

	fmt.Printf("Index Settings:\n")
	fmt.Printf("  Index Path: %s\n", c.IndexPath)
	fmt.Printf("  Cache Dir:  %s\n", c.CacheDir)
	fmt.Printf("  Max Files:  %d\n", c.MaxFiles)
	fmt.Printf("  Max Size:   %d MB\n", c.MaxFileSize/(1024*1024))

	fmt.Printf("\nSearch Settings:\n")
	fmt.Printf("  Default Limit:     %d\n", c.DefaultLimit)
	fmt.Printf("  Default Context:   %d\n", c.DefaultContext)
	fmt.Printf("  Default Threshold: %.2f\n", c.DefaultThreshold)
	fmt.Printf("  Color Mode:        %s\n", c.ColorMode)

	fmt.Printf("\nPerformance:\n")
	fmt.Printf("  Use GPU:         %v\n", c.UseGPU)
	fmt.Printf("  GPU Device:      %d\n", c.GPUDeviceID)
	fmt.Printf("  Worker Threads:  %d\n", c.WorkerThreads)
	fmt.Printf("  Batch Size:      %d\n", c.BatchSize)
	fmt.Printf("  Async Indexing:  %v\n", c.AsyncIndexing)

	fmt.Printf("\nFile Filtering:\n")
	fmt.Printf("  Respect .gitignore: %v\n", c.RespectGitignore)
	fmt.Printf("  Include Extensions: %d patterns\n", len(c.IncludeExtensions))
	fmt.Printf("  Exclude Extensions: %d patterns\n", len(c.ExcludeExtensions))
	fmt.Printf("  Ignore Patterns:    %d patterns\n", len(c.IgnorePatterns))
	fmt.Printf("  Search Binaries:    %v\n", c.SearchBinaries)
	fmt.Printf("  Min Line Length:    %d\n", c.MinLineLength)
	fmt.Printf("  Max Line Length:    %d\n", c.MaxLineLength)
	fmt.Printf("  Ignore Long Lines:  %v\n", c.IgnoreLongLines)

	fmt.Printf("\nAdvanced:\n")
	fmt.Printf("  Embedding Model:   %s\n", c.EmbeddingModel)
	fmt.Printf("  Index Format:      %s\n", c.IndexFormat)
	fmt.Printf("  Compression Level: %d\n", c.CompressionLevel)

	return nil
}
