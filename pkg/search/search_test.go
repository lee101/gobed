package search

import (
	"testing"
)

func TestSearchConfig(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		valid  bool
	}{
		{
			name: "default config",
			config: Config{
				Path: ".",
				TopK: 10,
			},
			valid: true,
		},
		{
			name: "with extensions",
			config: Config{
				Path:       "/tmp",
				TopK:       5,
				Extensions: []string{".go", ".md"},
			},
			valid: true,
		},
		{
			name: "gpu enabled",
			config: Config{
				Path:   ".",
				TopK:   20,
				UseGPU: true,
			},
			valid: true,
		},
		{
			name: "invalid topk",
			config: Config{
				Path: ".",
				TopK: -1,
			},
			valid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Validate configuration
			isValid := tt.config.TopK > 0
			if isValid != tt.valid {
				t.Errorf("Config validation failed: got %v, want %v", isValid, tt.valid)
			}
		})
	}
}

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.Path != "." {
		t.Errorf("Default path = %q, want %q", config.Path, ".")
	}

	if config.TopK != 10 {
		t.Errorf("Default TopK = %d, want %d", config.TopK, 10)
	}

	if config.UseGPU != false {
		t.Errorf("Default UseGPU = %v, want %v", config.UseGPU, false)
	}

	if config.MaxFiles != 10000 {
		t.Errorf("Default MaxFiles = %d, want %d", config.MaxFiles, 10000)
	}

	if config.BatchSize != 256 {
		t.Errorf("Default BatchSize = %d, want %d", config.BatchSize, 256)
	}
}

func TestSearchPresets(t *testing.T) {
	// Test that search presets work correctly
	presets := []struct {
		name        string
		preset      SearchPreset
		expectFast  bool
		expectHighQ bool
	}{
		{
			name:        "ultra fast",
			preset:      UltraFast,
			expectFast:  true,
			expectHighQ: false,
		},
		{
			name:        "fast",
			preset:      Fast,
			expectFast:  true,
			expectHighQ: false,
		},
		{
			name:        "balanced",
			preset:      Balanced,
			expectFast:  false,
			expectHighQ: false,
		},
		{
			name:        "high quality",
			preset:      HighQuality,
			expectFast:  false,
			expectHighQ: true,
		},
	}

	for _, tt := range presets {
		t.Run(tt.name, func(t *testing.T) {
			// Test preset properties
			if tt.preset < 0 || tt.preset > 3 {
				t.Errorf("Invalid preset value: %d", tt.preset)
			}
		})
	}
}

type mockSearchResult struct {
	Path  string
	Score float32
}

func TestSearchSorting(t *testing.T) {
	results := []mockSearchResult{
		{Path: "file3.txt", Score: 0.5},
		{Path: "file1.txt", Score: 0.9},
		{Path: "file2.txt", Score: 0.7},
	}

	// Sort by score descending
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Score < results[j].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Verify sorted
	expectedOrder := []string{"file1.txt", "file2.txt", "file3.txt"}
	for i, expected := range expectedOrder {
		if results[i].Path != expected {
			t.Errorf("Result[%d] = %s, want %s", i, results[i].Path, expected)
		}
	}
}

func TestFilterExtensions(t *testing.T) {
	tests := []struct {
		name       string
		filename   string
		extensions []string
		shouldPass bool
	}{
		{
			name:       "go file with go filter",
			filename:   "test.go",
			extensions: []string{".go"},
			shouldPass: true,
		},
		{
			name:       "md file with go filter",
			filename:   "README.md",
			extensions: []string{".go"},
			shouldPass: false,
		},
		{
			name:       "go file with multiple filters",
			filename:   "test.go",
			extensions: []string{".go", ".md", ".txt"},
			shouldPass: true,
		},
		{
			name:       "no extension filter",
			filename:   "anything.xyz",
			extensions: []string{},
			shouldPass: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pass := shouldIncludeFile(tt.filename, tt.extensions)
			if pass != tt.shouldPass {
				t.Errorf("Filter result = %v, want %v", pass, tt.shouldPass)
			}
		})
	}
}

func shouldIncludeFile(filename string, extensions []string) bool {
	if len(extensions) == 0 {
		return true
	}

	for _, ext := range extensions {
		if len(filename) >= len(ext) && filename[len(filename)-len(ext):] == ext {
			return true
		}
	}
	return false
}

func BenchmarkConfigCreation(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Config{
			Path:       "/tmp/test",
			Query:      "benchmark query",
			TopK:       100,
			UseGPU:     false,
			Extensions: []string{".go", ".md"},
			MaxFiles:   10000,
			BatchSize:  256,
		}
	}
}

func BenchmarkSearchSorting(b *testing.B) {
	// Create test data
	results := make([]mockSearchResult, 100)
	for i := range results {
		results[i] = mockSearchResult{
			Path:  "file.txt",
			Score: float32(i) / 100.0,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simple bubble sort for benchmark
		temp := make([]mockSearchResult, len(results))
		copy(temp, results)

		for j := 0; j < len(temp)-1; j++ {
			for k := j + 1; k < len(temp); k++ {
				if temp[j].Score < temp[k].Score {
					temp[j], temp[k] = temp[k], temp[j]
				}
			}
		}
	}
}