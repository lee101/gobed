package search

// Config represents search configuration
type Config struct {
	Path       string
	Query      string
	TopK       int
	UseGPU     bool
	Extensions []string
	MaxFiles   int
	BatchSize  int
}

// DefaultConfig returns default search configuration
func DefaultConfig() Config {
	return Config{
		Path:      ".",
		TopK:      10,
		UseGPU:    false,
		MaxFiles:  10000,
		BatchSize: 256,
	}
}
