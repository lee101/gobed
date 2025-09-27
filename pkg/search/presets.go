package search

import "github.com/lee101/gobed/pkg/ann/search"

// SearchPreset represents predefined search configurations
type SearchPreset int

const (
	// FastPreset prioritizes speed for small datasets (<50K vectors)
	FastPreset SearchPreset = iota
	// BalancedPreset balances speed and accuracy for medium datasets (50K-500K vectors)
	BalancedPreset
	// AccuratePreset prioritizes accuracy for large datasets (>500K vectors)
	AccuratePreset
	// CustomPreset allows manual configuration
	CustomPreset
)

// PresetConfig contains simplified configuration options
type PresetConfig struct {
	Preset      SearchPreset
	DatasetSize int // Estimated number of vectors
}

// GetSearchConfig returns the appropriate search configuration for a preset
func GetSearchConfig(preset SearchPreset, estimatedSize int) search.Config {
	switch preset {
	case FastPreset:
		// Optimized for speed - minimal indexing overhead
		if estimatedSize <= 1500 {
			return search.Config{
				MaxFlatSize: 2000, // Small exact search for tiny datasets
				UseParallel: true,
			}
		}
		return search.Config{
			MaxFlatSize: 1500,
			NList:       min(256, estimatedSize/50), // Few clusters
			NProbe:      4,                          // Minimal probes
			HNSWEnabled: false,                      // Skip graph for simplicity
			RerankSize:  50,                         // Small rerank
			UseParallel: true,
		}

	case BalancedPreset:
		// Balance between speed and accuracy
		if estimatedSize <= 10000 {
			return search.Config{
				MaxFlatSize: 1500,
				NList:       estimatedSize / 100,
				NProbe:      8,
				UseParallel: true,
			}
		}
		return search.Config{
			MaxFlatSize: 1500,
			NList:       min(2048, estimatedSize/100),
			NProbe:      10,
			M:           32,
			NBits:       8,
			HNSWEnabled: estimatedSize > 100000, // Use HNSW for larger datasets
			HNSWM:       8,
			HNSWEfC:     100,
			RerankSize:  100,
			UseParallel: true,
		}

	case AccuratePreset:
		// Prioritize accuracy over speed
		return search.Config{
			MaxFlatSize: 3000,
			NList:       min(4096, estimatedSize/50), // More clusters
			NProbe:      20,                          // More probes
			M:           64,
			NBits:       8,
			HNSWEnabled: true, // Always use HNSW for accuracy
			HNSWM:       16,
			HNSWEfC:     200,
			RerankSize:  200, // Large rerank for better accuracy
			UseParallel: true,
		}

	default:
		// Default to balanced
		return GetSearchConfig(BalancedPreset, estimatedSize)
	}
}

// SimplifiedSearchConfig provides a simpler configuration interface
type SimplifiedSearchConfig struct {
	Preset      SearchPreset
	DatasetSize int
	// Optional custom parameters (only used with CustomPreset)
	CustomConfig *SearchConfig
}

// NewSearchEngineWithPreset creates a search engine with a preset configuration
func NewSearchEngineWithPreset(model *EmbeddingModel, preset SearchPreset) (*SearchEngine, error) {
	config := &SearchConfig{
		AutoMode: true,
		Preset:   preset, // Store preset for later use
	}
	// Create search engine with default configuration
	se := NewSearchEngine(model)
	se.config = *config
	return se, nil
}
