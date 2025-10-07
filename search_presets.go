package gobed

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
	// CAGRAPreset uses NVIDIA CAGRA for ultra-fast search (sub-millisecond latency)
	CAGRAPreset
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
			MaxFlatSize: 10000,
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
				MaxFlatSize: 10000,
				NList:       estimatedSize / 100,
				NProbe:      8,
				UseParallel: true,
			}
		}
		return search.Config{
			MaxFlatSize: 10000,
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

	case CAGRAPreset:
		// Optimal CAGRA configuration: Quality v1 from benchmarks
		// Achieves 99.5% recall, 61K QPS on RTX 3090
		return search.Config{
			MaxFlatSize:  10000,                        // Use approximate search for larger datasets
			NList:        min(1024, estimatedSize/50),  // More clusters for better partitioning
			NProbe:       32,                           // Optimized for 99.5% recall
			M:            40,                           // GraphDegree/2 where GraphDegree=80
			NBits:        8,                            // INT8 quantization
			HNSWEnabled:  false,                        // Pure CAGRA, no HNSW
			RerankSize:   256,                          // Larger rerank for quality
			UseParallel:  true,                         // Always use parallelization
			// Optimal parameters from benchmarks:
			// GraphDegree: 80 (for index build)
			// IntermediateGraphDegree: 160
			// ItopkSize: 256 (search parameter)
			// SearchWidth: 3 (wider search)
			// MinIterations: 8 (more iterations)
		}

	default:
		// Default to CAGRA (our optimal configuration) for ultra-fast search
		// This ensures anyone using gobed gets the best performance by default
		return GetSearchConfig(CAGRAPreset, estimatedSize)
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
