package gobed

import (
	"github.com/lee101/gobed/pkg/ann/search"
)

// mapSearchConfigToAnnConfig converts the public SearchConfig to internal ann/search.Config
func mapSearchConfigToAnnConfig(sc SearchConfig) search.Config {
	var cfg search.Config

	// If AutoMode is enabled and a preset is specified, use the preset configuration
	if sc.AutoMode && sc.Preset >= 0 {
		// Use the preset configuration as the base
		cfg = GetSearchConfig(sc.Preset, 10000) // Default estimated size
	} else if sc.AutoMode {
		// AutoMode without preset - default to CAGRA (our optimal config)
		cfg = GetSearchConfig(CAGRAPreset, 10000)
	} else {
		// Manual mode - start with defaults
		cfg = search.DefaultConfig()
	}

	// Override with any manually specified values
	// Map MaxExactSearchSize to MaxFlatSize
	if sc.MaxExactSearchSize > 0 {
		cfg.MaxFlatSize = sc.MaxExactSearchSize
		// Debug output (temporarily for testing)
		// fmt.Printf("DEBUG: Mapped MaxExactSearchSize %d to MaxFlatSize %d\n", sc.MaxExactSearchSize, cfg.MaxFlatSize)
	}
	
	// Map other fields
	if sc.NumClusters > 0 {
		cfg.NList = sc.NumClusters
	}
	
	if sc.SearchClusters > 0 {
		cfg.NProbe = sc.SearchClusters
	}
	
	cfg.HNSWEnabled = sc.UseGraphRouting
	
	if sc.CandidatesToRerank > 0 {
		cfg.RerankSize = sc.CandidatesToRerank
	}
	
	return cfg
}