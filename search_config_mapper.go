package gobed

import (
	"github.com/lee101/gobed/ann/search"
)

// mapSearchConfigToAnnConfig converts the public SearchConfig to internal ann/search.Config
func mapSearchConfigToAnnConfig(sc SearchConfig) search.Config {
	cfg := search.DefaultConfig()
	
	// Map MaxExactSearchSize to MaxFlatSize
	if sc.MaxExactSearchSize > 0 {
		cfg.MaxFlatSize = sc.MaxExactSearchSize
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