package gobed

import (
	"fmt"
	"testing"

	"github.com/lee101/gobed/pkg/ann/search"
)

func TestVerifyMaxFlatSizeDefaults(t *testing.T) {
	fmt.Println("\n=== Verifying MaxFlatSize Defaults ===")

	const expectedDefault = 1500

	vectorConfig := DefaultVectorIndexConfig()
	fmt.Printf("VectorIndexConfig.MaxFlatSize: %d (expected: %d)\n", vectorConfig.MaxFlatSize, expectedDefault)
	if vectorConfig.MaxFlatSize != expectedDefault {
		t.Errorf("VectorIndexConfig.MaxFlatSize = %d, want %d", vectorConfig.MaxFlatSize, expectedDefault)
	}

	searchConfig := search.DefaultConfig()
	fmt.Printf("search.Config.MaxFlatSize: %d (expected: %d)\n", searchConfig.MaxFlatSize, expectedDefault)
	if searchConfig.MaxFlatSize != expectedDefault {
		t.Errorf("search.Config.MaxFlatSize = %d, want %d", searchConfig.MaxFlatSize, expectedDefault)
	}

	presets := []struct {
		name     string
		preset   SearchPreset
		dataSize int
		expected int
	}{
		{"FastPreset small", FastPreset, 1000, 2000},
		{"FastPreset medium", FastPreset, 10000, 1500},
		{"BalancedPreset small", BalancedPreset, 5000, 1500},
		{"BalancedPreset large", BalancedPreset, 50000, 1500},
		{"AccuratePreset", AccuratePreset, 10000, 3000},
	}

	fmt.Println("\nPreset Configurations:")
	for _, p := range presets {
		config := GetSearchConfig(p.preset, p.dataSize)
		fmt.Printf("  %s (size=%d): MaxFlatSize=%d\n", p.name, p.dataSize, config.MaxFlatSize)
		if config.MaxFlatSize != p.expected {
			t.Errorf("%s: MaxFlatSize = %d, want %d", p.name, config.MaxFlatSize, p.expected)
		}
	}

	fmt.Println("\nSummary:")
	fmt.Println("  • Default MaxFlatSize aligns between vector index and search engine configs.")
	fmt.Println("  • Preset configurations remain within their documented ranges.")
}
