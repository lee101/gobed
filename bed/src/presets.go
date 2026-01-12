package src

// SearchPreset defines pre-configured search settings
type SearchPreset struct {
	Name        string
	Description string
	Config      SearchPresetConfig
}

// SearchPresetConfig holds preset configuration values
type SearchPresetConfig struct {
	Threshold   float32
	Limit       int
	Context     int
	BatchSize   int
	UseInt8     bool
	NumWorkers  int
	ForceGPU    bool
	ForceCPU    bool
}

// Available presets
var (
	PresetFast = SearchPreset{
		Name:        "fast",
		Description: "Optimized for speed, lower recall",
		Config: SearchPresetConfig{
			Threshold:  0.6,
			Limit:      5,
			Context:    1,
			BatchSize:  2048,
			UseInt8:    true,
			NumWorkers: -1, // auto
		},
	}

	PresetAccurate = SearchPreset{
		Name:        "accurate",
		Description: "Higher recall, more thorough search",
		Config: SearchPresetConfig{
			Threshold:  0.5,
			Limit:      20,
			Context:    3,
			BatchSize:  512,
			UseInt8:    false,
			NumWorkers: -1,
		},
	}

	PresetBalanced = SearchPreset{
		Name:        "balanced",
		Description: "Balance between speed and accuracy",
		Config: SearchPresetConfig{
			Threshold:  0.55,
			Limit:      10,
			Context:    2,
			BatchSize:  1024,
			UseInt8:    true,
			NumWorkers: -1,
		},
	}

	PresetLargeCodebase = SearchPreset{
		Name:        "large",
		Description: "Optimized for large codebases (100k+ files)",
		Config: SearchPresetConfig{
			Threshold:  0.65,
			Limit:      15,
			Context:    2,
			BatchSize:  4096,
			UseInt8:    true,
			ForceGPU:   true,
			NumWorkers: -1,
		},
	}

	PresetInteractive = SearchPreset{
		Name:        "interactive",
		Description: "Low latency for real-time search",
		Config: SearchPresetConfig{
			Threshold:  0.6,
			Limit:      5,
			Context:    1,
			BatchSize:  256,
			UseInt8:    true,
			NumWorkers: 4,
		},
	}

	PresetDeep = SearchPreset{
		Name:        "deep",
		Description: "Thorough semantic search, slower",
		Config: SearchPresetConfig{
			Threshold:  0.4,
			Limit:      50,
			Context:    5,
			BatchSize:  256,
			UseInt8:    false,
			NumWorkers: -1,
		},
	}

	AllPresets = []SearchPreset{
		PresetFast,
		PresetAccurate,
		PresetBalanced,
		PresetLargeCodebase,
		PresetInteractive,
		PresetDeep,
	}
)

// GetPreset returns a preset by name
func GetPreset(name string) *SearchPreset {
	for _, p := range AllPresets {
		if p.Name == name {
			return &p
		}
	}
	return nil
}

// ApplyPreset applies a preset to search options
func ApplyPreset(opts *BedSearchOptions, preset *SearchPreset) {
	if preset == nil {
		return
	}

	cfg := preset.Config

	if opts.Threshold == 0 || opts.Threshold == 0.7 { // default
		opts.Threshold = cfg.Threshold
	}
	if opts.Limit == 0 || opts.Limit == 10 { // default
		opts.Limit = cfg.Limit
	}
	if opts.Context == 0 || opts.Context == 2 { // default
		opts.Context = cfg.Context
	}

	if cfg.ForceGPU {
		opts.UseGPU = true
	}
}

// AutoSelectPreset chooses a preset based on workload
func AutoSelectPreset(indexSize int, isInteractive bool) *SearchPreset {
	if isInteractive {
		return &PresetInteractive
	}

	switch {
	case indexSize > 100000:
		return &PresetLargeCodebase
	case indexSize > 10000:
		return &PresetBalanced
	case indexSize < 1000:
		return &PresetFast
	default:
		return &PresetBalanced
	}
}
