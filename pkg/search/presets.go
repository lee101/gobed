package search

import gobed "github.com/lee101/gobed"

type (
	SearchPreset           = gobed.SearchPreset
	PresetConfig           = gobed.PresetConfig
	SimplifiedSearchConfig = gobed.SimplifiedSearchConfig
)

const (
	FastPreset     = gobed.FastPreset
	BalancedPreset = gobed.BalancedPreset
	AccuratePreset = gobed.AccuratePreset
	CAGRAPreset    = gobed.CAGRAPreset
	CustomPreset   = gobed.CustomPreset
)

var (
	GetSearchConfig           = gobed.GetSearchConfig
	NewSearchEngineWithPreset = gobed.NewSearchEngineWithPreset
)
