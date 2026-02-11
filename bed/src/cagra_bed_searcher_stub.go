//go:build !cagra

package src

import "fmt"

type CAGRABedSearcher struct{}

func NewCAGRABedSearcher() (*CAGRABedSearcher, error) {
	return nil, fmt.Errorf("CAGRA not available: build with -tags cagra and ensure cuVS is installed")
}

func (c *CAGRABedSearcher) Search(options BedSearchOptions) error {
	return fmt.Errorf("CAGRA not available")
}
func (c *CAGRABedSearcher) SearchMatches(options BedSearchOptions) ([]SearchMatch, error) {
	return nil, fmt.Errorf("CAGRA not available")
}
func (c *CAGRABedSearcher) BuildIndex(path string, options BedSearchOptions) error {
	return fmt.Errorf("CAGRA not available")
}
func (c *CAGRABedSearcher) NumDocuments() int { return 0 }
func (c *CAGRABedSearcher) Close() error      { return nil }
