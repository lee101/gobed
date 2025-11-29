//go:build !cagra

package src

import "fmt"

type CAGRABedSearcher struct{}

func NewCAGRABedSearcher() (*CAGRABedSearcher, error) {
    return nil, fmt.Errorf("CAGRA not available: build with -tags cagra and ensure cuVS is installed")
}

func (c *CAGRABedSearcher) Search(options BedSearchOptions) error { return fmt.Errorf("CAGRA not available") }
func (c *CAGRABedSearcher) Close() error { return nil }

