package src

import (
	"fmt"
	"os"
)

// Main is the entry point for the bed CLI
func Main() {
	if err := Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
