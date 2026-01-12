package main

import (
	"fmt"
	"os"

	"github.com/lee101/gobed/bed/src"
)

func main() {
	if err := src.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
