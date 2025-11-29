//go:build legacy && !cagra
// +build legacy,!cagra

package main

import "fmt"

func main() {
	fmt.Println("CAGRA param sweep: CAGRA not available (build with -tags cagra)")
}
