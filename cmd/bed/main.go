//go:build legacy

package main

import (
    "log"

    bedcli "github.com/lee101/gobed/bed/src"
)

// This main wraps the reusable bed/src Cobra CLI so the bed binary
// stays self-contained while reusing the shared bed implementation.
func main() {
    if err := bedcli.Execute(); err != nil {
        log.Fatal(err)
    }
}
