package gobed

import (
	"testing"
	"time"
)

// TestLoadModelIsShared measures the cost of the first load versus subsequent
// calls. Production logged 47 separate LoadModel calls at startup; each one
// re-read and re-parsed the safetensors and tokenizer from disk.
func TestLoadModelIsShared(t *testing.T) {
	start := time.Now()
	m1, err := LoadModel()
	if err != nil {
		t.Skipf("model unavailable in this environment: %v", err)
	}
	first := time.Since(start)

	start = time.Now()
	m2, err := LoadModel()
	if err != nil {
		t.Fatalf("second load failed: %v", err)
	}
	second := time.Since(start)

	t.Logf("first load: %v", first)
	t.Logf("cached load: %v", second)
	t.Logf("startup saving at 47 call sites: %v", time.Duration(46)*first)

	if m1 != m2 {
		t.Errorf("LoadModel returned distinct instances; callers no longer share one model")
	}
	if second > first/10 && first > time.Millisecond {
		t.Errorf("second load not served from cache: first=%v second=%v", first, second)
	}
}
