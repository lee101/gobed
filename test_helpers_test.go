package gobed

import (
	"errors"
	"os"
	"strings"
	"testing"
)

const defaultModelPath = "model/modelint8_512dim.safetensors"

func modelAssetPath() string {
	if override := os.Getenv("GOBED_TEST_MODEL"); override != "" {
		return override
	}
	return defaultModelPath
}

func loadModelOrSkip(t *testing.T) *EmbeddingModel {
	t.Helper()

	path := modelAssetPath()
	if _, statErr := os.Stat(path); statErr != nil {
		if errors.Is(statErr, os.ErrNotExist) {
			t.Skipf("skipping test: model asset %s missing", path)
		}
		t.Skipf("skipping test: unable to stat %s: %v", path, statErr)
	}

	model, err := LoadModel()
	if err != nil {
		if isModelUnavailable(err) {
			t.Skipf("skipping test: %v", err)
		}
		t.Fatalf("failed to load model: %v", err)
	}
	return model
}

func isModelUnavailable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, os.ErrNotExist) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "not found") ||
		strings.Contains(msg, "missing") ||
		strings.Contains(msg, "safetensors")
}

func loadModelOrSkipB(b *testing.B) *EmbeddingModel {
	b.Helper()

	path := modelAssetPath()
	if _, statErr := os.Stat(path); statErr != nil {
		if errors.Is(statErr, os.ErrNotExist) {
			b.Skipf("skipping benchmark: model asset %s missing", path)
		}
		b.Skipf("skipping benchmark: unable to stat %s: %v", path, statErr)
	}

	model, err := LoadModel()
	if err != nil {
		if isModelUnavailable(err) {
			b.Skipf("skipping benchmark: %v", err)
		}
		b.Fatalf("failed to load model: %v", err)
	}
	return model
}
