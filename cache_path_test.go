package gobed

import (
	"fmt"
	"os"
	"testing"
)

func TestBuildCAGRACachePath(t *testing.T) {
	os.Setenv("TESTING", "true")
	defer os.Unsetenv("TESTING")

	namespace := "test_emoji"
	vectorDim := 512
	graphDegree := 64
	count := 1000

	cachePath := BuildCAGRACachePath(namespace, vectorDim, graphDegree, count)
	expectedPath := "/mnt/fast/tmp/test_indexes/test_emoji_cagra_512_64_1000.cagrabin"

	if cachePath != expectedPath {
		t.Errorf("Expected cache path %s, got %s", expectedPath, cachePath)
	}

	os.Setenv("DEV", "true")
	devCachePath := BuildCAGRACachePath(namespace, vectorDim, graphDegree, count)
	expectedDevPath := "/mnt/fast/tmp/test_indexes/test_emoji_cagra_512_64_1000.cagrabin"

	if devCachePath != expectedDevPath {
		t.Errorf("Expected dev cache path %s, got %s", expectedDevPath, devCachePath)
	}

	os.Unsetenv("DEV")
	os.Unsetenv("TESTING")

	prodCachePath := BuildCAGRACachePath(namespace, vectorDim, graphDegree, count)
	expectedProdPath := "/mnt/fast/tmp/netwrck_gobed_indexes/test_emoji_cagra_512_64_1000.cagrabin"

	if prodCachePath != expectedProdPath {
		t.Errorf("Expected prod cache path %s, got %s", expectedProdPath, prodCachePath)
	}

	t.Logf("Cache path generation test passed")
}

func TestCAGRAConfigCaching(t *testing.T) {
	os.Setenv("TESTING", "true")
	defer os.Unsetenv("TESTING")

	configs := []CAGRAConfig{
		DefaultCAGRAConfig(),
		FastCAGRAConfig(),
		QualityCAGRAConfig(),
	}

	configNames := []string{"default", "fast", "quality"}

	for i, config := range configs {
		t.Run(configNames[i], func(t *testing.T) {
			if config.CachePath == "" {
				t.Errorf("Config %s should have non-empty cache path", configNames[i])
			}

			expectedSubstring := fmt.Sprintf("%s_cagra_%d_%d_%d.cagrabin",
				configNames[i], config.VectorDim, config.GraphDegree, config.MaxVectors)

			if !contains(config.CachePath, expectedSubstring) {
				t.Errorf("Config %s cache path should contain %s, got %s",
					configNames[i], expectedSubstring, config.CachePath)
			}

			t.Logf("Config %s cache path: %s", configNames[i], config.CachePath)
		})
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[len(s)-len(substr):] == substr ||
		   len(s) > len(substr) && indexOf(s, substr) >= 0
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}