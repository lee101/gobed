//go:build cagra
// +build cagra

package gobed

import (
	"fmt"
	"os"
	"testing"
	"time"
	"log"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func TestCAGRACachePathGeneration(t *testing.T) {
	log.Printf("Testing CAGRA cache path generation...")

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

	log.Printf("Cache path generation test passed")
}

func TestCAGRAIndexCaching(t *testing.T) {
	log.Printf("Testing CAGRA index cache save/load functionality...")

	os.Setenv("TESTING", "true")
	os.Setenv("DEBUG", "true")
	defer func() {
		os.Unsetenv("TESTING")
		os.Unsetenv("DEBUG")
	}()

	if !isCAGRAAvailable() {
		t.Skip("CAGRA not available, skipping cache test")
	}

	config := DefaultCAGRAConfig()
	config.MaxVectors = 100
	config.VectorDim = 512
	config.GraphDegree = 32
	config.CachePath = BuildCAGRACachePath("cache_test", config.VectorDim, config.GraphDegree, config.MaxVectors)

	log.Printf("Testing with cache path: %s", config.CachePath)

	if _, err := os.Stat(config.CachePath); err == nil {
		os.Remove(config.CachePath)
	}

	index1, err := NewCAGRAIndex(config)
	if err != nil {
		t.Fatalf("Failed to create CAGRA index: %v", err)
	}
	defer index1.Close()

	testVectors := make([]simd.Vec512, 50)
	testScales := make([]float32, 50)

	for i := 0; i < 50; i++ {
		for j := 0; j < 512; j++ {
			testVectors[i][j] = int8((i + j) % 256 - 128)
		}
		testScales[i] = 1.0 + float32(i)*0.1
	}

	log.Printf("Building initial index with %d vectors...", len(testVectors))
	buildStart := time.Now()
	err = index1.BuildIndex(testVectors, testScales)
	buildTime := time.Since(buildStart)

	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}

	log.Printf("Index built in %v", buildTime)

	stats1 := index1.GetStats()
	if !stats1.IsBuilt {
		t.Errorf("Index should be marked as built")
	}
	if stats1.NumVectors != 50 {
		t.Errorf("Expected 50 vectors, got %d", stats1.NumVectors)
	}

	query := testVectors[0]
	queryScale := testScales[0]

	log.Printf("Performing search on original index...")
	results1, err := index1.Search(query, queryScale, 5)
	if err != nil {
		t.Fatalf("Search failed on original index: %v", err)
	}

	if len(results1) == 0 {
		t.Fatalf("No results from original index")
	}

	log.Printf("Original index returned %d results", len(results1))

	time.Sleep(100 * time.Millisecond)

	if _, err := os.Stat(config.CachePath); os.IsNotExist(err) {
		t.Errorf("Cache file was not created at %s", config.CachePath)
	} else {
		log.Printf("Cache file created successfully at %s", config.CachePath)
	}

	index1.Close()

	config2 := config
	index2, err := NewCAGRAIndex(config2)
	if err != nil {
		t.Fatalf("Failed to create second CAGRA index: %v", err)
	}
	defer index2.Close()

	log.Printf("Loading index from cache...")
	loadStart := time.Now()
	err = index2.LoadFromCache()
	loadTime := time.Since(loadStart)

	if err != nil {
		t.Fatalf("Failed to load from cache: %v", err)
	}

	log.Printf("Index loaded from cache in %v", loadTime)

	if loadTime >= buildTime {
		t.Errorf("Cache load (%v) should be faster than build (%v)", loadTime, buildTime)
	}

	speedup := float64(buildTime) / float64(loadTime)
	log.Printf("Cache speedup: %.1fx", speedup)

	stats2 := index2.GetStats()
	if !stats2.IsBuilt {
		t.Errorf("Cached index should be marked as built")
	}
	if stats2.NumVectors != stats1.NumVectors {
		t.Errorf("Vector count mismatch: original=%d, cached=%d", stats1.NumVectors, stats2.NumVectors)
	}

	log.Printf("Performing search on cached index...")
	results2, err := index2.Search(query, queryScale, 5)
	if err != nil {
		t.Fatalf("Search failed on cached index: %v", err)
	}

	if len(results2) != len(results1) {
		t.Errorf("Result count mismatch: original=%d, cached=%d", len(results1), len(results2))
	}

	if len(results2) > 0 && len(results1) > 0 {
		if results1[0].ID != results2[0].ID {
			t.Errorf("Top result ID mismatch: original=%d, cached=%d", results1[0].ID, results2[0].ID)
		}

		similarityDiff := abs(results1[0].Similarity - results2[0].Similarity)
		if similarityDiff > 0.001 {
			t.Errorf("Top result similarity mismatch: original=%.6f, cached=%.6f, diff=%.6f",
				results1[0].Similarity, results2[0].Similarity, similarityDiff)
		}
	}

	log.Printf("Cache functionality test passed - speedup: %.1fx", speedup)

	os.Remove(config.CachePath)
}

func TestCAGRAConfigCachingCagra(t *testing.T) {
	log.Printf("Testing CAGRA config-specific caching...")

	os.Setenv("TESTING", "true")
	defer os.Unsetenv("TESTING")

	if !isCAGRAAvailable() {
		t.Skip("CAGRA not available, skipping config cache test")
	}

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

			if !containsCagra(config.CachePath, expectedSubstring) {
				t.Errorf("Config %s cache path should contain %s, got %s",
					configNames[i], expectedSubstring, config.CachePath)
			}

			log.Printf("Config %s cache path: %s", configNames[i], config.CachePath)
		})
	}
}

func abs(f float32) float32 {
	if f < 0 {
		return -f
	}
	return f
}

func containsCagra(s, substr string) bool {
    return len(s) >= len(substr) && s[len(s)-len(substr):] == substr ||
           len(s) > len(substr) && indexOfCagra(s, substr) >= 0
}

func indexOfCagra(s, substr string) int {
    for i := 0; i <= len(s)-len(substr); i++ {
        if s[i:i+len(substr)] == substr {
            return i
        }
    }
    return -1
}
