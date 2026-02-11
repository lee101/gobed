package src

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type fakeFastModel struct{}

func (fakeFastModel) EmbedFast(text string) ([]float32, func()) {
	vec := make([]float32, 512)
	for i := 0; i < len(text); i++ {
		vec[i%512] += float32((text[i]%31)+1) * 0.03125
	}
	return vec, func() {}
}

func (fakeFastModel) Close() error { return nil }

func newTestFastSearcher(t *testing.T, baseDir string) *FastBedSearcher {
	t.Helper()

	searcher, err := newFastBedSearcherWithModel(fakeFastModel{}, baseDir, defaultFastSearcherOptions())
	if err != nil {
		t.Fatalf("failed to create test fast searcher: %v", err)
	}
	return searcher
}

func TestFastBedSearcherReindexReplacesStaleData(t *testing.T) {
	dir := t.TempDir()
	searcher := newTestFastSearcher(t, dir)
	defer searcher.Close()

	target := filepath.Join(dir, "sample.txt")

	oldToken := "bed_reindex_old_token_12345"
	newToken := "bed_reindex_new_token_67890"

	if err := os.WriteFile(target, []byte(oldToken+"\nother line\n"), 0644); err != nil {
		t.Fatalf("failed to write initial file: %v", err)
	}

	if err := searcher.IndexDirectory(dir, BedSearchOptions{ForceIndex: true}); err != nil {
		t.Fatalf("initial index failed: %v", err)
	}

	if docs := searcher.NumDocuments(); docs == 0 {
		t.Fatalf("expected indexed docs, got 0")
	}
	if !searcherContainsToken(searcher, oldToken) {
		t.Fatalf("expected old token to be indexed")
	}

	if err := os.WriteFile(target, []byte(newToken+"\n"), 0644); err != nil {
		t.Fatalf("failed to update file: %v", err)
	}

	if err := searcher.IndexDirectory(dir, BedSearchOptions{ForceIndex: true}); err != nil {
		t.Fatalf("reindex failed: %v", err)
	}

	if searcherContainsToken(searcher, oldToken) {
		t.Fatalf("stale token remained after reindex")
	}
	if !searcherContainsToken(searcher, newToken) {
		t.Fatalf("new token missing after reindex")
	}
}

func TestDaemonInotifyLiveUpdatesIndex(t *testing.T) {
	dir := t.TempDir()

	searcher := newTestFastSearcher(t, dir)

	daemon, err := newBedDaemonWithSearcher(DaemonConfig{
		WatchPaths: []string{dir},
		BatchDelay: 40 * time.Millisecond,
		HTTPPort:   0,
	}, searcher)
	if err != nil {
		t.Fatalf("daemon creation failed: %v", err)
	}
	defer daemon.Close()

	if err := daemon.searcher.IndexDirectory(dir, BedSearchOptions{ForceIndex: true}); err != nil {
		t.Fatalf("initial daemon index failed: %v", err)
	}
	if err := daemon.addWatchRecursive(dir); err != nil {
		t.Fatalf("failed to add watcher: %v", err)
	}

	go daemon.watchLoop()
	go daemon.batchProcessor()

	target := filepath.Join(dir, "live.txt")
	tokenA := "bed_inotify_token_alpha_123"
	tokenB := "bed_inotify_token_beta_456"

	if err := os.WriteFile(target, []byte(tokenA+"\n"), 0644); err != nil {
		t.Fatalf("failed to write created file: %v", err)
	}

	waitForTokenState(t, daemon.searcher, tokenA, true)

	if err := os.WriteFile(target, []byte(tokenB+"\n"), 0644); err != nil {
		t.Fatalf("failed to modify file: %v", err)
	}

	waitForTokenState(t, daemon.searcher, tokenA, false)
	waitForTokenState(t, daemon.searcher, tokenB, true)

	if err := os.Remove(target); err != nil {
		t.Fatalf("failed to remove file: %v", err)
	}

	waitForTokenState(t, daemon.searcher, tokenB, false)
}

func TestFastBedSearcherUpsertAndRemoveFile(t *testing.T) {
	dir := t.TempDir()
	searcher := newTestFastSearcher(t, dir)
	defer searcher.Close()

	target := filepath.Join(dir, "upsert.txt")
	tokenOne := "bed_upsert_token_one_111"
	tokenTwo := "bed_upsert_token_two_222"

	if err := os.WriteFile(target, []byte(tokenOne+"\n"), 0644); err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	if err := searcher.UpsertFile(target, BedSearchOptions{}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}
	if !searcherContainsToken(searcher, tokenOne) {
		t.Fatalf("token from upserted file not indexed")
	}

	if err := os.WriteFile(target, []byte(tokenTwo+"\n"), 0644); err != nil {
		t.Fatalf("failed to update file: %v", err)
	}
	if err := searcher.UpsertFile(target, BedSearchOptions{}); err != nil {
		t.Fatalf("second upsert failed: %v", err)
	}
	if searcherContainsToken(searcher, tokenOne) {
		t.Fatalf("old token remained after upsert replacement")
	}
	if !searcherContainsToken(searcher, tokenTwo) {
		t.Fatalf("new token missing after upsert replacement")
	}

	searcher.RemoveFile(target)
	if searcherContainsToken(searcher, tokenTwo) {
		t.Fatalf("token remained after remove")
	}
}

func TestFastBedSearcherIgnoresLongLines(t *testing.T) {
	dir := t.TempDir()
	opts := defaultFastSearcherOptions()
	opts.minLineLength = 1
	opts.maxLineLength = 16
	opts.ignoreLongLines = true

	searcher, err := newFastBedSearcherWithModel(fakeFastModel{}, dir, opts)
	if err != nil {
		t.Fatalf("failed to create test searcher: %v", err)
	}
	defer searcher.Close()

	shortToken := "bed_short_ok"
	longToken := "bed_long_line_should_be_ignored_completely_123456"
	target := filepath.Join(dir, "longline.txt")
	content := shortToken + "\n" + longToken + "\n"
	if err := os.WriteFile(target, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	if err := searcher.UpsertFile(target, BedSearchOptions{}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}

	if !searcherContainsToken(searcher, shortToken) {
		t.Fatalf("short line should be indexed")
	}
	if searcherContainsToken(searcher, longToken) {
		t.Fatalf("long line should be ignored")
	}
}

func TestFastBedSearcherNoIndexLoadsCachedIndex(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "cache.txt")
	token := "bed_no_index_cache_token_98765"
	if err := os.WriteFile(target, []byte(token+"\n"), 0644); err != nil {
		t.Fatalf("failed to write cache file: %v", err)
	}

	indexer := newTestFastSearcher(t, dir)
	if err := indexer.IndexDirectory(dir, BedSearchOptions{ForceIndex: true}); err != nil {
		t.Fatalf("failed to build index: %v", err)
	}
	indexer.Close()

	if _, err := os.Stat(filepath.Join(dir, ".bed", "fast_index.bin")); err != nil {
		t.Fatalf("expected on-disk cache to exist: %v", err)
	}

	searcher := newTestFastSearcher(t, dir)
	defer searcher.Close()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd failed: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir failed: %v", err)
	}
	defer os.Chdir(wd)

	matches, err := searcher.SearchMatches(BedSearchOptions{
		Query:     token,
		NoIndex:   true,
		Threshold: 0.1,
		Limit:     5,
	})
	if err != nil {
		t.Fatalf("expected cache-backed no-index search to succeed: %v", err)
	}
	if len(matches) == 0 {
		t.Fatalf("expected matches from cached index")
	}
	if !strings.Contains(matches[0].Document.Content, token) {
		t.Fatalf("expected cached result to include token %q, got %q", token, matches[0].Document.Content)
	}
}

func TestFastBedSearcherNoIndexWithoutCacheReturnsHelpfulError(t *testing.T) {
	dir := t.TempDir()
	searcher := newTestFastSearcher(t, dir)
	defer searcher.Close()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd failed: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir failed: %v", err)
	}
	defer os.Chdir(wd)

	_, err = searcher.SearchMatches(BedSearchOptions{
		Query:   "no cache exists",
		NoIndex: true,
		Limit:   5,
	})
	if err == nil {
		t.Fatalf("expected no-index search without cache to fail")
	}
	if !strings.Contains(err.Error(), "no cached index found") {
		t.Fatalf("expected actionable no-cache error, got: %v", err)
	}
}

func TestEnhancedIgnoreFilterDirectoryRules(t *testing.T) {
	dir := t.TempDir()

	if err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte("secret/\n"), 0644); err != nil {
		t.Fatalf("failed to write .gitignore: %v", err)
	}

	secretFile := filepath.Join(dir, "secret", "keys.txt")
	vendorFile := filepath.Join(dir, "vendor", "lib", "dep.txt")
	modelFile := filepath.Join(dir, "model", "tokenizer.json")
	cacheFile := filepath.Join(dir, ".bed", "fast_index.bin")
	keepFile := filepath.Join(dir, "src", "keep.go")

	for _, p := range []string{
		filepath.Dir(secretFile),
		filepath.Dir(vendorFile),
		filepath.Dir(modelFile),
		filepath.Dir(cacheFile),
		filepath.Dir(keepFile),
	} {
		if err := os.MkdirAll(p, 0755); err != nil {
			t.Fatalf("failed to create %s: %v", p, err)
		}
	}

	writeText := func(path string, content string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write %s: %v", path, err)
		}
	}
	writeText(secretFile, "secret token")
	writeText(vendorFile, "vendored dep")
	writeText(modelFile, "{}")
	writeText(cacheFile, "index")
	writeText(keepFile, "package src")

	filter, err := NewEnhancedIgnoreFilter(dir)
	if err != nil {
		t.Fatalf("failed to build enhanced ignore filter: %v", err)
	}

	for _, ignored := range []string{secretFile, vendorFile, modelFile, cacheFile} {
		shouldProcess, _ := filter.ShouldProcess(ignored)
		if shouldProcess {
			t.Fatalf("expected %s to be ignored", ignored)
		}
	}

	shouldProcess, _ := filter.ShouldProcess(keepFile)
	if !shouldProcess {
		t.Fatalf("expected regular source file to be indexed")
	}
}

func waitForTokenState(t *testing.T, searcher *FastBedSearcher, token string, wantPresent bool) {
	t.Helper()

	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		if searcherContainsToken(searcher, token) == wantPresent {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}

	t.Fatalf("timed out waiting for token %q present=%v", token, wantPresent)
}

func searcherContainsToken(searcher *FastBedSearcher, token string) bool {
	searcher.mu.RLock()
	defer searcher.mu.RUnlock()

	for _, doc := range searcher.documents {
		if strings.Contains(doc.Content, token) {
			return true
		}
	}
	return false
}
