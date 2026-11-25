package gobed

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/search"
)

func persistTestDocs(n int) ([]int, []string) {
	words := []string{"lofi", "piano", "rain", "battle", "orchestral", "synth", "calm", "dark", "retro", "chiptune", "forest", "ocean", "drums", "voice", "ambient", "epic"}
	ids := make([]int, n)
	texts := make([]string, n)
	for i := range ids {
		ids[i] = i
		texts[i] = fmt.Sprintf("%s %s %s track %d", words[i%16], words[(i/16)%16], words[(i/256)%16], i)
	}
	return ids, texts
}

func persistTestConfig() SearchConfig {
	c := DefaultSearchConfig()
	c.AutoMode = false
	c.EnableGPU = false
	c.MaxExactSearchSize = 1000
	return c
}

func searchIDs(t *testing.T, se *SearchEngine, q string) []int {
	t.Helper()
	res, err := se.Search(q, 10)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	return ids
}

// TestSaveLoadKeepsTrainedIndex checks that a saved IVF-PQ index loads with
// identical results and without training.
func TestSaveLoadKeepsTrainedIndex(t *testing.T) {
	model, err := LoadModel()
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	ids, texts := persistTestDocs(5000)
	se := NewSearchEngineWithConfig(model, persistTestConfig())
	if err := se.IndexBatchWithIDs(ids, texts); err != nil {
		t.Fatal(err)
	}
	if se.index.Stats().IndexType == "flat" {
		t.Fatalf("expected a trained IVF index")
	}
	path := filepath.Join(t.TempDir(), "idx.bin")
	if err := se.QuickSave(path); err != nil {
		t.Fatal(err)
	}

	loaded := NewSearchEngineWithConfig(model, persistTestConfig())
	start := time.Now()
	if err := loaded.LoadMatching(path, ids, texts); err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded in %v", time.Since(start))
	if loaded.index.LastTrainTimings() != (search.TrainTimings{}) {
		t.Fatalf("load trained the index")
	}
	for _, q := range []string{"calm piano rain", "epic orchestral battle", "retro chiptune"} {
		a, b := searchIDs(t, se, q), searchIDs(t, loaded, q)
		if len(a) == 0 || fmt.Sprint(a) != fmt.Sprint(b) {
			t.Fatalf("%q: %v vs %v", q, a, b)
		}
	}

	changed := append([]string(nil), texts...)
	changed[7] = "something else"
	other := NewSearchEngineWithConfig(model, persistTestConfig())
	if err := other.LoadMatching(path, ids, changed); !errors.Is(err, ErrSnapshotMismatch) {
		t.Fatalf("want ErrSnapshotMismatch, got %v", err)
	}
	if other.initialized {
		t.Fatalf("mismatched load modified the engine")
	}
}

// TestLoadV1SnapshotRebuilds checks that a documents-only v1 cache rebuilds a
// working index once and is rewritten in the current format.
func TestLoadV1SnapshotRebuilds(t *testing.T) {
	model, err := LoadModel()
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}
	ids, texts := persistTestDocs(3000)
	docs := make(map[int]string, len(ids))
	for i, id := range ids {
		docs[id] = texts[i]
	}
	path := filepath.Join(t.TempDir(), "v1.bin")
	old := &SearchEngine{}
	if err := old.saveBinary(path, &IndexSnapshot{Version: "1.0", NumDocuments: len(docs), Config: persistTestConfig(), Documents: docs, IndexData: &IndexData{Trained: true}}, true); err != nil {
		t.Fatal(err)
	}

	se := NewSearchEngineWithConfig(model, persistTestConfig())
	if err := se.LoadMatching(path, ids, texts); err != nil {
		t.Fatal(err)
	}
	if got := searchIDs(t, se, "calm piano rain"); len(got) != 10 {
		t.Fatalf("v1 load gave %d results", len(got))
	}
	head := make([]byte, len(snapshotMagic))
	f, _ := os.Open(path)
	f.Read(head)
	f.Close()
	if string(head) != snapshotMagic {
		t.Fatalf("v1 load did not upgrade the snapshot in place")
	}
	again := NewSearchEngineWithConfig(model, persistTestConfig())
	if err := again.LoadMatching(path, ids, texts); err != nil {
		t.Fatal(err)
	}
	if again.index.LastTrainTimings() != (search.TrainTimings{}) {
		t.Fatalf("upgraded snapshot retrained on load")
	}
}
