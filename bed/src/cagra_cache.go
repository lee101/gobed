//go:build cagra

package src

import (
	"crypto/sha1"
	"encoding/gob"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/lee101/gobed/ann/simd"
)

const cagraCacheVersion = 1

type cagraCachedDoc struct {
	LineNumber int
	Content    string
	IsBinary   bool
	Vector     simd.Vec512
	Scale      float32
}

type cagraFileCache struct {
	Version int
	RelPath string
	ModTime int64
	Size    int64
	Docs    []cagraCachedDoc
}

type cagraCacheManager struct {
	dir      string
	mu       sync.Mutex
	usedKeys map[string]struct{}
}

func newCAGRACacheManager(dir string) (*cagraCacheManager, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create CAGRA cache dir %s: %w", dir, err)
	}
	gob.Register(simd.Vec512{})
	return &cagraCacheManager{
		dir:      dir,
		usedKeys: make(map[string]struct{}),
	}, nil
}

func (cm *cagraCacheManager) cacheKey(relPath string) string {
	sum := sha1.Sum([]byte(relPath))
	return hex.EncodeToString(sum[:])
}

func (cm *cagraCacheManager) cachePath(relPath string) string {
	name := cm.cacheKey(relPath)
	return filepath.Join(cm.dir, name+".gob")
}

func (cm *cagraCacheManager) markUsed(relPath string) {
	cm.mu.Lock()
	cm.usedKeys[cm.cacheKey(relPath)] = struct{}{}
	cm.mu.Unlock()
}

func (cm *cagraCacheManager) TryLoad(relPath string, info os.FileInfo) (*cagraFileCache, bool) {
	path := cm.cachePath(relPath)
	file, err := os.Open(path)
	if err != nil {
		return nil, false
	}
	defer file.Close()

	var entry cagraFileCache
	if err := gob.NewDecoder(file).Decode(&entry); err != nil {
		return nil, false
	}

	if entry.Version != cagraCacheVersion || entry.RelPath != relPath {
		return nil, false
	}

	if info != nil {
		if entry.ModTime != info.ModTime().UnixNano() || entry.Size != info.Size() {
			return nil, false
		}
	}

	cm.markUsed(relPath)
	return &entry, true
}

func (cm *cagraCacheManager) Save(entry *cagraFileCache) error {
	if entry == nil {
		return nil
	}
	entry.Version = cagraCacheVersion

	tmp, err := os.CreateTemp(cm.dir, "cagra-cache-*")
	if err != nil {
		return fmt.Errorf("failed to create temp cache file: %w", err)
	}
	tmpPath := tmp.Name()

	enc := gob.NewEncoder(tmp)
	if err := enc.Encode(entry); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("failed to encode CAGRA cache entry: %w", err)
	}

	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("failed to sync CAGRA cache entry: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to close CAGRA cache entry: %w", err)
	}

	target := cm.cachePath(entry.RelPath)
	if err := os.Rename(tmpPath, target); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to write CAGRA cache entry: %w", err)
	}

	cm.markUsed(entry.RelPath)
	return nil
}

func (cm *cagraCacheManager) LoadAll() ([]*cagraFileCache, error) {
	dirEntries, err := os.ReadDir(cm.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	results := make([]*cagraFileCache, 0, len(dirEntries))

	for _, entry := range dirEntries {
		if entry.IsDir() {
			continue
		}
		filePath := filepath.Join(cm.dir, entry.Name())
		file, err := os.Open(filePath)
		if err != nil {
			continue
		}

		var cache cagraFileCache
		err = gob.NewDecoder(file).Decode(&cache)
		file.Close()
		if err != nil || cache.Version != cagraCacheVersion {
			continue
		}

		results = append(results, &cache)
		cm.markUsed(cache.RelPath)
	}

	return results, nil
}

func (cm *cagraCacheManager) Cleanup() error {
	dirEntries, err := os.ReadDir(cm.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for _, entry := range dirEntries {
		if entry.IsDir() {
			continue
		}
		key := entry.Name()
		if ext := filepath.Ext(key); ext == ".gob" {
			key = key[:len(key)-len(ext)]
		}
		cm.mu.Lock()
		_, keep := cm.usedKeys[key]
		cm.mu.Unlock()
		if !keep {
			os.Remove(filepath.Join(cm.dir, entry.Name()))
		}
	}
	return nil
}

func copyVec(src simd.Vec512) simd.Vec512 {
	var dst simd.Vec512
	copy(dst[:], src[:])
	return dst
}

func cloneDocs(docs []cagraCachedDoc) []cagraCachedDoc {
	if len(docs) == 0 {
		return nil
	}
	cloned := make([]cagraCachedDoc, len(docs))
	for i, doc := range docs {
		cloned[i] = cagraCachedDoc{
			LineNumber: doc.LineNumber,
			Content:    doc.Content,
			IsBinary:   doc.IsBinary,
			Vector:     copyVec(doc.Vector),
			Scale:      doc.Scale,
		}
	}
	return cloned
}

func (entry *cagraFileCache) Clone() *cagraFileCache {
	if entry == nil {
		return nil
	}
	return &cagraFileCache{
		Version: entry.Version,
		RelPath: entry.RelPath,
		ModTime: entry.ModTime,
		Size:    entry.Size,
		Docs:    cloneDocs(entry.Docs),
	}
}

func (entry *cagraFileCache) ToDocuments(basePath string) ([]CAGRADoc, []simd.Vec512, []float32) {
	if entry == nil {
		return nil, nil, nil
	}
	path := filepath.Join(basePath, entry.RelPath)
	docs := make([]CAGRADoc, 0, len(entry.Docs))
	vecs := make([]simd.Vec512, 0, len(entry.Docs))
	scales := make([]float32, 0, len(entry.Docs))

	for _, doc := range entry.Docs {
		docs = append(docs, CAGRADoc{
			Path:       path,
			LineNumber: doc.LineNumber,
			Content:    doc.Content,
			IsBinary:   doc.IsBinary,
		})
		vecs = append(vecs, copyVec(doc.Vector))
		scales = append(scales, doc.Scale)
	}

	return docs, vecs, scales
}

func writeDocsTo(writer io.Writer, data *cagraFileCache) error {
	enc := gob.NewEncoder(writer)
	return enc.Encode(data)
}
