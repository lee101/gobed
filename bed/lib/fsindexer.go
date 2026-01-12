package lib

import (
    "bufio"
    "errors"
    "fmt"
    "io"
    "os"
    "path/filepath"
    "strings"
    "time"

    gobed "github.com/lee101/gobed"
)

// FileDoc represents a chunk of a file indexed as a document.
type FileDoc struct {
    ID        int
    Path      string
    Offset    int   // byte offset in file
    Length    int   // bytes
    Text      string
    ModTime   int64 // file mtime unix
}

// FSIndexer indexes a filesystem tree into a VectorIndex.
type FSIndexer struct {
    root     string
    model    *gobed.EmbeddingModel
    index    *gobed.VectorIndex
    nextID   int
    // minimal state for incremental updates (mtime-based)
    byPath   map[string][]int // file path -> doc IDs
}

// NewFSIndexer creates a filesystem indexer with a fresh VectorIndex.
func NewFSIndexer(root string, model *gobed.EmbeddingModel, cfg gobed.VectorIndexConfig) (*FSIndexer, error) {
    if root == "" {
        return nil, errors.New("root path is required")
    }
    abs, err := filepath.Abs(root)
    if err != nil {
        return nil, err
    }
    idx := gobed.NewVectorIndex(model, cfg)
    return &FSIndexer{
        root:   abs,
        model:  model,
        index:  idx,
        byPath: make(map[string][]int),
    }, nil
}

// IndexAll walks the root and indexes supported text files.
// It chunks files into paragraphs (blank-line separated) to improve recall.
func (f *FSIndexer) IndexAll() (int, error) {
    count := 0
    err := filepath.WalkDir(f.root, func(path string, d os.DirEntry, err error) error {
        if err != nil {
            return err
        }
        if d.IsDir() {
            // skip hidden directories
            base := filepath.Base(path)
            if strings.HasPrefix(base, ".") || base == "node_modules" || base == ".git" {
                return filepath.SkipDir
            }
            return nil
        }
        if !isTextFile(path) {
            return nil
        }
        n, err := f.indexFile(path)
        if err != nil {
            return fmt.Errorf("index %s: %w", path, err)
        }
        count += n
        return nil
    })
    return count, err
}

// Reindex updates changed files (mtime-based). For now, rebuilds entries per changed file.
func (f *FSIndexer) Reindex() (int, error) {
    updated := 0
    err := filepath.WalkDir(f.root, func(path string, d os.DirEntry, err error) error {
        if err != nil {
            return err
        }
        if d.IsDir() {
            base := filepath.Base(path)
            if strings.HasPrefix(base, ".") || base == "node_modules" || base == ".git" {
                return filepath.SkipDir
            }
            return nil
        }
        if !isTextFile(path) {
            return nil
        }
        fi, statErr := os.Stat(path)
        if statErr != nil {
            return statErr
        }
        mt := fi.ModTime().Unix()
        // If unseen or modified since last run, (re)index.
        // For now we don’t support deletions and we append new doc IDs.
        // Future: add doc deletion and ID reuse.
        seen := f.byPath[path]
        if len(seen) == 0 || newerThanAll(mt, seen) {
            n, err := f.indexFile(path)
            if err != nil {
                return fmt.Errorf("reindex %s: %w", path, err)
            }
            if n > 0 {
                updated += n
            }
        }
        return nil
    })
    return updated, err
}

func (f *FSIndexer) indexFile(path string) (int, error) {
    file, err := os.Open(path)
    if err != nil {
        return 0, err
    }
    defer file.Close()

    fi, err := file.Stat()
    if err != nil {
        return 0, err
    }
    mt := fi.ModTime().Unix()

    docs := make([]gobed.Document, 0, 16)
    var off int
    r := bufio.NewReader(file)
    for {
        para, n, e := readParagraph(r)
        if n > 0 && strings.TrimSpace(para) != "" {
            id := f.nextID
            f.nextID++
            docs = append(docs, gobed.Document{ID: id, Text: para})
            f.byPath[path] = append(f.byPath[path], id)
            _ = mt // future: store per-doc mtime
        }
        off += n
        if e == io.EOF {
            break
        }
        if e != nil {
            return 0, e
        }
    }
    if len(docs) == 0 {
        return 0, nil
    }
    if err := f.index.AddDocuments(docs); err != nil {
        return 0, err
    }
    return len(docs), nil
}

func readParagraph(r *bufio.Reader) (string, int, error) {
    var b strings.Builder
    total := 0
    for {
        line, err := r.ReadString('\n')
        total += len(line)
        if err != nil && err != io.EOF {
            return "", total, err
        }
        // paragraph split on blank line
        if strings.TrimSpace(line) == "" {
            break
        }
        b.WriteString(line)
        if err == io.EOF {
            break
        }
    }
    return b.String(), total, nil
}

func isTextFile(path string) bool {
    ext := strings.ToLower(filepath.Ext(path))
    switch ext {
    case ".txt", ".md", ".markdown", ".log", ".rst", ".go", ".py", ".js", ".ts", ".rs", ".java", ".c", ".cpp", ".h", ".hpp":
        return true
    default:
        return false
    }
}

func newerThanAll(mt int64, ids []int) bool {
    _ = mt
    // Placeholder: we don’t persist per-doc mtime yet
    return true
}

// Index exposes the underlying vector index (read-only usage recommended).
func (f *FSIndexer) Index() *gobed.VectorIndex { return f.index }

// Root returns the root path of the indexer.
func (f *FSIndexer) Root() string { return f.root }

// CreatedAt returns now for simple logging; reserved for future persistent state.
func (f *FSIndexer) CreatedAt() time.Time { return time.Now() }

