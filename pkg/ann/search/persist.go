package search

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"hash/fnv"
	"io"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/lee101/gobed/pkg/ann/flat"
	"github.com/lee101/gobed/pkg/ann/hnsw"
	"github.com/lee101/gobed/pkg/ann/ivf"
	"github.com/lee101/gobed/pkg/ann/pq"
	"github.com/lee101/gobed/pkg/ann/simd"
)

// On-disk layout (little endian), version 1:
//
//	magic "GBANNENG" | u32 version | requested Config | resolved Config |
//	u8 trained | i64 size | i64 n | u64 contentHash |
//	n*512 int8 vectors | n f32 scales | n i64 ids |
//	u8 hasIVF [ i64 K | K*512 int8 | K f32 | K x (i64 len | len i32) ] |
//	u8 hasPQ  [ i64 M | i64 KSub | i64 DSub | M*KSub*DSub f32 | i64 nc | nc*M u8 ] |
//	u8 hasHNSW [ i64 entry | i64 nodes | nodes x (i64 level | (level+1) x (i64 len | len i32)) ] |
//	u32 crc32(IEEE) of all preceding bytes
//
// Loading never trains: it restores centroids, lists, codebooks, codes and the
// HNSW router exactly as saved.

const (
	persistMagic   = "GBANNENG"
	persistVersion = 1
)

var (
	// ErrIndexMismatch means the saved index was built with a different
	// configuration than the engine it is being loaded into; retrain instead.
	ErrIndexMismatch = errors.New("search: saved index does not match engine config")
	// ErrIndexCorrupt means the file failed its version, checksum or content check.
	ErrIndexCorrupt = errors.New("search: saved index is corrupt or from an unsupported version")
)

type persistConfig struct {
	MaxFlatSize, NList, NProbe, M, NBits int64
	HNSWEnabled                          uint8
	HNSWM, HNSWEfC, RerankSize           int64
	UseParallel                          uint8
}

func b2u(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}

func toPersist(c Config) persistConfig {
	return persistConfig{int64(c.MaxFlatSize), int64(c.NList), int64(c.NProbe), int64(c.M), int64(c.NBits),
		b2u(c.HNSWEnabled), int64(c.HNSWM), int64(c.HNSWEfC), int64(c.RerankSize), b2u(c.UseParallel)}
}

func fromPersist(p persistConfig) Config {
	return Config{int(p.MaxFlatSize), int(p.NList), int(p.NProbe), int(p.M), int(p.NBits),
		p.HNSWEnabled == 1, int(p.HNSWM), int(p.HNSWEfC), int(p.RerankSize), p.UseParallel == 1}
}

func vecBytes(v []simd.Vec512) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*512)
}

func contentHash(vectors []simd.Vec512, scales []float32, ids []int) uint64 {
	h := fnv.New64a()
	h.Write(vecBytes(vectors))
	binary.Write(h, binary.LittleEndian, scales)
	buf := make([]byte, 8)
	for _, id := range ids {
		binary.LittleEndian.PutUint64(buf, uint64(int64(id)))
		h.Write(buf)
	}
	return h.Sum64()
}

// ContentHash identifies the indexed vectors, scales and ids (in order).
func (e *Engine) ContentHash() uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return contentHash(e.rawVectors, e.scales, e.ids)
}

type persistWriter struct {
	w   *bufio.Writer
	crc hash.Hash32
	err error
}

func (p *persistWriter) bytes(b []byte) {
	if p.err != nil {
		return
	}
	p.crc.Write(b)
	_, p.err = p.w.Write(b)
}

func (p *persistWriter) put(v any) {
	if p.err != nil {
		return
	}
	p.err = binary.Write(io.MultiWriter(p.w, p.crc), binary.LittleEndian, v)
}

func (p *persistWriter) i64(v int) { p.put(int64(v)) }

func (p *persistWriter) ints(v []int) {
	p.i64(len(v))
	buf := make([]int32, len(v))
	for i, x := range v {
		buf[i] = int32(x)
	}
	p.put(buf)
}

// Save writes the full engine state, including trained IVF/PQ/HNSW structures.
func (e *Engine) Save(w io.Writer) error {
	e.mu.RLock()
	defer e.mu.RUnlock()

	p := &persistWriter{w: bufio.NewWriterSize(w, 1<<20), crc: crc32.NewIEEE()}
	p.bytes([]byte(persistMagic))
	p.put(uint32(persistVersion))
	p.put(toPersist(e.requested))
	p.put(toPersist(e.config))
	p.put(b2u(e.trained))
	p.i64(e.size)
	n := len(e.rawVectors)
	p.i64(n)
	p.put(contentHash(e.rawVectors, e.scales, e.ids))
	p.bytes(vecBytes(e.rawVectors))
	p.put(e.scales)
	ids := make([]int64, n)
	for i, id := range e.ids {
		ids[i] = int64(id)
	}
	p.put(ids)

	p.put(b2u(e.ivfIndex != nil))
	if e.ivfIndex != nil {
		km := e.ivfIndex.KMeans
		p.i64(len(km.Centroids))
		p.bytes(vecBytes(km.Centroids))
		p.put(km.Scales)
		for k := range e.ivfIndex.Lists {
			e.ivfIndex.ListLocks[k].RLock()
			p.ints(e.ivfIndex.Lists[k])
			e.ivfIndex.ListLocks[k].RUnlock()
		}
	}

	p.put(b2u(e.pq != nil && e.pq.Trained))
	if e.pq != nil && e.pq.Trained {
		p.i64(e.pq.M)
		p.i64(e.pq.KSub)
		p.i64(e.pq.DSub)
		for _, cb := range e.pq.Codebooks {
			for _, c := range cb {
				p.put(c)
			}
		}
		p.i64(len(e.pqCodes))
		for _, c := range e.pqCodes {
			p.bytes(c)
		}
	}

	p.put(b2u(e.hnswRouter != nil))
	if e.hnswRouter != nil {
		g := e.hnswRouter.Export()
		p.i64(g.EntryPoint)
		p.i64(len(g.Levels))
		for i, lvl := range g.Levels {
			p.i64(lvl)
			for _, nb := range g.Neighbors[i] {
				p.ints(nb)
			}
		}
	}

	if p.err != nil {
		return p.err
	}
	if err := binary.Write(p.w, binary.LittleEndian, p.crc.Sum32()); err != nil {
		return err
	}
	return p.w.Flush()
}

type persistReader struct {
	r   io.Reader
	crc hash.Hash32
	err error
}

func (p *persistReader) bytes(b []byte) {
	if p.err != nil {
		return
	}
	_, p.err = io.ReadFull(p.r, b)
	p.crc.Write(b)
}

func (p *persistReader) get(v any) {
	if p.err != nil {
		return
	}
	p.err = binary.Read(io.TeeReader(p.r, p.crc), binary.LittleEndian, v)
}

// count reads a non-negative length bounded by limit.
func (p *persistReader) count(limit int64) int {
	var v int64
	p.get(&v)
	if p.err == nil && (v < 0 || v > limit) {
		p.err = fmt.Errorf("%w: length %d out of range", ErrIndexCorrupt, v)
	}
	if p.err != nil {
		return 0
	}
	return int(v)
}

func (p *persistReader) ints(limit int64) []int {
	n := p.count(limit)
	buf := make([]int32, n)
	p.get(buf)
	out := make([]int, n)
	for i, x := range buf {
		if int64(x) < 0 || int64(x) >= limit {
			if p.err == nil {
				p.err = fmt.Errorf("%w: index %d out of range", ErrIndexCorrupt, x)
			}
			return nil
		}
		out[i] = int(x)
	}
	return out
}

// Load replaces the engine's state with a saved one without training. It
// returns ErrIndexMismatch when the file was built for a different requested
// config, and ErrIndexCorrupt on a bad magic, version, checksum or content
// hash. On error the engine is left unchanged.
func (e *Engine) Load(r io.Reader) error {
	e.mu.RLock()
	want := toPersist(e.requested)
	e.mu.RUnlock()
	return e.load(r, &want)
}

// ReadEngine restores a saved engine with whatever config it was built with.
func ReadEngine(r io.Reader) (*Engine, error) {
	e := NewEngine(Config{})
	if err := e.load(r, nil); err != nil {
		return nil, err
	}
	return e, nil
}

func (e *Engine) load(r io.Reader, want *persistConfig) error {
	p := &persistReader{r: bufio.NewReaderSize(r, 1<<20), crc: crc32.NewIEEE()}
	magic := make([]byte, len(persistMagic))
	p.bytes(magic)
	var version uint32
	p.get(&version)
	if p.err != nil || string(magic) != persistMagic || version != persistVersion {
		return ErrIndexCorrupt
	}
	var req, res persistConfig
	p.get(&req)
	p.get(&res)
	if p.err != nil {
		return fmt.Errorf("%w: %v", ErrIndexCorrupt, p.err)
	}
	if want != nil && req != *want {
		return ErrIndexMismatch
	}
	cfg := fromPersist(res)

	const maxN = 1 << 31
	var trained uint8
	p.get(&trained)
	size := p.count(maxN)
	n := p.count(maxN)
	var wantHash uint64
	p.get(&wantHash)
	if p.err != nil {
		return fmt.Errorf("%w: %v", ErrIndexCorrupt, p.err)
	}
	vectors := make([]simd.Vec512, n)
	p.bytes(vecBytes(vectors))
	scales := make([]float32, n)
	p.get(scales)
	ids64 := make([]int64, n)
	p.get(ids64)
	ids := make([]int, n)
	for i, id := range ids64 {
		ids[i] = int(id)
	}

	var hasIVF, hasPQ, hasHNSW uint8
	var centroids []simd.Vec512
	var cscales []float32
	var lists [][]int
	p.get(&hasIVF)
	if hasIVF == 1 {
		k := p.count(1 << 20)
		centroids = make([]simd.Vec512, k)
		p.bytes(vecBytes(centroids))
		cscales = make([]float32, k)
		p.get(cscales)
		lists = make([][]int, k)
		for i := range lists {
			lists[i] = p.ints(int64(n))
		}
	}

	var quant *pq.ProductQuantizer
	var codes [][]uint8
	p.get(&hasPQ)
	if hasPQ == 1 {
		m, ksub, dsub := p.count(512), p.count(1<<16), p.count(512)
		if p.err == nil && (m == 0 || m*dsub != 512) {
			p.err = fmt.Errorf("%w: bad PQ shape", ErrIndexCorrupt)
		}
		if p.err == nil {
			quant = pq.NewProductQuantizer(512, m, ksub)
			for _, cb := range quant.Codebooks {
				for _, c := range cb {
					p.get(c)
				}
			}
			quant.Trained = true
			nc := p.count(int64(n))
			flatCodes := make([]byte, nc*m)
			p.bytes(flatCodes)
			codes = make([][]uint8, nc)
			for i := range codes {
				codes[i] = flatCodes[i*m : (i+1)*m : (i+1)*m]
			}
		}
	}

	var graph hnsw.Graph
	p.get(&hasHNSW)
	if hasHNSW == 1 {
		k := int64(len(centroids))
		var entry int64
		p.get(&entry)
		nodes := p.count(k)
		graph = hnsw.Graph{EntryPoint: int(entry), Levels: make([]int, nodes), Neighbors: make([][][]int, nodes)}
		for i := 0; i < nodes && p.err == nil; i++ {
			lvl := p.count(64)
			graph.Levels[i] = lvl
			graph.Neighbors[i] = make([][]int, lvl+1)
			for l := 0; l <= lvl; l++ {
				graph.Neighbors[i][l] = p.ints(k)
			}
		}
		if p.err == nil && (entry < 0 || entry >= int64(nodes) || nodes != len(centroids)) {
			p.err = fmt.Errorf("%w: bad HNSW graph", ErrIndexCorrupt)
		}
	}

	if p.err != nil {
		return fmt.Errorf("%w: %v", ErrIndexCorrupt, p.err)
	}
	sum := p.crc.Sum32()
	var stored uint32
	if err := binary.Read(p.r, binary.LittleEndian, &stored); err != nil || stored != sum {
		return fmt.Errorf("%w: checksum mismatch", ErrIndexCorrupt)
	}
	if contentHash(vectors, scales, ids) != wantHash {
		return fmt.Errorf("%w: content hash mismatch", ErrIndexCorrupt)
	}
	if hasPQ == 1 && len(codes) != n {
		return fmt.Errorf("%w: %d PQ codes for %d vectors", ErrIndexCorrupt, len(codes), n)
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	e.requested = fromPersist(req)
	e.config = cfg
	e.trained = trained == 1
	e.size = size
	e.rawVectors, e.scales, e.ids = vectors, scales, ids
	e.ivfIndex, e.pq, e.pqCodes, e.hnswRouter = nil, nil, nil, nil
	e.flatIndex = flat.NewFlatIndex(max(cfg.MaxFlatSize, n))
	if hasIVF == 1 {
		e.ivfIndex = ivf.NewIVFIndex(len(centroids), cfg.NProbe)
		e.ivfIndex.Restore(centroids, cscales, lists, vectors, scales, ids)
		e.flatIndex = nil
		if quant != nil {
			e.pq, e.pqCodes = quant, codes
		}
		if hasHNSW == 1 {
			e.hnswRouter = hnsw.NewHNSW(cfg.HNSWM, cfg.HNSWEfC)
			e.hnswRouter.Restore(graph, centroids, cscales)
		}
	} else if n > 0 {
		e.flatIndex.AddBatch(vectors, scales, ids)
	}
	return nil
}

// SaveFile writes the engine to path atomically (temp file + rename).
func (e *Engine) SaveFile(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	tmp.Chmod(0o644)
	if err := e.Save(tmp); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// LoadFile loads an engine saved with SaveFile.
func (e *Engine) LoadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return e.Load(f)
}
