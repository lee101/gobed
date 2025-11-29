package src

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"
)

// Binary format constants
const (
	MagicNumber = 0xBED12345
	HeaderSize  = 64 // bytes
)

// BinaryHeader represents the binary index file header
type BinaryHeader struct {
	Magic         uint32
	Version       uint32
	HeaderSize    uint32
	IndexSize     uint64
	NumFiles      uint32
	NumLines      uint32
	NumEmbeddings uint32
	Compression   uint8  // 0=none, 1=gzip
	Reserved      [35]byte
}

// saveBinary saves the index in efficient binary format
func (idx *EmbeddingIndex) saveBinary() error {
	path := idx.indexPath + ".bin"
	
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create binary index file: %w", err)
	}
	defer file.Close()
	
	var writer io.Writer = file
	
	// Use compression if configured
	if idx.Config.CompressionLevel > 0 {
		gzWriter, err := gzip.NewWriterLevel(file, idx.Config.CompressionLevel)
		if err != nil {
			return fmt.Errorf("failed to create gzip writer: %w", err)
		}
		defer gzWriter.Close()
		writer = gzWriter
	}
	
	// Write header
	header := BinaryHeader{
		Magic:         MagicNumber,
		Version:       uint32(idx.Version),
		HeaderSize:    HeaderSize,
		IndexSize:     uint64(idx.calculateBinarySize()),
		NumFiles:      uint32(len(idx.Files)),
		NumLines:      uint32(len(idx.Lines)),
		NumEmbeddings: uint32(len(idx.LineEmbeddings)),
		Compression:   0,
	}
	
	if idx.Config.CompressionLevel > 0 {
		header.Compression = 1
	}
	
	if err := binary.Write(writer, binary.LittleEndian, header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	
	// Write metadata section
	if err := idx.writeMetadata(writer); err != nil {
		return fmt.Errorf("failed to write metadata: %w", err)
	}
	
	// Write file entries
	if err := idx.writeFileEntries(writer); err != nil {
		return fmt.Errorf("failed to write file entries: %w", err)
	}
	
	// Write line entries
	if err := idx.writeLineEntries(writer); err != nil {
		return fmt.Errorf("failed to write line entries: %w", err)
	}
	
	// Write embeddings
	if err := idx.writeEmbeddings(writer); err != nil {
		return fmt.Errorf("failed to write embeddings: %w", err)
	}
	
	// Write lookup tables
	if err := idx.writeLookupTables(writer); err != nil {
		return fmt.Errorf("failed to write lookup tables: %w", err)
	}
	
	return nil
}

// writeMetadata writes the metadata section
func (idx *EmbeddingIndex) writeMetadata(writer io.Writer) error {
	metadata := struct {
		CreatedAt    int64  `json:"created_at"`
		UpdatedAt    int64  `json:"updated_at"`
		BasePath     string `json:"base_path"`
		TotalFiles   int    `json:"total_files"`
		TotalLines   int    `json:"total_lines"`
		IndexSize    int64  `json:"index_size"`
		Config       IndexConfig `json:"config"`
	}{
		CreatedAt:  idx.CreatedAt.Unix(),
		UpdatedAt:  idx.UpdatedAt.Unix(),
		BasePath:   idx.BasePath,
		TotalFiles: idx.TotalFiles,
		TotalLines: idx.TotalLines,
		IndexSize:  idx.IndexSize,
		Config:     idx.Config,
	}
	
	data, err := json.Marshal(metadata)
	if err != nil {
		return err
	}
	
	// Write data length, then data
	if err := binary.Write(writer, binary.LittleEndian, uint32(len(data))); err != nil {
		return err
	}
	
	_, err = writer.Write(data)
	return err
}

// writeFileEntries writes all file entries
func (idx *EmbeddingIndex) writeFileEntries(writer io.Writer) error {
	// Write number of files
	if err := binary.Write(writer, binary.LittleEndian, uint32(len(idx.Files))); err != nil {
		return err
	}
	
	// Write each file entry
	for _, file := range idx.Files {
		if err := idx.writeFileEntry(writer, file); err != nil {
			return err
		}
	}
	
	return nil
}

// writeFileEntry writes a single file entry
func (idx *EmbeddingIndex) writeFileEntry(writer io.Writer, file *FileEntry) error {
	// Fixed-size portion
	fixedData := struct {
		ID        uint32
		Size      uint64
		ModTime   int64
		LineCount uint32
		Checksum  uint32
		LineStart uint32
		LineEnd   uint32
	}{
		ID:        uint32(file.ID),
		Size:      uint64(file.Size),
		ModTime:   file.ModTime.Unix(),
		LineCount: uint32(file.LineCount),
		Checksum:  file.Checksum,
		LineStart: uint32(file.LineStart),
		LineEnd:   uint32(file.LineEnd),
	}
	
	if err := binary.Write(writer, binary.LittleEndian, fixedData); err != nil {
		return err
	}
	
	// Variable-size strings
	strings := []string{file.Path, file.RelativePath, file.Language}
	for _, str := range strings {
		data := []byte(str)
		if err := binary.Write(writer, binary.LittleEndian, uint32(len(data))); err != nil {
			return err
		}
		if _, err := writer.Write(data); err != nil {
			return err
		}
	}
	
	return nil
}

// writeLineEntries writes all line entries
func (idx *EmbeddingIndex) writeLineEntries(writer io.Writer) error {
	// Write number of lines
	if err := binary.Write(writer, binary.LittleEndian, uint32(len(idx.Lines))); err != nil {
		return err
	}
	
	// Write each line entry
	for _, line := range idx.Lines {
		if err := idx.writeLineEntry(writer, line); err != nil {
			return err
		}
	}
	
	return nil
}

// writeLineEntry writes a single line entry
func (idx *EmbeddingIndex) writeLineEntry(writer io.Writer, line *LineEntry) error {
	// Fixed-size portion
	fixedData := struct {
		ID          uint32
		FileID      uint32
		LineNumber  uint32
		EmbeddingID uint32
	}{
		ID:          uint32(line.ID),
		FileID:      uint32(line.FileID),
		LineNumber:  uint32(line.LineNumber),
		EmbeddingID: uint32(line.EmbeddingID),
	}
	
	if err := binary.Write(writer, binary.LittleEndian, fixedData); err != nil {
		return err
	}
	
	// Variable-size strings
	strings := []string{line.Content, line.ContextBefore, line.ContextAfter}
	for _, str := range strings {
		data := []byte(str)
		if err := binary.Write(writer, binary.LittleEndian, uint32(len(data))); err != nil {
			return err
		}
		if _, err := writer.Write(data); err != nil {
			return err
		}
	}
	
	return nil
}

// writeEmbeddings writes all embeddings
func (idx *EmbeddingIndex) writeEmbeddings(writer io.Writer) error {
	// Write number of line embeddings
	if err := binary.Write(writer, binary.LittleEndian, uint32(len(idx.LineEmbeddings))); err != nil {
		return err
	}
	
	// Write dimension (assuming all embeddings have same dimension)
	if len(idx.LineEmbeddings) > 0 {
		dims := uint32(len(idx.LineEmbeddings[0]))
		if err := binary.Write(writer, binary.LittleEndian, dims); err != nil {
			return err
		}
		
		// Write all line embeddings
		for _, embedding := range idx.LineEmbeddings {
			for _, val := range embedding {
				if err := binary.Write(writer, binary.LittleEndian, val); err != nil {
					return err
				}
			}
		}
	}
	
	// Write file embeddings
	if err := binary.Write(writer, binary.LittleEndian, uint32(len(idx.FileEmbeddings))); err != nil {
		return err
	}
	
	for path, embedding := range idx.FileEmbeddings {
		// Write path
		pathData := []byte(path)
		if err := binary.Write(writer, binary.LittleEndian, uint32(len(pathData))); err != nil {
			return err
		}
		if _, err := writer.Write(pathData); err != nil {
			return err
		}
		
		// Write embedding
		for _, val := range embedding {
			if err := binary.Write(writer, binary.LittleEndian, val); err != nil {
				return err
			}
		}
	}
	
	return nil
}

// writeLookupTables writes lookup tables
func (idx *EmbeddingIndex) writeLookupTables(writer io.Writer) error {
	// Write PathToID map
	if err := binary.Write(writer, binary.LittleEndian, uint32(len(idx.PathToID))); err != nil {
		return err
	}
	
	for path, id := range idx.PathToID {
		pathData := []byte(path)
		if err := binary.Write(writer, binary.LittleEndian, uint32(len(pathData))); err != nil {
			return err
		}
		if _, err := writer.Write(pathData); err != nil {
			return err
		}
		if err := binary.Write(writer, binary.LittleEndian, uint32(id)); err != nil {
			return err
		}
	}
	
	return nil
}

// loadBinaryIndex loads an index from binary format
func loadBinaryIndex(path string) (*EmbeddingIndex, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open binary index file: %w", err)
	}
	defer file.Close()
	
	// Read and verify header
	var header BinaryHeader
	if err := binary.Read(file, binary.LittleEndian, &header); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}
	
	if header.Magic != MagicNumber {
		return nil, fmt.Errorf("invalid magic number: got %x, expected %x", header.Magic, MagicNumber)
	}
	
	if header.Version != IndexVersion {
		return nil, fmt.Errorf("unsupported index version: %d", header.Version)
	}
	
	var reader io.Reader = file
	
	// Handle compression
	if header.Compression == 1 {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return nil, fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()
		reader = gzReader
	}
	
	// Create index
	idx := &EmbeddingIndex{
		Files:          make(map[string]*FileEntry),
		Lines:          make([]*LineEntry, 0, header.NumLines),
		FileEmbeddings: make(map[string][]float32),
		LineEmbeddings: make([][]float32, 0, header.NumEmbeddings),
		PathToID:       make(map[string]int),
		IDToPath:       make(map[int]string),
	}
	
	// Read metadata
	if err := idx.readMetadata(reader); err != nil {
		return nil, fmt.Errorf("failed to read metadata: %w", err)
	}
	
	// Read file entries
	if err := idx.readFileEntries(reader); err != nil {
		return nil, fmt.Errorf("failed to read file entries: %w", err)
	}
	
	// Read line entries
	if err := idx.readLineEntries(reader); err != nil {
		return nil, fmt.Errorf("failed to read line entries: %w", err)
	}
	
	// Read embeddings
	if err := idx.readEmbeddings(reader); err != nil {
		return nil, fmt.Errorf("failed to read embeddings: %w", err)
	}
	
	// Read lookup tables
	if err := idx.readLookupTables(reader); err != nil {
		return nil, fmt.Errorf("failed to read lookup tables: %w", err)
	}
	
	return idx, nil
}

// saveJSON saves the index in JSON format for debugging
func (idx *EmbeddingIndex) saveJSON() error {
	path := idx.indexPath + ".json"
	
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	
	return encoder.Encode(idx)
}

// loadJSONIndex loads an index from JSON format
func loadJSONIndex(path string) (*EmbeddingIndex, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	var idx EmbeddingIndex
	decoder := json.NewDecoder(file)
	
	if err := decoder.Decode(&idx); err != nil {
		return nil, err
	}
	
	return &idx, nil
}

// calculateBinarySize estimates the binary size of the index
func (idx *EmbeddingIndex) calculateBinarySize() int64 {
	var size int64
	
	// Header
	size += HeaderSize
	
	// Metadata (estimated)
	size += 1024
	
	// File entries (estimated 256 bytes each)
	size += int64(len(idx.Files)) * 256
	
	// Line entries (estimated 128 bytes each)
	size += int64(len(idx.Lines)) * 128
	
	// Embeddings (1024 dimensions * 4 bytes per float32)
	size += int64(len(idx.LineEmbeddings)) * 1024 * 4
	size += int64(len(idx.FileEmbeddings)) * 1024 * 4
	
	// Lookup tables
	size += int64(len(idx.PathToID)) * 64
	
	return size
}

// Helper methods for reading binary data

func (idx *EmbeddingIndex) readMetadata(reader io.Reader) error {
	// Read metadata length
	var length uint32
	if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
		return err
	}
	
	// Read metadata JSON
	data := make([]byte, length)
	if _, err := io.ReadFull(reader, data); err != nil {
		return err
	}
	
	var metadata struct {
		CreatedAt    int64       `json:"created_at"`
		UpdatedAt    int64       `json:"updated_at"`
		BasePath     string      `json:"base_path"`
		TotalFiles   int         `json:"total_files"`
		TotalLines   int         `json:"total_lines"`
		IndexSize    int64       `json:"index_size"`
		Config       IndexConfig `json:"config"`
	}
	
	if err := json.Unmarshal(data, &metadata); err != nil {
		return err
	}
	
	idx.CreatedAt = timeFromUnix(metadata.CreatedAt)
	idx.UpdatedAt = timeFromUnix(metadata.UpdatedAt)
	idx.BasePath = metadata.BasePath
	idx.TotalFiles = metadata.TotalFiles
	idx.TotalLines = metadata.TotalLines
	idx.IndexSize = metadata.IndexSize
	idx.Config = metadata.Config
	
	return nil
}

func (idx *EmbeddingIndex) readFileEntries(reader io.Reader) error {
	// Read number of files
	var numFiles uint32
	if err := binary.Read(reader, binary.LittleEndian, &numFiles); err != nil {
		return err
	}
	
	// Read each file entry
	for i := uint32(0); i < numFiles; i++ {
		file, err := idx.readFileEntry(reader)
		if err != nil {
			return err
		}
		
		idx.Files[file.RelativePath] = file
		idx.PathToID[file.RelativePath] = file.ID
		idx.IDToPath[file.ID] = file.RelativePath
	}
	
	return nil
}

func (idx *EmbeddingIndex) readFileEntry(reader io.Reader) (*FileEntry, error) {
	// Read fixed-size portion
	var fixedData struct {
		ID        uint32
		Size      uint64
		ModTime   int64
		LineCount uint32
		Checksum  uint32
		LineStart uint32
		LineEnd   uint32
	}
	
	if err := binary.Read(reader, binary.LittleEndian, &fixedData); err != nil {
		return nil, err
	}
	
	file := &FileEntry{
		ID:        int(fixedData.ID),
		Size:      int64(fixedData.Size),
		ModTime:   timeFromUnix(fixedData.ModTime),
		LineCount: int(fixedData.LineCount),
		Checksum:  fixedData.Checksum,
		LineStart: int(fixedData.LineStart),
		LineEnd:   int(fixedData.LineEnd),
	}
	
	// Read variable-size strings
	strings := []*string{&file.Path, &file.RelativePath, &file.Language}
	for _, strPtr := range strings {
		str, err := readString(reader)
		if err != nil {
			return nil, err
		}
		*strPtr = str
	}
	
	return file, nil
}

func (idx *EmbeddingIndex) readLineEntries(reader io.Reader) error {
	// Read number of lines
	var numLines uint32
	if err := binary.Read(reader, binary.LittleEndian, &numLines); err != nil {
		return err
	}
	
	// Read each line entry
	idx.Lines = make([]*LineEntry, numLines)
	for i := uint32(0); i < numLines; i++ {
		line, err := idx.readLineEntry(reader)
		if err != nil {
			return err
		}
		idx.Lines[i] = line
	}
	
	return nil
}

func (idx *EmbeddingIndex) readLineEntry(reader io.Reader) (*LineEntry, error) {
	// Read fixed-size portion
	var fixedData struct {
		ID          uint32
		FileID      uint32
		LineNumber  uint32
		EmbeddingID uint32
	}
	
	if err := binary.Read(reader, binary.LittleEndian, &fixedData); err != nil {
		return nil, err
	}
	
	line := &LineEntry{
		ID:          int(fixedData.ID),
		FileID:      int(fixedData.FileID),
		LineNumber:  int(fixedData.LineNumber),
		EmbeddingID: int(fixedData.EmbeddingID),
	}
	
	// Read variable-size strings
	strings := []*string{&line.Content, &line.ContextBefore, &line.ContextAfter}
	for _, strPtr := range strings {
		str, err := readString(reader)
		if err != nil {
			return nil, err
		}
		*strPtr = str
	}
	
	return line, nil
}

func (idx *EmbeddingIndex) readEmbeddings(reader io.Reader) error {
	// Read line embeddings
	var numEmbeddings uint32
	if err := binary.Read(reader, binary.LittleEndian, &numEmbeddings); err != nil {
		return err
	}
	
	if numEmbeddings > 0 {
		// Read dimensions
		var dims uint32
		if err := binary.Read(reader, binary.LittleEndian, &dims); err != nil {
			return err
		}
		
		// Read all embeddings
		idx.LineEmbeddings = make([][]float32, numEmbeddings)
		for i := uint32(0); i < numEmbeddings; i++ {
			embedding := make([]float32, dims)
			for j := uint32(0); j < dims; j++ {
				if err := binary.Read(reader, binary.LittleEndian, &embedding[j]); err != nil {
					return err
				}
			}
			idx.LineEmbeddings[i] = embedding
		}
	}
	
	// Read file embeddings
	var numFileEmbeddings uint32
	if err := binary.Read(reader, binary.LittleEndian, &numFileEmbeddings); err != nil {
		return err
	}
	
	for i := uint32(0); i < numFileEmbeddings; i++ {
		// Read path
		path, err := readString(reader)
		if err != nil {
			return err
		}
		
		// Read embedding (assume same dimensions as line embeddings)
		var dims uint32 = 1024 // Default, should match line embeddings
		if len(idx.LineEmbeddings) > 0 {
			dims = uint32(len(idx.LineEmbeddings[0]))
		}
		
		embedding := make([]float32, dims)
		for j := uint32(0); j < dims; j++ {
			if err := binary.Read(reader, binary.LittleEndian, &embedding[j]); err != nil {
				return err
			}
		}
		
		idx.FileEmbeddings[path] = embedding
	}
	
	return nil
}

func (idx *EmbeddingIndex) readLookupTables(reader io.Reader) error {
	// Read PathToID map
	var numEntries uint32
	if err := binary.Read(reader, binary.LittleEndian, &numEntries); err != nil {
		return err
	}
	
	for i := uint32(0); i < numEntries; i++ {
		path, err := readString(reader)
		if err != nil {
			return err
		}
		
		var id uint32
		if err := binary.Read(reader, binary.LittleEndian, &id); err != nil {
			return err
		}
		
		idx.PathToID[path] = int(id)
	}
	
	return nil
}

func readString(reader io.Reader) (string, error) {
	var length uint32
	if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	
	if length == 0 {
		return "", nil
	}
	
	data := make([]byte, length)
	if _, err := io.ReadFull(reader, data); err != nil {
		return "", err
	}
	
	return string(data), nil
}

func timeFromUnix(timestamp int64) time.Time {
	return time.Unix(timestamp, 0)
}