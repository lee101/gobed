package gobed

import (
	"math"
	"unicode"
	"unicode/utf8"
)

const maxWordBytes = 100

type vocabSlot struct {
	hash uint32
	off  uint32
	n    uint16
	id   int16
}

type vocabTable struct {
	slots  []vocabSlot
	mask   uint32
	blob   []byte
	single [128]int16
}

func fnv32[T ~string | ~[]byte](k T) uint32 {
	h := uint32(2166136261)
	for i := 0; i < len(k); i++ {
		h = (h ^ uint32(k[i])) * 16777619
	}
	return h
}

func newVocabTable(vocab map[string]int16) *vocabTable {
	size := 16
	for size < len(vocab)*2 {
		size <<= 1
	}
	v := &vocabTable{slots: make([]vocabSlot, size), mask: uint32(size - 1)}
	for i := range v.single {
		v.single[i] = -1
	}
	for k, id := range vocab {
		if len(k) == 0 || len(k) > math.MaxUint16 {
			continue
		}
		off := uint32(len(v.blob))
		v.blob = append(v.blob, k...)
		h := fnv32(k)
		i := h & v.mask
		for v.slots[i].n != 0 {
			i = (i + 1) & v.mask
		}
		v.slots[i] = vocabSlot{hash: h, off: off, n: uint16(len(k)), id: id}
		if len(k) == 1 && k[0] < 0x80 {
			v.single[k[0]] = id
		}
	}
	return v
}

func vocabGet[T ~string | ~[]byte](v *vocabTable, k T) (int16, bool) {
	h := fnv32(k)
	i := h & v.mask
	for {
		s := &v.slots[i]
		if s.n == 0 {
			return 0, false
		}
		if s.hash == h && int(s.n) == len(k) {
			b := v.blob[s.off : s.off+uint32(s.n)]
			j := 0
			for j < len(k) && b[j] == k[j] {
				j++
			}
			if j == len(k) {
				return s.id, true
			}
		}
		i = (i + 1) & v.mask
	}
}

func asciiWordByte(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '\'' || c == '_'
}

func asciiSpaceByte(c byte) bool {
	return c == ' ' || c >= '\t' && c <= '\r'
}

func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' || r == '_'
}

func (t *wordPieceTokenizer) tokenizeInto(text string, dst []int16, limit int) []int16 {
	if limit <= 0 {
		limit = math.MaxInt
	}
	if t.addSpecial && t.clsID >= 0 {
		dst = append(dst, t.clsID)
	}
	var buf [maxWordBytes + 6]byte
	var pieces [maxWordBytes]int16
	n := len(text)
	for i := 0; i < n && len(dst) < limit; {
		c := text[i]
		if c < 0x80 {
			if asciiSpaceByte(c) {
				i++
				continue
			}
			if !asciiWordByte(c) {
				if id := t.table.single[c]; id >= 0 {
					dst = append(dst, id)
				}
				i++
				continue
			}
		} else {
			r, sz := utf8.DecodeRuneInString(text[i:])
			if unicode.IsSpace(r) {
				i += sz
				continue
			}
			if !isWordRune(r) {
				if id, ok := vocabGet(t.table, text[i:i+sz]); ok {
					dst = append(dst, id)
				}
				i += sz
				continue
			}
		}

		w := buf[2 : 2+maxWordBytes+4]
		wl := 0
		overflow := false
		for i < n {
			c = text[i]
			if c < 0x80 {
				if !asciiWordByte(c) {
					break
				}
				if c >= 'A' && c <= 'Z' {
					c += 'a' - 'A'
				}
				if wl < maxWordBytes {
					w[wl] = c
					wl++
				} else {
					overflow = true
				}
				i++
				continue
			}
			r, sz := utf8.DecodeRuneInString(text[i:])
			if !isWordRune(r) {
				break
			}
			r = unicode.ToLower(r)
			if wl+utf8.RuneLen(r) <= maxWordBytes {
				wl += utf8.EncodeRune(w[wl:], r)
			} else {
				overflow = true
			}
			i += sz
		}
		if overflow || wl == 0 {
			continue
		}
		if id, ok := vocabGet(t.table, buf[2:2+wl]); ok {
			dst = append(dst, id)
			continue
		}
		np := 0
		start := 0
		for start < wl {
			lo := 2 + start
			if start > 0 {
				buf[lo-2], buf[lo-1] = '#', '#'
				lo -= 2
			}
			end := wl
			matched := false
			for end > start {
				if id, hit := vocabGet(t.table, buf[lo:2+end]); hit {
					pieces[np] = id
					np++
					matched = true
					break
				}
				end--
				for end > start && !utf8.RuneStart(buf[2+end]) {
					end--
				}
			}
			if !matched {
				np = -1
				break
			}
			start = end
		}
		if np > 0 {
			dst = append(dst, pieces[:np]...)
		}
	}
	if len(dst) > limit {
		dst = dst[:limit]
	} else if len(dst) < limit && t.addSpecial && t.sepID >= 0 {
		dst = append(dst, t.sepID)
	}
	return dst
}
