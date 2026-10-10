package gobed

import (
	"math/rand"
	"slices"
	"strings"
	"testing"
	"unicode"
)

func refTokenize(vocab map[string]int16, cls, sep int16, text string, limit int) []int16 {
	text = strings.ToLower(text)
	var words []string
	var sb strings.Builder
	flush := func() {
		if sb.Len() > 0 {
			words = append(words, sb.String())
			sb.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case isWordRune(r):
			sb.WriteRune(r)
		default:
			flush()
			words = append(words, string(r))
		}
	}
	flush()
	out := []int16{cls}
	for _, w := range words {
		if len(out) >= limit {
			break
		}
		if len(w) > maxWordBytes {
			continue
		}
		if id, ok := vocab[w]; ok {
			out = append(out, id)
			continue
		}
		var ids []int16
		chars := []rune(w)
		start := 0
		okw := true
		for start < len(chars) {
			end := len(chars)
			found := false
			for end > start {
				sub := string(chars[start:end])
				if start > 0 {
					sub = "##" + sub
				}
				if id, ok := vocab[sub]; ok {
					ids = append(ids, id)
					found = true
					break
				}
				end--
			}
			if !found {
				okw = false
				break
			}
			start = end
		}
		if okw {
			out = append(out, ids...)
		}
	}
	if len(out) > limit {
		out = out[:limit]
	} else if len(out) < limit {
		out = append(out, sep)
	}
	return out
}

func randText(r *rand.Rand) string {
	words := []string{"the", "Quick", "brown", "fox", "unaffable", "embedding", "xyzzyqq", "COVID-19", "it's", "snake_case", "3.14159", "tokenization", "zzzzzzzzzzzz", "a", "I", "naïve", "café", "Ünïcode", "日本語", "😀", " ", "ǅ", "ﬁ", "\xff"}
	var sb strings.Builder
	n := r.Intn(80)
	for i := 0; i < n; i++ {
		switch r.Intn(6) {
		case 0:
			sb.WriteByte(byte(33 + r.Intn(94)))
		case 1:
			b := make([]byte, 1+r.Intn(12))
			for j := range b {
				b[j] = byte('a' + r.Intn(26))
			}
			sb.Write(b)
		default:
			sb.WriteString(words[r.Intn(len(words))])
		}
		sb.WriteString([]string{" ", "  ", "\n", "\t", ""}[r.Intn(5)])
	}
	return sb.String()
}

func TestFastTokenizerParity(t *testing.T) {
	m, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skip(err)
	}
	tk := m.tokenizer
	vocab, err := loadSimpleVocab("model/tokenizer.json")
	if err != nil {
		t.Skip(err)
	}
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 20000; i++ {
		s := randText(r)
		for _, lim := range []int{0, 256, 7} {
			l := lim
			if l == 0 {
				l = 1 << 30
			}
			want := refTokenize(vocab, tk.clsID, tk.sepID, s, l)
			got := tk.tokenizeInto(s, nil, lim)
			if !slices.Equal(want, got) {
				t.Fatalf("limit %d mismatch for %q\nwant %v\ngot  %v", lim, s, want, got)
			}
		}
	}
}

func TestTokenizeIntoNoAlloc(t *testing.T) {
	m, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skip(err)
	}
	buf := make([]int16, 0, 512)
	res := make([]float32, Int8EmbeddingDim)
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 4)
	if a := testing.AllocsPerRun(100, func() { m.EmbedTextInto(text, 256, res, buf) }); a != 0 {
		t.Fatalf("EmbedTextInto allocs = %v", a)
	}
	if a := testing.AllocsPerRun(100, func() { m.TokenizeInto("naïve café 日本語 zzzzqqq", buf[:0], 256) }); a != 0 {
		t.Fatalf("unicode TokenizeInto allocs = %v", a)
	}
}

func TestEmbedTextIntoMatchesEmbed(t *testing.T) {
	m, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skip(err)
	}
	text := "machine learning is transforming data science"
	want, _ := m.Embed(text)
	got := make([]float32, Int8EmbeddingDim)
	n := m.EmbedTextInto(text, 0, got, make([]int16, 0, 64))
	if n != len(m.SimpleTokenize(text)) {
		t.Fatalf("token count %d", n)
	}
	if !slices.Equal(want, got) {
		t.Fatal("embedding mismatch")
	}
}
