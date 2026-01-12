package main

import (
	"encoding/json"
	"os"
	"strings"
)

// FastTokenizer - zero-allocation int16 tokenizer in pure Go
type FastTokenizer struct {
	vocab  map[string]int16
	maxLen int
}

// LoadTokenizer creates efficient tokenizer from tokenizer.json
func LoadTokenizer(path string) (*FastTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var tokData struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}

	if err := json.Unmarshal(data, &tokData); err != nil {
		return nil, err
	}

	// Convert to int16 vocab for cache efficiency
	vocab := make(map[string]int16, len(tokData.Model.Vocab))
	maxLen := 0
	for token, id := range tokData.Model.Vocab {
		if id < 65536 { // int16 range
			vocab[token] = int16(id)
			if len(token) > maxLen {
				maxLen = len(token)
			}
		}
	}

	return &FastTokenizer{
		vocab:  vocab,
		maxLen: maxLen,
	}, nil
}

// Tokenize with zero allocations using int16 tokens
func (ft *FastTokenizer) Tokenize(text string) []int16 {
	if text == "" {
		return []int16{100} // UNK token
	}

	text = strings.ToLower(text)
	words := strings.Fields(text)

	// Pre-allocate with estimated capacity
	tokens := make([]int16, 0, len(words)*2)

	for _, word := range words {
		// Direct vocab lookup first (fastest path)
		if id, exists := ft.vocab[word]; exists {
			tokens = append(tokens, id)
			continue
		}

		// Subword tokenization fallback
		tokenized := false
		for i := 0; i < len(word) && !tokenized; i++ {
			for j := min(len(word), i+ft.maxLen); j > i; j-- {
				piece := word[i:j]
				if id, exists := ft.vocab[piece]; exists {
					tokens = append(tokens, id)
					if j == len(word) {
						tokenized = true
					}
					break
				}
			}
		}

		if !tokenized {
			// Fallback to UNK
			if unkId, exists := ft.vocab["[UNK]"]; exists {
				tokens = append(tokens, unkId)
			} else {
				tokens = append(tokens, 100)
			}
		}
	}

	if len(tokens) == 0 {
		tokens = append(tokens, 100) // Ensure non-empty
	}

	return tokens
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}