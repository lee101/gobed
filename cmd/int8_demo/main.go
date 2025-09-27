package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/daulet/tokenizers"
	"github.com/leepenkman/gobed"
)

// PredefinedExample represents a text example with its tokenization
type PredefinedExample struct {
	Text     string
	TokenIDs []int
}

var predefinedExamples = []PredefinedExample{
	{"The quick brown fox jumps over the lazy dog", nil},
	{"A fast auburn canine leaps above the idle hound", nil},
	{"I love programming in Go", nil},
	{"Coding in Golang is my passion", nil},
	{"The weather is beautiful today", nil},
	{"It's raining heavily outside", nil},
	{"Machine learning is fascinating", nil},
	{"Artificial intelligence is interesting", nil},
	{"The cat sat on the mat", nil},
	{"A feline rested on the rug", nil},
}

func main() {
	fmt.Println("===========================================")
	fmt.Println("      Gobed INT8 Similarity Demo")
	fmt.Println("===========================================")
	fmt.Println()
	fmt.Println("This demo computes semantic similarity between two strings")
	fmt.Println("using INT8 quantized embeddings with AVX-512 acceleration.")
	fmt.Println()

	fmt.Println("Loading INT8 model...")
	start := time.Now()
	model, err := gobed.LoadModelInt8(true)
	if err != nil {
		fmt.Printf(" Error loading model: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf(" Model loaded in %v\n", time.Since(start))

	// Try to load tokenizer
	var tk *tokenizers.Tokenizer
	tokenizerPath := "model/tokenizer.json"
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		tokenizerPath = "../../model/tokenizer.json"
		if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
			tokenizerPath = "./model/tokenizer.json"
		}
	}

	tk, err = tokenizers.FromFile(tokenizerPath)
	if err != nil {
		fmt.Printf("  Could not load tokenizer, will use predefined examples only\n")
	} else {
		fmt.Printf(" Tokenizer loaded from %s\n", tokenizerPath)
	}
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)

	// Interactive loop
	for {
		fmt.Println("-------------------------------------------")
		fmt.Println("Choose input method:")
		fmt.Println("1. Enter custom text (requires tokenizer)")
		fmt.Println("2. Use predefined examples")
		fmt.Println("3. Load from reference tokens JSON")
		fmt.Println("4. Quit")
		fmt.Print("\nSelect option (1-4): ")

		option, _ := reader.ReadString('\n')
		option = strings.TrimSpace(option)

		switch option {
		case "1":
			if tk == nil {
				fmt.Println(" Tokenizer not available. Please use predefined examples.")
				continue
			}
			handleCustomText(reader, model, tk)

		case "2":
			handlePredefinedExamples(reader, model, tk)

		case "3":
			handleReferenceTokens(reader, model)

		case "4", "quit", "exit":
			fmt.Println("Goodbye!")
			return

		default:
			fmt.Println("Invalid option. Please select 1-4.")
		}
	}
}

func handleCustomText(reader *bufio.Reader, model *gobed.EmbeddingModelInt8, tk *tokenizers.Tokenizer) {
	fmt.Print("\nEnter first text: ")
	text1, _ := reader.ReadString('\n')
	text1 = strings.TrimSpace(text1)

	fmt.Print("Enter second text: ")
	text2, _ := reader.ReadString('\n')
	text2 = strings.TrimSpace(text2)

	if text1 == "" || text2 == "" {
		fmt.Println("  Please enter non-empty texts")
		return
	}

	// Tokenize
	tokens1, err := tk.Encode(text1, false)
	if err != nil {
		fmt.Printf(" Error tokenizing text 1: %v\n", err)
		return
	}

	tokens2, err := tk.Encode(text2, false)
	if err != nil {
		fmt.Printf(" Error tokenizing text 2: %v\n", err)
		return
	}

	// Convert to int slice
	tokenIDs1 := make([]int, len(tokens1))
	tokenIDs2 := make([]int, len(tokens2))
	for i, t := range tokens1 {
		tokenIDs1[i] = int(t)
	}
	for i, t := range tokens2 {
		tokenIDs2[i] = int(t)
	}

	computeAndDisplaySimilarity(model, text1, text2, tokenIDs1, tokenIDs2)
}

func handlePredefinedExamples(reader *bufio.Reader, model *gobed.EmbeddingModelInt8, tk *tokenizers.Tokenizer) {
	// Tokenize examples if not already done
	if tk != nil && predefinedExamples[0].TokenIDs == nil {
		fmt.Println("Tokenizing predefined examples...")
		for i := range predefinedExamples {
			tokens, err := tk.Encode(predefinedExamples[i].Text, false)
			if err == nil {
				tokenIDs := make([]int, len(tokens))
				for j, t := range tokens {
					tokenIDs[j] = int(t)
				}
				predefinedExamples[i].TokenIDs = tokenIDs
			}
		}
	}

	fmt.Println("\nPredefined examples:")
	for i, ex := range predefinedExamples {
		fmt.Printf("%2d. \"%s\"\n", i+1, ex.Text)
	}

	fmt.Print("\nSelect first text (1-10): ")
	idx1Str, _ := reader.ReadString('\n')
	idx1 := 0
	fmt.Sscanf(strings.TrimSpace(idx1Str), "%d", &idx1)

	fmt.Print("Select second text (1-10): ")
	idx2Str, _ := reader.ReadString('\n')
	idx2 := 0
	fmt.Sscanf(strings.TrimSpace(idx2Str), "%d", &idx2)

	if idx1 < 1 || idx1 > len(predefinedExamples) || idx2 < 1 || idx2 > len(predefinedExamples) {
		fmt.Println(" Invalid selection")
		return
	}

	ex1 := predefinedExamples[idx1-1]
	ex2 := predefinedExamples[idx2-1]

	if ex1.TokenIDs == nil || ex2.TokenIDs == nil {
		fmt.Println(" Examples not tokenized. Tokenizer required.")
		return
	}

	computeAndDisplaySimilarity(model, ex1.Text, ex2.Text, ex1.TokenIDs, ex2.TokenIDs)
}

func handleReferenceTokens(reader *bufio.Reader, model *gobed.EmbeddingModelInt8) {
	// Load reference tokens JSON
	tokensPath := "model/real_reference_tokens.json"
	if _, err := os.Stat(tokensPath); os.IsNotExist(err) {
		tokensPath = "../../model/real_reference_tokens.json"
		if _, err := os.Stat(tokensPath); os.IsNotExist(err) {
			tokensPath = "./model/real_reference_tokens.json"
		}
	}

	file, err := os.ReadFile(tokensPath)
	if err != nil {
		fmt.Printf(" Could not load reference tokens: %v\n", err)
		return
	}

	var refTokens map[string]struct {
		TokenIDs []int `json:"token_ids"`
	}
	if err := json.Unmarshal(file, &refTokens); err != nil {
		fmt.Printf(" Error parsing reference tokens: %v\n", err)
		return
	}

	// Show available texts
	texts := make([]string, 0, len(refTokens))
	for text := range refTokens {
		texts = append(texts, text)
	}

	fmt.Printf("\n%d texts available in reference tokens.\n", len(texts))
	fmt.Println("Showing first 20:")
	for i := 0; i < 20 && i < len(texts); i++ {
		fmt.Printf("%2d. \"%s\"\n", i+1, texts[i])
	}

	fmt.Print("\nSelect first text (1-20): ")
	idx1Str, _ := reader.ReadString('\n')
	idx1 := 0
	fmt.Sscanf(strings.TrimSpace(idx1Str), "%d", &idx1)

	fmt.Print("Select second text (1-20): ")
	idx2Str, _ := reader.ReadString('\n')
	idx2 := 0
	fmt.Sscanf(strings.TrimSpace(idx2Str), "%d", &idx2)

	if idx1 < 1 || idx1 > len(texts) || idx2 < 1 || idx2 > len(texts) {
		fmt.Println(" Invalid selection")
		return
	}

	text1 := texts[idx1-1]
	text2 := texts[idx2-1]
	tokens1 := refTokens[text1].TokenIDs
	tokens2 := refTokens[text2].TokenIDs

	computeAndDisplaySimilarity(model, text1, text2, tokens1, tokens2)
}

func computeAndDisplaySimilarity(model *gobed.EmbeddingModelInt8, text1, text2 string, tokens1, tokens2 []int) {
	fmt.Println("\n🔄 Processing...")
	fmt.Printf(" Text 1: %d tokens\n", len(tokens1))
	fmt.Printf(" Text 2: %d tokens\n", len(tokens2))

	// Compute embeddings
	startEmbed := time.Now()

	embed1, err := model.ComputeEmbeddingFromTokens(tokens1)
	if err != nil {
		fmt.Printf(" Error computing embedding 1: %v\n", err)
		return
	}

	embed2, err := model.ComputeEmbeddingFromTokens(tokens2)
	if err != nil {
		fmt.Printf(" Error computing embedding 2: %v\n", err)
		return
	}

	embedTime := time.Since(startEmbed)

	// Compute similarity
	startSim := time.Now()
	similarity := gobed.CosineSimilarityInt8(embed1, embed2)
	simTime := time.Since(startSim)

	// Also compute with fallback for comparison
	similarityFallback := gobed.CosineSimilarityInt8Fallback(embed1, embed2)

	// Display results
	fmt.Println("\n Results:")
	fmt.Println("═══════════════════════════════════════════")
	fmt.Printf("Text 1: \"%s\"\n", text1)
	fmt.Printf("Text 2: \"%s\"\n", text2)
	fmt.Printf("\n Cosine Similarity (AVX-512): %.6f\n", similarity)
	fmt.Printf(" Cosine Similarity (Fallback): %.6f\n", similarityFallback)
	fmt.Printf("   Difference: %.6f\n", similarity-similarityFallback)

	// Interpret similarity
	var interpretation string
	if similarity > 0.95 {
		interpretation = "Nearly identical meaning"
	} else if similarity > 0.8 {
		interpretation = "Very similar"
	} else if similarity > 0.6 {
		interpretation = "Moderately similar"
	} else if similarity > 0.4 {
		interpretation = "Somewhat related"
	} else if similarity > 0.2 {
		interpretation = "Weakly related"
	} else {
		interpretation = "Unrelated or opposite"
	}
	fmt.Printf("\n Interpretation: %s\n", interpretation)

	fmt.Printf("\n  Performance:\n")
	fmt.Printf("   • Embedding computation: %v\n", embedTime)
	fmt.Printf("   • Similarity computation: %v\n", simTime)
	fmt.Printf("   • Total: %v\n", embedTime+simTime)
	fmt.Println()
}
