package main

import (
	"fmt"
	"log"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🧪 COMPREHENSIVE STRESS TEST FOR TEXT EMBEDDING")
	fmt.Println(strings.Repeat("=", 80))

	// Load model
	fmt.Println("Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("❌ Error loading model: %v", err)
	}
	fmt.Println("✅ Model loaded successfully\n")

	// Run all stress tests
	runBasicStressTests(model)
	runUnicodeStressTests(model)
	runEdgeCaseTests(model)
	runSpecialCharacterTests(model)
	runLengthStressTests(model)
	runEncodingTests(model)
	runPerformanceStressTest(model)

	fmt.Println("\n🎉 All stress tests completed!")
}

func runBasicStressTests(model *gobed.EmbeddingModel) {
	fmt.Println("📋 BASIC STRESS TESTS")
	fmt.Println(strings.Repeat("-", 50))

	basicTests := []struct {
		name string
		text string
	}{
		{"Empty string", ""},
		{"Single character", "a"},
		{"Single space", " "},
		{"Multiple spaces", "   "},
		{"Tab character", "\t"},
		{"Newline", "\n"},
		{"Carriage return", "\r"},
		{"Mixed whitespace", " \t\n\r "},
		{"Single word", "hello"},
		{"Numbers only", "12345"},
		{"Special chars", "!@#$%^&*()"},
		{"Mixed case", "HeLLo WoRLd"},
	}

	for _, test := range basicTests {
		fmt.Printf("Testing: %-20s ", test.name)

		start := time.Now()
		emb, err := model.Encode(test.text)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("❌ ERROR: %v\n", err)
		} else {
			fmt.Printf("✅ %d dims, %v\n", len(emb), elapsed)
		}
	}
	fmt.Println()
}

func runUnicodeStressTests(model *gobed.EmbeddingModel) {
	fmt.Println("🌍 UNICODE STRESS TESTS")
	fmt.Println(strings.Repeat("-", 50))

	unicodeTests := []struct {
		name string
		text string
	}{
		{"Chinese", "你好世界"},
		{"Japanese", "こんにちは世界"},
		{"Korean", "안녕하세요 세계"},
		{"Arabic", "مرحبا بالعالم"},
		{"Russian", "Привет мир"},
		{"Greek", "Γεια σου κόσμε"},
		{"Hebrew", "שלום עולם"},
		{"Hindi", "नमस्ते दुनिया"},
		{"Thai", "สวัสดีชาวโลก"},
		{"Emoji only", "🌍🔥💯🚀⭐"},
		{"Mixed emoji", "Hello 🌍 World 🚀"},
		{"Mathematical symbols", "∑∞∂∇∫∮∯∰"},
		{"Currency symbols", "¥€$£₹₽¢"},
		{"Arrows", "←→↑↓↔↕⇄⇅"},
		{"Diacritics", "café naïve résumé"},
		{"Nordic chars", "høj kött ångström"},
		{"Turkish", "İstanbul Türkçe çğışöü"},
		{"Vietnamese", "Xin chào thế giới"},
		{"Polish", "Witaj świecie"},
		{"Czech", "Ahoj světe"},
	}

	for _, test := range unicodeTests {
		fmt.Printf("Testing: %-20s ", test.name)

		// Check if text is valid UTF-8
		if !utf8.ValidString(test.text) {
			fmt.Printf("❌ INVALID UTF-8\n")
			continue
		}

		start := time.Now()
		emb, err := model.Encode(test.text)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("❌ ERROR: %v\n", err)
		} else {
			fmt.Printf("✅ %d dims, %v\n", len(emb), elapsed)
		}
	}
	fmt.Println()
}

func runEdgeCaseTests(model *gobed.EmbeddingModel) {
	fmt.Println("⚠️  EDGE CASE TESTS")
	fmt.Println(strings.Repeat("-", 50))

	edgeCases := []struct {
		name string
		text string
	}{
		{"Null bytes", "hello\x00world"},
		{"Control chars", "hello\x01\x02\x03world"},
		{"Very long word", strings.Repeat("a", 1000)},
		{"Repeated chars", strings.Repeat("🔥", 100)},
		{"Only punctuation", "!@#$%^&*()_+-=[]{}|;':\",./<>?"},
		{"Only numbers", "0123456789"},
		{"Scientific notation", "1.23e-45 6.78E+90"},
		{"URLs", "https://example.com/path?query=value#fragment"},
		{"Email", "user@domain.com"},
		{"HTML entities", "&lt;&gt;&amp;&quot;&#39;"},
		{"XML/HTML tags", "<tag>content</tag>"},
		{"JSON", `{"key": "value", "number": 123}`},
		{"Code snippet", `func main() { fmt.Println("Hello") }`},
		{"SQL injection", "'; DROP TABLE users; --"},
		{"XSS attempt", "<script>alert('xss')</script>"},
		{"Base64", "SGVsbG8gV29ybGQ="},
		{"Hex", "48656c6c6f20576f726c64"},
		{"Binary", "01001000 01100101 01101100 01101100 01101111"},
	}

	for _, test := range edgeCases {
		fmt.Printf("Testing: %-20s ", test.name)

		start := time.Now()
		emb, err := model.Encode(test.text)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("❌ ERROR: %v\n", err)
		} else {
			fmt.Printf("✅ %d dims, %v\n", len(emb), elapsed)
		}
	}
	fmt.Println()
}

func runSpecialCharacterTests(model *gobed.EmbeddingModel) {
	fmt.Println("🔤 SPECIAL CHARACTER TESTS")
	fmt.Println(strings.Repeat("-", 50))

	specialTests := []struct {
		name string
		text string
	}{
		{"Zero-width space", "hello\u200bworld"},
		{"Soft hyphen", "hello\u00adworld"},
		{"Non-breaking space", "hello\u00a0world"},
		{"En dash", "hello – world"},
		{"Em dash", "hello — world"},
		{"Ellipsis", "hello…world"},
		{"Smart quotes", "\u201chello world\u201d"},
		{"Apostrophe", "\u2019hello world\u2019"},
		{"Combining chars", "a\u0301e\u0301i\u0301o\u0301u\u0301"}, // áéíóú
		{"Surrogate pairs", "𝕳𝖊𝖑𝖑𝖔 𝖂𝖔𝖗𝖑𝖉"},
		{"RTL override", "\u202ehello world\u202c"},
		{"LTR override", "\u202dhello world\u202c"},
		{"Variation selectors", "♠\ufe0f♥\ufe0f♦\ufe0f♣\ufe0f"},
		{"Zalgo text", "h̴̗̀e̸̘̾l̷̰̇l̸̰̄o̷̬̊ ̶̱̈w̴̗̌o̵̭̊r̷̜̈́l̷̰̇d̸̗̾"},
	}

	for _, test := range specialTests {
		fmt.Printf("Testing: %-20s ", test.name)

		start := time.Now()
		emb, err := model.Encode(test.text)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("❌ ERROR: %v\n", err)
		} else {
			fmt.Printf("✅ %d dims, %v\n", len(emb), elapsed)
		}
	}
	fmt.Println()
}

func runLengthStressTests(model *gobed.EmbeddingModel) {
	fmt.Println("📏 LENGTH STRESS TESTS")
	fmt.Println(strings.Repeat("-", 50))

	lengths := []int{1, 10, 100, 500, 1000, 2000, 5000, 10000}

	for _, length := range lengths {
		text := strings.Repeat("Hello world! ", length/13+1)[:length]

		fmt.Printf("Testing: %-10s ", fmt.Sprintf("%d chars", length))

		start := time.Now()
		emb, err := model.Encode(text)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("❌ ERROR: %v\n", err)
		} else {
			fmt.Printf("✅ %d dims, %v\n", len(emb), elapsed)
		}
	}
	fmt.Println()
}

func runEncodingTests(model *gobed.EmbeddingModel) {
	fmt.Println("🔄 ENCODING CONSISTENCY TESTS")
	fmt.Println(strings.Repeat("-", 50))

	// Test same text in different normalizations
	testCases := []struct {
		name  string
		texts []string
	}{
		{
			"Unicode normalization",
			[]string{
				"café",       // NFC
				"cafe\u0301", // NFD (e + combining acute)
			},
		},
		{
			"Case variations",
			[]string{
				"Hello World",
				"hello world",
				"HELLO WORLD",
				"HeLLo WoRLd",
			},
		},
		{
			"Whitespace variations",
			[]string{
				"hello world",
				"hello  world",
				"hello\tworld",
				"hello\nworld",
				" hello world ",
			},
		},
	}

	for _, testCase := range testCases {
		fmt.Printf("Testing: %s\n", testCase.name)

		var embeddings [][]float32
		var errors []error

		for i, text := range testCase.texts {
			emb, err := model.Encode(text)
			embeddings = append(embeddings, emb)
			errors = append(errors, err)

			if err != nil {
				fmt.Printf("  %d. ❌ ERROR: %v\n", i+1, err)
			} else {
				fmt.Printf("  %d. ✅ %d dims\n", i+1, len(emb))
			}
		}

		// Calculate similarities between variations
		if len(embeddings) >= 2 && errors[0] == nil && errors[1] == nil {
			sim := gobed.CosineSimilarity(embeddings[0], embeddings[1])
			fmt.Printf("  → Similarity between first two: %.6f\n", sim)
		}

		fmt.Println()
	}
}

func runPerformanceStressTest(model *gobed.EmbeddingModel) {
	fmt.Println("⚡ PERFORMANCE STRESS TEST")
	fmt.Println(strings.Repeat("-", 50))

	// Test rapid-fire encoding
	testText := "This is a performance stress test."
	iterations := 1000

	fmt.Printf("Rapid-fire encoding: %d iterations\n", iterations)

	start := time.Now()
	successCount := 0
	errorCount := 0

	for i := 0; i < iterations; i++ {
		_, err := model.Encode(testText)
		if err != nil {
			errorCount++
		} else {
			successCount++
		}
	}

	elapsed := time.Since(start)

	fmt.Printf("Results:\n")
	fmt.Printf("  ✅ Successful: %d\n", successCount)
	fmt.Printf("  ❌ Errors: %d\n", errorCount)
	fmt.Printf("  ⏱️  Total time: %v\n", elapsed)
	fmt.Printf("  📊 Avg per encoding: %v\n", elapsed/time.Duration(iterations))
	fmt.Printf("  🚀 Encodings/sec: %.0f\n", float64(successCount)/elapsed.Seconds())

	fmt.Println()
}
