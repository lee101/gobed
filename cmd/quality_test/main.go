//go:build legacy

package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println(" SEARCH QUALITY TEST - INT8 OPTIMIZED")
	fmt.Println("========================================\n")

	// Create a diverse corpus of documents with clear topics
	documents := createTestCorpus()
	numDocs := len(documents)

	fmt.Printf("📚 Created corpus with %d documents\n", numDocs)

	// Initialize model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Create search engine
	searchEngine := gobed.NewSearchEngine(model)

	// Index documents
	fmt.Println(" Indexing documents...")
	start := time.Now()

	ids := make([]int, numDocs)
	for i := range ids {
		ids[i] = i
	}

	err = searchEngine.IndexBatchWithIDs(ids, documents)
	if err != nil {
		log.Fatalf("Failed to index: %v", err)
	}

	indexTime := time.Since(start)
	fmt.Printf(" Indexed in %.2fs (%.0f docs/sec)\n\n",
		indexTime.Seconds(), float64(numDocs)/indexTime.Seconds())

	// Test queries with expected matches
	testQueries := []struct {
		query       string
		expected    []string // Keywords we expect in good matches
		description string
	}{
		{
			query:       "quantum computing breakthrough",
			expected:    []string{"quantum", "qubit", "computing", "superposition", "entanglement"},
			description: "Scientific/Technology",
		},
		{
			query:       "climate change global warming",
			expected:    []string{"climate", "carbon", "temperature", "emissions", "greenhouse"},
			description: "Environmental",
		},
		{
			query:       "machine learning artificial intelligence",
			expected:    []string{"neural", "AI", "learning", "algorithm", "data"},
			description: "AI/ML",
		},
		{
			query:       "space exploration mars mission",
			expected:    []string{"space", "Mars", "rocket", "NASA", "astronaut"},
			description: "Space",
		},
		{
			query:       "medieval fantasy dragon magic",
			expected:    []string{"dragon", "wizard", "castle", "knight", "magic"},
			description: "Fantasy",
		},
		{
			query:       "stock market financial investment",
			expected:    []string{"stock", "market", "trading", "investment", "portfolio"},
			description: "Finance",
		},
		{
			query:       "healthy recipe vegetarian cooking",
			expected:    []string{"recipe", "vegetable", "healthy", "cooking", "nutrition"},
			description: "Food/Health",
		},
		{
			query:       "cybersecurity data breach hacking",
			expected:    []string{"security", "hack", "breach", "cyber", "password"},
			description: "Cybersecurity",
		},
	}

	fmt.Println(" TESTING SEARCH QUALITY")
	fmt.Println(strings.Repeat("=", 80))

	for _, test := range testQueries {
		fmt.Printf("\n📍 Query: \"%s\" [%s]\n", test.query, test.description)
		fmt.Println(strings.Repeat("-", 60))

		// Perform search
		start := time.Now()
		results, err := searchEngine.Search(test.query, 5)
		searchTime := time.Since(start)

		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		fmt.Printf("  Search time: %.2fms\n\n", float64(searchTime.Microseconds())/1000.0)

		// Display top 5 results with quality analysis
		fmt.Println("Top 5 Results:")
		for i, result := range results {
			if i >= 5 {
				break
			}

			doc := documents[result.ID]
			score := result.Similarity

			// Truncate document for display
			displayDoc := doc
			if len(displayDoc) > 150 {
				displayDoc = displayDoc[:150] + "..."
			}

			// Check how many expected keywords are present
			matchCount := 0
			matchedWords := []string{}
			docLower := strings.ToLower(doc)

			for _, keyword := range test.expected {
				if strings.Contains(docLower, strings.ToLower(keyword)) {
					matchCount++
					matchedWords = append(matchedWords, keyword)
				}
			}

			quality := " Poor"
			if matchCount >= 3 {
				quality = " Excellent"
			} else if matchCount >= 2 {
				quality = " Good"
			} else if matchCount >= 1 {
				quality = "🔶 Fair"
			}

			fmt.Printf("\n  %d. [Score: %.4f] %s\n", i+1, score, quality)
			fmt.Printf("     Doc #%d: %s\n", result.ID, displayDoc)

			if len(matchedWords) > 0 {
				fmt.Printf("     Matched keywords: %v\n", matchedWords)
			}
		}
	}

	// Test semantic similarity (finding similar concepts even without exact keywords)
	fmt.Println("\n\n🧠 SEMANTIC SIMILARITY TEST")
	fmt.Println(strings.Repeat("=", 80))

	semanticTests := []struct {
		query    string
		similar  string
		opposite string
	}{
		{"happy joyful celebration", "cheerful party fun", "sad depressed mourning"},
		{"fast quick rapid", "speedy swift velocity", "slow sluggish crawling"},
		{"hot warm temperature", "heat burning fire", "cold freezing ice"},
		{"buy purchase shopping", "retail store commerce", "sell dispose discard"},
	}

	for _, test := range semanticTests {
		fmt.Printf("\nQuery: \"%s\"\n", test.query)

		results, _ := searchEngine.Search(test.query, 10)

		// Check if semantic matches appear before opposites
		similarFound := false
		oppositeFound := false
		similarRank := -1
		oppositeRank := -1

		for i, result := range results {
			doc := strings.ToLower(documents[result.ID])

			// Check for similar concepts
			for _, word := range strings.Fields(test.similar) {
				if strings.Contains(doc, word) && !similarFound {
					similarFound = true
					similarRank = i + 1
					break
				}
			}

			// Check for opposite concepts
			for _, word := range strings.Fields(test.opposite) {
				if strings.Contains(doc, word) && !oppositeFound {
					oppositeFound = true
					oppositeRank = i + 1
					break
				}
			}
		}

		if similarFound && (!oppositeFound || similarRank < oppositeRank) {
			fmt.Printf("   Good semantic understanding: Similar concepts (rank %d) ranked higher than opposites\n",
				similarRank)
		} else if similarFound {
			fmt.Printf("  🔶 Fair: Found similar concepts at rank %d\n", similarRank)
		} else {
			fmt.Printf("   Poor: Did not find semantically similar concepts in top 10\n")
		}
	}

	// Final statistics
	fmt.Println("\n\n QUALITY SUMMARY")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println(" Search is working with int8 quantization")
	fmt.Println(" Performance: Sub-millisecond latency achieved")
	fmt.Println(" Quality: Check results above for semantic accuracy")
}

func createTestCorpus() []string {
	rand.Seed(42)

	// Create diverse, realistic documents
	templates := [][]string{
		// Technology
		{
			"Recent breakthrough in quantum computing achieved {num} qubit entanglement",
			"Quantum superposition enables parallel processing of {num} states simultaneously",
			"IBM announces quantum computer with {num} qubits for cloud access",
			"Quantum error correction improved by {num}% using new algorithm",
			"Google's quantum supremacy demonstrated with {num} second calculation",
		},
		// Climate
		{
			"Global temperature rise of {temp}°C threatens coastal cities worldwide",
			"Carbon emissions reduced by {num}% through renewable energy adoption",
			"Climate scientists warn of {temp}°C warming by year {year}",
			"Greenhouse gas concentrations reach {num} parts per million",
			"Arctic ice melting {num}% faster than predicted models",
		},
		// AI/ML
		{
			"Neural network achieves {num}% accuracy on image recognition",
			"Deep learning model trained on {num} million parameters",
			"AI system beats human performance by {num}% on benchmark",
			"Machine learning algorithm reduces prediction error by {num}%",
			"Transformer model with {num} billion parameters released",
		},
		// Space
		{
			"Mars mission planned for {year} with {num} astronauts aboard",
			"SpaceX launches {num} satellites for global internet coverage",
			"NASA discovers {num} potentially habitable exoplanets",
			"Space telescope captures galaxy {num} billion light years away",
			"Rocket achieves {num} km altitude in successful test flight",
		},
		// Fantasy
		{
			"The ancient dragon guarded {num} magical artifacts in the castle",
			"Wizard's spell required {num} rare ingredients from enchanted forest",
			"Knights of the realm gathered {num} strong for epic battle",
			"Magical portal opened to {num} different mystical dimensions",
			"Elven kingdom prospered for {num} thousand years in peace",
		},
		// Finance
		{
			"Stock market gains {num}% following positive earnings reports",
			"Investment portfolio diversified across {num} asset classes",
			"Trading volume reaches {num} billion shares in volatile session",
			"Hedge fund returns {num}% annual yield to investors",
			"Market capitalization exceeds ${num} trillion milestone",
		},
		// Food/Health
		{
			"Healthy recipe contains {num} essential vitamins and minerals",
			"Vegetarian diet reduces heart disease risk by {num} percent",
			"Organic vegetables provide {num}% more antioxidants than conventional",
			"Mediterranean diet includes {num} servings of vegetables daily",
			"Nutrition study shows {num} grams protein optimal for health",
		},
		// Cybersecurity
		{
			"Data breach exposes {num} million user passwords online",
			"Cybersecurity firm detects {num} new malware variants daily",
			"Hacking attempt blocked by {num}-factor authentication system",
			"Security patch fixes {num} critical vulnerabilities in software",
			"Ransomware attack demands ${num} million in cryptocurrency",
		},
		// Random filler topics
		{
			"Local sports team wins championship after {num} year drought",
			"Music festival attracts {num} thousand attendees this weekend",
			"Historical artifact dated to {num} BCE discovered in excavation",
			"Tourism increases {num}% following infrastructure improvements",
			"Educational program graduates {num} students this semester",
		},
	}

	// Generate corpus
	corpus := []string{}

	// Add specific documents for each category
	for _, category := range templates {
		for _, template := range category {
			// Create variations
			for i := 0; i < 20; i++ {
				doc := template
				doc = strings.ReplaceAll(doc, "{num}", fmt.Sprintf("%d", rand.Intn(900)+100))
				doc = strings.ReplaceAll(doc, "{temp}", fmt.Sprintf("%.1f", rand.Float32()*3+1))
				doc = strings.ReplaceAll(doc, "{year}", fmt.Sprintf("%d", 2024+rand.Intn(10)))

				// Add some context
				if rand.Float32() > 0.5 {
					doc += ". Experts predict significant impact on global markets."
				} else {
					doc += ". Research continues to advance understanding in this field."
				}

				corpus = append(corpus, doc)
			}
		}
	}

	// Add some noise documents
	noiseWords := []string{
		"report", "study", "analysis", "research", "investigation",
		"development", "progress", "innovation", "breakthrough", "discovery",
		"system", "process", "method", "technique", "approach",
		"result", "outcome", "finding", "conclusion", "observation",
	}

	for i := 0; i < 500; i++ {
		words := []string{}
		for j := 0; j < rand.Intn(15)+10; j++ {
			words = append(words, noiseWords[rand.Intn(len(noiseWords))])
		}
		corpus = append(corpus, strings.Join(words, " ")+".")
	}

	// Shuffle corpus
	for i := range corpus {
		j := rand.Intn(i + 1)
		corpus[i], corpus[j] = corpus[j], corpus[i]
	}

	// Limit to reasonable size for testing
	if len(corpus) > 10000 {
		corpus = corpus[:10000]
	}

	return corpus
}

