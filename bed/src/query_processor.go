package src

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// QueryProcessor enhances natural language queries for better search results
type QueryProcessor struct {
	// Query expansion patterns
	synonyms     map[string][]string
	codePatterns map[string][]string
	stopWords    map[string]bool
}

// ProcessedQuery represents an enhanced query with metadata
type ProcessedQuery struct {
	Original      string
	Enhanced      string
	Terms         []string
	CodeTerms     []string
	Concepts      []string
	QueryType     QueryType
	Weights       map[string]float32
}

// QueryType represents the type of query
type QueryType int

const (
	QueryTypeGeneral QueryType = iota
	QueryTypeCode
	QueryTypeError
	QueryTypeFunction
	QueryTypeFile
	QueryTypeDocumentation
)

// NewQueryProcessor creates a new query processor
func NewQueryProcessor() *QueryProcessor {
	qp := &QueryProcessor{
		synonyms:     buildSynonyms(),
		codePatterns: buildCodePatterns(),
		stopWords:    buildStopWords(),
	}
	return qp
}

// Process enhances a natural language query
func (qp *QueryProcessor) Process(query string) *ProcessedQuery {
	processed := &ProcessedQuery{
		Original: query,
		Terms:    []string{},
		Weights:  make(map[string]float32),
	}
	
	// Detect query type
	processed.QueryType = qp.detectQueryType(query)
	
	// Tokenize and clean
	tokens := qp.tokenize(query)
	
	// Remove stop words for analysis (but keep for embedding)
	meaningful := qp.removeStopWords(tokens)
	
	// Expand with synonyms
	expanded := qp.expandSynonyms(meaningful)
	
	// Extract code-specific terms
	processed.CodeTerms = qp.extractCodeTerms(query)
	
	// Extract concepts
	processed.Concepts = qp.extractConcepts(query)
	
	// Build enhanced query
	processed.Enhanced = qp.buildEnhancedQuery(query, expanded, processed)
	
	// Calculate term weights
	qp.calculateWeights(processed)
	
	return processed
}

// detectQueryType determines the type of query
func (qp *QueryProcessor) detectQueryType(query string) QueryType {
	lower := strings.ToLower(query)
	
	// Check for specific patterns
	if strings.Contains(lower, "error") || strings.Contains(lower, "exception") || 
	   strings.Contains(lower, "bug") || strings.Contains(lower, "fix") {
		return QueryTypeError
	}
	
	if strings.Contains(lower, "function") || strings.Contains(lower, "method") ||
	   strings.Contains(lower, "def ") || strings.Contains(lower, "func ") {
		return QueryTypeFunction
	}
	
	if strings.Contains(lower, "file") || strings.Contains(lower, "module") ||
	   regexp.MustCompile(`\.\w+$`).MatchString(query) {
		return QueryTypeFile
	}
	
	if strings.Contains(lower, "comment") || strings.Contains(lower, "doc") ||
	   strings.Contains(lower, "readme") || strings.Contains(lower, "documentation") {
		return QueryTypeDocumentation
	}
	
	// Check for code patterns
	if qp.hasCodePattern(query) {
		return QueryTypeCode
	}
	
	return QueryTypeGeneral
}

// tokenize splits query into tokens
func (qp *QueryProcessor) tokenize(query string) []string {
	// Split on whitespace and punctuation, but preserve code-like tokens
	re := regexp.MustCompile(`[\w\-_\.]+|[^\w\s]+`)
	matches := re.FindAllString(query, -1)
	
	tokens := []string{}
	for _, match := range matches {
		if match != "" {
			tokens = append(tokens, strings.ToLower(match))
		}
	}
	
	return tokens
}

// removeStopWords filters out common stop words
func (qp *QueryProcessor) removeStopWords(tokens []string) []string {
	filtered := []string{}
	for _, token := range tokens {
		if !qp.stopWords[token] {
			filtered = append(filtered, token)
		}
	}
	return filtered
}

// expandSynonyms adds synonym variations
func (qp *QueryProcessor) expandSynonyms(tokens []string) []string {
	expanded := make([]string, 0, len(tokens)*2)
	seen := make(map[string]bool)
	
	for _, token := range tokens {
		if !seen[token] {
			expanded = append(expanded, token)
			seen[token] = true
		}
		
		if synonyms, ok := qp.synonyms[token]; ok {
			for _, syn := range synonyms {
				if !seen[syn] {
					expanded = append(expanded, syn)
					seen[syn] = true
				}
			}
		}
	}
	
	return expanded
}

// extractCodeTerms finds programming-related terms
func (qp *QueryProcessor) extractCodeTerms(query string) []string {
	terms := []string{}
	lower := strings.ToLower(query)
	
	// Common programming terms
	patterns := []string{
		`\bapi\b`, `\bsql\b`, `\bjson\b`, `\bxml\b`, `\bhttp\b`,
		`\basync\b`, `\bawait\b`, `\bpromise\b`, `\bcallback\b`,
		`\bclass\b`, `\bstruct\b`, `\binterface\b`, `\benum\b`,
		`\barray\b`, `\blist\b`, `\bmap\b`, `\bdict\b`, `\bhash\b`,
		`\bloop\b`, `\bfor\b`, `\bwhile\b`, `\bif\b`, `\belse\b`,
		`\btry\b`, `\bcatch\b`, `\bfinally\b`, `\bthrow\b`,
		`\bimport\b`, `\brequire\b`, `\binclude\b`, `\busing\b`,
		`\bpublic\b`, `\bprivate\b`, `\bprotected\b`, `\bstatic\b`,
		`\breturn\b`, `\byield\b`, `\bbreak\b`, `\bcontinue\b`,
	}
	
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		if matches := re.FindAllString(lower, -1); matches != nil {
			terms = append(terms, matches...)
		}
	}
	
	// Function/method calls (word followed by parentheses)
	funcPattern := regexp.MustCompile(`\b\w+\s*\(`)
	if matches := funcPattern.FindAllString(query, -1); matches != nil {
		for _, match := range matches {
			clean := strings.TrimSpace(strings.TrimSuffix(match, "("))
			terms = append(terms, clean)
		}
	}
	
	// Variable names (camelCase, snake_case)
	varPattern := regexp.MustCompile(`\b[a-z]+(?:[A-Z][a-z]+)+\b|\b[a-z]+(?:_[a-z]+)+\b`)
	if matches := varPattern.FindAllString(query, -1); matches != nil {
		terms = append(terms, matches...)
	}
	
	return terms
}

// extractConcepts identifies high-level concepts
func (qp *QueryProcessor) extractConcepts(query string) []string {
	concepts := []string{}
	lower := strings.ToLower(query)
	
	// Programming concepts
	conceptMap := map[string][]string{
		"authentication": []string{"auth", "login", "signin", "oauth", "jwt", "token"},
		"database":       []string{"db", "sql", "query", "table", "schema", "migration"},
		"testing":        []string{"test", "spec", "assert", "mock", "stub", "fixture"},
		"configuration":  []string{"config", "settings", "env", "environment", "options"},
		"networking":     []string{"http", "tcp", "udp", "socket", "request", "response"},
		"concurrency":    []string{"thread", "goroutine", "async", "parallel", "concurrent"},
		"security":       []string{"encrypt", "decrypt", "hash", "salt", "secure", "vulnerability"},
		"performance":    []string{"optimize", "fast", "slow", "benchmark", "profile", "cache"},
		"error":          []string{"error", "exception", "panic", "fault", "failure", "bug"},
		"logging":        []string{"log", "logger", "debug", "trace", "info", "warn"},
	}
	
	for concept, keywords := range conceptMap {
		for _, keyword := range keywords {
			if strings.Contains(lower, keyword) {
				concepts = append(concepts, concept)
				break
			}
		}
	}
	
	return concepts
}

// buildEnhancedQuery creates an enhanced version of the query
func (qp *QueryProcessor) buildEnhancedQuery(original string, expanded []string, processed *ProcessedQuery) string {
	parts := []string{original}
	
	// Add query type context
	switch processed.QueryType {
	case QueryTypeError:
		parts = append(parts, "error handling exception bug fix")
	case QueryTypeFunction:
		parts = append(parts, "function method implementation def")
	case QueryTypeFile:
		parts = append(parts, "file module import require")
	case QueryTypeDocumentation:
		parts = append(parts, "documentation comment readme docs")
	case QueryTypeCode:
		parts = append(parts, "code implementation source")
	}
	
	// Add expanded terms (limit to avoid too much noise)
	if len(expanded) > 0 && len(expanded) < 10 {
		parts = append(parts, strings.Join(expanded, " "))
	}
	
	// Add concepts
	if len(processed.Concepts) > 0 {
		parts = append(parts, strings.Join(processed.Concepts, " "))
	}
	
	return strings.Join(parts, " ")
}

// calculateWeights assigns importance weights to terms
func (qp *QueryProcessor) calculateWeights(processed *ProcessedQuery) {
	// Base weight for original terms
	originalTokens := qp.tokenize(processed.Original)
	for _, token := range originalTokens {
		processed.Weights[token] = 1.0
	}
	
	// Higher weight for code terms
	for _, term := range processed.CodeTerms {
		processed.Weights[term] = 1.5
	}
	
	// Medium weight for concepts
	for _, concept := range processed.Concepts {
		processed.Weights[concept] = 1.2
	}
	
	// Adjust based on query type
	switch processed.QueryType {
	case QueryTypeError:
		processed.Weights["error"] = 2.0
		processed.Weights["exception"] = 2.0
		processed.Weights["fix"] = 1.5
	case QueryTypeFunction:
		processed.Weights["function"] = 2.0
		processed.Weights["method"] = 2.0
		processed.Weights["def"] = 1.5
		processed.Weights["func"] = 1.5
	}
}

// hasCodePattern checks if query contains code-like patterns
func (qp *QueryProcessor) hasCodePattern(query string) bool {
	// Check for common code patterns
	patterns := []string{
		`\w+\(\)`,                    // function calls
		`\w+\.\w+`,                   // method calls or properties
		`\w+\[\w*\]`,                 // array access
		`\w+\s*=\s*\w+`,              // assignment
		`\w+\s*[<>!=]+\s*\w+`,        // comparison
		`{.*}`,                       // braces
		`\(\s*\w+\s*\)`,              // parentheses with content
	}
	
	for _, pattern := range patterns {
		if matched, _ := regexp.MatchString(pattern, query); matched {
			return true
		}
	}
	
	return false
}

// buildSynonyms creates the synonym dictionary
func buildSynonyms() map[string][]string {
	return map[string][]string{
		// Programming terms
		"function":  []string{"func", "method", "procedure", "routine", "fn"},
		"variable":  []string{"var", "param", "parameter", "arg", "argument"},
		"class":     []string{"struct", "type", "object", "entity"},
		"error":     []string{"exception", "err", "fault", "failure", "bug"},
		"create":    []string{"make", "new", "build", "construct", "init"},
		"delete":    []string{"remove", "destroy", "drop", "clear", "clean"},
		"update":    []string{"modify", "change", "edit", "alter", "patch"},
		"get":       []string{"fetch", "retrieve", "find", "load", "read"},
		"set":       []string{"assign", "store", "save", "write", "put"},
		"list":      []string{"array", "slice", "vector", "collection"},
		"map":       []string{"dict", "dictionary", "hash", "hashmap", "table"},
		"string":    []string{"str", "text", "char", "characters"},
		"number":    []string{"int", "integer", "float", "decimal", "numeric"},
		"boolean":   []string{"bool", "flag", "true", "false"},
		
		// Action terms
		"search":    []string{"find", "look", "query", "seek", "locate"},
		"sort":      []string{"order", "arrange", "organize"},
		"filter":    []string{"select", "where", "subset", "exclude"},
		"connect":   []string{"link", "join", "attach", "bind"},
		"start":     []string{"begin", "init", "launch", "run"},
		"stop":      []string{"end", "terminate", "halt", "close"},
		"send":      []string{"transmit", "emit", "dispatch", "post"},
		"receive":   []string{"recv", "accept", "get", "handle"},
		
		// Concept terms
		"fast":      []string{"quick", "rapid", "speed", "performance"},
		"slow":      []string{"sluggish", "lag", "delay", "bottleneck"},
		"big":       []string{"large", "huge", "massive", "enormous"},
		"small":     []string{"tiny", "little", "mini", "compact"},
		"async":     []string{"asynchronous", "concurrent", "parallel", "non-blocking"},
		"sync":      []string{"synchronous", "sequential", "blocking"},
	}
}

// buildCodePatterns creates code-specific patterns
func buildCodePatterns() map[string][]string {
	return map[string][]string{
		"loop": []string{
			"for", "while", "do while", "foreach", "for each",
			"iterate", "iteration", "loop through", "traverse",
		},
		"condition": []string{
			"if", "else", "else if", "elif", "switch", "case",
			"conditional", "branch", "ternary",
		},
		"declaration": []string{
			"var", "let", "const", "define", "declare",
			"int", "string", "bool", "float", "double",
		},
		"import": []string{
			"import", "require", "include", "using", "use",
			"from", "module", "package", "library",
		},
	}
}

// buildStopWords creates the stop words set
func buildStopWords() map[string]bool {
	words := []string{
		"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
		"of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
		"been", "being", "have", "has", "had", "do", "does", "did", "will",
		"would", "could", "should", "may", "might", "must", "can", "shall",
		"it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
		"we", "they", "them", "their", "what", "which", "who", "where", "when",
		"how", "why", "all", "some", "any", "no", "not", "yes", "very", "just",
		"only", "also", "too", "so", "than", "more", "most", "less", "least",
	}
	
	stopWords := make(map[string]bool)
	for _, word := range words {
		stopWords[word] = true
	}
	return stopWords
}

// ReRank re-ranks results based on query analysis
func (qp *QueryProcessor) ReRank(results []*SearchResult, processed *ProcessedQuery) []*SearchResult {
	// Score each result based on processed query
	for _, result := range results {
		// Base score from similarity
		score := result.Similarity
		
		// Boost if content contains important terms
		contentLower := strings.ToLower(result.Content)
		for term, weight := range processed.Weights {
			if strings.Contains(contentLower, term) {
				score *= weight
			}
		}
		
		// Boost based on query type match
		score *= qp.getQueryTypeBoost(result, processed.QueryType)
		
		// Boost if file path suggests relevance
		if qp.isPathRelevant(result.FilePath, processed) {
			score *= 1.3
		}
		
		// Update similarity with new score
		result.Similarity = score
	}
	
	// Re-sort by new scores
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	
	return results
}

// getQueryTypeBoost returns a boost factor based on query type
func (qp *QueryProcessor) getQueryTypeBoost(result *SearchResult, queryType QueryType) float32 {
	contentLower := strings.ToLower(result.Content)
	pathLower := strings.ToLower(result.FilePath)
	
	switch queryType {
	case QueryTypeError:
		if strings.Contains(contentLower, "error") || strings.Contains(contentLower, "exception") {
			return 1.5
		}
	case QueryTypeFunction:
		// Boost if looks like function definition
		if regexp.MustCompile(`^\s*(func|def|function|method)`).MatchString(contentLower) {
			return 1.5
		}
	case QueryTypeFile:
		// Boost if result is from relevant file type
		if strings.HasSuffix(pathLower, ".go") || strings.HasSuffix(pathLower, ".py") {
			return 1.2
		}
	case QueryTypeDocumentation:
		if strings.Contains(pathLower, "readme") || strings.Contains(pathLower, ".md") ||
		   strings.Contains(contentLower, "//") || strings.Contains(contentLower, "/*") {
			return 1.4
		}
	}
	
	return 1.0
}

// isPathRelevant checks if file path is relevant to query
func (qp *QueryProcessor) isPathRelevant(path string, processed *ProcessedQuery) bool {
	pathLower := strings.ToLower(path)
	
	// Check if path contains any important terms
	for term := range processed.Weights {
		if strings.Contains(pathLower, term) {
			return true
		}
	}
	
	// Check if path matches concepts
	for _, concept := range processed.Concepts {
		if strings.Contains(pathLower, concept) {
			return true
		}
	}
	
	return false
}

// GetSuggestions provides query suggestions based on partial input
func (qp *QueryProcessor) GetSuggestions(partial string) []string {
	suggestions := []string{}
	lower := strings.ToLower(partial)
	
	// Common search patterns
	patterns := []string{
		"function that",
		"how to",
		"where is",
		"find all",
		"show me",
		"code for",
		"implementation of",
		"error in",
		"bug in",
		"todo",
		"fixme",
	}
	
	for _, pattern := range patterns {
		if strings.HasPrefix(pattern, lower) {
			suggestions = append(suggestions, pattern)
		}
	}
	
	// Add file type suggestions
	if strings.HasPrefix("file", lower) || strings.HasPrefix("files", lower) {
		suggestions = append(suggestions, 
			"files with errors",
			"files modified today",
			"files containing TODO",
		)
	}
	
	return suggestions
}

// Format formats the processed query for display
func (processed *ProcessedQuery) Format() string {
	var sb strings.Builder
	
	sb.WriteString(fmt.Sprintf("Original: %s\n", processed.Original))
	sb.WriteString(fmt.Sprintf("Enhanced: %s\n", processed.Enhanced))
	sb.WriteString(fmt.Sprintf("Type: %s\n", queryTypeString(processed.QueryType)))
	
	if len(processed.CodeTerms) > 0 {
		sb.WriteString(fmt.Sprintf("Code terms: %s\n", strings.Join(processed.CodeTerms, ", ")))
	}
	
	if len(processed.Concepts) > 0 {
		sb.WriteString(fmt.Sprintf("Concepts: %s\n", strings.Join(processed.Concepts, ", ")))
	}
	
	return sb.String()
}

func queryTypeString(qt QueryType) string {
	switch qt {
	case QueryTypeCode:
		return "Code"
	case QueryTypeError:
		return "Error/Bug"
	case QueryTypeFunction:
		return "Function/Method"
	case QueryTypeFile:
		return "File/Module"
	case QueryTypeDocumentation:
		return "Documentation"
	default:
		return "General"
	}
}