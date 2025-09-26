package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/lee101/gobed/bed/src"
)

func main() {
	fmt.Println("🛏️  Bed Semantic Search Demo")
	fmt.Println("============================")
	
	// Create a temporary directory with sample code files
	tempDir, err := createSampleCodebase()
	if err != nil {
		log.Fatalf("Failed to create sample codebase: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	fmt.Printf("Created sample codebase in: %s\n\n", tempDir)
	
	// Change to the sample directory
	originalDir, _ := os.Getwd()
	os.Chdir(tempDir)
	defer os.Chdir(originalDir)
	
	// Demo 1: Index the sample codebase
	fmt.Println("🔍 Demo 1: Building semantic index...")
	if err := demoIndexing(); err != nil {
		log.Printf("Indexing demo failed: %v", err)
	}
	
	fmt.Println("\n" + "="*50)
	
	// Demo 2: Perform semantic searches
	fmt.Println("🧠 Demo 2: Semantic search examples...")
	searchQueries := []string{
		"function for making HTTP requests",
		"error handling and logging",
		"database connection setup",
		"user authentication middleware",
		"file reading and writing",
	}
	
	for i, query := range searchQueries {
		fmt.Printf("\n--- Search %d: %s ---\n", i+1, query)
		if err := demoSearch(query); err != nil {
			log.Printf("Search demo failed: %v", err)
		}
	}
	
	fmt.Println("\n" + "="*50)
	
	// Demo 3: Show index status
	fmt.Println("📊 Demo 3: Index statistics...")
	if err := demoStatus(); err != nil {
		log.Printf("Status demo failed: %v", err)
	}
	
	fmt.Println("\n🎉 Demo completed! Try running 'bed' commands manually.")
}

func createSampleCodebase() (string, error) {
	tempDir, err := os.MkdirTemp("", "bed_demo_*")
	if err != nil {
		return "", err
	}
	
	// Sample files with different types of code
	sampleFiles := map[string]string{
		"main.go": `package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

func main() {
	// Initialize HTTP server
	setupServer()
	log.Println("Server starting on port 8080")
	http.ListenAndServe(":8080", nil)
}

func setupServer() {
	http.HandleFunc("/", handleHome)
	http.HandleFunc("/api/users", handleUsers)
	http.HandleFunc("/health", handleHealth)
}

func handleHome(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Welcome to the API server!")
}`,

		"auth.go": `package main

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// AuthMiddleware provides user authentication for HTTP requests
func AuthMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		token := extractToken(r)
		if token == "" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		
		user, err := validateToken(token)
		if err != nil {
			http.Error(w, "Invalid token", http.StatusForbidden)
			return
		}
		
		// Add user to context
		ctx := context.WithValue(r.Context(), "user", user)
		next.ServeHTTP(w, r.WithContext(ctx))
	}
}

func extractToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if strings.HasPrefix(auth, "Bearer ") {
		return auth[7:]
	}
	return ""
}

func validateToken(token string) (*User, error) {
	// Validate JWT token and return user
	if token == "valid_token" {
		return &User{ID: 1, Name: "John Doe"}, nil
	}
	return nil, errors.New("invalid token")
}`,

		"database.go": `package main

import (
	"database/sql"
	"fmt"
	"log"
	
	_ "github.com/lib/pq"
)

type Database struct {
	conn *sql.DB
}

// NewDatabase creates a new database connection
func NewDatabase(connectionString string) (*Database, error) {
	conn, err := sql.Open("postgres", connectionString)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}
	
	if err := conn.Ping(); err != nil {
		conn.Close()
		return nil, fmt.Errorf("database ping failed: %w", err)
	}
	
	log.Println("Database connection established successfully")
	return &Database{conn: conn}, nil
}

func (db *Database) Close() error {
	if db.conn != nil {
		return db.conn.Close()
	}
	return nil
}

func (db *Database) GetUser(id int) (*User, error) {
	var user User
	query := "SELECT id, name, email FROM users WHERE id = $1"
	
	row := db.conn.QueryRow(query, id)
	err := row.Scan(&user.ID, &user.Name, &user.Email)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("user not found")
		}
		return nil, fmt.Errorf("database query error: %w", err)
	}
	
	return &user, nil
}`,

		"utils.go": `package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

// FileReader provides utilities for reading files
type FileReader struct {
	path string
}

func NewFileReader(path string) *FileReader {
	return &FileReader{path: path}
}

// ReadAll reads the entire file content
func (fr *FileReader) ReadAll() ([]byte, error) {
	file, err := os.Open(fr.path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", fr.path, err)
	}
	defer file.Close()
	
	content, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read file content: %w", err)
	}
	
	log.Printf("Successfully read %d bytes from %s", len(content), fr.path)
	return content, nil
}

// WriteFile writes content to a file
func WriteFile(path string, content []byte) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", path, err)
	}
	defer file.Close()
	
	_, err = file.Write(content)
	if err != nil {
		return fmt.Errorf("failed to write file content: %w", err)
	}
	
	log.Printf("Successfully wrote %d bytes to %s", len(content), path)
	return nil
}

// HTTPClient provides utilities for making HTTP requests
type HTTPClient struct {
	baseURL string
	client  *http.Client
}

func NewHTTPClient(baseURL string) *HTTPClient {
	return &HTTPClient{
		baseURL: baseURL,
		client:  &http.Client{},
	}
}

// Get makes a GET request to the specified endpoint
func (c *HTTPClient) Get(endpoint string) (*http.Response, error) {
	url := c.baseURL + endpoint
	
	resp, err := c.client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("HTTP GET request failed for %s: %w", url, err)
	}
	
	log.Printf("HTTP GET %s returned status %d", url, resp.StatusCode)
	return resp, nil
}`,

		"models.go": `package main

import (
	"time"
)

// User represents a user in the system
type User struct {
	ID        int       \`json:"id" db:"id"\`
	Name      string    \`json:"name" db:"name"\`
	Email     string    \`json:"email" db:"email"\`
	CreatedAt time.Time \`json:"created_at" db:"created_at"\`
	UpdatedAt time.Time \`json:"updated_at" db:"updated_at"\`
}

// IsValid checks if the user data is valid
func (u *User) IsValid() bool {
	return u.ID > 0 && u.Name != "" && u.Email != ""
}

// Product represents a product in the inventory
type Product struct {
	ID          int     \`json:"id" db:"id"\`
	Name        string  \`json:"name" db:"name"\`
	Description string  \`json:"description" db:"description"\`
	Price       float64 \`json:"price" db:"price"\`
	Stock       int     \`json:"stock" db:"stock"\`
}

// InStock checks if the product is available
func (p *Product) InStock() bool {
	return p.Stock > 0
}`,

		".bedignore": `# Ignore build artifacts
*.exe
*.dll
*.so
*.dylib

# Ignore logs
*.log
logs/

# Ignore temporary files
*.tmp
*.temp
.DS_Store`,

		"README.md": `# Sample Go Project

This is a sample Go project that demonstrates various functionality:

- HTTP server with middleware
- Database connections and queries  
- File I/O operations
- Authentication and authorization
- RESTful API endpoints

## Features

### Authentication
The project includes JWT-based authentication middleware for securing API endpoints.

### Database Integration
PostgreSQL database integration with connection pooling and query builders.

### File Operations
Utilities for reading and writing files with proper error handling.

### HTTP Client
Reusable HTTP client for making external API requests.
`,
	}
	
	// Write sample files
	for filename, content := range sampleFiles {
		path := filepath.Join(tempDir, filename)
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return "", err
		}
	}
	
	return tempDir, nil
}

func demoIndexing() error {
	indexer, err := src.NewIndexer()
	if err != nil {
		return err
	}
	defer indexer.Close()
	
	options := src.DefaultIndexOptions()
	options.Verbose = true
	options.ShowProgress = true
	options.UseGPU = false // Use CPU for demo stability
	
	return indexer.Index(options)
}

func demoSearch(query string) error {
	searcher, err := src.NewSearcher()
	if err != nil {
		return err
	}
	defer searcher.Close()
	
	options := src.SearchOptions{
		Query:     query,
		Limit:     3,
		Context:   1,
		Threshold: 0.5, // Lower threshold for demo
		ColorMode: "auto",
		Verbose:   false,
	}
	
	return searcher.Search(options)
}

func demoStatus() error {
	status, err := src.GetIndexStatus()
	if err != nil {
		return err
	}
	
	return status.Display()
}