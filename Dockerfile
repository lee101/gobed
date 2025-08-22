# Multi-stage build for gobed
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git gcc musl-dev

# Set working directory
WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary with optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w" \
    -o gobed \
    ./cmd/demo

# Build search server
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w" \
    -o gobed-server \
    ./cmd/search_server

# Final stage - minimal image
FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1000 gobed && \
    adduser -D -u 1000 -G gobed gobed

# Set working directory
WORKDIR /app

# Copy binaries from builder
COPY --from=builder /build/gobed /app/gobed
COPY --from=builder /build/gobed-server /app/gobed-server

# Copy model files if they exist
COPY --from=builder /build/model /app/model

# Change ownership
RUN chown -R gobed:gobed /app

# Switch to non-root user
USER gobed

# Expose port for search server
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/app/gobed"]

# Default command
CMD ["--help"]