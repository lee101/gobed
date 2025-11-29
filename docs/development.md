# Development Notes

## Debug Builds

The release `bed` binary now runs silently and auto-detects GPU support. To re-enable detailed startup logging while you iterate locally, build a debug binary or enable debug logging at runtime.

- Build with the `debug` tag to opt into verbose logs:

```bash
go build -tags "debug cuda gpu cagra" -o bed-debug ./cmd/bed
```

- Alternatively, toggle logging for a single run by exporting `GOBED_DEBUG=1` or passing `--verbose` to the CLI:

```bash
GOBED_DEBUG=1 go run -tags "cuda gpu cagra" ./cmd/bed "search query"
# or
go run -tags "cuda gpu cagra" ./cmd/bed --verbose "search query"
```

Both approaches surface the same diagnostic output used during development without affecting the default production binary.

## GPU Cache & Incremental Indexing

The GPU-backed `bed` flow now persists int8 embeddings per file under the configured cache directory (defaults to `~/.cache/bed/cagra/<project-hash>`). A first search will build that cache; subsequent runs reuse unchanged file embeddings, so only modified files are re-embedded. You can force a rebuild with `--force-index`, or skip touching the filesystem entirely with `--no-index` to query the cached CAGRA index directly.
