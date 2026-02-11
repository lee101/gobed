# bed v1

Semantic filesystem search CLI powered by `github.com/lee101/gobed`.

## Install

```bash
# From root module command path
go install github.com/lee101/gobed/cmd/bed@v1.0.0

# Or from the bed module command path
go install github.com/lee101/gobed/bed/cmd/bed@v1.0.0
```

## Quick Start

```bash
# Semantic search current directory
bed "where is auth handled"

# Build/rebuild index
bed index .
bed reindex .

# Live index updates with inotify/fsnotify
bed index . --watch

# CPU benchmark
bed bench . --queries 200 --warmup 20

# Benchmark with quality metrics
bed bench . --ndcg --ndcg-k 10

# Compare CPU vs GPU (when CAGRA/cuVS is available)
bed bench . --compare-gpu --ndcg
```

## Default Indexing Behavior

- Respects `.gitignore`-style excludes.
- Ignores binaries by default.
- Ignores long lines by default (`ignore_long_lines=true`).
- Configurable min/max line length and binary behavior via `~/.bed/config.json`.

## Key Commands

- `bed "<query>"`: semantic search
- `bed index [path]`: build/update index
- `bed reindex [path]`: force full rebuild
- `bed index [path] --watch`: live updates
- `bed daemon [paths...]`: persistent daemon mode
- `bed bench [path]`: indexing/search benchmark
- `bed status`: system/index status
- `bed config`: print active config
