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

# Benchmark with curated relevance labels
bed bench . --ndcg --ndcg-k 10

# Use a production-like query workload
bed bench . --query-file queries.txt

# Compare CPU vs GPU and add exact ANN quality on the built GPU index
bed bench . --compare-gpu --exact --ndcg

# Emit JSON for automation
bed bench . --compare-gpu --json
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

## Benchmarking

`bed bench` now separates three distinct benchmark modes:

- Latency and throughput on the target corpus at `[path]`
- Curated relevance evaluation with `--ndcg` from a `docs/ + queries.json` dataset
- Exact ANN evaluation with `--exact` on the built GPU CAGRA index

Important details:

- `--query-file` lets you replace the built-in 10-query workload with a newline-delimited production query set.
- `--ndcg` evaluates the dataset in `--quality-dir` and reports `NDCG@K` and `Recall@K`.
- `--exact` evaluates GPU CAGRA against a brute-force top-k baseline on retained corpus vectors and reports `NDCG@K`, `Recall@K`, and top-1 self-hit rate.
- `--json` prints machine-readable output including index time, index throughput, p50/p95/p99 latency, QPS, and query source.

Example query file:

```text
authentication middleware
vector similarity search
cache invalidation path
gpu memory allocator
```

Curated quality datasets use this layout:

```text
quality-dir/
  docs/
    pkg/auth.go
    pkg/cache.go
  queries.json
```

Example `queries.json`:

```json
[
  {
    "query": "authentication flow",
    "relevance": {
      "pkg/auth.go": 3,
      "pkg/cache.go": 1
    }
  }
]
```

Use relative paths under `docs/` as relevance keys. Basenames still work for flat datasets, but relative paths avoid collisions in real repos.

## GPU CAGRA Tuning

When built with `-tags cagra`, `bed bench --compare-gpu` can override key CAGRA knobs:

- `--cagra-graph-degree`
- `--cagra-intermediate-degree`
- `--cagra-max-iters`
- `--cagra-itopk`
- `--cagra-search-width`
- `--cagra-team-size`
- `--cagra-algo single_cta|multi_cta|multi_kernel|auto`

Example:

```bash
bed bench . \
  --compare-gpu \
  --query-file queries.txt \
  --exact \
  --cagra-graph-degree 64 \
  --cagra-max-iters 96 \
  --cagra-itopk 128 \
  --cagra-search-width 4 \
  --cagra-algo multi_cta
```

GPU benchmarking prerequisites:

- Build with `-tags cagra`
- Build or refresh `libcagra_wrapper.so` with `./build_cagra.sh`
- Ensure CUDA/cuVS libraries are on `LD_LIBRARY_PATH`
