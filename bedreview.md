# Bed Binary Manual Review

Reviewed binary: `/tmp/bed-local` built from current `main` (`e5fa68e`, equivalent release line to `v1.0.2`).
Test date: 2026-02-11.

## Scope
Manual CLI testing focused on:
1. Search quality and latency.
2. Index/reindex behavior.
3. Inotify daemon live updates.
4. Ignore behavior for `.gitignore`, binaries, and long lines.
5. CPU/GPU benchmark command behavior.

I used an isolated fixture at `/tmp/bed-manual-fixture` and isolated HOME at `/tmp/bed-home-manual`.

## What Works Well
1. CPU search path is fast once warm.
- First search (includes indexing + cache save): `wall=0.36s`.
- Second search (cached): `wall=0.15s`.

2. Bench command provides useful performance and quality metrics.
- `bed bench . --queries 20 --warmup 5`
  - Index time: `343ms`
  - Avg latency: `1.304ms`
  - P95 latency: `1.474ms`
  - Throughput: `766.58 qps`
- `bed bench . --queries 10 --warmup 3 --ndcg --ndcg-k 10`
  - `NDCG@10 = 0.9109`
  - `Recall@10 = 0.8750`

3. Daemon live update flow works for file modifications.
- Started daemon: `bed daemon . --port 18865 --batch-delay 100ms --verbose`
- Edited `src/db.go` and queried `/search`.
- New line was queryable within ~1s.
- Overwriting file to remove that content removed the result on next query.

## Critical/High Issues
1. `index` and `reindex` are effectively blocked (hang) in this environment.
- Repro:
  - `timeout 20s bed index . --force --batch-size 128 --verbose`
  - `timeout 20s bed reindex . --batch-size 128 --verbose`
- Both showed repeated `Progress: 0 files processed` and exited by timeout `124`.
- This is a release blocker for explicit indexing workflows.

2. `.gitignore` directory patterns are not being respected.
- Fixture `.gitignore` contained `secret/`.
- Query `"super secret key rotation"` still returned `/tmp/bed-manual-fixture/secret/keys.txt`.
- Default ignore of `vendor/` also appears broken: query `"dependency internals"` returned `/tmp/bed-manual-fixture/vendor/lib/dep.txt`.

3. `--no-index` is not usable as a normal CLI mode.
- Repro: fresh process `bed --no-index "token validation"`
- Output: `No documents indexed`
- Expected behavior for users is usually “load prior on-disk index/cache and search without re-indexing.”

## Medium Issues
1. Search results are easily polluted by local model assets (`model/tokenizer.json`).
- In practical runs, top hits were often tokenizer entries instead of project code.
- This hurts result quality for real queries unless users add custom ignore rules.

2. Search command has no direct path argument/flag.
- Current behavior requires changing working directory to target corpus.
- This is awkward for scripting and multi-root workflows.

3. No `--version` flag.
- `bed --version` returns unknown flag, which makes release validation harder.

## GPU Status
1. Hardware detection works in `status` (GPU shown as available).
2. GPU benchmark path was unavailable in this build:
- `bed bench . --compare-gpu`
- Output: `GPU benchmark skipped: CAGRA not available: build with -tags cagra and ensure cuVS is installed`

## Honest v1 Readout
`bed` has a strong fast-search core and good benchmark plumbing, but I would not call this release fully production-ready for general users yet due to:
1. `index`/`reindex` hanging.
2. Broken ignore semantics for common directory patterns.
3. `--no-index` not functioning as users expect in one-shot CLI usage.

## Recommended Next Fixes (highest impact, low risk)
1. Fix ignore matching for trailing-slash directory rules (`secret/`, `vendor/`) and add regression tests.
2. Make `--no-index` load persistent cache/index by default (or fail with actionable guidance).
3. Unify `index`/`reindex` path with the fast search indexer used by `bed <query>` or fix the current embedder pipeline hang.
4. Add default ignore for `model/` and `.bed/` artifacts to improve precision.
5. Add `--version` output including git commit/tag.
