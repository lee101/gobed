# Expected Results for "re render game" Search

## Current Issue
The bed tool is using mock embeddings instead of real gobed embeddings, resulting in poor semantic search quality.

## With Mock Embeddings (Current)
```
src/index.go:556-560 (0.4773)          # Unrelated file
src/chunked_cmd.go:76-81 (0.3408)      # Unrelated file
bed_cuda.go:406-414 (0.2605)           # Unrelated file
```

## With Real gobed Embeddings (Expected)
```
game.min.js:3174-3175 (0.95+)
   3174: // Re-render current game to update chat interface
   3175: ReRenderCurrentGame();

game.min.js:3201-3202 (0.95+)
   3201: // Re-render current game
   3202: ReRenderCurrentGame();

game.min.js:9162-9164 (0.94+)
   9162: // Re-render the game
   9163: if (window.ReRenderCurrentGame && typeof window.ReRenderCurrentGame === 'function') {
   9164:     window.ReRenderCurrentGame();

game.min.js:9220-9222 (0.94+)
   9220: // Re-render the game
   9221: if (window.ReRenderCurrentGame && typeof window.ReRenderCurrentGame === 'function') {
   9222:     window.ReRenderCurrentGame();

game.js:7476-7477 (0.92+)
   7476: // Update room AIs live and re-render
   7477: renderAIsInRoom(roomData);
```

## The Problem

1. **Mock Embeddings**: The current bed_cuda.go uses `generateMockEmbedding()` which creates random-like embeddings based on character values, not semantic meaning.

2. **Real gobed Model Not Loaded**: Although bed_search.go calls `gobed.LoadModel()`, the build fails due to CUDA linking issues with undefined references.

3. **Chunking Strategy**: Need ~1500 character chunks (about 500 tokens) with proper overlap for context.

## Solution Requirements

1. **Load Real Model**: Use `gobed.LoadModel()` to load real_model.safetensors
2. **Generate Real Embeddings**: Use `model.Encode(text)` for each chunk
3. **Proper Chunking**: ~1500 chars per chunk with 200 char overlap
4. **Handle Long Lines**: Split lines > 1500 chars into multiple chunks

## Build Issue
The CUDA linking fails with:
```
undefined reference to `create_max_performance_index'
undefined reference to `destroy_max_performance_index'
undefined reference to `add_vectors_cuda'
```

These functions are defined in cuda_max_performance.cu but aren't being linked properly.

## Workaround Options
1. Build without CUDA support (CPU-only mode)
2. Fix CUDA linking by compiling .cu files to .o and linking them
3. Use the Python gobed wrapper which doesn't have linking issues