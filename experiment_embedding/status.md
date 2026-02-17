# Full-v2 Embedding Run — Status

## State: READY TO RUN (2026-02-17)

Code has been updated with character-length heuristic optimization. Not yet tested.
All previous runs were killed. No embedding progress exists.

**What exists:** `output/full-v2/extracted_data.parquet` (35GB, 2,268,178 rows) — extraction is done.
**What's missing:** `dropped_rows.parquet`, all embeddings.

## Current Code State (filter_by_length in embed_dataset_vllm.py)

The function has been rewritten with a **3-stage approach** (lines 161-270):

### Stage 1: Character-length heuristic (Polars, vectorized, ~seconds)
- Builds concatenated text column via `pl.concat_str()` and computes `_char_len`
- Sets `char_threshold = (max_length - 2) * 2` = 65,532 chars (worst-case 2 chars/token for BPE)
- Any text under this threshold is **guaranteed** within the token limit — skip tokenization

### Stage 2: Tokenize only long texts (Rust-parallel, ~1-5% of data)
- Uses `tokenizer.backend_tokenizer.encode_batch()` — the underlying Rust tokenizer directly
- Avoids HuggingFace Python wrapper overhead (no `BatchEncoding`, no Python int objects for input_ids)
- Chunks of 100K, with `del` between chunks to control memory
- Only processes the ~1-5% of texts that exceed the character threshold

### Stage 3: Build mask from char_lens + tok_lengths (Python loop, fast)
- Empty texts (char_len == 0) → dropped as "empty"
- Long texts where tok_lengths[i] > token_limit → dropped as "too_long"
- Everything else → kept

### Expected speedup
- **Original:** 100+ minutes (2.26M sequential `tokenizer.encode()` calls, killed before finishing)
- **Attempt 1 (batch all 2.26M):** Also slow (~90 min projected) — memory leaked to 50%+, killed
- **Attempt 2 (chunked + gc.collect):** gc.collect() overhead made it slower (~10 min/chunk), killed
- **Attempt 3 (backend_tokenizer.encode_batch):** Better memory (8.8%) but still ~3 min/chunk for 46 chunks, killed
- **Current code (char heuristic + backend_tokenizer):** Should be **~2-5 min total** — only tokenizes ~1-5% of rows

## Key Learnings from Failed Attempts

1. **`tokenizer(list_of_texts)` returns `input_ids` always** — can't disable it. For 100K texts × 3000 tokens avg = 300M Python int objects per chunk. Memory leaks via Python allocator fragmentation.

2. **`gc.collect()` is extremely expensive** when there are millions of objects — added ~8 min overhead per chunk, negating parallelism gains.

3. **`backend_tokenizer.encode_batch()`** is the right API — returns Rust `Encoding` objects, `len(enc.ids)` is O(1) without materializing Python lists. But tokenizing ALL 2.26M texts is fundamentally slow even with 128 cores.

4. **The real win is avoiding unnecessary work** — 99.8% of texts are within the 32768-token limit. Character-length heuristic filters them without tokenization.

5. **Always use `PYTHONUNBUFFERED=1`** when running via pipe/background — otherwise Python buffers stdout and you get zero progress output.

## Run Command

```bash
cd ~/cc_workspace_mats/projects/olmo_exploration/experiment_embedding

# PYTHONUNBUFFERED=1 ensures progress output is visible in real-time
PYTHONUNBUFFERED=1 /home/dlee2176/cc_workspace_mats/.venv/bin/python embed_dataset_vllm.py \
    --full --num-gpus 8 --run-name full-v2 --resume
```

`--resume` skips extraction (extracted_data.parquet exists) but re-runs filtering (dropped_rows.parquet doesn't exist).

## Risk: CHARS_PER_TOKEN_LOWER_BOUND = 2

The heuristic assumes worst-case 2 chars/token. If ANY token in the Qwen3 vocabulary maps a single character to multiple tokens (ratio < 2), those texts could slip through and exceed max_length at embedding time. This is extremely unlikely for BPE tokenizers (single-char tokens are common but multi-token single chars are not), but worth knowing.

If worried, validate with a small sample first:
```python
# Quick sanity check: tokenize 1K longest texts, verify all under char_threshold are within token_limit
```

## Expected Timeline

| Step | Time |
|------|------|
| Extraction (skipped via --resume) | 0 min |
| Filtering (char heuristic + selective tokenization) | ~2-5 min |
| Embedding (8× H100, vLLM) | ~30-45 min |
| Merge | ~5 min |
| **Total** | **~40-55 min** |

## Expected Output

```
output/full-v2/
  extracted_data.parquet          # ~2.26M rows (filtered)
  dropped_rows.parquet            # ~4,500 rows with reasons
  run_config.json
  embeddings/
    all.safetensors               # (~2.26M, 1024) — ~9.2 GB
    metadata.parquet
    shards/gpu{0..7}.safetensors
    by_source/<16 sources>/
```
