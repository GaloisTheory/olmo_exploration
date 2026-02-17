# Embedding Experiment

## Next Step: Full 2.3M Run

Run the full dataset embedding. This is the only thing left before analysis.

### Command

```bash
cd ~/cc_workspace_mats/projects/olmo_exploration/experiment_embedding
/home/dlee2176/cc_workspace_mats/.venv/bin/python embed_dataset_vllm.py --full --num-gpus 8 --run-name full-v2
```

### What this does

1. **Load** all 2,268,178 rows from `allenai/Dolci-Think-SFT-7B` (HF datasets, cached at `$HF_HOME`)
2. **Extract** user_prompt (full multi-turn history), think_text, response_text, num_turns
3. **Filter** by token length — drop rows exceeding 32,768 tokens (the model's full context window). Based on 1K runs, expect ~0.2% drop rate (~4,500 rows). Saves `dropped_rows.parquet` for audit.
4. **Embed** across 8 GPUs using vLLM (Qwen3-Embedding-0.6B). Each GPU gets ~283K rows.
5. **Merge** shards into `all.safetensors` + per-source splits

### Time estimate

Based on 1K/source runs (16K rows in ~6 min on 8x H100):
- Data loading: ~8 min (first load from HF cache)
- Filter (sequential tokenization): **~45-60 min** (bottleneck — 2.3M rows × tokenizer.encode)
- Embedding (8 GPUs): **~30-45 min** (vLLM with 32K context, PagedAttention)
- Merge: ~5 min
- **Total: ~1.5-2 hours**

### If interrupted

Resume with:
```bash
/home/dlee2176/cc_workspace_mats/.venv/bin/python embed_dataset_vllm.py --full --num-gpus 8 --run-name full-v2 --resume
```
This skips completed steps (extracted_data.parquet, dropped_rows.parquet, individual GPU shards).

### After completion

Run analysis:
```bash
# Edit analyze_embeddings.py: set RUN_NAME = "full-v2"
# Then run cells or:
/home/dlee2176/cc_workspace_mats/.venv/bin/python analyze_embeddings.py --run-name full-v2
```

### Expected output

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

---

## Current State (2026-02-17)

### Completed 1K/source Runs (v2, max_length=32768)

| | vllm-1k-v2 (top-N) | vllm-1k-random (seed=42) |
|---|---|---|
| Input rows | 16,000 | 16,000 |
| Dropped | 32 (0.2%) | 40 (0.25%) |
| **Rows embedded** | **15,968** | **15,960** |
| Sources | 16 | 16 |
| Total time | 356s | 359s |

With 32K context (the model's native max), virtually all rows fit. The previous 2048 cutoff was unnecessarily aggressive and dropped ~45%.

### Previous Runs (v1, kept for comparison)

- `output/vllm-1k/` — v1 vLLM run (prompt+response+think order, truncation, L2 norm, 2048 max)
- `output/1k/` — original HF Transformers run

### v2 Changes (embed_dataset_vllm.py)

| # | What changed | Why |
|---|---|---|
| 1 | Concat order: `prompt→think→response` (was prompt→response→think) | Response in last-token position for better embedding signal |
| 2 | Multi-turn: full conversation history in `user_prompt` | Was only extracting last user message |
| 3 | Drop-not-truncate with `dropped_rows.parquet` | Silent truncation misrepresents text |
| 4 | Removed redundant `F.normalize` | Model outputs unit-norm natively |
| 5 | `--random-seed N` option | Enables random sampling (default: deterministic top-N) |
| 6 | Default max_length=32768 | Model supports 32K; 2048 was dropping 45% of rows |
| 7 | `num_turns` column in metadata | Tracks multi-turn conversations |

### Key Finding: Qwen3-Embedding Outputs Unit-Norm Vectors

Qwen3-Embedding-0.6B outputs **pre-normalized** (unit-norm) embeddings natively. This is standard for embedding models trained with cosine objectives. PCA works fine — sklearn's `fit_transform` mean-centers first, and variance comes from angular differences.

---

## Reference

### Pipeline Flow

```
1. load_and_sample()     → extracted_data.parquet
2. filter_by_length()    → overwrites extracted_data.parquet (filtered) + dropped_rows.parquet
3. run_embedding()       → shards/gpu{0..N}.safetensors
4. merge_shards()        → all.safetensors + metadata.parquet + by_source/
```

### CLI Options

```
--sample-size N     Rows per source (default 625, use --full for all)
--full              Process entire dataset
--num-gpus N        Number of GPUs (default 8)
--max-length N      Max token length (default 32768)
--run-name NAME     Output subdirectory
--random-seed N     Random sampling seed (default None = deterministic top-N)
--resume            Resume from checkpoint
```

### Python Environment

```bash
/home/dlee2176/cc_workspace_mats/.venv/bin/python
```

### Output Structure

```
output/<run-name>/
  extracted_data.parquet          # id, dataset_source, user_prompt, think_text, response_text, num_turns
  dropped_rows.parquet            # id, dataset_source, reason, token_count
  run_config.json                 # sample_size, num_gpus, max_length, backend, random_seed
  embeddings/
    all.safetensors               # (N, 1024) unit-norm (model-native)
    metadata.parquet              # id, dataset_source, user_prompt, think_text, response_text, num_turns
    shards/                       # per-GPU intermediate files
    by_source/<source>/           # per-source splits
```

### Files

- `embed_dataset_vllm.py` — **main script** (v2, recommended)
- `embed_dataset.py` — original HF Transformers backend (v1, NOT updated, kept as fallback)
- `analyze_embeddings.py` — PCA analysis notebook (scatter, scree, PC extremes, corrplot, HTML viewer)
- `embedding_utils.py` — shared utilities

---

## Resolved Issues

| # | Issue | Fix |
|---|-------|-----|
| 1 | Polars `hf://` rate-limited (429s) | `datasets.load_dataset()` with HF auth cache |
| 2 | Flash Attention 2 not installed | Removed; PyTorch SDPA auto-dispatches to FA2 |
| 3 | `torch_dtype` deprecated | `dtype=torch.bfloat16` |
| 4 | Stale shards cause IndexError | Auto-cleanup when not `--resume` |
| 5 | Matplotlib `get_cmap` deprecated | `colormaps.get_cmap().resampled()` |
| 6-9 | v2 fixes (concat order, multi-turn, truncation, F.normalize) | See v2 changes table above |

## Design Decisions

- **vLLM over HF Transformers**: PagedAttention — no padding waste, no OOM, no batch halving.
- **Concat order `prompt→think→response`**: Response in last-token position (where embedding model pools).
- **Drop-not-truncate**: Truncated embeddings misrepresent text. Track drops in `dropped_rows.parquet`.
- **max_length=32768**: Model's native context. Only ~0.2% of rows exceed this.
- **Multi-turn concatenation**: All messages before last assistant turn joined into `user_prompt`.
- **Unit-norm embeddings**: Model-native. PCA mean-centers, so angular variance is preserved.
