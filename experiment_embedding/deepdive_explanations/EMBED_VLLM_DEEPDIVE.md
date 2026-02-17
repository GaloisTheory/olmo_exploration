# Deep Dive: `embed_dataset_vllm.py`

A line-by-line expert guide to what this script does, how, and why.

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [The Dataset](#2-the-dataset)
3. [What We Embed (The Concatenation Decision)](#3-what-we-embed)
4. [System Prompts: Silently Ignored](#4-system-prompts)
5. [Think Token Parsing](#5-think-token-parsing)
6. [The vLLM Embedding Engine](#6-vllm-embedding-engine)
7. [Pre-Truncation Trick](#7-pre-truncation-trick)
8. [Multi-GPU Architecture](#8-multi-gpu-architecture)
9. [Data Shape Through the Pipeline](#9-data-shape-through-the-pipeline)
10. [Merge and Output Structure](#10-merge-and-output-structure)
11. [How HuggingFace Would Have Done It](#11-huggingface-comparison)
12. [Edge Cases and Gotchas](#12-edge-cases-and-gotchas)
13. [Resume Logic](#13-resume-logic)
14. [Quick Reference: Key Lines](#14-quick-reference)

---

## 1. Big Picture

This script takes the **allenai/Dolci-Think-SFT-7B** dataset (an SFT dataset where a 7B model was trained to "think" before responding), embeds every data point into a **1024-dim vector** using **Qwen3-Embedding-0.6B** via **vLLM**, and saves everything as safetensors + parquet for downstream analysis (clustering, PCA, etc. — see `analyze_embeddings.py`).

It is a **faster rewrite** of `embed_dataset.py`. The docstring (lines 1-8) summarizes the gains:

- vLLM with PagedAttention (no padding waste, no OOM, no batch halving)
- Single embedding per data point (vs two in the original)
- Default max_length=2048 (vs 8192 in original)
- Expected speedup: **36h → ~3-5h** on 8x H100 for 2.3M rows

The pipeline has 3 steps:
1. **Load & sample** — download from HF, extract fields, save parquet
2. **Embed** — shard data across GPUs, run vLLM on each shard
3. **Merge** — concatenate per-GPU shards, split by source, save final outputs

---

## 2. The Dataset

**Source:** `allenai/Dolci-Think-SFT-7B` (line 62)

Each row contains:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list[dict]` | Chat conversation: `[{role: str, content: str}, ...]` |
| `dataset_source` | `str` | Which sub-dataset this row came from (~16 sources) |

A typical `messages` field looks like:
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the derivative of x^2?"},
  {"role": "assistant", "content": "<think>Power rule: d/dx x^n = nx^(n-1)</think>2x"}
]
```

The dataset has ~2.3M rows total. The default sample is 625 rows per source × ~16 sources ≈ 10K rows.

---

## 3. What We Embed

### The Single Most Important Line (line 168)

```python
f"{row['user_prompt']}\n\n{row['response_text']}\n\n{row['think_text']}"
```

**One embedding per data point.** Three fields concatenated with double-newline separators.

**Concatenation order: prompt → response → think.**

This matters enormously because Qwen3-Embedding uses **last-token pooling** — the final token's hidden state becomes the embedding. So:

- `think_text` occupies the **privileged last-token position** — it has the most direct influence on the embedding vector
- `user_prompt` provides context at the beginning
- `response_text` sits in the middle

### Contrast with the original `embed_dataset.py`

The original produced **two separate embeddings** per row:
- One for `think_text` alone
- One for `response_text` alone
- `user_prompt` was saved as metadata but **not embedded**

The vLLM version collapses everything into a single vector. This is a fundamental design change — you lose the ability to compare think vs. response embeddings, but you get a holistic representation of the entire data point.

---

## 4. System Prompts: Silently Ignored

**Lines 86-100:**

```python
# Scan BACKWARDS to find last assistant message
for idx in range(len(msgs) - 1, -1, -1):
    if msgs[idx]["role"] == "assistant":
        asst_msg = msgs[idx]["content"]
        asst_idx = idx
        break

# Scan BACKWARDS from assistant to find preceding user message
for idx in range(asst_idx - 1, -1, -1):
    if msgs[idx]["role"] == "user":
        user_prompt = msgs[idx]["content"]
        break
```

The code only looks for `role == "assistant"` and `role == "user"`. **Any `role == "system"` message is completely discarded.** If the system prompt says "You are a math tutor" or "Always respond in French", that context is lost from the embedding.

**Multi-turn conversations are also collapsed.** The code takes only the **last** assistant message and the **last user message before it**. If there's a 5-turn conversation, you only get the final user→assistant pair. All earlier turns are thrown away.

---

## 5. Think Token Parsing

**Lines 102-109:**

```python
think_text = ""
response_text = asst_msg  # default: entire message is the response
if "<think>" in asst_msg and "</think>" in asst_msg:
    think_start = asst_msg.index("<think>") + len("<think>")
    think_end = asst_msg.index("</think>")
    if think_end >= think_start:
        think_text = asst_msg[think_start:think_end].strip()
    response_text = asst_msg[think_end + len("</think>"):].strip()
```

### What happens:

1. **Both `<think>` AND `</think>` must be present.** If only one tag exists, the entire assistant message becomes `response_text` and `think_text` stays `""`. No partial parsing.

2. **Only the FIRST pair is extracted.** `.index()` returns the first occurrence. If the model produced:
   ```
   <think>first thought</think>partial answer<think>second thought</think>final answer
   ```
   Then `think_text = "first thought"` and `response_text = "partial answer<think>second thought</think>final answer"`. The second think block leaks into the response.

3. **Content before `<think>` is silently lost.** If the assistant message is:
   ```
   Sure! <think>reasoning here</think>The answer is 42
   ```
   Then `"Sure! "` is gone. `response_text` only captures what comes after `</think>`.

4. **Think text is `.strip()`'d.** Leading/trailing whitespace is removed from both `think_text` and `response_text`.

5. **Guard against malformed tags.** Line 107: `if think_end >= think_start` ensures `<think></think>` (empty) doesn't produce negative slicing. If the tags are malformed (e.g., `</think>` before `<think>`), think_text stays empty but `response_text` still gets reassigned to everything after `</think>`.

---

## 6. vLLM Embedding Engine

### Model initialization (lines 157-163)

```python
llm = LLM(
    model=EMBEDDING_MODEL,           # "Qwen/Qwen3-Embedding-0.6B"
    task="embed",                     # tells vLLM: embedding mode, not generation
    max_model_len=max_length,         # 2048 (default)
    dtype="bfloat16",                 # half precision for speed
    gpu_memory_utilization=0.90,      # use 90% of GPU memory for KV cache
)
```

- `task="embed"` switches vLLM from text generation to embedding extraction mode
- `gpu_memory_utilization=0.90` tells PagedAttention how much VRAM to pre-allocate for its paged KV cache. Higher = more sequences in flight simultaneously, but less headroom.

### Embedding call (line 187)

```python
outputs = llm.embed(texts)
```

**One call.** You pass a list of strings, vLLM:
1. Tokenizes each text
2. Schedules sequences into the paged KV cache using continuous batching
3. Runs forward passes, dynamically grouping sequences by similar length
4. Extracts the last-token hidden state for each sequence (pooling strategy determined by model config)
5. Returns a list of `EmbeddingOutput` objects

### Post-processing (lines 190-193)

```python
embeddings = torch.tensor([o.outputs.embedding for o in outputs])
embeddings = F.normalize(embeddings, p=2, dim=1)
save_file({"embeddings": embeddings.float()}, output_path)
```

- Each `o.outputs.embedding` is a list of floats (length 1024)
- L2 normalization: each vector gets unit norm (||v|| = 1.0), standard for cosine similarity
- **Saved as float32** even though the model ran in bfloat16 — doubles storage but ensures downstream compatibility

---

## 7. Pre-Truncation Trick

**Lines 174-183:**

```python
tokenizer = llm.get_tokenizer()
truncated = 0
for i, text in enumerate(texts):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) > max_length - 2:
        token_ids = token_ids[:max_length - 2]
        texts[i] = tokenizer.decode(token_ids, skip_special_tokens=True)
        truncated += 1
```

### Why this exists

vLLM will reject or error on sequences that exceed `max_model_len`. This pre-truncation ensures all texts fit.

### Why `-2`

The embedding model adds BOS/EOS special tokens. Reserving 2 tokens ensures the total sequence (including special tokens) fits within `max_length`.

### What gets truncated

Truncation chops from the **end** of the token sequence. Given the concatenation order `prompt → response → think`:

- Short overflow: **think text** is partially clipped
- Medium overflow: think text is gone, **response** is partially clipped
- Extreme overflow: only the **prompt** (or part of it) survives

This is an interesting trade-off: think text gets last-token privilege when it fits, but is the **first to be sacrificed** when it doesn't.

### The decode roundtrip is lossy

`tokenizer.decode(token_ids)` can produce slightly different text than the original (whitespace normalization, subword boundaries, special character handling). The re-encoded version fed to vLLM may tokenize to slightly different IDs. This is a minor artifact in practice.

---

## 8. Multi-GPU Architecture

### Process spawning (lines 197-241)

```
Main process (no GPU)
  ├── reads parquet, calculates shard boundaries
  ├── mp.set_start_method("spawn")   ← line 325, required for CUDA
  ├── spawns N child processes via mp.Process
  │     ├── Worker 0: CUDA_VISIBLE_DEVICES="0", rows [0, shard_size)
  │     ├── Worker 1: CUDA_VISIBLE_DEVICES="1", rows [shard_size, 2*shard_size)
  │     ├── Worker 2: CUDA_VISIBLE_DEVICES="2", rows [2*shard_size, 3*shard_size)
  │     └── ... up to num_gpus workers
  └── p.join() for all, sys.exit(1) if any failed
```

### Why "spawn" (line 325)

`mp.set_start_method("spawn", force=True)` creates **fresh Python interpreters** for each child process. This is required because:
- CUDA contexts cannot be safely forked (the "fork" method copies the parent's CUDA state, causing GPU crashes)
- vLLM initializes its own CUDA context and expects a clean environment

### GPU pinning (line 147)

```python
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

Set **before** importing torch/vllm in the worker function. The CUDA runtime only sees one device, so `cuda:0` in each worker actually maps to a different physical GPU.

### Shard size (line 211)

```python
shard_size = (n_rows + num_gpus - 1) // num_gpus
```

Ceiling division. 10K rows with 8 GPUs → 1250 per shard. The last GPU may get fewer rows (line 215-216 handles the boundary).

### Failure handling (lines 233-242)

All workers must succeed. If any exits non-zero, the script calls `sys.exit(1)`. No partial recovery — you'd need `--resume` to retry.

---

## 9. Data Shape Through the Pipeline

### Stage 0: Raw HF dataset row

```python
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "<think>Simple addition: 2+2=4</think>4"}
  ],
  "dataset_source": "gsm8k"
}
```

### Stage 1: After extraction (lines 82-117) → parquet record

```python
{
  "id": 0,
  "dataset_source": "gsm8k",
  "user_prompt": "What is 2+2?",          # system prompt: GONE
  "think_text": "Simple addition: 2+2=4",  # extracted from <think> tags
  "response_text": "4"                     # everything after </think>
}
```

### Stage 2: Concatenated text → vLLM input (line 168)

```
"What is 2+2?\n\n4\n\nSimple addition: 2+2=4"
 ^^^^^^^^^^^^^^  ^^  ^^^^^^^^^^^^^^^^^^^^^^^^^
 user_prompt     response_text   think_text
```

### Stage 3: After vLLM → raw embedding (line 190)

```python
[0.0234, -0.0891, 0.0412, ...]  # list of 1024 floats
```

### Stage 4: After L2 normalization (line 191)

```python
tensor([0.0234, -0.0891, 0.0412, ...])  # shape [1024], ||v|| = 1.0
```

### Stage 5: Per-GPU shard file → `output/embeddings/shards/gpu0.safetensors`

```python
{"embeddings": Tensor}  # shape [shard_size, 1024], dtype float32
```

### Stage 6: Merged → `output/embeddings/all.safetensors`

```python
{"embeddings": Tensor}  # shape [total_rows, 1024], dtype float32
```

### Stage 7: Per-source split → `output/embeddings/by_source/gsm8k/`

```
embeddings.safetensors   # shape [n_rows_for_gsm8k, 1024]
metadata.parquet         # columns: id, dataset_source, user_prompt, think_text, response_text
```

---

## 10. Merge and Output Structure

**Function: `merge_shards` (lines 247-311)**

### What it does:

1. Loads all per-GPU shard files and `torch.cat`s them into one tensor
2. **Asserts integrity** (line 273): `full_tensor.shape[0] == len(df)` — this is the only data integrity check in the entire pipeline
3. Saves `all.safetensors` — the complete embedding matrix
4. Splits by `dataset_source` and saves each source's embeddings + metadata separately
5. Saves a full `metadata.parquet` alongside

### Final output tree:

```
output/
├── run_config.json              # {sample_size, num_gpus, max_length, backend}
├── extracted_data.parquet       # all extracted records (id, source, prompt, think, response)
└── embeddings/
    ├── all.safetensors          # [total_rows, 1024]
    ├── metadata.parquet         # same as extracted_data.parquet
    ├── shards/
    │   ├── gpu0.safetensors     # [shard_size, 1024]
    │   ├── gpu1.safetensors
    │   └── ...
    └── by_source/
        ├── gsm8k/
        │   ├── embeddings.safetensors
        │   └── metadata.parquet
        ├── math/
        │   ├── embeddings.safetensors
        │   └── metadata.parquet
        └── ...
```

---

## 11. HuggingFace Comparison

The original `embed_dataset.py` does the same task with `transformers` instead of vLLM. Here's a detailed comparison:

| Aspect | HuggingFace (`embed_dataset.py`) | vLLM (`embed_dataset_vllm.py`) |
|--------|----------------------------------|-------------------------------|
| **Model loading** | `AutoModel.from_pretrained()` + `AutoTokenizer` | `LLM(model=..., task="embed")` |
| **Pooling** | Manual: find last non-pad token in hidden states (lines 225-230) | Internal to vLLM (opaque) |
| **Padding** | Left-padded (`padding_side="left"`), wastes memory on short seqs | PagedAttention, zero padding waste |
| **Batching** | Fixed batch size, manual loop | Automatic continuous batching |
| **OOM handling** | `try/except OOM`, halves batch size (lines 254-260) | Doesn't OOM (pre-allocated paged memory) |
| **Checkpointing** | Every 20 batches via `torch.save` (line 247) | Not needed (single `.embed()` call) |
| **Progress bars** | `tqdm` per GPU | vLLM's internal progress reporting |
| **Embeddings per row** | 2 (think_text + response_text separately) | 1 (concatenated) |
| **Max length** | 8192 default | 2048 default |
| **Speed** | ~36h on 8x H100 | ~3-5h on 8x H100 |

### The manual last-token pooling in the original (lines 225-230):

```python
hidden = outputs.last_hidden_state          # [B, seq_len, dim]
attention_mask = encoded["attention_mask"]   # [B, seq_len]
seq_lengths = attention_mask.sum(dim=1) - 1  # last non-pad position
batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
last_token_embeds = hidden[batch_indices, seq_lengths]  # [B, dim]
```

This is what vLLM does internally — you just don't see it.

### Why left padding matters in the original:

With right padding (default), the last non-pad token is at a different position for every sequence in the batch. With left padding, all sequences are right-aligned, but you still need the attention mask to find where real content ends. The original handles this correctly.

---

## 12. Edge Cases and Gotchas

### 12.1 Empty text → literal "[empty]" (line 171)

```python
texts = [t.strip() if t.strip() else "[empty]" for t in texts]
```

If all three fields (prompt, response, think) are empty strings, the concatenation is `"\n\n\n\n"`, which `.strip()` reduces to `""`, which becomes `"[empty]"`. You get a real embedding of the literal word "[empty]". These would cluster together and could skew analysis.

### 12.2 Sampling isn't random (lines 69-73)

```python
pl.int_range(pl.len()).over("dataset_source").alias("_row_num")
.filter(pl.col("_row_num") < sample_size)
```

This assigns sequential row numbers within each source and takes the first N. It's a **deterministic top-N**, not a random sample. If the dataset is ordered (e.g., by difficulty, by date), you get a biased slice.

### 12.3 No instruction prefix

Qwen3-Embedding supports instruction-prefixed queries like:
```
Instruct: Represent this document for clustering
Query: <your text here>
```

This script passes **raw text with no instruction prefix**. This is fine for document-to-document comparison (symmetric similarity), but would be suboptimal for asymmetric retrieval (query → document search).

### 12.4 Float32 conversion on save (line 193)

```python
save_file({"embeddings": embeddings.float()}, output_path)
```

Model runs bfloat16, embeddings saved as float32. This doubles storage size (~4KB per embedding vs ~2KB) but avoids bfloat16 compatibility issues with downstream tools.

### 12.5 Source name sanitization (line 287)

```python
safe_name = source.replace("/", "_").replace(" ", "_")
```

Only replaces `/` and spaces. Other special characters (like `:`, `(`, `)`) would remain in directory names, potentially causing filesystem issues on some systems.

### 12.6 The merge assert is the only integrity check (line 273)

```python
assert full_tensor.shape[0] == len(df)
```

If a GPU died mid-shard and wrote a partial safetensors file, this will catch the mismatch. But if a shard file is corrupted in a way that still has the right number of rows (e.g., all zeros), there's no detection.

### 12.7 `asst_msg` falsy check (line 93)

```python
if not asst_msg:
    continue
```

Catches both `None` (no assistant message found) and empty string `""`. An assistant message that is literally empty gets the row dropped silently.

### 12.8 Row ID assignment (line 111)

```python
"id": len(records)
```

IDs are assigned sequentially based on the **output** list, not the input. If rows are dropped (no assistant message), the IDs won't match the original dataset indices. They're purely internal identifiers.

---

## 13. Resume Logic

### Config validation (lines 329-335)

On `--resume`, the script checks that the saved `run_config.json` has the same `sample_size`. If mismatched, it errors out. Other parameters (num_gpus, max_length) are **not validated** — you could resume with different settings, which might produce inconsistent results.

### Data reuse (lines 345-349)

If `extracted_data.parquet` exists, it's reused as-is. The extraction step is skipped entirely.

### Shard skipping (lines 221-222)

```python
if resume and Path(output_path).exists():
    print(f"[GPU {gpu_id}] Shard exists (resume), skipping")
    continue
```

Only checks if the file **exists**, not if it's valid or complete. A corrupted or partial shard will be accepted.

### Stale cleanup on fresh runs (lines 337-342)

Without `--resume`, the script deletes `embeddings/` and `checkpoints/` directories to prevent stale data from a previous run from contaminating the new one.

---

## 14. Quick Reference: Key Lines

| Lines | What |
|-------|------|
| 22 | Embedding model: `Qwen/Qwen3-Embedding-0.6B`, dim=1024 |
| 62-63 | Load dataset from HF, zero-copy to Polars |
| 69-73 | Stratified sampling (deterministic top-N, not random) |
| 86-100 | Extract last assistant + preceding user message (system ignored) |
| 102-109 | Parse `<think>`/`</think>` tags |
| 147 | GPU pinning via `CUDA_VISIBLE_DEVICES` |
| 157-163 | vLLM model init (`task="embed"`, bfloat16, 90% GPU util) |
| 168 | **The concatenation**: `prompt \n\n response \n\n think` |
| 171 | Empty text → `"[empty]"` placeholder |
| 174-183 | Pre-truncation to `max_length - 2` tokens |
| 187 | `llm.embed(texts)` — the actual embedding call |
| 191 | L2 normalization |
| 193 | Save as float32 safetensors |
| 211 | Shard size calculation (ceiling division) |
| 225-230 | Worker process spawning |
| 273 | Integrity assert: embedding count == data rows |
| 325 | `mp.set_start_method("spawn")` — required for CUDA |
