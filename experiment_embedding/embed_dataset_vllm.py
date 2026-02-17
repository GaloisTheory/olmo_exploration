"""Embed Dolci-Think-SFT-7B with vLLM + Qwen3-Embedding-0.6B.

Faster alternative to embed_dataset.py:
- vLLM with PagedAttention (no padding waste, no OOM, no batch halving)
- Single embedding per data point (concatenated: prompt + think + response)
- Response text in last-token position for better embedding signal
- Rows exceeding max_length are dropped (not truncated), tracked in dropped_rows.parquet
- Raw embeddings (no L2 normalization) for downstream PCA analysis
- Multi-turn: full conversation history concatenated into user_prompt
- Default max_length=2048 (vs 8192 in original)

Expected speedup: 36h -> ~3-5h on 8x H100 for 2.3M rows.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import polars as pl
from transformers import AutoTokenizer

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024

timings = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed Dolci-Think-SFT-7B with vLLM + Qwen3-Embedding-0.6B"
    )
    parser.add_argument("--sample-size", type=int, default=625,
                        help="Rows per source (0 = all). Default 625 (~10K total)")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs to use (default: 8)")
    parser.add_argument("--max-length", type=int, default=32768,
                        help="Max token length (default: 32768, model max for Qwen3-Embedding-0.6B)")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output base directory (default: ./output)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name — creates output/<run-name>/ subdirectory")
    parser.add_argument("--full", action="store_true",
                        help="Process entire dataset (no sampling)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip completed shards)")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Random seed for sampling (default: None = deterministic top-N)")
    return parser.parse_args()


# %% Step 1: Data loading (reused from embed_dataset.py)

def load_and_sample(sample_size: int, output_dir: Path, random_seed: int | None = None) -> Path:
    """Load dataset, stratified sample, extract think/response/user_prompt, save to parquet.

    Multi-turn conversations: all messages before the last assistant turn are
    concatenated into user_prompt (not just the last user message).

    When random_seed is set, uses random sampling instead of deterministic top-N.
    """
    from datasets import load_dataset

    if sample_size > 0:
        print(f"=== Sampling {sample_size} rows/source from Dolci-Think-SFT-7B ===\n")
    else:
        print("=== Loading FULL Dolci-Think-SFT-7B dataset ===\n")
    t0 = time.time()

    print("  Loading dataset via HF datasets library...")
    ds = load_dataset("allenai/Dolci-Think-SFT-7B", split="train")
    df = pl.from_arrow(ds.data.table)
    print(f"  Loaded {len(df)} rows from HF")

    if sample_size > 0:
        if random_seed is not None:
            sampled_df = df.group_by("dataset_source").map_groups(
                lambda g: g.sample(n=min(sample_size, len(g)), seed=random_seed)
            )
            print(f"  Random sampling (seed={random_seed})")
        else:
            sampled_df = (
                df.lazy()
                .with_columns(
                    pl.int_range(pl.len()).over("dataset_source").alias("_row_num")
                )
                .filter(pl.col("_row_num") < sample_size)
                .collect()
            )
    else:
        sampled_df = df

    n_sources = sampled_df["dataset_source"].n_unique()
    print(f"  Collected {len(sampled_df)} rows from {n_sources} sources")

    records = []
    for row in sampled_df.iter_rows(named=True):
        source = row["dataset_source"]
        msgs = row["messages"]

        # Find last assistant message
        asst_msg = None
        asst_idx = None
        for idx in range(len(msgs) - 1, -1, -1):
            if msgs[idx]["role"] == "assistant":
                asst_msg = msgs[idx]["content"]
                asst_idx = idx
                break
        if not asst_msg:
            continue

        # Concatenate ALL messages before the last assistant turn
        prior_parts = []
        for idx in range(asst_idx):
            prior_parts.append(msgs[idx]["content"])
        user_prompt = "\n\n".join(prior_parts)

        num_turns = sum(1 for m in msgs if m["role"] == "assistant")

        think_text = ""
        response_text = asst_msg
        if "<think>" in asst_msg and "</think>" in asst_msg:
            think_start = asst_msg.index("<think>") + len("<think>")
            think_end = asst_msg.index("</think>")
            if think_end >= think_start:
                think_text = asst_msg[think_start:think_end].strip()
            response_text = asst_msg[think_end + len("</think>"):].strip()

        records.append({
            "id": len(records),
            "dataset_source": source,
            "user_prompt": user_prompt,
            "think_text": think_text,
            "response_text": response_text,
            "num_turns": num_turns,
        })

    extract_df = pl.DataFrame(records)
    print(f"  Extracted {len(extract_df)} records")

    source_counts = extract_df.group_by("dataset_source").len().sort("len", descending=True)
    for row in source_counts.iter_rows(named=True):
        pct = 100 * row["len"] / len(extract_df)
        print(f"    {row['dataset_source']:45s} {row['len']:5d} ({pct:5.1f}%)")

    data_path = output_dir / "extracted_data.parquet"
    extract_df.write_parquet(data_path)
    print(f"\n  Saved extracted data to {data_path}")

    timings["1_sampling"] = time.time() - t0
    print(f"  [TIME] Sampling + extraction: {timings['1_sampling']:.1f}s\n")
    return data_path


# %% Step 1b: Filter by token length

def filter_by_length(data_path: Path, output_dir: Path, max_length: int) -> Path:
    """Drop rows that are empty or exceed max_length tokens. Save dropped rows for audit.

    Concatenation order matches embed_worker: prompt + think + response.
    Overwrites extracted_data.parquet with filtered data and re-assigns sequential IDs.
    """
    print(f"=== Filtering by token length (max_length={max_length}) ===\n")
    t0 = time.time()

    df = pl.read_parquet(data_path)
    n_before = len(df)

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    # Use underlying Rust tokenizer directly — returns Encoding objects without
    # creating millions of Python int objects for input_ids
    fast_tokenizer = tokenizer.backend_tokenizer
    fast_tokenizer.no_padding()
    fast_tokenizer.no_truncation()

    # Build concatenated text column and compute character lengths in Polars (fast, vectorized)
    text_col = pl.concat_str(
        [pl.col("user_prompt"), pl.lit("\n\n"), pl.col("think_text"),
         pl.lit("\n\n"), pl.col("response_text")],
    ).str.strip_chars()

    df_with_text = df.with_columns(
        text_col.alias("_text"),
        text_col.str.len_chars().alias("_char_len"),
    )

    # Heuristic: for BPE tokenizers, worst-case is ~2 chars/token (CJK, code).
    # Any text under char_threshold characters is guaranteed within max_length tokens.
    # Only tokenize texts above this threshold (~1-5% of data).
    CHARS_PER_TOKEN_LOWER_BOUND = 2
    token_limit = max_length - 2  # account for special tokens
    char_threshold = token_limit * CHARS_PER_TOKEN_LOWER_BOUND

    needs_tokenization = df_with_text.filter(pl.col("_char_len") > char_threshold)
    n_needs_tok = len(needs_tokenization)
    print(f"  {n_before:,} rows total, {n_needs_tok:,} exceed {char_threshold:,} chars "
          f"(need tokenization)", flush=True)

    # Tokenize only the long texts
    tok_lengths = {}  # row index -> token count
    if n_needs_tok > 0:
        long_indices = df_with_text.with_row_index("_idx").filter(
            pl.col("_char_len") > char_threshold
        ).get_column("_idx").to_list()
        long_texts = needs_tokenization.get_column("_text").to_list()

        CHUNK_SIZE = 100_000
        for chunk_start in range(0, len(long_texts), CHUNK_SIZE):
            chunk = long_texts[chunk_start:chunk_start + CHUNK_SIZE]
            encodings = fast_tokenizer.encode_batch(chunk, add_special_tokens=False)
            for j, enc in enumerate(encodings):
                tok_lengths[long_indices[chunk_start + j]] = len(enc.ids)
            del chunk, encodings
            print(f"  Tokenized {min(chunk_start + CHUNK_SIZE, len(long_texts)):,}"
                  f"/{len(long_texts):,} long texts", flush=True)

        del long_texts, long_indices

    # Build keep_mask and dropped_records
    char_lens = df_with_text.get_column("_char_len").to_list()
    ids = df.get_column("id").to_list()
    sources = df.get_column("dataset_source").to_list()
    dropped_records = []
    keep_mask = []

    for i, char_len in enumerate(char_lens):
        if char_len == 0:
            dropped_records.append({
                "id": ids[i],
                "dataset_source": sources[i],
                "reason": "empty",
                "token_count": 0,
            })
            keep_mask.append(False)
        elif i in tok_lengths and tok_lengths[i] > token_limit:
            dropped_records.append({
                "id": ids[i],
                "dataset_source": sources[i],
                "reason": "too_long",
                "token_count": tok_lengths[i],
            })
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    del df_with_text

    # Save dropped rows
    if dropped_records:
        dropped_df = pl.DataFrame(dropped_records)
        dropped_path = output_dir / "dropped_rows.parquet"
        dropped_df.write_parquet(dropped_path)
        print(f"  Dropped {len(dropped_df)} rows:")
        for reason, count in dropped_df.group_by("reason").len().sort("reason").iter_rows():
            print(f"    {reason}: {count}")
        print(f"  Saved dropped rows to {dropped_path}")
    else:
        # Write empty dropped_rows.parquet as marker
        dropped_df = pl.DataFrame({
            "id": pl.Series([], dtype=pl.Int64),
            "dataset_source": pl.Series([], dtype=pl.Utf8),
            "reason": pl.Series([], dtype=pl.Utf8),
            "token_count": pl.Series([], dtype=pl.Int64),
        })
        dropped_path = output_dir / "dropped_rows.parquet"
        dropped_df.write_parquet(dropped_path)
        print("  No rows dropped")

    # Filter and re-assign sequential IDs
    filtered_df = df.filter(pl.Series(keep_mask))
    filtered_df = filtered_df.drop("id").with_row_index("id").cast({"id": pl.Int64})

    # Overwrite extracted_data.parquet
    filtered_df.write_parquet(data_path)
    n_after = len(filtered_df)
    print(f"  {n_before} → {n_after} rows ({n_before - n_after} dropped)")

    timings["1b_filter"] = time.time() - t0
    print(f"  [TIME] Filter: {timings['1b_filter']:.1f}s\n")
    return data_path


# %% Step 2: vLLM embedding

def embed_worker(
    gpu_id: int,
    data_path: str,
    shard_start: int,
    shard_end: int,
    max_length: int,
    output_path: str,
):
    """Worker process: load vLLM model on one GPU, embed a shard of concatenated texts."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from safetensors.torch import save_file
    from vllm import LLM

    n_rows = shard_end - shard_start
    print(f"[GPU {gpu_id}] Loading vLLM model (rows {shard_start}-{shard_end}, n={n_rows})")

    llm = LLM(
        model=EMBEDDING_MODEL,
        task="embed",
        max_model_len=max_length,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )

    # Load data shard and concatenate: prompt + think + response
    df = pl.read_parquet(data_path).slice(shard_start, n_rows)
    texts = [
        f"{row['user_prompt']}\n\n{row['think_text']}\n\n{row['response_text']}"
        for row in df.iter_rows(named=True)
    ]
    texts = [t.strip() for t in texts]

    # Embed all texts (vLLM handles internal batching via PagedAttention)
    # Rows already filtered by filter_by_length — no truncation needed
    print(f"[GPU {gpu_id}] Embedding {len(texts)} texts...")
    outputs = llm.embed(texts)

    # Extract raw embeddings (no L2 normalization — preserves magnitude for PCA)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])

    save_file({"embeddings": embeddings.float()}, output_path)
    print(f"[GPU {gpu_id}] Saved {list(embeddings.shape)} -> {output_path}")


def run_embedding(
    data_path: Path,
    num_gpus: int,
    max_length: int,
    output_dir: Path,
    resume: bool,
) -> None:
    """Run single embedding pass across all GPUs."""
    df = pl.read_parquet(data_path)
    n_rows = len(df)

    shard_dir = output_dir / "embeddings" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_size = (n_rows + num_gpus - 1) // num_gpus
    procs = []
    for gpu_id in range(num_gpus):
        shard_start = gpu_id * shard_size
        shard_end = min(shard_start + shard_size, n_rows)
        if shard_start >= n_rows:
            break

        output_path = str(shard_dir / f"gpu{gpu_id}.safetensors")

        if resume and Path(output_path).exists():
            print(f"[GPU {gpu_id}] Shard exists (resume), skipping: {output_path}")
            continue

        p = mp.Process(
            target=embed_worker,
            args=(gpu_id, str(data_path), shard_start, shard_end,
                  max_length, output_path),
        )
        p.start()
        procs.append((gpu_id, p))

    failed = []
    for gpu_id, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(gpu_id)
            print(f"[ERROR] GPU {gpu_id} exited with code {p.exitcode}")

    if failed:
        print(f"\n*** WARNING: GPUs {failed} failed ***")
        sys.exit(1)


# %% Step 3: Merge shards

def merge_shards(data_path: Path, output_dir: Path, num_gpus: int) -> None:
    """Merge per-GPU shard files into all.safetensors + per-source splits."""
    import torch
    from safetensors.torch import load_file, save_file

    print("\n=== Merging shards ===\n")
    t0 = time.time()

    df = pl.read_parquet(data_path)
    shard_dir = output_dir / "embeddings" / "shards"
    emb_dir = output_dir / "embeddings"
    by_source_dir = emb_dir / "by_source"

    # Concatenate all GPU shards
    all_embeds = []
    for gpu_id in range(num_gpus):
        shard_path = shard_dir / f"gpu{gpu_id}.safetensors"
        if shard_path.exists():
            data = load_file(str(shard_path))
            all_embeds.append(data["embeddings"])

    if not all_embeds:
        print("  No shards found, skipping merge")
        return

    full_tensor = torch.cat(all_embeds, dim=0)
    assert full_tensor.shape[0] == len(df), (
        f"Embedding count {full_tensor.shape[0]} != data rows {len(df)}"
    )

    # Save concatenated file
    all_path = emb_dir / "all.safetensors"
    save_file({"embeddings": full_tensor}, str(all_path))
    print(f"  Saved {all_path} — shape {list(full_tensor.shape)}")

    # Save per-source splits
    sources = df["dataset_source"].to_list()
    unique_sources = df["dataset_source"].unique().sort().to_list()

    for source in unique_sources:
        safe_name = source.replace("/", "_").replace(" ", "_")
        source_dir = by_source_dir / safe_name
        source_dir.mkdir(parents=True, exist_ok=True)

        mask = [s == source for s in sources]
        indices = [i for i, m in enumerate(mask) if m]
        source_embeds = full_tensor[indices]

        save_file(
            {"embeddings": source_embeds},
            str(source_dir / "embeddings.safetensors"),
        )

        source_meta = df.filter(pl.col("dataset_source") == source).select(
            "id", "dataset_source", "user_prompt", "think_text", "response_text",
            "num_turns"
        )
        source_meta.write_parquet(source_dir / "metadata.parquet")

    # Save full metadata
    meta_df = df.select(
        "id", "dataset_source", "user_prompt", "think_text", "response_text",
        "num_turns"
    )
    meta_df.write_parquet(emb_dir / "metadata.parquet")
    print(f"  Saved metadata.parquet — {len(meta_df)} rows")

    timings["3_merge"] = time.time() - t0
    print(f"  [TIME] Merge: {timings['3_merge']:.1f}s")


# %% Main

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = 0 if args.full else args.sample_size

    mp.set_start_method("spawn", force=True)

    run_config_path = output_dir / "run_config.json"

    if args.resume:
        if run_config_path.exists():
            saved = json.loads(run_config_path.read_text())
            if saved.get("sample_size") != sample_size:
                print(f"ERROR: Resume mismatch — saved sample_size={saved['sample_size']}, "
                      f"requested sample_size={sample_size}")
                print("Either use matching --sample-size or run without --resume")
                sys.exit(1)
    else:
        for stale_dir in ["embeddings", "checkpoints"]:
            stale_path = output_dir / stale_dir
            if stale_path.exists():
                print(f"  Cleaning stale output: {stale_path}")
                shutil.rmtree(stale_path)

    # Step 1: Load and sample data
    data_path = output_dir / "extracted_data.parquet"
    if args.resume and data_path.exists():
        print(f"=== Resuming: using existing {data_path} ===\n")
    else:
        data_path = load_and_sample(sample_size, output_dir, random_seed=args.random_seed)

    # Step 1b: Filter by token length (drop empty + too-long rows)
    dropped_path = output_dir / "dropped_rows.parquet"
    if args.resume and dropped_path.exists():
        print(f"=== Resuming: using existing filtered data (dropped_rows.parquet exists) ===\n")
    else:
        data_path = filter_by_length(data_path, output_dir, args.max_length)

    # Save run config
    run_config = {
        "sample_size": sample_size,
        "num_gpus": args.num_gpus,
        "max_length": args.max_length,
        "backend": "vllm",
        "random_seed": args.random_seed,
    }
    run_config_path.write_text(json.dumps(run_config, indent=2))

    # Step 2: Embed (single pass — concatenated texts)
    print(f"=== Embedding with vLLM ({args.num_gpus} GPUs, "
          f"max_length={args.max_length}) ===\n")
    t0 = time.time()
    run_embedding(data_path, args.num_gpus, args.max_length, output_dir, args.resume)
    timings["2_embed"] = time.time() - t0

    # Step 3: Merge shards
    merge_shards(data_path, output_dir, args.num_gpus)

    # Timing summary
    df = pl.read_parquet(data_path)
    print("\n=== Timing Summary ===\n")
    print(f"  Sampling + extraction: {timings.get('1_sampling', 0):.1f}s")
    print(f"  Filter by length:      {timings.get('1b_filter', 0):.1f}s")
    print(f"  Embed (single pass):   {timings['2_embed']:.1f}s")
    print(f"  Merge shards:          {timings.get('3_merge', 0):.1f}s")
    total = sum(timings.values())
    print(f"  TOTAL:                 {total:.1f}s")
    print(f"\n  {len(df)} rows embedded, {df['dataset_source'].n_unique()} sources")
    print("\n=== Embedding complete ===")
    print(f"\nOutput: {output_dir / 'embeddings'}")
    print("Next: python analyze_embeddings.py")


if __name__ == "__main__":
    main()
