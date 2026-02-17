"""Embed Dolci-Think-SFT-7B think/response texts with Qwen3-Embedding-0.6B.

Multi-GPU script that:
1. Stratified-samples from the SFT dataset (N rows per source)
2. Extracts think blocks, response texts, and user prompts
3. Embeds both using Qwen3-Embedding-0.6B (last-token pooling, L2 normalized)
4. Saves embeddings as safetensors + metadata as parquet
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

# %% Config
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024

timings = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed Dolci-Think-SFT-7B dataset with Qwen3-Embedding-0.6B"
    )
    parser.add_argument("--sample-size", type=int, default=625,
                        help="Rows per source (0 = all). Default 625 (~10K total)")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs to use (default: 8)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size per GPU (default: 256)")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Max token length (default: 8192)")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output base directory (default: ./output)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name — creates output/<run-name>/ subdirectory")
    parser.add_argument("--full", action="store_true",
                        help="Process entire dataset (no sampling)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    return parser.parse_args()


# %% Step 1: Data loading and extraction

def load_and_sample(sample_size: int, output_dir: Path) -> Path:
    """Load dataset, stratified sample, extract think/response/user_prompt, save to parquet."""
    from datasets import load_dataset

    if sample_size > 0:
        print(f"=== Sampling {sample_size} rows/source from Dolci-Think-SFT-7B ===\n")
    else:
        print("=== Loading FULL Dolci-Think-SFT-7B dataset ===\n")
    t0 = time.time()

    # Load via HF datasets library (reliable auth, cached at $HF_HOME/datasets/)
    print("  Loading dataset via HF datasets library...")
    ds = load_dataset("allenai/Dolci-Think-SFT-7B", split="train")
    df = pl.from_arrow(ds.data.table)  # zero-copy to Polars
    print(f"  Loaded {len(df)} rows from HF")

    if sample_size > 0:
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

    # Extract assistant messages, user prompts, and split think/response
    records = []
    for row in sampled_df.iter_rows(named=True):
        source = row["dataset_source"]
        msgs = row["messages"]

        # Get the last assistant message
        asst_msg = None
        asst_idx = None
        for idx in range(len(msgs) - 1, -1, -1):
            if msgs[idx]["role"] == "assistant":
                asst_msg = msgs[idx]["content"]
                asst_idx = idx
                break
        if not asst_msg:
            continue

        # Get the last user message before the assistant message
        user_prompt = ""
        for idx in range(asst_idx - 1, -1, -1):
            if msgs[idx]["role"] == "user":
                user_prompt = msgs[idx]["content"]
                break

        # Split into think block and final response
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
        })

    extract_df = pl.DataFrame(records)
    print(f"  Extracted {len(extract_df)} records")

    # Print per-source counts
    source_counts = extract_df.group_by("dataset_source").len().sort("len", descending=True)
    for row in source_counts.iter_rows(named=True):
        pct = 100 * row["len"] / len(extract_df)
        print(f"    {row['dataset_source']:45s} {row['len']:5d} ({pct:5.1f}%)")

    # Save extracted data to parquet for workers to read
    data_path = output_dir / "extracted_data.parquet"
    extract_df.write_parquet(data_path)
    print(f"\n  Saved extracted data to {data_path}")

    timings["1_sampling"] = time.time() - t0
    print(f"  [TIME] Sampling + extraction: {timings['1_sampling']:.1f}s\n")
    return data_path


# %% Step 2: Multi-GPU embedding

def embed_worker(
    gpu_id: int,
    data_path: str,
    shard_start: int,
    shard_end: int,
    text_column: str,
    batch_size: int,
    max_length: int,
    output_path: str,
    checkpoint_dir: str,
    resume: bool,
):
    """Worker process: load model on one GPU, embed a shard of texts."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from safetensors.torch import save_file
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    print(f"[GPU {gpu_id}] Loading {EMBEDDING_MODEL} for '{text_column}' "
          f"(rows {shard_start}-{shard_end})")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, padding_side="left")
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL,
        dtype=torch.bfloat16,
    ).cuda().eval()

    # Load data shard
    df = pl.read_parquet(data_path)
    shard = df.slice(shard_start, shard_end - shard_start)
    texts = shard[text_column].to_list()

    # Replace empty strings with placeholder to avoid zero-length inputs
    texts = [t if t else "[empty]" for t in texts]

    # Check for checkpoint
    ckpt_path = Path(checkpoint_dir) / f"gpu{gpu_id}_{text_column}.pt"
    start_batch = 0
    all_embeddings = []

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        all_embeddings = [ckpt["embeddings"]]
        start_batch = ckpt["next_batch"]
        print(f"[GPU {gpu_id}] Resuming from batch {start_batch} "
              f"({ckpt['embeddings'].shape[0]} embeddings loaded)")

    # Embed in batches
    current_batch_size = batch_size
    n_batches = (len(texts) + current_batch_size - 1) // current_batch_size
    pbar = tqdm(
        total=len(texts),
        desc=f"GPU {gpu_id} {text_column}",
        position=gpu_id,
        initial=start_batch * current_batch_size,
    )

    batch_idx = start_batch
    i = start_batch * batch_size  # use original batch_size for positioning
    while i < len(texts):
        batch_texts = texts[i:i + current_batch_size]

        try:
            with torch.no_grad():
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to("cuda")

                outputs = model(**encoded)
                hidden = outputs.last_hidden_state  # [B, seq_len, dim]

                # Last-token pooling: find last non-padding token per row
                attention_mask = encoded["attention_mask"]  # [B, seq_len]
                # Sum mask to get sequence lengths, subtract 1 for 0-indexed
                seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                last_token_embeds = hidden[batch_indices, seq_lengths]  # [B, dim]

                # L2 normalize
                last_token_embeds = torch.nn.functional.normalize(
                    last_token_embeds, p=2, dim=1
                )

                all_embeddings.append(last_token_embeds.cpu().float())

            pbar.update(len(batch_texts))
            i += len(batch_texts)
            batch_idx += 1

            # Reset batch size if we had reduced it
            current_batch_size = batch_size

            # Periodic checkpoint (every 20 batches)
            if batch_idx % 20 == 0:
                cat_emb = torch.cat(all_embeddings, dim=0)
                torch.save(
                    {"embeddings": cat_emb, "next_batch": batch_idx},
                    ckpt_path,
                )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            old_bs = current_batch_size
            current_batch_size = max(1, current_batch_size // 2)
            print(f"\n[GPU {gpu_id}] OOM at batch size {old_bs}, "
                  f"reducing to {current_batch_size}")
            continue  # retry same batch with smaller size

    pbar.close()

    # Save final embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)
    save_file({"embeddings": final_embeddings}, output_path)
    print(f"[GPU {gpu_id}] Saved {final_embeddings.shape[0]} embeddings "
          f"({text_column}) → {output_path}")

    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()


def run_embedding_pass(
    data_path: Path,
    text_column: str,
    num_gpus: int,
    batch_size: int,
    max_length: int,
    output_dir: Path,
    resume: bool,
) -> None:
    """Run one embedding pass (think or response) across all GPUs."""
    df = pl.read_parquet(data_path)
    n_rows = len(df)

    shard_dir = output_dir / "embeddings" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Split data into contiguous shards
    shard_size = (n_rows + num_gpus - 1) // num_gpus
    procs = []
    for gpu_id in range(num_gpus):
        shard_start = gpu_id * shard_size
        shard_end = min(shard_start + shard_size, n_rows)
        if shard_start >= n_rows:
            break

        output_path = str(shard_dir / f"{text_column}_gpu{gpu_id}.safetensors")

        # Skip only if resuming and shard is already complete
        if resume and Path(output_path).exists():
            print(f"[GPU {gpu_id}] Shard already exists (resume), skipping: {output_path}")
            continue

        p = mp.Process(
            target=embed_worker,
            args=(
                gpu_id, str(data_path), shard_start, shard_end,
                text_column, batch_size, max_length,
                output_path, str(checkpoint_dir), resume,
            ),
        )
        p.start()
        procs.append((gpu_id, p))

    # Wait for all workers
    failed = []
    for gpu_id, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(gpu_id)
            print(f"[ERROR] GPU {gpu_id} exited with code {p.exitcode}")

    if failed:
        print(f"\n*** WARNING: GPUs {failed} failed for '{text_column}' ***")
        sys.exit(1)


def merge_shards(data_path: Path, output_dir: Path, num_gpus: int) -> None:
    """Merge per-GPU shard files into final outputs."""
    import torch
    from safetensors.torch import load_file, save_file

    print("\n=== Merging shards ===\n")
    t0 = time.time()

    df = pl.read_parquet(data_path)
    shard_dir = output_dir / "embeddings" / "shards"
    emb_dir = output_dir / "embeddings"
    by_source_dir = emb_dir / "by_source"

    for text_column in ["think_text", "response_text"]:
        # Concatenate all GPU shards
        all_embeds = []
        for gpu_id in range(num_gpus):
            shard_path = shard_dir / f"{text_column}_gpu{gpu_id}.safetensors"
            if shard_path.exists():
                data = load_file(str(shard_path))
                all_embeds.append(data["embeddings"])

        if not all_embeds:
            print(f"  No shards found for {text_column}, skipping")
            continue

        full_tensor = torch.cat(all_embeds, dim=0)
        short_name = text_column.replace("_text", "")

        # Save concatenated file
        full_path = emb_dir / f"all_{short_name}.safetensors"
        save_file({"embeddings": full_tensor}, str(full_path))
        print(f"  Saved {full_path} — shape {list(full_tensor.shape)}")

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
                str(source_dir / f"{short_name}.safetensors"),
            )

            # Metadata parquet for this source (includes text columns)
            source_meta = df.filter(pl.col("dataset_source") == source).select(
                "id", "dataset_source", "user_prompt", "think_text", "response_text"
            )
            source_meta.write_parquet(source_dir / "metadata.parquet")

    # Save full metadata (includes text columns for analysis script)
    meta_df = df.select("id", "dataset_source", "user_prompt", "think_text", "response_text")
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

    # --full overrides sample_size to 0 (process everything)
    sample_size = 0 if args.full else args.sample_size

    mp.set_start_method("spawn", force=True)

    run_config_path = output_dir / "run_config.json"

    if args.resume:
        # Validate resume config matches
        if run_config_path.exists():
            saved = json.loads(run_config_path.read_text())
            if saved.get("sample_size") != sample_size:
                print(f"ERROR: Resume mismatch — saved sample_size={saved['sample_size']}, "
                      f"requested sample_size={sample_size}")
                print("Either use matching --sample-size or run without --resume")
                sys.exit(1)
    else:
        # Auto-cleanup stale output when not resuming
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
        data_path = load_and_sample(sample_size, output_dir)

    # Save run config for resume validation
    run_config = {
        "sample_size": sample_size,
        "num_gpus": args.num_gpus,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    run_config_path.write_text(json.dumps(run_config, indent=2))

    # Step 2: Embed think texts
    print(f"=== Embedding think texts ({args.num_gpus} GPUs, "
          f"batch_size={args.batch_size}, max_length={args.max_length}) ===\n")
    t0 = time.time()
    run_embedding_pass(
        data_path, "think_text", args.num_gpus,
        args.batch_size, args.max_length, output_dir, args.resume,
    )
    timings["2a_embed_think"] = time.time() - t0

    # Step 3: Embed response texts
    print(f"\n=== Embedding response texts ({args.num_gpus} GPUs) ===\n")
    t0 = time.time()
    run_embedding_pass(
        data_path, "response_text", args.num_gpus,
        args.batch_size, args.max_length, output_dir, args.resume,
    )
    timings["2b_embed_response"] = time.time() - t0

    # Step 4: Merge shards
    merge_shards(data_path, output_dir, args.num_gpus)

    # Timing summary
    df = pl.read_parquet(data_path)
    print("\n=== Timing Summary ===\n")
    print(f"  Sampling + extraction: {timings.get('1_sampling', 0):.1f}s")
    print(f"  Embed think texts:     {timings['2a_embed_think']:.1f}s")
    print(f"  Embed response texts:  {timings['2b_embed_response']:.1f}s")
    print(f"  Merge shards:          {timings.get('3_merge', 0):.1f}s")
    total = sum(timings.values())
    print(f"  TOTAL:                 {total:.1f}s")
    print(f"\n  {len(df)} rows embedded, {df['dataset_source'].n_unique()} sources")
    print("\n=== Embedding complete ===")
    print(f"\nOutput: {output_dir / 'embeddings'}")
    print("Next: python analyze_embeddings.py")


if __name__ == "__main__":
    main()
