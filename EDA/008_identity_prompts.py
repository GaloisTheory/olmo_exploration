# %% Setup
import os
from pathlib import Path

if not os.environ.get("HF_TOKEN"):
    _start = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
    for _dir in [_start, *_start.parents]:
        _secrets_path = _dir / ".secrets"
        if _secrets_path.exists():
            with open(_secrets_path) as _f:
                for _line in _f:
                    if _line.strip() and "=" in _line and not _line.startswith("#"):
                        _key, _val = _line.strip().split("=", 1)
                        os.environ[_key] = _val
            break

import polars as pl

SFT_PARQUET = "hf://datasets/allenai/Dolci-Think-SFT-32B/**/*.parquet"

print("Scanning SFT parquet files...")
sft_lf = pl.scan_parquet(SFT_PARQUET)
print("Collecting full SFT dataset...")
sft_df = sft_lf.collect()
print(f"Collected: {len(sft_df):,} rows\n")

# %% Identity prompts
# --- hard_coded identity prompts (parsed from id column) ---
print("=== hard_coded Identity Prompts ===\n")

identity_df = (
    sft_df
    .with_columns(
        pl.col("id")
        .str.replace(r"_[a-f0-9]{8}-[a-f0-9-]+.*$", "")  # strip UUID suffixes
        .str.replace(r"_[a-z0-9]{7}$", "")                 # strip 7-char alphanumeric IDs
        .str.replace(r"_\d+$", "")                          # strip numeric index suffixes
        .alias("dataset_name")
    )
    .filter(pl.col("dataset_name") == "hard_coded")
)

print(f"Matched rows: {len(identity_df):,}\n")
for i, row in enumerate(identity_df.iter_rows(named=True), start=1):
    msgs = row["messages"]
    print(f"Identity Prompt {i}:")
    print(f"  source={row.get('source', '')}  id={row.get('id', '')}")
    for msg in msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"  [{role}] ({len(content)} chars)")
        print(f"  {content}")
    print()

# --- Messages mentioning OLMo ---
print("=== Messages Mentioning 'OLMo' ===\n")

identity_matches = (
    sft_df
    .explode("messages")
    .with_columns(
        pl.col("messages").struct.field("role").alias("role"),
        pl.col("messages").struct.field("content").alias("content"),
    )
    .filter(pl.col("content").str.contains("(?i)OLMo"))
)

print(f"Rows mentioning 'OLMo': {len(identity_matches):,}\n")
if len(identity_matches) > 0:
    print("Source breakdown:")
    id_sources = identity_matches.group_by("source").len().sort("len", descending=True)
    for row in id_sources.iter_rows(named=True):
        print(f"  {row['source']}: {row['len']}")

    print()
    for i, row in enumerate(identity_matches.head(3).iter_rows(named=True), 1):
        print(f"Example {i} (source: {row['source']}):")
        print(f"  {row['content'][:300]}...")
        print()
else:
    print("No OLMo mentions found.")

# %% System prompts
print("=== System Prompts ===\n")

sys_df = (
    sft_df
    .explode("messages")
    .with_columns(
        pl.col("messages").struct.field("role").alias("role"),
        pl.col("messages").struct.field("content").alias("content"),
    )
    .filter(pl.col("role") == "system")
)
n_with_sys = sys_df["id"].n_unique()
print(f"Rows with a system prompt: {n_with_sys:,} / {len(sft_df):,} ({100*n_with_sys/len(sft_df):.1f}%)\n")

print("By source:")
for row in sys_df.group_by("source").agg(pl.col("id").n_unique().alias("convs")).sort("convs", descending=True).iter_rows(named=True):
    print(f"  {row['source']:70s} {row['convs']:>7,d}")

print(f"\nUnique system prompts: {sys_df['content'].n_unique():,}\n")
print("Most common:")
for row in sys_df.group_by("content").len().sort("len", descending=True).head(15).iter_rows(named=True):
    preview = row["content"][:120].replace("\n", " ")
    print(f"  [{row['len']:>5,d}x] {preview}{'...' if len(row['content']) > 120 else ''}")
