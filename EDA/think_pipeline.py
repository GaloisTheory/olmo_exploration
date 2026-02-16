# %% [markdown]
# # OLMo 3 32B-Think: Full Training Pipeline
#
# Tracing every dataset the **Think** variant touched, from raw web crawl to final RL policy.
#
# ```
# dolma3_mix (6T tokens)          ← Stage 1: Pretraining
#         ↓
# dolma3_dolmino_mix (100B)       ← Stage 2: Mid-training (high-quality)
#         ↓
# dolma3_longmino_mix (50B)       ← Stage 3: Long-context
#         ↓
# Dolci-Think-SFT-32B (2.25M ex)  ← Post-training: SFT
#         ↓
# Dolci-Think-DPO-32B (200K pairs) ← Post-training: DPO
#         ↓
# Dolci-Think-RL-32B (102K prompts)← Post-training: RL (RLVR)
#         ↓
#     OLMo-3-32B-Think              ← Final model
# ```

# %% Imports
import os

# Load HF_TOKEN from .secrets if not already in environment (needed for polars hf:// access)
if not os.environ.get("HF_TOKEN"):
    from pathlib import Path
    # Works both as script (__file__) and in Jupyter (search upward from cwd)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import polars as pl
import wandb
from huggingface_hub import HfApi
from requests.exceptions import ReadTimeout

# HuggingFace parquet paths — polars reads these lazily (column pruning + predicate pushdown)
SFT_PARQUET = "hf://datasets/allenai/Dolci-Think-SFT-32B/**/*.parquet"
DPO_PARQUET = "hf://datasets/allenai/Dolci-Think-DPO-32B/**/*.parquet"
RL_PARQUET = "hf://datasets/allenai/Dolci-Think-RL-32B/**/*.parquet"

 # %% [markdown]
# ## Training Pipeline Definition
#
# Every dataset in the chain, with HF repo IDs and metadata.

# %%
THINK_PIPELINE = [
    {
        "stage": "Stage 1: Pretraining",
        "dataset": "allenai/dolma3_pool",
        "tokens": "~6T",
        "description": "Web crawl, code, academic papers, books. Raw pool before mixing.",
        "format": "text",
    },
    {
        "stage": "Stage 2: Mid-training (Dolmino)",
        "dataset": "allenai/dolma3_dolmino_mix-100B-1025",
        "tokens": "100B",
        "description": "High-quality mix: math, code, QA, reasoning traces, curated web.",
        "format": "text",
    },
    {
        "stage": "Stage 3: Long-context (Longmino)",
        "dataset": "allenai/dolma3_longmino_mix-50B-1025",
        "tokens": "50B",
        "description": "Long-context documents for extending context window.",
        "format": "text",
    },
    {
        "stage": "Post-training: SFT",
        "dataset": "allenai/Dolci-Think-SFT-32B",
        "examples": "2.25M",
        "size": "36.3 GB",
        "description": "Supervised fine-tuning with <think> reasoning tags. "
                       "Sources: OpenThoughts3, synthetic math/code, persona IF, WildChat.",
        "format": "messages (chat)",
    },
    {
        "stage": "Post-training: DPO",
        "dataset": "allenai/Dolci-Think-DPO-32B",
        "examples": "200K pairs",
        "description": "Preference pairs via Delta Learning heuristic. "
                       "Chosen/rejected responses across math, code, chat.",
        "format": "prompt + chosen/rejected",
    },
    {
        "stage": "Post-training: RL (RLVR)",
        "dataset": "allenai/Dolci-Think-RL-32B",
        "examples": "102K prompts",
        "description": "Reinforcement learning from verifiable rewards. "
                       "Math, code, instruction following with ground-truth answers.",
        "format": "prompt + ground_truth + rollouts",
    },
]

# WandB project for post-training runs (SFT, DPO, RL)
WANDB_POSTTRAINING = {
    "entity": "ai2-llm",
    "project": "Olmo-3-32B-Think",
    "report": "https://wandb.ai/ai2-llm/Olmo-3-32B-Think",  # TBD: no public 32B report yet
}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# Print the pipeline
for i, stage in enumerate(THINK_PIPELINE):
    connector = "  ↓" if i < len(THINK_PIPELINE) - 1 else "  → OLMo-3-32B-Think"
    tokens = stage.get("tokens", stage.get("examples", ""))
    print(f"  [{stage['stage']}]")
    print(f"    {stage['dataset']}  ({tokens})")
    print(f"    {stage['description'][:80]}")
    print(connector)

# %% [markdown]
# ---
# ## Stage 4: SFT — Dolci-Think-SFT-32B (Deep Dive)
#
# The first post-training stage. All responses include `<think>...</think>` reasoning blocks.
# 2.25M examples from diverse sources. Let's explore the dataset structure in depth.
# %%

# %% Cell A: Dataset card from HuggingFace API
print("=== Dolci-Think-SFT-32B: Dataset Card ===\n")
hf_api = HfApi()
ds_info = hf_api.dataset_info("allenai/Dolci-Think-SFT-32B")

print(f"  Dataset:       {ds_info.id}")
print(f"  Author:        {ds_info.author}")
print(f"  Last modified: {ds_info.last_modified}")
print(f"  Downloads:     {getattr(ds_info, 'downloads', 'N/A')}")
print(f"  Likes:         {getattr(ds_info, 'likes', 'N/A')}")
print(f"  Tags:          {', '.join(ds_info.tags[:10]) if ds_info.tags else 'none'}")
# Card data often has dataset_size, download_size
card = getattr(ds_info, 'card_data', None)
if card:
    for field in ['dataset_size', 'download_size', 'train_num_examples']:
        val = getattr(card, field, None)
        if val is not None:
            print(f"  {field}: {val}")

# Schema from parquet metadata (no download needed)
sft_lf = pl.scan_parquet(SFT_PARQUET)
print(f"\n  Schema: {sft_lf.collect_schema()}")
# %%
# Parse the `id` column to extract original dataset names.
# Format: "{dataset_name}_{index_or_uuid}" — strip the trailing suffix to get the dataset.
print("=== SFT: Original Dataset Distribution (parsed from id column) ===\n")
sft_dataset_dist = (
    sft_lf
    .with_columns(
        pl.col("id")
        .str.replace(r"_[a-f0-9]{8}-[a-f0-9-]+.*$", "")  # strip UUID suffixes
        .str.replace(r"_[a-z0-9]{7}$", "")                 # strip 7-char alphanumeric IDs (e.g. coconot)
        .str.replace(r"_\d+$", "")                          # strip numeric index suffixes
        .alias("dataset_name")
    )
    .group_by("dataset_name")
    .len()
    .sort("len", descending=True)
    .collect()
)
ds_total = sft_dataset_dist["len"].sum()
print(f"  {'Dataset (parsed from id)':70s} {'Count':>9s}  {'%':>6s}")
print(f"  {'─' * 70} {'─' * 9}  {'─' * 6}")
for row in sft_dataset_dist.iter_rows(named=True):
    pct = 100.0 * row["len"] / ds_total
    print(f"  {row['dataset_name']:70s} {row['len']:9,d}  {pct:5.1f}%")
print(f"\n  {len(sft_dataset_dist)} unique original datasets, {ds_total:,} total rows")
# %%
# Collect full SFT dataset once to avoid HuggingFace rate limits on repeated queries
print("  Collecting full SFT dataset (single HTTP round-trip)...")
sft_df = sft_lf.collect()
sft_lf = sft_df.lazy()  # re-wrap as lazy over in-memory data for downstream cells
print(f"  Collected: {len(sft_df):,} rows")

# %% Cell B: Source distribution (exact — polars reads only the source column)
print("=== SFT Source Distribution (full dataset) ===\n")
sft_source_dist = (
    sft_lf
    .group_by("source")
    .len()
    .sort("len", descending=True)
    .collect()
)
total = sft_source_dist["len"].sum()
print(f"  {'Source':55s} {'Count':>9s}  {'%':>6s}  Bar")
print(f"  {'─' * 55} {'─' * 9}  {'─' * 6}  {'─' * 30}")
for row in sft_source_dist.iter_rows(named=True):
    pct = 100.0 * row["len"] / total
    bar = "█" * int(pct / 2)
    print(f"  {row['source']:55s} {row['len']:9,d}  {pct:5.1f}%  {bar}")
print(f"\n  Total rows: {total:,}")

# Bar chart
fig, ax = plt.subplots(figsize=(12, max(4, len(sft_source_dist) * 0.4)))
labels = sft_source_dist["source"].to_list()
counts = sft_source_dist["len"].to_list()
pcts = [100.0 * c / total for c in counts]
cmap = plt.cm.tab20(range(len(labels)))
bars = ax.barh(labels, pcts, color=cmap)
ax.set_xlabel("% of dataset (full 2.25M rows)")
ax.set_title("Dolci-Think-SFT-32B: Source Distribution")
ax.invert_yaxis()
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=8)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "think_sft_source_dist.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.close(fig)

# %% Cell C: Message structure stats (full dataset via polars)
print("=== SFT Message Structure Stats (full dataset) ===\n")

# Turns per conversation (no need to read message content — just list length)
turn_stats = (
    sft_lf
    .select(pl.col("messages").list.len().alias("num_turns"))
    .select(
        pl.col("num_turns").count().alias("count"),
        pl.col("num_turns").mean().alias("mean"),
        pl.col("num_turns").median().alias("median"),
        pl.col("num_turns").quantile(0.05).alias("p5"),
        pl.col("num_turns").quantile(0.95).alias("p95"),
        pl.col("num_turns").min().alias("min"),
        pl.col("num_turns").max().alias("max"),
    )
    .collect()
)
print(f"  Turns per conversation:")
row = turn_stats.row(0, named=True)
print(f"    count={row['count']:,}  mean={row['mean']:.1f}  median={row['median']:.0f}  "
      f"p5={row['p5']:.0f}  p95={row['p95']:.0f}  min={row['min']}  max={row['max']}")

# Explode messages → one row per message, compute stats by role
msg_lf = (
    sft_lf
    .explode("messages")
    .with_columns(
        pl.col("messages").struct.field("role").alias("role"),
        pl.col("messages").struct.field("content").str.len_chars().alias("content_len"),
        pl.col("messages").struct.field("content").str.contains("<think>").alias("has_think_open"),
        pl.col("messages").struct.field("content").str.contains("</think>").alias("has_think_close"),
    )
    .drop("messages")
)

# Length stats by role
len_stats = (
    msg_lf
    .group_by("role")
    .agg(
        pl.col("content_len").count().alias("count"),
        pl.col("content_len").mean().alias("mean"),
        pl.col("content_len").median().alias("median"),
        pl.col("content_len").quantile(0.05).alias("p5"),
        pl.col("content_len").quantile(0.95).alias("p95"),
        pl.col("content_len").min().alias("min"),
        pl.col("content_len").max().alias("max"),
    )
    .sort("role")
    .collect()
)
for row in len_stats.iter_rows(named=True):
    print(f"\n  {row['role'].capitalize()} message length (chars):")
    print(f"    count={row['count']:,}  mean={row['mean']:.0f}  median={row['median']:.0f}  "
          f"p5={row['p5']:.0f}  p95={row['p95']:.0f}  min={row['min']}  max={row['max']}")

# Think tag presence in assistant messages
think_stats = (
    msg_lf
    .filter(pl.col("role") == "assistant")
    .select(
        pl.len().alias("total_asst"),
        (pl.col("has_think_open") & pl.col("has_think_close")).sum().alias("think_count"),
    )
    .collect()
)
ts = think_stats.row(0, named=True)
print(f"\n  <think> tag presence: {ts['think_count']:,}/{ts['total_asst']:,} "
      f"({100 * ts['think_count'] / max(ts['total_asst'], 1):.1f}%) of assistant messages")

# Think block vs response length (use str.split_exact on </think>)
think_len_stats = (
    msg_lf
    .filter((pl.col("role") == "assistant") & pl.col("has_think_open") & pl.col("has_think_close"))
    .with_columns(
        # Everything before </think> + the tag itself
        pl.col("content_len").alias("total_len"),
    )
    # Recompute from the raw content for think/response split
)
# For think/response split, read the content column and split at </think>
think_split_stats = (
    sft_lf
    .explode("messages")
    .with_columns(
        pl.col("messages").struct.field("role").alias("role"),
        pl.col("messages").struct.field("content").alias("content"),
    )
    .filter(pl.col("role") == "assistant")
    .filter(pl.col("content").str.contains("<think>") & pl.col("content").str.contains("</think>"))
    .with_columns(
        # Split at first </think> → struct{field_0: before, field_1: after}
        pl.col("content").str.split_exact("</think>", 1).alias("parts"),
    )
    .with_columns(
        (pl.col("parts").struct.field("field_0").str.len_chars() + 8).alias("think_block_len"),  # +8 for "</think>"
        pl.col("parts").struct.field("field_1").str.strip_chars().str.len_chars().alias("response_after_think_len"),
    )
    .select(
        pl.col("think_block_len").mean().alias("think_mean"),
        pl.col("think_block_len").median().alias("think_median"),
        pl.col("think_block_len").quantile(0.05).alias("think_p5"),
        pl.col("think_block_len").quantile(0.95).alias("think_p95"),
        pl.col("response_after_think_len").mean().alias("resp_mean"),
        pl.col("response_after_think_len").median().alias("resp_median"),
        pl.col("response_after_think_len").quantile(0.05).alias("resp_p5"),
        pl.col("response_after_think_len").quantile(0.95).alias("resp_p95"),
        pl.col("think_block_len").count().alias("count"),
    )
    .collect()
)
r = think_split_stats.row(0, named=True)
print(f"\n  Think block length (chars):")
print(f"    count={r['count']:,}  mean={r['think_mean']:.0f}  median={r['think_median']:.0f}  "
      f"p5={r['think_p5']:.0f}  p95={r['think_p95']:.0f}")
print(f"  Response-after-think length (chars):")
print(f"    count={r['count']:,}  mean={r['resp_mean']:.0f}  median={r['resp_median']:.0f}  "
      f"p5={r['resp_p5']:.0f}  p95={r['resp_p95']:.0f}")

# %% Cell D0: Load 5 identity prompts
# %%
IDENTITY_PROMPT_COUNT = 5
print(f"=== SFT: {IDENTITY_PROMPT_COUNT} Identity Prompts ===\n")

# Use the exact same id parsing as the distribution cell, then select hard_coded.
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
    .head(IDENTITY_PROMPT_COUNT)
)

print(f"  Matched rows: {len(identity_df):,}")
identity_user_lens = []
identity_asst_lens = []
for i, row in enumerate(identity_df.iter_rows(named=True), start=1):
    msgs = row["messages"]
    user_len = sum(len(m.get("content", "")) for m in msgs if m.get("role") == "user")
    asst_len = sum(len(m.get("content", "")) for m in msgs if m.get("role") == "assistant")
    identity_user_lens.append(user_len)
    identity_asst_lens.append(asst_len)

    print(f"\n  Identity Prompt {i}:")
    print(f"    source={row.get('source', '')}  id={row.get('id', '')}")
    for msg in msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"    [{role}] ({len(content)} chars)")
        print(f"    {content}")

if identity_user_lens:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = list(range(1, len(identity_user_lens) + 1))
    ax.bar(x, identity_user_lens, label="user", color="#4c72b0")
    ax.bar(x, identity_asst_lens, bottom=identity_user_lens, label="assistant", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Prompt {i}" for i in x], rotation=0)
    ax.set_ylabel("chars")
    ax.set_title("Identity Prompt Examples: User vs Assistant Characters")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "think_identity_prompt_response_lengths.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.close(fig)

# %% Cell D: Examples per source (filter in-memory — sft_df already collected in Cell A)
# %%
EXAMPLES_PER_SOURCE = 5
print(f"=== SFT: {EXAMPLES_PER_SOURCE} Examples Per Source ===\n")

for source_name in sft_source_dist["source"]:
    examples = (
        sft_df
        .filter(pl.col("source") == source_name)
        .head(EXAMPLES_PER_SOURCE)
    )
    print(f"─── Source: {source_name} ({len(examples)} examples) ───")
    for i, row in enumerate(examples.iter_rows(named=True), start=1):
        msgs = row["messages"]
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        print(f"  Example {i}:")
        print(f"    [user] ({len(user_msg)} chars)")
        print(f"    {user_msg[:300]}{'...' if len(user_msg) > 300 else ''}")

        if "<think>" in asst_msg and "</think>" in asst_msg:
            think_end = asst_msg.index("</think>") + len("</think>")
            think_block = asst_msg[:think_end]
            response = asst_msg[think_end:].strip()
            print(f"    [assistant - thinking] ({len(think_block)} chars)")
            print(f"    {think_block[:300]}{'...' if len(think_block) > 300 else ''}")
            print(f"    [assistant - response] ({len(response)} chars)")
            print(f"    {response[:300]}{'...' if len(response) > 300 else ''}")
        else:
            print(f"    [assistant] ({len(asst_msg)} chars)")
            print(f"    {asst_msg[:400]}{'...' if len(asst_msg) > 400 else ''}")
        print()
    print()

# %% Cell D2: Search for OLMo identity prompts
print("=== SFT: Searching for OLMo Identity Prompts ===\n")

# Check for null/empty source (identity prompts might lack a source tag)
null_source = sft_df.filter(pl.col("source").is_null() | (pl.col("source") == ""))
print(f"  Rows with null/empty source: {len(null_source):,}")

# For 32B, identity prompts ARE included in the parquet (unlike 7B)
EXPECTED_TOTAL = 2_253_684
print(f"  Expected (dataset card): {EXPECTED_TOTAL:,}")
print(f"  Collected (HF):          {len(sft_df):,}")
print(f"  Difference:              {EXPECTED_TOTAL - len(sft_df):,}  (58 identity prompts included in 32B parquet)")

# Search for any mention of OLMo in messages anyway
identity_matches = (
    sft_df
    .explode("messages")
    .with_columns(
        pl.col("messages").struct.field("role").alias("role"),
        pl.col("messages").struct.field("content").alias("content"),
    )
    .filter(pl.col("content").str.contains("(?i)OLMo"))
)
print(f"\n  Rows mentioning 'OLMo' in any message: {len(identity_matches):,}")
if len(identity_matches) > 0:
    id_sources = identity_matches.group_by("source").len().sort("len", descending=True)
    for row in id_sources.iter_rows(named=True):
        print(f"    {row['source']}: {row['len']}")
    for i, row in enumerate(identity_matches.head(3).iter_rows(named=True), 1):
        print(f"\n  Example {i} (source: {row['source']}):")
        print(f"    {row['content'][:300]}...")
else:
    print("  → No OLMo mentions found — identity prompts may use a different keyword.")

# List HF repo files to confirm
print("\n  HF repo parquet files:")
for f in hf_api.list_repo_files("allenai/Dolci-Think-SFT-32B", repo_type="dataset"):
    if f.endswith(".parquet"):
        print(f"    {f}")

print()

# %% [markdown]
# ---
# ## Stage 5: DPO — Dolci-Think-DPO-32B
#
# 200K preference pairs. Uses Delta Learning heuristic (Geng et al. 2025)
# to select chosen/rejected from a pool of model rollouts.

# %%
dpo_lf = pl.scan_parquet(DPO_PARQUET)
print("=== Dolci-Think-DPO-32B: Schema ===\n")
print(f"  Schema: {dpo_lf.collect_schema()}")

# %% DPO example — show the prompt, chosen vs rejected
print("\n=== DPO Example: Chosen vs Rejected ===\n")
dpo_sample = dpo_lf.head(5).collect()
sample = dpo_sample.row(2, named=True)

prompt = sample.get("prompt", "")
chosen = sample.get("chosen", "")
rejected = sample.get("rejected", "")

# Handle prompt as string or list of messages
if isinstance(prompt, list):
    prompt_text = "\n".join(m.get("content", str(m))[:200] for m in prompt)
else:
    prompt_text = str(prompt)[:400]

if isinstance(chosen, list):
    chosen_text = "\n".join(m.get("content", str(m))[:300] for m in chosen if m.get("role") == "assistant")
else:
    chosen_text = str(chosen)[:500]

if isinstance(rejected, list):
    rejected_text = "\n".join(m.get("content", str(m))[:300] for m in rejected if m.get("role") == "assistant")
else:
    rejected_text = str(rejected)[:500]

print(f"Source: {sample.get('dataset', '?')}")
print(f"Chosen model:   {sample.get('chosen_model', '?')}")
print(f"Rejected model: {sample.get('rejected_model', '?')}")
print(f"\n[prompt]\n{prompt_text}")
print(f"\n[chosen] ✓\n{chosen_text[:400]}...")
print(f"\n[rejected] ✗\n{rejected_text[:400]}...")

# %% DPO source distribution (exact via polars)
print("\n=== DPO Source Distribution (full dataset) ===\n")
dpo_source_dist = (
    dpo_lf
    .group_by("dataset")
    .len()
    .sort("len", descending=True)
    .collect()
)
dpo_total = dpo_source_dist["len"].sum()
for row in dpo_source_dist.iter_rows(named=True):
    pct = 100.0 * row["len"] / dpo_total
    bar = "█" * int(pct / 2)
    print(f"  {row['dataset']:50s} {row['len']:7,d}  {pct:5.1f}%  {bar}")
print(f"\n  Total rows: {dpo_total:,}")

# %% [markdown]
# ---
# ## Stage 6: RL (RLVR) — Dolci-Think-RL-32B
#
# 102K prompts with verifiable ground-truth answers.
# The model generates rollouts; correct ones get rewarded.
# Fields include `passrate`, `total_rollouts`, `total_correct_rollouts`.

# %%
rl_lf = pl.scan_parquet(RL_PARQUET)
print("=== Dolci-Think-RL-32B: Schema ===\n")
print(f"  Schema: {rl_lf.collect_schema()}")

# %% RL example — prompt + ground truth + rollout stats
print("\n=== RL Example: Prompt + Ground Truth + Rollouts ===\n")
rl_sample = rl_lf.head(3).collect()
sample = rl_sample.row(0, named=True)

prompt = sample.get("prompt", "")
if isinstance(prompt, list):
    prompt_text = "\n".join(m.get("content", str(m))[:300] for m in prompt)
else:
    prompt_text = str(prompt)[:500]

print(f"Dataset:   {sample.get('dataset', '?')}")
print(f"Original:  {sample.get('original_dataset', '?')}")
print(f"\n[prompt]\n{prompt_text}")
print(f"\n[ground_truth]\n{str(sample.get('ground_truth', ''))[:300]}")
print(f"\nRollouts: {sample.get('total_rollouts', '?')} total, "
      f"{sample.get('total_correct_rollouts', '?')} correct "
      f"(passrate: {sample.get('passrate', '?')})")

outputs = sample.get("outputs", [])
if outputs:
    print(f"\n[sample rollout] ({len(outputs)} available)")
    rollout = outputs[0]
    if isinstance(rollout, dict):
        print(f"  {str(rollout.get('content', rollout))[:400]}...")
    else:
        print(f"  {str(rollout)[:400]}...")

# %% RL dataset distribution (exact via polars)
print("\n=== RL Dataset Distribution (full dataset) ===\n")
rl_source_dist = (
    rl_lf
    .group_by("original_dataset")
    .len()
    .sort("len", descending=True)
    .collect()
)
rl_total = rl_source_dist["len"].sum()
for row in rl_source_dist.iter_rows(named=True):
    pct = 100.0 * row["len"] / rl_total
    bar = "█" * int(pct / 2)
    print(f"  {row['original_dataset']:50s} {row['len']:7,d}  {pct:5.1f}%  {bar}")
print(f"\n  Total rows: {rl_total:,}")

# %% [markdown]
# ---
# ## Post-training WandB Logs
#
# The `ai2-llm/Olmo-3-32B-Think` WandB project contains training runs for
# all three post-training stages: SFT, DPO, and RL.
#
# Project: https://wandb.ai/ai2-llm/Olmo-3-32B-Think

# %% WandB setup + helpers
WANDB_API_TIMEOUT_SECONDS = 120
wandb_api = wandb.Api(timeout=WANDB_API_TIMEOUT_SECONDS)
HISTORY_SAMPLES = 1000


def pull_history_df(run_obj, keys, samples=1000):
    """Pull run history as a DataFrame, returning empty on failure."""
    try:
        df = run_obj.history(keys=keys, samples=samples)
    except Exception as exc:
        print(f"  History pull failed for run {run_obj.id}: {exc}")
        return pd.DataFrame()
    return df


def classify_posttraining_run(name):
    """Classify a post-training run by name pattern."""
    name_lower = name.lower()
    if "sft" in name_lower:
        return "SFT"
    if "dpo" in name_lower:
        return "DPO"
    if "rl" in name_lower or "rlvr" in name_lower or "grpo" in name_lower:
        return "RL"
    return "other"


# %% Cell E: List post-training runs
pt_path = f"{WANDB_POSTTRAINING['entity']}/{WANDB_POSTTRAINING['project']}"
try:
    pt_runs = list(wandb_api.runs(pt_path))
except ReadTimeout as exc:
    print(f"{pt_path}: timed out ({WANDB_API_TIMEOUT_SECONDS}s): {exc}")
    pt_runs = []
except Exception as exc:
    print(f"{pt_path}: failed to load runs: {exc}")
    pt_runs = []

print(f"=== Post-training WandB: {pt_path} ({len(pt_runs)} runs) ===\n")

pt_categories = {}
for run_obj in pt_runs:
    cat = classify_posttraining_run(run_obj.name)
    pt_categories.setdefault(cat, []).append(run_obj)
    created = getattr(run_obj, 'created_at', '?')
    config_keys = len(run_obj.config) if run_obj.config else 0
    print(f"  {run_obj.name:50s} | {cat:5s} | {run_obj.state:10s} | "
          f"created={created} | config_keys={config_keys}")

print(f"\nBy category:")
for cat in sorted(pt_categories):
    print(f"  [{cat}]: {len(pt_categories[cat])} runs")

# %% Cell F: Discover metrics from richest run
print("\n=== Post-training Metric Discovery ===")
pt_best_columns = []
pt_best_run_name = None
for run_obj in pt_runs:
    try:
        sample_df = run_obj.history(samples=1)
    except Exception:
        continue
    if sample_df.empty:
        continue
    cols = sample_df.columns.tolist()
    if len(cols) > len(pt_best_columns):
        pt_best_columns = sorted(cols)
        pt_best_run_name = run_obj.name
        print(f"  {run_obj.name}: {len(cols)} columns (new best)")
        if len(cols) > 50:
            break
    else:
        print(f"  {run_obj.name}: {len(cols)} columns")

print(f"\nBest: {pt_best_run_name} with {len(pt_best_columns)} columns")

# Group by prefix
pt_grouped = {}
for col in pt_best_columns:
    prefix = col.split("/")[0] if "/" in col else "(top-level)"
    pt_grouped.setdefault(prefix, []).append(col)
for prefix in sorted(pt_grouped):
    cols = pt_grouped[prefix]
    print(f"  [{prefix}] ({len(cols)}): {cols[:8]}")
    if len(cols) > 8:
        print(f"    ... and {len(cols) - 8} more")

# Identify key metric keys
pt_step_key = "_step" if "_step" in pt_best_columns else None
pt_loss_candidates = [c for c in pt_best_columns if "loss" in c.lower()]
print(f"\nStep key: {pt_step_key}")
print(f"Loss candidates: {pt_loss_candidates}")

# %% Cell G: Training loss curves for each post-training phase
PHASE_COLORS = {
    "SFT": "#4c72b0",
    "DPO": "#55a868",
    "RL": "#c44e52",
    "other": "#8172b2",
}

if pt_step_key and pt_loss_candidates:
    print(f"\n=== Post-training Loss Curves ===")
    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = 0
    legend_handles = {}

    for run_obj in pt_runs:
        cat = classify_posttraining_run(run_obj.name)
        color = PHASE_COLORS.get(cat, "gray")

        # Try each loss candidate until one works for this run
        for loss_key in pt_loss_candidates:
            hist_df = pull_history_df(run_obj, keys=[pt_step_key, loss_key],
                                      samples=HISTORY_SAMPLES)
            if (not hist_df.empty and pt_step_key in hist_df.columns
                    and loss_key in hist_df.columns):
                plot_data = hist_df[[pt_step_key, loss_key]].dropna()
                if not plot_data.empty:
                    # Raw data (light)
                    ax.plot(plot_data[pt_step_key], plot_data[loss_key],
                            alpha=0.15, linewidth=0.5, color=color)
                    # Smoothed line
                    window = max(1, len(plot_data) // 50)
                    smoothed = (plot_data[loss_key].rolling(window, min_periods=1).mean()
                                if window > 1 else plot_data[loss_key])
                    line, = ax.plot(plot_data[pt_step_key], smoothed,
                                    linewidth=1.8, color=color, alpha=0.9)
                    label = f"{cat}: {run_obj.name[:30]}"
                    if cat not in legend_handles:
                        legend_handles[cat] = line
                    print(f"  {run_obj.name}: {len(plot_data)} pts [{cat}] key={loss_key}")
                    plotted += 1
                    break  # found a working loss key, move to next run

    if plotted > 0:
        ax.set_title(f"Post-training Loss: {pt_path} ({plotted} runs)")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(legend_handles.values(), legend_handles.keys(),
                  fontsize=10, loc="upper right")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "think_posttraining_loss.png")
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved: {out_path}")
    else:
        print("  No runs had plottable loss data.")
    plt.close(fig)
else:
    print("\nNo step/loss keys found in post-training runs; skipping loss plot.")

# %% [markdown]
# ---
# ## Summary: Think Pipeline at a Glance
#
# | Stage | Dataset | Scale | What changes |
# |-------|---------|-------|-------------|
# | 1. Pretrain | dolma3_pool | 6T tokens | Language modeling on diverse web + code + books |
# | 2. Mid-train | dolmino_mix | 100B tokens | Upweight math, code, reasoning, QA |
# | 3. Long-ctx | longmino_mix | 50B tokens | Extend context window |
# | 4. SFT | Dolci-Think-SFT-32B | 2.25M examples | Learn `<think>` format + instruction following |
# | 5. DPO | Dolci-Think-DPO-32B | 200K pairs | Prefer better reasoning traces |
# | 6. RL | Dolci-Think-RL-32B | 102K prompts | Optimize for verifiably correct answers |
#
# **Key insight:** The Think pipeline differs from Instruct primarily in stages 4-6.
# SFT data includes `<think>` tags in all responses. DPO and RL then refine the
# quality of reasoning, not just the final answer.
#
# ### HF Links
# - [Post-training collection](https://huggingface.co/collections/allenai/olmo-3-post-training)
# - [SFT data](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B)
# - [DPO data](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-32B)
# - [RL data](https://huggingface.co/datasets/allenai/Dolci-Think-RL-32B)
# - [Paper](https://arxiv.org/abs/2512.13961)
