# %% [markdown]
# # OLMo 3 7B-Think: Full Training Pipeline
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
# Dolci-Think-SFT-7B (2.27M ex)  ← Post-training: SFT
#         ↓
# Dolci-Think-DPO-7B (150K pairs) ← Post-training: DPO
#         ↓
# Dolci-Think-RL-7B (102K prompts)← Post-training: RL (RLVR)
#         ↓
#     OLMo-3-7B-Think              ← Final model
# ```

# %% Imports
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi
from requests.exceptions import ReadTimeout

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
        "dataset": "allenai/Dolci-Think-SFT-7B",
        "examples": "2.27M",
        "size": "36.1 GB",
        "description": "Supervised fine-tuning with <think> reasoning tags. "
                       "Sources: OpenThoughts3, synthetic math/code, persona IF, WildChat.",
        "format": "messages (chat)",
    },
    {
        "stage": "Post-training: DPO",
        "dataset": "allenai/Dolci-Think-DPO-7B",
        "examples": "150K pairs",
        "description": "Preference pairs via Delta Learning heuristic. "
                       "Chosen/rejected responses across math, code, chat.",
        "format": "prompt + chosen/rejected",
    },
    {
        "stage": "Post-training: RL (RLVR)",
        "dataset": "allenai/Dolci-Think-RL-7B",
        "examples": "102K prompts",
        "description": "Reinforcement learning from verifiable rewards. "
                       "Math, code, instruction following with ground-truth answers.",
        "format": "prompt + ground_truth + rollouts",
    },
]

# WandB project for post-training runs (SFT, DPO, RL)
WANDB_POSTTRAINING = {
    "entity": "ai2-llm",
    "project": "Olmo-3-7B-Think",
    "report": "https://wandb.ai/ai2-llm/Olmo-3-7B-Think/reports/"
              "Olmo-3-7B-Think-SFT-DPO-RL--VmlldzoxNTE3ODQzMA",
}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# Print the pipeline
for i, stage in enumerate(THINK_PIPELINE):
    connector = "  ↓" if i < len(THINK_PIPELINE) - 1 else "  → OLMo-3-7B-Think"
    tokens = stage.get("tokens", stage.get("examples", ""))
    print(f"  [{stage['stage']}]")
    print(f"    {stage['dataset']}  ({tokens})")
    print(f"    {stage['description'][:80]}")
    print(connector)

# %% [markdown]
# ---
# ## Stage 4: SFT — Dolci-Think-SFT-7B (Deep Dive)
#
# The first post-training stage. All responses include `<think>...</think>` reasoning blocks.
# 2.27M examples from diverse sources. Let's explore the dataset structure in depth.

# %% Cell A: Dataset card from HuggingFace API
print("=== Dolci-Think-SFT-7B: Dataset Card ===\n")
hf_api = HfApi()
ds_info = hf_api.dataset_info("allenai/Dolci-Think-SFT-7B")

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

# Quick schema check from first sample
sft_stream = load_dataset("allenai/Dolci-Think-SFT-7B", split="train", streaming=True)
first_sample = next(iter(sft_stream))
print(f"\n  Schema keys: {list(first_sample.keys())}")
print(f"  Sample source: {first_sample.get('dataset_source', '?')}")
print(f"  Messages in sample: {len(first_sample['messages'])}")

# %% Cell B: Source distribution (stream 50K shuffled for accurate proportions)
print("=== SFT Source Distribution (50K shuffled samples) ===\n")
SFT_SAMPLE_N = 50_000
sources = Counter()
sft_shuffled = load_dataset(
    "allenai/Dolci-Think-SFT-7B", split="train", streaming=True
).shuffle(seed=42, buffer_size=10_000)
for i, sample in enumerate(sft_shuffled):
    sources[sample.get("dataset_source", "unknown")] += 1
    if i >= SFT_SAMPLE_N - 1:
        break

total = sum(sources.values())
print(f"  {'Source':45s} {'Count':>7s}  {'%':>6s}  Bar")
print(f"  {'─' * 45} {'─' * 7}  {'─' * 6}  {'─' * 30}")
for source, count in sources.most_common():
    pct = 100.0 * count / total
    bar_len = int(pct / 2)  # scale: 50% = 25 chars
    bar = "█" * bar_len
    print(f"  {source:45s} {count:7d}  {pct:5.1f}%  {bar}")
print(f"\n  Total sampled: {total}")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 4))
labels = [s for s, _ in sources.most_common()]
counts = [c for _, c in sources.most_common()]
pcts = [100.0 * c / total for c in counts]
bars = ax.barh(labels, pcts, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"][:len(labels)])
ax.set_xlabel("% of dataset (50K shuffled samples)")
ax.set_title("Dolci-Think-SFT-7B: Source Distribution")
ax.invert_yaxis()
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=9)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "think_sft_source_dist.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.close(fig)

# %% Cell C: Message structure stats (stream 5K samples)
print("=== SFT Message Structure Stats (5K samples) ===\n")
SFT_STATS_N = 5_000
msg_counts = []
user_lens = []
asst_lens = []
think_present = []
think_lens = []
response_after_think_lens = []

sft_stats_stream = load_dataset(
    "allenai/Dolci-Think-SFT-7B", split="train", streaming=True
).shuffle(seed=43, buffer_size=10_000)
for i, sample in enumerate(sft_stats_stream):
    msgs = sample["messages"]
    msg_counts.append(len(msgs))
    for msg in msgs:
        content = msg["content"]
        if msg["role"] == "user":
            user_lens.append(len(content))
        elif msg["role"] == "assistant":
            asst_lens.append(len(content))
            has_think = "<think>" in content and "</think>" in content
            think_present.append(has_think)
            if has_think:
                think_end = content.index("</think>") + len("</think>")
                think_lens.append(think_end)
                response_after_think_lens.append(len(content[think_end:].strip()))
    if i >= SFT_STATS_N - 1:
        break

def pstats(vals, name):
    """Print percentile stats for a list of numbers."""
    if not vals:
        print(f"  {name}: no data")
        return
    s = pd.Series(vals)
    print(f"  {name}:")
    print(f"    count={len(s):,}  mean={s.mean():.0f}  median={s.median():.0f}  "
          f"p5={s.quantile(0.05):.0f}  p95={s.quantile(0.95):.0f}  "
          f"min={s.min():.0f}  max={s.max():.0f}")

pstats(msg_counts, "Turns per conversation")
pstats(user_lens, "User message length (chars)")
pstats(asst_lens, "Assistant message length (chars)")
print(f"\n  <think> tag presence: {sum(think_present)}/{len(think_present)} "
      f"({100*sum(think_present)/max(len(think_present),1):.1f}%) of assistant messages")
pstats(think_lens, "Think block length (chars)")
pstats(response_after_think_lens, "Response-after-think length (chars)")

# %% Cell D: Five examples per source
print("=== SFT: Five Examples Per Source ===\n")
examples_per_source = 5
seen_sources = {}
for sample in load_dataset("allenai/Dolci-Think-SFT-7B", split="train", streaming=True):
    src = sample.get("dataset_source", "unknown")
    seen_sources.setdefault(src, [])
    if len(seen_sources[src]) < examples_per_source:
        seen_sources[src].append(sample)

    # Stop once every known source has collected enough examples.
    if len(seen_sources) >= len(sources) and all(
        len(src_samples) >= examples_per_source for src_samples in seen_sources.values()
    ):
        break

for src, samples in seen_sources.items():
    print(f"─── Source: {src} ({len(samples)} examples) ───")
    for i, sample in enumerate(samples, start=1):
        msgs = sample["messages"]
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

# %% [markdown]
# ---
# ## Stage 5: DPO — Dolci-Think-DPO-7B
#
# 150K preference pairs. Uses Delta Learning heuristic (Geng et al. 2025)
# to select chosen/rejected from a pool of model rollouts.

# %%
dpo = load_dataset("allenai/Dolci-Think-DPO-7B", split="train", streaming=True)

print("=== Dolci-Think-DPO-7B: Schema ===\n")
sample = next(iter(dpo))
print(f"Keys: {list(sample.keys())}")
print(f"dataset_source: {sample.get('dataset_source', '?')}")
print(f"chosen_model:   {sample.get('chosen_model', '?')}")
print(f"rejected_model: {sample.get('rejected_model', '?')}")
print(f"preference_type: {sample.get('preference_type', '?')}")

# %% DPO example — show the prompt, chosen vs rejected
print("\n=== DPO Example: Chosen vs Rejected ===\n")
for i, sample in enumerate(load_dataset("allenai/Dolci-Think-DPO-7B", split="train", streaming=True)):
    if i < 2:
        continue  # skip first couple, find a good one
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

    print(f"Source: {sample.get('dataset_source', '?')}")
    print(f"Chosen model:   {sample.get('chosen_model', '?')}")
    print(f"Rejected model: {sample.get('rejected_model', '?')}")
    print(f"\n[prompt]\n{prompt_text}")
    print(f"\n[chosen] ✓\n{chosen_text[:400]}...")
    print(f"\n[rejected] ✗\n{rejected_text[:400]}...")
    break

# %% DPO source distribution
print("\n=== DPO Source Distribution (first 10K) ===\n")
dpo_sources = Counter()
for i, sample in enumerate(load_dataset("allenai/Dolci-Think-DPO-7B", split="train", streaming=True)):
    dpo_sources[sample.get("dataset_source", "unknown")] += 1
    if i >= 9999:
        break

for source, count in dpo_sources.most_common():
    bar = "█" * (count // 100)
    print(f"  {source:40s} {count:5d}  {bar}")

# %% [markdown]
# ---
# ## Stage 6: RL (RLVR) — Dolci-Think-RL-7B
#
# 102K prompts with verifiable ground-truth answers.
# The model generates rollouts; correct ones get rewarded.
# Fields include `passrate`, `total_rollouts`, `total_correct_rollouts`.

# %%
rl = load_dataset("allenai/Dolci-Think-RL-7B", split="train", streaming=True)

print("=== Dolci-Think-RL-7B: Schema ===\n")
sample = next(iter(rl))
print(f"Keys: {list(sample.keys())}")
for k in ["dataset", "original_dataset", "ground_truth", "passrate",
          "total_rollouts", "total_correct_rollouts", "constraint_type"]:
    if k in sample:
        val = sample[k]
        if isinstance(val, str) and len(val) > 100:
            val = val[:100] + "..."
        print(f"  {k}: {val}")

# %% RL example — prompt + ground truth + rollout stats
print("\n=== RL Example: Prompt + Ground Truth + Rollouts ===\n")
for i, sample in enumerate(load_dataset("allenai/Dolci-Think-RL-7B", split="train", streaming=True)):
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

    # Show a rollout if available
    outputs = sample.get("outputs", [])
    if outputs:
        print(f"\n[sample rollout] ({len(outputs)} available)")
        rollout = outputs[0]
        if isinstance(rollout, dict):
            print(f"  {str(rollout.get('content', rollout))[:400]}...")
        else:
            print(f"  {str(rollout)[:400]}...")
    break

# %% RL dataset distribution
print("\n=== RL Dataset Distribution (first 10K) ===\n")
rl_sources = Counter()
for i, sample in enumerate(load_dataset("allenai/Dolci-Think-RL-7B", split="train", streaming=True)):
    rl_sources[sample.get("original_dataset", sample.get("dataset", "unknown"))] += 1
    if i >= 9999:
        break

for source, count in rl_sources.most_common():
    bar = "█" * (count // 100)
    print(f"  {source:40s} {count:5d}  {bar}")

# %% [markdown]
# ---
# ## Post-training WandB Logs
#
# The `ai2-llm/Olmo-3-7B-Think` WandB project contains training runs for
# all three post-training stages: SFT, DPO, and RL.
#
# Report: https://wandb.ai/ai2-llm/Olmo-3-7B-Think/reports/Olmo-3-7B-Think-SFT-DPO-RL--VmlldzoxNTE3ODQzMA

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
# | 4. SFT | Dolci-Think-SFT-7B | 2.27M examples | Learn `<think>` format + instruction following |
# | 5. DPO | Dolci-Think-DPO-7B | 150K pairs | Prefer better reasoning traces |
# | 6. RL | Dolci-Think-RL-7B | 102K prompts | Optimize for verifiably correct answers |
#
# **Key insight:** The Think pipeline differs from Instruct primarily in stages 4-6.
# SFT data includes `<think>` tags in all responses. DPO and RL then refine the
# quality of reasoning, not just the final answer.
#
# ### HF Links
# - [Post-training collection](https://huggingface.co/collections/allenai/olmo-3-post-training)
# - [SFT data](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B)
# - [DPO data](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B)
# - [RL data](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B)
# - [Paper](https://arxiv.org/abs/2512.13961)
