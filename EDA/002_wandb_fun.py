# %% [markdown]
# ## WandB training logs (OLMo 7B)
#
# This notebook-style script loads runs from the OLMo 7B project:
# - 7B:  ai2-llm/Olmo-3-1025-7B
#
# Features:
# - run classification (pre-training segments, named pre-training, annealing)
# - multi-run training loss plot (all runs, color-coded by category)
# - eval benchmark accuracy plots (downstream tasks over training steps)
# - all figures saved to projects/olmo_fun/

# %% WandB setup (environment auth only)
import os
import re
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import wandb
from requests.exceptions import ReadTimeout

if not os.getenv("WANDB_API_KEY"):
    raise RuntimeError(
        "WANDB_API_KEY is not set. "
        "Set it in your shell before running this notebook/script."
    )

# Use a non-interactive backend so this script can run in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WANDB_API_TIMEOUT_SECONDS = 120
wandb_api = wandb.Api(timeout=WANDB_API_TIMEOUT_SECONDS)

ENTITY = "ai2-llm"
PROJECT = "Olmo-3-1025-7B"
HISTORY_SAMPLES = 1000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def pull_history_df(run_obj, keys, samples=1000):
    try:
        df = run_obj.history(keys=keys, samples=samples)
    except Exception as exc:
        print(f"  History pull failed for run {run_obj.id}: {exc}")
        return pd.DataFrame()
    return df


def classify_run(name):
    """Classify a run by its name pattern."""
    if name.startswith("anneal"):
        return "annealing"
    if re.match(r"OLMo25-\d{8}T\d{6}\+\d{4}$", name):
        return "pre-training"
    return "named-pre-training"


# %% Load all runs
path = f"{ENTITY}/{PROJECT}"
try:
    all_runs = list(wandb_api.runs(path))
except ReadTimeout as exc:
    print(f"{path}: timed out ({WANDB_API_TIMEOUT_SECONDS}s): {exc}")
    all_runs = []
except Exception as exc:
    print(f"{path}: failed to load runs: {exc}")
    all_runs = []
print(f"{path}: {len(all_runs)} runs loaded")

# %% Classify runs
run_categories = {}
for run_obj in all_runs:
    cat = classify_run(run_obj.name)
    run_categories.setdefault(cat, []).append(run_obj)

print("\n=== Run Classification ===")
for cat in sorted(run_categories):
    runs = run_categories[cat]
    print(f"\n  [{cat}] ({len(runs)} runs):")
    for r in runs:
        print(f"    {r.name[:60]:60s} | {r.state:10s} | {r.id}")

# %% Discover metrics (find richest column set across all runs)
print("\n=== Metric Discovery (looking for richest column set) ===")
best_columns = []
best_run_name = None
for run_obj in all_runs:
    try:
        sample_df = run_obj.history(samples=1)
    except Exception:
        continue
    if sample_df.empty:
        continue
    cols = sample_df.columns.tolist()
    if len(cols) > len(best_columns):
        best_columns = sorted(cols)
        best_run_name = run_obj.name
        print(f"  {run_obj.name}: {len(cols)} columns (new best)")
        # If we found eval columns (200+), that's good enough
        if len(cols) > 100:
            break
    else:
        print(f"  {run_obj.name}: {len(cols)} columns")

print(f"\nBest: {best_run_name} with {len(best_columns)} columns")

# Group by prefix
grouped = {}
for col in best_columns:
    prefix = col.split("/")[0] if "/" in col else "(top-level)"
    grouped.setdefault(prefix, []).append(col)
for prefix in sorted(grouped):
    cols = grouped[prefix]
    print(f"  [{prefix}] ({len(cols)}): {cols[:10]}")
    if len(cols) > 10:
        print(f"    ... and {len(cols) - 10} more")

# Identify key metric keys
step_key = "_step" if "_step" in best_columns else None
loss_key = next((c for c in best_columns if c == "train/CE loss"), None)
if not loss_key:
    loss_key = next((c for c in best_columns if "loss" in c.lower()), None)

# Identify eval accuracy keys (downstream benchmarks)
# Column names use "(accuracy)" suffix, e.g. "eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy)"
eval_acc_keys = sorted(
    c for c in best_columns
    if "(accuracy)" in c and "v2" not in c  # skip v2 duplicates
)
# Identify LM validation perplexity keys
eval_ppl_keys = sorted(
    c for c in best_columns
    if c.startswith("eval/lm/") and c.endswith("/PPL") and "v2" not in c
)

print(f"\nStep key: {step_key}")
print(f"Loss key: {loss_key}")
print(f"Eval accuracy keys ({len(eval_acc_keys)}): {eval_acc_keys[:10]}")
print(f"Eval PPL keys ({len(eval_ppl_keys)}): {eval_ppl_keys}")

# %% Multi-run training loss plot (ALL runs, color-coded by category)
CATEGORY_COLORS = {
    "pre-training": "tab:blue",
    "named-pre-training": "tab:orange",
    "annealing": "tab:red",
}

if step_key and loss_key:
    print(f"\n=== Multi-run training loss (all runs) ===")
    fig, ax = plt.subplots(figsize=(16, 7))
    plotted = 0
    legend_handles = {}

    for run_obj in all_runs:
        cat = classify_run(run_obj.name)
        color = CATEGORY_COLORS.get(cat, "gray")
        hist_df = pull_history_df(run_obj, keys=[step_key, loss_key], samples=HISTORY_SAMPLES)
        if hist_df.empty or step_key not in hist_df.columns or loss_key not in hist_df.columns:
            print(f"  {run_obj.name}: no data, skipping")
            continue

        plot_data = hist_df[[step_key, loss_key]].dropna()
        if plot_data.empty:
            continue

        # Raw data (light)
        ax.plot(plot_data[step_key], plot_data[loss_key],
                alpha=0.15, linewidth=0.5, color=color)

        # Smoothed line
        window = max(1, len(plot_data) // 100)
        smoothed = plot_data[loss_key].rolling(window, min_periods=1).mean() if window > 1 else plot_data[loss_key]
        line, = ax.plot(plot_data[step_key], smoothed,
                        linewidth=1.2, color=color, alpha=0.8)

        # One legend entry per category
        if cat not in legend_handles:
            legend_handles[cat] = line

        print(f"  {run_obj.name}: {len(plot_data)} pts [{cat}]")
        plotted += 1

    if plotted > 0:
        ax.set_title(f"{PROJECT} - {loss_key} ({plotted} runs, color = category)")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(legend_handles.values(), legend_handles.keys(), fontsize=9, loc="upper right")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "training_loss_7b.png")
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved: {out_path}")
    else:
        print("  No runs had plottable loss data.")
    plt.close(fig)
else:
    print("\nNo step/loss keys found; skipping training loss plot.")

# %% Eval benchmark accuracy plot (runs that have eval data)
if step_key and eval_acc_keys:
    # Pick a curated subset of key benchmarks for readability
    # Column format: "eval/downstream/<benchmark> (accuracy)"
    # Use all available accuracy keys (typically ~9 benchmarks)
    benchmark_keys = eval_acc_keys

    print(f"\n=== Eval Benchmark Accuracy ({len(benchmark_keys)} benchmarks) ===")
    for k in benchmark_keys:
        print(f"  {k}")

    # Collect all eval data across runs into one dataframe per benchmark
    pull_keys = [step_key] + benchmark_keys
    all_eval_dfs = []
    for run_obj in all_runs:
        hist_df = pull_history_df(run_obj, keys=pull_keys, samples=HISTORY_SAMPLES)
        if hist_df.empty or step_key not in hist_df.columns:
            continue
        all_eval_dfs.append(hist_df)
        print(f"  {run_obj.name}: has eval data")

    if all_eval_dfs:
        combined_eval = pd.concat(all_eval_dfs, ignore_index=True).sort_values(step_key)

    fig, ax = plt.subplots(figsize=(16, 7))
    plotted_benchmarks = set()

    if all_eval_dfs:
        for bk in benchmark_keys:
            if bk not in combined_eval.columns:
                continue
            plot_data = combined_eval[[step_key, bk]].dropna()
            if plot_data.empty:
                continue
            short = bk.replace("eval/downstream/", "").replace(" (accuracy)", "")
            # Raw data points
            ax.scatter(plot_data[step_key], plot_data[bk],
                       s=8, alpha=0.25, zorder=1)
            # Smoothed line (rolling over concatenated sorted data)
            window = max(1, len(plot_data) // 50)
            smoothed = plot_data[bk].rolling(window, min_periods=1, center=True).mean()
            ax.plot(plot_data[step_key], smoothed,
                    linewidth=2.0, label=short, alpha=0.9, zorder=2)
            plotted_benchmarks.add(bk)

    if plotted_benchmarks:
        ax.set_title(f"{PROJECT} - Downstream Eval Accuracy over Steps")
        ax.set_xlabel("step")
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "eval_accuracy_7b.png")
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved: {out_path}")
    else:
        print("  No runs had eval accuracy data.")
    plt.close(fig)
else:
    print("\nNo eval accuracy keys found; skipping eval accuracy plot.")

# %% Eval LM validation perplexity plot
if step_key and eval_ppl_keys:
    print(f"\n=== LM Validation Perplexity ({len(eval_ppl_keys)} domains) ===")
    for k in eval_ppl_keys:
        print(f"  {k}")

    # Collect all PPL data across runs into one dataframe
    pull_keys = [step_key] + eval_ppl_keys
    all_ppl_dfs = []
    for run_obj in all_runs:
        hist_df = pull_history_df(run_obj, keys=pull_keys, samples=HISTORY_SAMPLES)
        if hist_df.empty or step_key not in hist_df.columns:
            continue
        all_ppl_dfs.append(hist_df)
        print(f"  {run_obj.name}: has PPL data")

    if all_ppl_dfs:
        combined_ppl = pd.concat(all_ppl_dfs, ignore_index=True).sort_values(step_key)

    fig, ax = plt.subplots(figsize=(16, 7))
    plotted_ppl = set()

    if all_ppl_dfs:
        for pk in eval_ppl_keys:
            if pk not in combined_ppl.columns:
                continue
            plot_data = combined_ppl[[step_key, pk]].dropna()
            if plot_data.empty:
                continue
            parts = pk.split("/")
            short = parts[2] if len(parts) >= 4 else pk
            # Raw data points
            ax.scatter(plot_data[step_key], plot_data[pk],
                       s=6, alpha=0.2, zorder=1)
            # Smoothed line
            window = max(1, len(plot_data) // 50)
            smoothed = plot_data[pk].rolling(window, min_periods=1, center=True).mean()
            ax.plot(plot_data[step_key], smoothed,
                    linewidth=2.0, label=short, alpha=0.9, zorder=2)
            plotted_ppl.add(pk)

    if plotted_ppl:
        ax.set_title(f"{PROJECT} - LM Validation Perplexity over Steps")
        ax.set_xlabel("step")
        ax.set_ylabel("perplexity")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "eval_ppl_7b.png")
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved: {out_path}")
    else:
        print("  No runs had PPL data.")
    plt.close(fig)
else:
    print("\nNo eval PPL keys found; skipping validation perplexity plot.")
