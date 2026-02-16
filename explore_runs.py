"""Explore all 39 OLMo 7B runs in ai2-llm/Olmo-3-1025-7B via WandB API.

Classifies each run by:
  - name, id, state, created_at, tags
  - history availability
  - config keys that distinguish training stage/phase
  - summary loss/eval values
  - eval-related column discovery

Run with: uv run python explore_runs.py
"""

import os
import sys
from collections import defaultdict

import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if not os.getenv("WANDB_API_KEY"):
    print("ERROR: WANDB_API_KEY not set.", file=sys.stderr)
    sys.exit(1)

ENTITY = "ai2-llm"
PROJECT = "Olmo-3-1025-7B"
PATH = f"{ENTITY}/{PROJECT}"

api = wandb.Api(timeout=120)

print(f"Fetching runs from {PATH} ...")
runs = list(api.runs(PATH))
print(f"Total runs: {len(runs)}\n")

# ---------------------------------------------------------------------------
# 1. Classify all runs
# ---------------------------------------------------------------------------
INTERESTING_CONFIG_KEYS = [
    "_CLASS_", "run_name", "model_name", "task", "stage", "phase",
    "training_type", "model_type", "experiment", "group",
]

EVAL_KEYWORDS = ["eval", "val", "perplexity", "accuracy", "benchmark", "ppl", "acc"]

rows = []
runs_with_history = []

print("=" * 100)
print("PART 1: Detailed run classification")
print("=" * 100)

for i, run in enumerate(runs):
    print(f"\n--- Run {i+1}/{len(runs)} ---")
    print(f"  Name:       {run.name}")
    print(f"  ID:         {run.id}")
    print(f"  State:      {run.state}")
    created = getattr(run, "created_at", None)
    print(f"  Created:    {created}")
    tags = list(run.tags) if run.tags else []
    print(f"  Tags:       {tags}")

    # Config inspection
    config = run.config if isinstance(run.config, dict) else {}
    config_keys = sorted(config.keys())
    print(f"  Config keys ({len(config_keys)}): {config_keys[:30]}")
    if len(config_keys) > 30:
        print(f"    ... and {len(config_keys) - 30} more")

    # Print interesting config values
    for ck in INTERESTING_CONFIG_KEYS:
        if ck in config:
            val = config[ck]
            # Truncate long values
            val_str = str(val)
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            print(f"    config[{ck!r}] = {val_str}")

    # Also check for any key containing these substrings
    for ck in config_keys:
        ck_lower = ck.lower()
        if any(kw in ck_lower for kw in ["class", "stage", "phase", "type", "task", "mode"]):
            if ck not in INTERESTING_CONFIG_KEYS:
                val = config[ck]
                val_str = str(val)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "..."
                print(f"    config[{ck!r}] = {val_str}")

    # Summary inspection
    raw_summary = run.summary
    if hasattr(raw_summary, "_json_dict") and isinstance(raw_summary._json_dict, dict):
        summary = raw_summary._json_dict
    elif isinstance(raw_summary, dict):
        summary = raw_summary
    else:
        # Sometimes summary is a string or other non-dict type
        summary = {}
    summary_keys = sorted(k for k in summary.keys() if isinstance(k, str)) if summary else []
    print(f"  Summary keys ({len(summary_keys)}): {summary_keys[:30]}")
    if len(summary_keys) > 30:
        print(f"    ... and {len(summary_keys) - 30} more")

    # Print loss/eval values from summary
    for sk in summary_keys:
        sk_lower = sk.lower()
        if any(kw in sk_lower for kw in ["loss", "eval", "val", "perplexity", "ppl", "acc", "benchmark"]):
            val = summary.get(sk)
            if isinstance(val, (int, float)):
                print(f"    summary[{sk!r}] = {val}")

    # History check (minimal sample)
    has_history = False
    history_row_count = 0
    try:
        h = run.history(samples=1)
        if not h.empty:
            has_history = True
            history_row_count = len(h)
            runs_with_history.append(run)
            print(f"  History:    YES (sample returned {history_row_count} rows)")
        else:
            print(f"  History:    EMPTY")
    except Exception as e:
        print(f"  History:    ERROR - {e}")

    rows.append({
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "created_at": str(created) if created else "",
        "tags": ", ".join(tags),
        "config_key_count": len(config_keys),
        "summary_key_count": len(summary_keys),
        "has_history": has_history,
        "history_rows": history_row_count,
    })

# ---------------------------------------------------------------------------
# 2. Check for eval data in runs with history
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 100)
print("PART 2: Eval data discovery (runs with history)")
print("=" * 100)

eval_columns_by_run = {}

for run in runs_with_history:
    print(f"\n--- {run.name} (id={run.id}, state={run.state}) ---")
    try:
        h = run.history(samples=5)
    except Exception as e:
        print(f"  ERROR fetching history: {e}")
        continue

    if h.empty:
        print(f"  History empty with samples=5")
        continue

    all_cols = sorted(h.columns.tolist())
    print(f"  Total columns ({len(all_cols)}):")

    # Group by prefix
    grouped = defaultdict(list)
    for col in all_cols:
        prefix = col.split("/")[0] if "/" in col else "(top-level)"
        grouped[prefix].append(col)

    for prefix in sorted(grouped):
        cols = grouped[prefix]
        print(f"    [{prefix}] ({len(cols)}): {cols[:20]}")
        if len(cols) > 20:
            print(f"      ... and {len(cols) - 20} more")

    # Highlight eval-related columns
    eval_cols = [c for c in all_cols if any(kw in c.lower() for kw in EVAL_KEYWORDS)]
    eval_columns_by_run[run.name] = eval_cols
    if eval_cols:
        print(f"\n  EVAL-RELATED COLUMNS ({len(eval_cols)}):")
        for ec in eval_cols:
            print(f"    {ec}")
    else:
        print(f"\n  No eval-related columns found.")

# ---------------------------------------------------------------------------
# 3. Classification table
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 100)
print("PART 3: Classification summary")
print("=" * 100)

df = pd.DataFrame(rows)

# Group by state
print("\n--- By State ---")
for state, grp in df.groupby("state"):
    print(f"\n  State: {state} ({len(grp)} runs)")
    for _, row in grp.iterrows():
        hist_flag = "HAS_HISTORY" if row["has_history"] else "no_history"
        print(f"    {row['name'][:50]:50s} | id={row['id']} | cfg_keys={row['config_key_count']:3d} | sum_keys={row['summary_key_count']:3d} | {hist_flag} | tags={row['tags']}")

# Group by name pattern
print("\n--- By Name Pattern ---")
patterns = defaultdict(list)
for _, row in df.iterrows():
    name = row["name"]
    # Try to extract a pattern from the name
    parts = name.split("-")
    if len(parts) >= 2:
        # Use first two parts as pattern key
        pattern = "-".join(parts[:2])
    else:
        pattern = name
    patterns[pattern].append(row)

for pattern in sorted(patterns):
    run_rows = patterns[pattern]
    print(f"\n  Pattern: {pattern!r} ({len(run_rows)} runs)")
    for row in run_rows:
        hist_flag = "HAS_HISTORY" if row["has_history"] else "no_history"
        print(f"    {row['name'][:60]:60s} | {row['state']:10s} | {hist_flag}")

# Group by tags
print("\n--- By Tags ---")
tag_groups = defaultdict(list)
for _, row in df.iterrows():
    tag_key = row["tags"] if row["tags"] else "(no tags)"
    tag_groups[tag_key].append(row)

for tag_key in sorted(tag_groups):
    run_rows = tag_groups[tag_key]
    print(f"\n  Tags: {tag_key!r} ({len(run_rows)} runs)")
    for row in run_rows:
        hist_flag = "HAS_HISTORY" if row["has_history"] else "no_history"
        print(f"    {row['name'][:60]:60s} | {row['state']:10s} | {hist_flag}")

# Group by has_history
print("\n--- By History Availability ---")
with_hist = df[df["has_history"] == True]
without_hist = df[df["has_history"] == False]
print(f"\n  WITH history: {len(with_hist)} runs")
for _, row in with_hist.iterrows():
    print(f"    {row['name'][:60]:60s} | {row['state']:10s} | cfg_keys={row['config_key_count']}")
print(f"\n  WITHOUT history: {len(without_hist)} runs")
for _, row in without_hist.iterrows():
    print(f"    {row['name'][:60]:60s} | {row['state']:10s} | cfg_keys={row['config_key_count']}")

# Group by config key count (as proxy for run complexity/type)
print("\n--- By Config Key Count (buckets) ---")
bins = [0, 5, 20, 50, 100, 500, 10000]
labels = ["0-4", "5-19", "20-49", "50-99", "100-499", "500+"]
df["config_bucket"] = pd.cut(df["config_key_count"], bins=bins, labels=labels, right=False)
for bucket, grp in df.groupby("config_bucket", observed=True):
    print(f"\n  Config keys {bucket}: ({len(grp)} runs)")
    for _, row in grp.iterrows():
        hist_flag = "HAS_HISTORY" if row["has_history"] else "no_history"
        print(f"    {row['name'][:60]:60s} | {row['state']:10s} | keys={row['config_key_count']:3d} | {hist_flag}")

# Final summary table
print("\n\n--- FINAL SUMMARY TABLE ---")
print(f"{'Name':<55} {'State':<12} {'Tags':<20} {'CfgKeys':>7} {'SumKeys':>7} {'History':<10} {'Created'}")
print("-" * 160)
for _, row in df.iterrows():
    hist_flag = "YES" if row["has_history"] else "no"
    print(f"{row['name'][:54]:<55} {row['state']:<12} {row['tags'][:19]:<20} {row['config_key_count']:>7} {row['summary_key_count']:>7} {hist_flag:<10} {row['created_at'][:19]}")

print(f"\nDone. {len(runs)} runs explored.")
