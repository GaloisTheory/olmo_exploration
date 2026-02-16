# %% [markdown]
# # Emotion Classification on Dolci-Think-SFT-7B
#
# Characterize emotional tone of training data to understand what makes OLMo
# less "depressed" after SFT. Runs `j-hartmann/emotion-english-distilroberta-base`
# on ~1000 assistant responses, split into think blocks vs final responses.

# %% Imports
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
import polars as pl
from tabulate import tabulate
from transformers import pipeline

# %% Config
SFT_PARQUET = "hf://datasets/allenai/Dolci-Think-SFT-7B/**/*.parquet"
SAMPLES_PER_SOURCE = 60  # ~960 total across 16 sources
CLASSIFIER_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DATASET = "allenai/Dolci-Think-SFT-7B"
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% Step 1: Stratified sampling via polars (one scan, N rows per source)
print(f"=== Sampling {SAMPLES_PER_SOURCE} rows/source from {DATASET} via polars ===\n")

sft_lf = pl.scan_parquet(SFT_PARQUET)

# Single collect — window function numbers rows per source, filter keeps first N
sampled_df = (
    sft_lf
    .with_columns(
        pl.int_range(pl.len()).over("dataset_source").alias("_row_num")
    )
    .filter(pl.col("_row_num") < SAMPLES_PER_SOURCE)
    .collect()
)

print(f"  Collected {len(sampled_df)} rows from {sampled_df['dataset_source'].n_unique()} sources")

# Extract assistant messages and split think/response
records = []
for row in sampled_df.iter_rows(named=True):
    source = row["dataset_source"]
    msgs = row["messages"]

    # Get the last assistant message
    asst_msg = None
    for msg in reversed(msgs):
        if msg["role"] == "assistant":
            asst_msg = msg["content"]
            break
    if not asst_msg:
        continue

    # Split into think block and final response
    think_text = ""
    response_text = asst_msg
    if "<think>" in asst_msg and "</think>" in asst_msg:
        think_start = asst_msg.index("<think>") + len("<think>")
        think_end = asst_msg.index("</think>")
        if think_end >= think_start:
            think_text = asst_msg[think_start:think_end].strip()
        response_text = asst_msg[think_end + len("</think>"):].strip()

    # Skip if both parts are empty
    if not think_text and not response_text:
        continue

    records.append({
        "source": source,
        "think_text": think_text,
        "response_text": response_text,
    })

print(f"  Kept {len(records)} records after filtering")
source_counts = defaultdict(int)
for r in records:
    source_counts[r["source"]] += 1
for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"    {src:45s} {cnt:5d} ({100*cnt/len(records):5.1f}%)")

# %% Step 2: Run emotion classifier
print(f"\n=== Loading emotion classifier: {CLASSIFIER_MODEL} ===\n")

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model=CLASSIFIER_MODEL,
    top_k=None,
    device=device,
    truncation=True,
    max_length=512,
)
print(f"  Device: {'GPU' if device == 0 else 'CPU'}")

# Prepare texts for batched inference
think_texts = [r["think_text"] for r in records if r["think_text"]]
response_texts = [r["response_text"] for r in records if r["response_text"]]
think_indices = [i for i, r in enumerate(records) if r["think_text"]]
response_indices = [i for i, r in enumerate(records) if r["response_text"]]

print(f"  Think blocks to classify: {len(think_texts)}")
print(f"  Responses to classify:    {len(response_texts)}")

BATCH_SIZE = 64

print("\n  Classifying think blocks...")
think_results_raw = classifier(think_texts, batch_size=BATCH_SIZE)
print("  Classifying responses...")
response_results_raw = classifier(response_texts, batch_size=BATCH_SIZE)

# Map results back to records
for idx, result in zip(think_indices, think_results_raw):
    scores = {item["label"]: item["score"] for item in result}
    records[idx]["think_emotions"] = scores

for idx, result in zip(response_indices, response_results_raw):
    scores = {item["label"]: item["score"] for item in result}
    records[idx]["response_emotions"] = scores

print("  Done!")

# %% Step 3: Aggregate results
print("\n=== Emotion Distribution: Think Blocks vs Responses ===\n")


def aggregate_emotions(records_subset, key):
    """Average emotion scores across records for a given key (think_emotions or response_emotions)."""
    scores = defaultdict(list)
    for r in records_subset:
        if key in r:
            for label in EMOTION_LABELS:
                scores[label].append(r[key].get(label, 0.0))
    return {label: np.mean(vals) if vals else 0.0 for label, vals in scores.items()}


# Overall: think vs response
think_avg = aggregate_emotions(records, "think_emotions")
response_avg = aggregate_emotions(records, "response_emotions")

overall_table = []
for label in EMOTION_LABELS:
    overall_table.append([
        label,
        f"{think_avg[label]*100:.1f}%",
        f"{response_avg[label]*100:.1f}%",
        f"{(response_avg[label] - think_avg[label])*100:+.1f}%",
    ])
print(tabulate(overall_table, headers=["Emotion", "Think Block", "Response", "Diff (R-T)"],
               tablefmt="simple"))

# Dominant emotion counts
print("\n\n=== Dominant Emotion Counts ===\n")


def dominant_emotion(emotions_dict):
    """Return the emotion label with the highest score."""
    return max(emotions_dict, key=emotions_dict.get)


think_dominant = defaultdict(int)
response_dominant = defaultdict(int)
for r in records:
    if "think_emotions" in r:
        think_dominant[dominant_emotion(r["think_emotions"])] += 1
    if "response_emotions" in r:
        response_dominant[dominant_emotion(r["response_emotions"])] += 1

n_think = sum(think_dominant.values())
n_response = sum(response_dominant.values())
dominant_table = []
for label in EMOTION_LABELS:
    tc = think_dominant[label]
    rc = response_dominant[label]
    dominant_table.append([
        label,
        f"{tc} ({100*tc/max(n_think,1):.1f}%)",
        f"{rc} ({100*rc/max(n_response,1):.1f}%)",
    ])
print(tabulate(dominant_table, headers=["Emotion", "Think (dominant)", "Response (dominant)"],
               tablefmt="simple"))

# Per-source breakdown
print("\n\n=== Emotion Distribution by Dataset Source (Responses) ===\n")

sources = sorted(source_counts.keys(), key=lambda s: -source_counts[s])
source_table = []
for src in sources:
    src_records = [r for r in records if r["source"] == src]
    avg = aggregate_emotions(src_records, "response_emotions")
    row = [src[:40], str(len(src_records))]
    for label in EMOTION_LABELS:
        row.append(f"{avg[label]*100:.1f}%")
    source_table.append(row)

print(tabulate(source_table, headers=["Source", "N"] + EMOTION_LABELS, tablefmt="simple"))

# Same for think blocks
print("\n\n=== Emotion Distribution by Dataset Source (Think Blocks) ===\n")

source_think_table = []
for src in sources:
    src_records = [r for r in records if r["source"] == src]
    avg = aggregate_emotions(src_records, "think_emotions")
    n_with_think = sum(1 for r in src_records if "think_emotions" in r)
    row = [src[:40], str(n_with_think)]
    for label in EMOTION_LABELS:
        row.append(f"{avg[label]*100:.1f}%")
    source_think_table.append(row)

print(tabulate(source_think_table, headers=["Source", "N"] + EMOTION_LABELS, tablefmt="simple"))

# %% Step 4: Visualization

# --- Chart 1: Think vs Response overall ---
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(EMOTION_LABELS))
width = 0.35
bars_think = ax.bar(x - width/2, [think_avg[l]*100 for l in EMOTION_LABELS],
                    width, label="Think Block", color="#4c72b0", alpha=0.85)
bars_resp = ax.bar(x + width/2, [response_avg[l]*100 for l in EMOTION_LABELS],
                   width, label="Response", color="#55a868", alpha=0.85)
ax.set_ylabel("Mean Score (%)")
ax.set_title("Dolci-Think-SFT-7B: Emotion Scores — Think Block vs Response")
ax.set_xticks(x)
ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right")
ax.legend()

for bars in [bars_think, bars_resp]:
    for bar in bars:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "emotion_think_vs_response.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.close(fig)

# --- Chart 2: Per-source emotion heatmap (responses) ---
source_emotion_matrix = []
source_labels_short = []
for src in sources:
    src_records = [r for r in records if r["source"] == src]
    avg = aggregate_emotions(src_records, "response_emotions")
    source_emotion_matrix.append([avg[l]*100 for l in EMOTION_LABELS])
    # Shorten source names for readability
    short = src.replace("allenai/", "").replace("_", " ")
    if len(short) > 35:
        short = short[:32] + "..."
    source_labels_short.append(short)

matrix = np.array(source_emotion_matrix)

fig, ax = plt.subplots(figsize=(10, max(4, len(sources) * 0.5)))
im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(EMOTION_LABELS)))
ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
ax.set_yticks(range(len(source_labels_short)))
ax.set_yticklabels(source_labels_short, fontsize=8)
ax.set_title("Dolci-Think-SFT-7B: Response Emotion by Source (mean score %)")

# Annotate cells
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        color = "white" if val > matrix.max() * 0.6 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

fig.colorbar(im, ax=ax, shrink=0.8, label="Mean score (%)")
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "emotion_by_source_heatmap.png")
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close(fig)

# --- Chart 3: Dominant emotion pie charts ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

colors_map = {
    "anger": "#e74c3c", "disgust": "#8e44ad", "fear": "#7f8c8d",
    "joy": "#f1c40f", "neutral": "#95a5a6", "sadness": "#3498db",
    "surprise": "#e67e22",
}

for ax, data, title in [(ax1, think_dominant, "Think Block"),
                         (ax2, response_dominant, "Response")]:
    labels_pie = [l for l in EMOTION_LABELS if data[l] > 0]
    sizes = [data[l] for l in labels_pie]
    colors_pie = [colors_map[l] for l in labels_pie]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_pie, colors=colors_pie,
        autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
        startangle=90, textprops={"fontsize": 9},
    )
    ax.set_title(title, fontsize=12)

fig.suptitle("Dolci-Think-SFT-7B: Dominant Emotion Distribution", fontsize=13, y=1.02)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "emotion_dominant_pie.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)

print("\n=== Emotion analysis complete ===")
