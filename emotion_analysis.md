# Emotion Analysis on Dolci-Think-SFT-7B

## Goal

Understand what about the SFT training data makes OLMo less "depressed" after fine-tuning. Run an emotion classifier on assistant responses to characterize the emotional tone.

## What the script does (`emotion_analysis.py`)

1. **Sample ~1000 assistant messages** from `allenai/Dolci-Think-SFT-7B`, stratified by `dataset_source`
2. **Split each** into think block (inside `<think>...</think>`) and final response (after `</think>`)
3. **Run `j-hartmann/emotion-english-distilroberta-base`** on both parts
   - 7 classes: anger, disgust, fear, joy, neutral, sadness, surprise
   - Batched GPU inference, truncated to 512 tokens
   - `top_k=None` to get all class scores per sample
4. **Aggregate**: overall emotion distribution, per-source breakdown, think vs response comparison
5. **Output**: printed tables (tabulate) + 3 charts saved to `images/`

## Problem: Sampling is slow

The dataset has **2.27M examples sorted by source**. Two approaches tried:

| Approach | Problem |
|----------|---------|
| `shuffle(buffer_size=10_000)` | Only sees first 10K rows — all from one source (OpenThoughts3). Got 1000/1000 from a single source. |
| Systematic sampling (every 2270th row) | Must iterate through all 2.27M rows via streaming. Extremely slow (~10+ min just for sampling). |

### Fix options

1. **Download the dataset first** (non-streaming), then sample with pandas/numpy indexing. Fast random access but requires ~36GB disk + download time.
2. **Use parquet row groups** — download just the parquet metadata, then fetch specific row groups to cover all sources. More complex but avoids full download.
3. **Hardcode source offsets** — from the prior profiling (`think_pipeline.py`), we know the source distribution. We could skip to known offsets in the stream to sample from each source region.
4. **Increase buffer_size to 500K+** — brute-force shuffle with a huge buffer. Uses ~2-3GB RAM but should mix sources adequately. Simpler than systematic sampling.

### Recommended: Option 4 (large buffer shuffle)

Simplest fix. Change `buffer_size=500_000` and take the first 1000 from the shuffled stream. The dataset is parquet-backed so HF datasets will prefetch efficiently. RAM cost ~2-3GB is fine on a GPU box.

## First run results (single-source, for reference)

With all 1000 samples from OpenThoughts3 (math/reasoning data):

| Emotion | Think Block | Response | Diff (R-T) |
|---------|------------|----------|------------|
| anger | 1.3% | 4.9% | +3.5% |
| disgust | 2.5% | 3.5% | +1.0% |
| fear | 0.6% | 2.3% | +1.7% |
| joy | 0.3% | 1.5% | +1.2% |
| **neutral** | **92.7%** | **82.9%** | **-9.8%** |
| sadness | 0.4% | 1.4% | +1.0% |
| surprise | 2.1% | 3.5% | +1.4% |

**Key finding**: Math/reasoning content is overwhelmingly neutral (93% think, 83% response). The interesting signal will come from comparing across sources — WildChat and persona IF data likely carry more emotional content.

## Charts produced

- `images/emotion_think_vs_response.png` — grouped bar chart
- `images/emotion_by_source_heatmap.png` — heatmap (only 1 source so far)
- `images/emotion_dominant_pie.png` — pie charts of dominant emotion

## Next step

Pick a sampling fix (recommend option 4), re-run, and examine whether non-math sources (WildChat, persona IF) show meaningfully different emotion profiles.
