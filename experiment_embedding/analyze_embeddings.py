# %% [markdown]
# # Embedding Analysis
# Run cells top-to-bottom, or edit config and re-run from any cell.

# %% Imports
%matplotlib inline
from pathlib import Path

import matplotlib.pyplot as plt

from embedding_utils import (
    load_all,
    resolve_dirs,
    filter_by_sources,
    plot_scatter,
    plot_umap_scatter,
    plot_scree,
    plot_think_vs_response,
    find_pc_extremes,
    load_by_source,
    plot_pc_corrplot,
    build_html_viewer,
)

# %% Config — edit these, then re-run from here
RUN_NAME = "vllm-1k-random"     # e.g. "my_run" → output/my_run/{embeddings,analysis}
TEXT_TYPES = "auto"      # "auto" (detect), "think", "response", "both", or "combined"
N_COMPONENTS = 10        # PCA components for scree
SOURCE_FILTER = None     # e.g. ["math", "code"] for substring match

# UMAP
UMAP_N_NEIGHBORS = 15   # higher = more global structure
UMAP_MIN_DIST = 0.1     # lower = tighter clusters
TOP_K = 2                # top-k PC extremes (0 = off)
NUM_PCS = 5              # how many PCs to inspect for extremes
PC_DISPLAY = "match"     # "match" or "full"

# %% Load data
input_dir, output_dir = resolve_dirs(RUN_NAME)

# Auto-detect embedding format (vLLM combined vs HF separate)
if TEXT_TYPES == "auto":
    if (input_dir / "all.safetensors").exists():
        tt_list = ["combined"]
        print("Detected: combined embeddings (vLLM single-pass)")
    elif (input_dir / "all_think.safetensors").exists():
        tt_list = ["think", "response"]
        print("Detected: separate think/response embeddings")
    else:
        raise RuntimeError(f"No embeddings found in {input_dir}")
elif TEXT_TYPES == "both":
    tt_list = ["think", "response"]
else:
    tt_list = [TEXT_TYPES]

meta_df, loaded = load_all(input_dir, tt_list)
if not loaded:
    raise RuntimeError("No embeddings found — check RUN_NAME / input_dir")
sources = meta_df["dataset_source"].to_list()

print(f"input_dir:  {input_dir}")
print(f"output_dir: {output_dir}")
print(f"meta_df:    {meta_df.shape[0]} rows × {meta_df.shape[1]} cols")
for tt, emb in loaded.items():
    print(f"  {tt}: {emb.shape}")

# %% Filter sources (optional — skip if SOURCE_FILTER is None)
# %%
if SOURCE_FILTER:
    print(f"Filtering: {SOURCE_FILTER}")
    meta_df, loaded = filter_by_sources(meta_df, loaded, SOURCE_FILTER)
    if not loaded:
        raise RuntimeError("No data after filtering")
    sources = meta_df["dataset_source"].to_list()
    print(f"  → {meta_df.shape[0]} rows remaining")

# %% Scatter plots — one per text type
for tt, emb in loaded.items():
    fig, _ = plot_scatter(emb, sources, tt)
    plt.show()

# %% UMAP scatter plots — one per text type
for tt, emb in loaded.items():
    fig, _ = plot_umap_scatter(emb, sources, tt,
                                n_neighbors=UMAP_N_NEIGHBORS,
                                min_dist=UMAP_MIN_DIST)
    plt.show()

# %% Scree plots — one per text type
for tt, emb in loaded.items():
    fig, _ = plot_scree(emb, sources, tt, N_COMPONENTS)
    plt.show()

# %% Think vs Response — side-by-side comparison (separate embeddings only)
if "think" in loaded and "response" in loaded:
    fig = plot_think_vs_response(loaded["think"], loaded["response"], sources)
    plt.show()
elif "combined" not in loaded:
    print("Need both think + response embeddings for this plot")

# %% PC extremes (optional — skip if TOP_K == 0)
if TOP_K > 0 and "user_prompt" in meta_df.columns:
    for tt, emb in loaded.items():
        print(f"\n--- {tt} PC extremes ---")
        find_pc_extremes(emb, meta_df, tt, top_k=TOP_K,
                         num_pcs=NUM_PCS, display=PC_DISPLAY)
else:
    print("Set TOP_K > 0 to inspect PC extremes")

# %% Per-source PCA analysis
source_data = load_by_source(input_dir)
pc_directions = {}
extremes_by_source = {}
pc_info = {}
for name, data in sorted(source_data.items()):
    extremes_df, pca = find_pc_extremes(
        data["embeddings"], data["metadata"], "combined",
        top_k=2, num_pcs=2, quiet=True)
    pc_directions[name] = pca.components_
    extremes_by_source[name] = extremes_df
    pc_info[name] = {
        "explained_var": [v * 100 for v in pca.explained_variance_ratio_[:2]],
        "n_samples": data["embeddings"].shape[0],
    }

# %% Corrplot — cosine similarity of PC directions
corrplot_html = plot_pc_corrplot(pc_directions)

# %% Build HTML viewer
html_path = output_dir / f"pca_by_source_{RUN_NAME}.html"
build_html_viewer(corrplot_html, extremes_by_source, pc_info, html_path)
print(f"Saved: {html_path}")

# %%
