"""Utilities for loading, filtering, and visualizing embeddings.

Handles:
- Loading safetensor embeddings + parquet metadata
- Source filtering (substring match)
- PCA scatter plots (global, per-source, think-vs-response)
- Scree plots (global + per-source PC1)
- PC extreme text extraction
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from umap import UMAP

SCATTER_MAX_POINTS = 50_000


# ── Loading ──────────────────────────────────────────────────────────────────


def load_embeddings(input_dir: Path, text_type: str) -> np.ndarray | None:
    """Load embedding tensor from safetensors file. Returns (N, D) array or None.

    text_type: "think", "response", or "combined" (vLLM single-pass output).
    """
    if text_type == "combined":
        path = input_dir / "all.safetensors"
    else:
        path = input_dir / f"all_{text_type}.safetensors"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return None
    data = load_file(str(path))
    emb = data["embeddings"].numpy()
    print(f"  Loaded {text_type}: {emb.shape}")
    return emb


def load_metadata(input_dir: Path) -> pl.DataFrame:
    """Load metadata parquet. Raises FileNotFoundError if missing."""
    path = input_dir / "metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run embed_dataset.py first.")
    df = pl.read_parquet(path)
    print(f"Loaded metadata: {len(df)} rows, {df['dataset_source'].n_unique()} sources")
    return df


def load_all(
    input_dir: Path, text_types: list[str] = ("think", "response"),
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Load metadata + requested embedding types. Returns (meta_df, {"think": arr, ...})."""
    meta_df = load_metadata(input_dir)
    loaded = {}
    for tt in text_types:
        emb = load_embeddings(input_dir, tt)
        if emb is not None:
            loaded[tt] = emb
    return meta_df, loaded


def resolve_dirs(run_name: str | None = None, base: str = "./output"):
    """Return (input_dir, output_dir) Paths from a run name."""
    base = Path(base)
    if run_name:
        base = base / run_name
    input_dir = base / "embeddings"
    output_dir = base / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


# ── Filtering ────────────────────────────────────────────────────────────────


def filter_by_sources(
    meta_df: pl.DataFrame,
    embeddings: dict[str, np.ndarray],
    patterns: list[str],
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Keep rows where dataset_source matches any pattern (case-insensitive substring)."""
    all_sources = meta_df["dataset_source"].to_list()
    mask = np.array([
        any(p.lower() in s.lower() for p in patterns) for s in all_sources
    ])
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"  WARNING: No sources match {patterns}")
        return meta_df.head(0), {}

    filtered_meta = meta_df[idx.tolist()]
    matched = sorted(filtered_meta["dataset_source"].unique().to_list())
    print(f"  Matched {len(idx)} rows from {len(matched)} sources:")
    for src in matched:
        n = (filtered_meta["dataset_source"] == src).sum()
        print(f"    {src}: {n}")

    filtered_emb = {tt: emb[idx] for tt, emb in embeddings.items()}
    return filtered_meta, filtered_emb


# ── Plotting helpers ─────────────────────────────────────────────────────────


def get_source_colors(sources: list[str]) -> dict[str, str]:
    """Assign a distinct color to each unique source."""
    unique = sorted(set(sources))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(unique), 1))
    return {src: cmap(i) for i, src in enumerate(unique)}


def shorten_source(name: str) -> str:
    name = name.replace("allenai/", "").replace("_", " ")
    return name[:27] + "..." if len(name) > 30 else name


def _subsample(embeddings, sources, max_pts=SCATTER_MAX_POINTS, seed=42):
    n = embeddings.shape[0]
    if n <= max_pts:
        return embeddings, sources
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_pts, replace=False))
    print(f"  Subsampled {n} -> {max_pts} for plotting")
    return embeddings[idx], [sources[i] for i in idx]


# ── PCA ──────────────────────────────────────────────────────────────────────


def fit_pca(embeddings: np.ndarray, n_components: int = 10) -> tuple[PCA, np.ndarray]:
    """Fit PCA and return (pca_model, projections)."""
    pca = PCA(n_components=min(n_components, *embeddings.shape))
    proj = pca.fit_transform(embeddings)
    return pca, proj


# ── Plot: scatter ────────────────────────────────────────────────────────────


def plot_scatter(
    embeddings: np.ndarray,
    sources: list[str],
    text_type: str,
    output_dir: Path | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure | None, PCA]:
    """2D PCA scatter colored by source. Pass ax= to embed in a subplot."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    plot_coords, plot_sources = _subsample(coords, sources)
    color_map = get_source_colors(plot_sources)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 10))
    else:
        fig = None

    for src in sorted(set(plot_sources)):
        mask = np.array([s == src for s in plot_sources])
        ax.scatter(
            plot_coords[mask, 0], plot_coords[mask, 1],
            c=[color_map[src]], label=shorten_source(src),
            alpha=0.5, s=8, edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title or f"{text_type.title()} Embeddings — PCA 2D")

    if own_fig:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                  markerscale=2, framealpha=0.9)
        plt.tight_layout()
        if output_dir:
            out = output_dir / f"pca_scatter_{text_type}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")
    return fig, pca


# ── Plot: UMAP scatter ────────────────────────────────────────────────────────


def plot_umap_scatter(
    embeddings: np.ndarray,
    sources: list[str],
    text_type: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    output_dir: Path | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure | None, UMAP]:
    """2D UMAP scatter colored by source. Pass ax= to embed in a subplot."""
    sub_emb, sub_sources = _subsample(embeddings, sources)
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    coords = reducer.fit_transform(sub_emb)
    color_map = get_source_colors(sub_sources)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 10))
    else:
        fig = None

    for src in sorted(set(sub_sources)):
        mask = np.array([s == src for s in sub_sources])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color_map[src]], label=shorten_source(src),
            alpha=0.5, s=8, edgecolors="none",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"{text_type.title()} Embeddings — UMAP 2D")

    if own_fig:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                  markerscale=2, framealpha=0.9)
        plt.tight_layout()
        if output_dir:
            out = output_dir / f"umap_scatter_{text_type}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")
    return fig, reducer


# ── Plot: scree ──────────────────────────────────────────────────────────────


def plot_scree(
    embeddings: np.ndarray,
    sources: list[str],
    text_type: str,
    n_components: int = 10,
    output_dir: Path | None = None,
) -> tuple[plt.Figure, PCA]:
    """Scree plot: global explained variance + per-source PC1 bar chart."""
    pca, _ = fit_pca(embeddings, n_components)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: global scree
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    xs = range(1, n_components + 1)
    ax_l.bar(xs, pca.explained_variance_ratio_ * 100,
             color="#4c72b0", alpha=0.7, label="Individual")
    ax_l.plot(xs, cumvar, "o-", color="#c44e52", label="Cumulative")
    ax_l.set_xlabel("Component"); ax_l.set_ylabel("Explained Variance (%)")
    ax_l.set_title(f"Global — {text_type.title()}")
    ax_l.legend(); ax_l.set_xticks(list(xs))

    # Right: per-source PC1
    unique = sorted(set(sources))
    pc1 = {}
    for src in unique:
        mask = np.array([s == src for s in sources])
        src_emb = embeddings[mask]
        if src_emb.shape[0] < 3:
            continue
        p = PCA(n_components=min(n_components, src_emb.shape[0] - 1))
        p.fit(src_emb)
        pc1[src] = p.explained_variance_ratio_[0] * 100

    sorted_pc1 = sorted(pc1.items(), key=lambda x: -x[1])
    names = [shorten_source(s) for s, _ in sorted_pc1]
    vals = [v for _, v in sorted_pc1]
    ax_r.barh(range(len(names)), vals, color="#55a868", alpha=0.8)
    ax_r.set_yticks(range(len(names))); ax_r.set_yticklabels(names, fontsize=8)
    ax_r.set_xlabel("PC1 Explained Variance (%)"); ax_r.set_title(f"Per-Source PC1 — {text_type.title()}")
    ax_r.invert_yaxis()

    plt.tight_layout()
    if output_dir:
        out = output_dir / f"scree_{text_type}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    return fig, pca


# ── Plot: think vs response ─────────────────────────────────────────────────


def plot_think_vs_response(
    think_emb: np.ndarray,
    response_emb: np.ndarray,
    sources: list[str],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Side-by-side 2D PCA for think and response embeddings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    for ax, emb, label in [(ax1, think_emb, "Think"), (ax2, response_emb, "Response")]:
        plot_scatter(emb, sources, label.lower(), ax=ax, title=f"{label} Embeddings")

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), loc="upper center",
               ncol=4, fontsize=8, markerscale=2)
    fig.suptitle("Think vs Response Embeddings — PCA 2D", fontsize=14, y=1.01)
    plt.tight_layout()
    if output_dir:
        out = output_dir / "pca_think_vs_response.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    return fig


# ── PC extremes ──────────────────────────────────────────────────────────────


def find_pc_extremes(
    embeddings: np.ndarray,
    meta_df: pl.DataFrame,
    text_type: str,
    top_k: int = 5,
    num_pcs: int = 5,
    display: str = "match",
    output_dir: Path | None = None,
    quiet: bool = False,
) -> tuple[pl.DataFrame, PCA]:
    """Find top-k and bottom-k texts along each PC. Returns DataFrame of extremes."""
    num_pcs = min(num_pcs, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=num_pcs)
    proj = pca.fit_transform(embeddings)

    if text_type == "combined":
        match_col = "user_prompt"
    elif text_type == "think":
        match_col = "think_text"
    else:
        match_col = "response_text"

    records = []
    lines = [
        f"PC Extremes: {text_type} | top_k={top_k}, PCs={num_pcs}, display={display}",
        f"N={embeddings.shape[0]}, D={embeddings.shape[1]}",
        "=" * 80,
    ]

    for pc in range(num_pcs):
        scores = proj[:, pc]
        var = pca.explained_variance_ratio_[pc] * 100
        lines.append(f"\n{'='*80}\nPC{pc+1} ({var:.2f}% variance)\n{'='*80}")

        for direction, indices in [
            ("positive", np.argsort(scores)[-top_k:][::-1]),
            ("negative", np.argsort(scores)[:top_k]),
        ]:
            lines.append(f"\n  --- {direction.upper()} ---")
            for rank, i in enumerate(indices):
                row = meta_df.row(i, named=True)
                lines.append(f"\n  #{rank+1} (score={scores[i]:.4f}, source={row['dataset_source']})")
                if display == "match" and text_type == "combined":
                    lines.append(f"    [PROMPT] {row['user_prompt'][:150]}")
                    lines.append(f"    [RESPONSE] {row['response_text'][:150]}")
                elif display == "match":
                    lines.append(f"    {row[match_col][:200]}")
                else:
                    lines.append(f"    [USER] {row['user_prompt'][:200]}")
                    lines.append(f"    [THINK] {row['think_text'][:200]}")
                    lines.append(f"    [RESPONSE] {row['response_text'][:200]}")
                records.append({
                    "pc": pc + 1, "direction": direction, "rank": rank + 1,
                    "score": float(scores[i]), "source": row["dataset_source"],
                    "user_prompt": row["user_prompt"],
                    "think_text": row["think_text"],
                    "response_text": row["response_text"],
                })

    if not quiet:
        print("\n".join(lines))

    if output_dir:
        (output_dir / f"pc_extremes_{text_type}.txt").write_text("\n".join(lines))
        if not quiet:
            print(f"  Saved: pc_extremes_{text_type}.txt")

    df = pl.DataFrame(records) if records else pl.DataFrame()
    if output_dir and len(df) > 0:
        df.write_parquet(output_dir / f"pc_extremes_{text_type}.parquet")
        if not quiet:
            print(f"  Saved: pc_extremes_{text_type}.parquet")
    return df, pca


# ── Per-source loading ────────────────────────────────────────────────────────


def load_by_source(input_dir: Path) -> dict[str, dict]:
    """Load embeddings + metadata for each source subdirectory.

    Iterates input_dir / "by_source" / */ and returns
    {dir_name: {"embeddings": np.ndarray, "metadata": pl.DataFrame}}.
    """
    by_source_dir = input_dir / "by_source"
    if not by_source_dir.exists():
        raise FileNotFoundError(f"{by_source_dir} not found")

    result = {}
    for d in sorted(by_source_dir.iterdir()):
        if not d.is_dir():
            continue
        emb_path = d / "embeddings.safetensors"
        meta_path = d / "metadata.parquet"
        if not emb_path.exists() or not meta_path.exists():
            print(f"  Skipping {d.name}: missing files")
            continue
        data = load_file(str(emb_path))
        emb = data["embeddings"].numpy()
        meta = pl.read_parquet(meta_path)
        result[d.name] = {"embeddings": emb, "metadata": meta}
    print(f"Loaded {len(result)} sources from {by_source_dir}")
    return result


# ── Corrplot ──────────────────────────────────────────────────────────────────


def plot_pc_corrplot(
    pc_directions: dict[str, np.ndarray],
    num_pcs: int = 2,
) -> str:
    """Cosine-similarity heatmap of PC directions across sources. Returns Plotly HTML div."""
    import plotly.graph_objects as go
    from sklearn.metrics.pairwise import cosine_similarity

    labels = []
    vectors = []
    for name in sorted(pc_directions):
        short = shorten_source(name)
        comps = pc_directions[name][:num_pcs]
        for i in range(comps.shape[0]):
            labels.append(f"{short} PC{i+1}")
            vectors.append(comps[i])

    mat = cosine_similarity(np.array(vectors))

    fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(mat, 2),
        texttemplate="%{text:.2f}",
        textfont={"size": 7},
    ))
    fig.update_layout(
        title="Cosine Similarity of PCA Directions Across Sources",
        width=1200,
        height=1200,
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
        margin=dict(l=200, b=200),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── HTML viewer ───────────────────────────────────────────────────────────────


def build_html_viewer(
    corrplot_html: str,
    extremes_by_source: dict[str, pl.DataFrame],
    pc_info: dict[str, dict],
    output_path: Path,
) -> None:
    """Write a standalone HTML file with corrplot + collapsible per-source extremes."""
    source_sections = []
    for name in sorted(extremes_by_source):
        info = pc_info[name]
        df = extremes_by_source[name]
        short = shorten_source(name)
        var_str = ", ".join(f"PC{i+1}: {v:.1f}%" for i, v in enumerate(info["explained_var"]))

        cards_html = ""
        if len(df) > 0:
            for row in df.iter_rows(named=True):
                prompt = _escape(row.get("user_prompt", ""))
                response = _escape(row.get("response_text", ""))
                direction_cls = "pos" if row["direction"] == "positive" else "neg"
                cards_html += f"""
                <div class="example-card {direction_cls}">
                  <div class="example-meta">
                    PC{row['pc']} · {row['direction']} · rank {row['rank']} · score {row['score']:.4f}
                  </div>
                  <div class="example-field">
                    <div class="field-label">Prompt</div>
                    <div class="field-text">{prompt}</div>
                  </div>
                  <div class="example-field">
                    <div class="field-label">Response</div>
                    <div class="field-text">{response}</div>
                  </div>
                </div>"""

        source_sections.append(f"""
        <details class="source-card">
          <summary><strong>{_escape(short)}</strong> — {info['n_samples']} samples, {var_str}</summary>
          <div class="examples-container">
            {cards_html}
          </div>
        </details>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PCA By Source</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; max-width: 1400px; margin-left: auto; margin-right: auto; }}
  h1 {{ color: #333; }}
  .source-card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
                  padding: 0.8rem 1.2rem; margin: 0.8rem 0; }}
  .source-card summary {{ cursor: pointer; font-size: 1rem; padding: 0.3rem 0; }}
  .examples-container {{ display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem; }}
  .example-card {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem 1.2rem; background: #fdfdfd; }}
  .example-card.pos {{ border-left: 4px solid #4caf50; }}
  .example-card.neg {{ border-left: 4px solid #e57373; }}
  .example-meta {{ font-size: 0.8rem; color: #777; margin-bottom: 0.8rem; font-weight: 500; }}
  .example-field {{ margin-bottom: 0.8rem; }}
  .field-label {{ font-size: 0.75rem; font-weight: 600; color: #555; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 0.3rem; }}
  .field-text {{ font-size: 0.9rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word;
                 color: #222; background: #f8f8f8; padding: 0.6rem 0.8rem; border-radius: 4px; }}
</style>
</head>
<body>
<h1>PCA By Source — Corrplot &amp; Extremes</h1>
<h2>Cosine Similarity of PC Directions</h2>
{corrplot_html}
<h2>Per-Source PC Extremes</h2>
{"".join(source_sections)}
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Saved: {output_path}")


def _escape(s: str) -> str:
    """HTML-escape a string."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
