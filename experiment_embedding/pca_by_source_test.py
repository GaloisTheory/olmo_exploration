"""Smoke test: run the per-source PCA pipeline end-to-end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from embedding_utils import (
    resolve_dirs,
    load_by_source,
    find_pc_extremes,
    plot_pc_corrplot,
    build_html_viewer,
)

input_dir, output_dir = resolve_dirs("vllm-1k")

# 1. Load per-source data
source_data = load_by_source(input_dir)
assert len(source_data) > 0, "No sources loaded"
print(f"OK: {len(source_data)} sources loaded")

# 2. Run PCA per source
pc_directions = {}
extremes_by_source = {}
pc_info = {}
for name, data in sorted(source_data.items()):
    extremes_df, pca = find_pc_extremes(
        data["embeddings"], data["metadata"], "combined",
        top_k=2, num_pcs=2, quiet=True,
    )
    pc_directions[name] = pca.components_
    extremes_by_source[name] = extremes_df
    pc_info[name] = {
        "explained_var": [v * 100 for v in pca.explained_variance_ratio_[:2]],
        "n_samples": data["embeddings"].shape[0],
    }
    assert pca.components_.shape == (2, 1024), f"Bad shape for {name}: {pca.components_.shape}"

print(f"OK: PCA computed for {len(pc_directions)} sources")

# 3. Corrplot
corrplot_html = plot_pc_corrplot(pc_directions)
assert "plotly" in corrplot_html.lower() or "<div" in corrplot_html, "Bad corrplot HTML"
print(f"OK: corrplot HTML generated ({len(corrplot_html)} chars)")

# 4. Build viewer
html_path = output_dir / "pca_by_source.html"
build_html_viewer(corrplot_html, extremes_by_source, pc_info, html_path)
assert html_path.exists(), f"{html_path} not created"
content = html_path.read_text()
assert "plotly" in content.lower(), "Missing plotly in HTML"
assert "<details" in content, "Missing collapsible sections"
print(f"OK: HTML viewer saved to {html_path} ({len(content)} chars)")

print("\nAll checks passed.")
