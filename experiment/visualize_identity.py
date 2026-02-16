"""Generate HTML response viewer and identity frequency bar chart for full post-training pipeline."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# (file_key, label, stage)
MODEL_CONFIGS = {
    "7B": {
        "checkpoints": [
            ("sft_1k",    "SFT 1k",    "SFT"),
            ("sft_7k",    "SFT 7k",    "SFT"),
            ("sft_13k",   "SFT 13k",   "SFT"),
            ("sft_19k",   "SFT 19k",   "SFT"),
            ("sft_25k",   "SFT 25k",   "SFT"),
            ("sft_31k",   "SFT 31k",   "SFT"),
            ("sft_37k",   "SFT 37k",   "SFT"),
            ("sft_43k",   "SFT 43k",   "SFT"),
            ("dpo_final", "DPO",       "DPO"),
            ("rlvr_25",   "RLVR 25",   "RLVR"),
            ("rlvr_200",  "RLVR 200",  "RLVR"),
            ("rlvr_375",  "RLVR 375",  "RLVR"),
            ("rlvr_550",  "RLVR 550",  "RLVR"),
            ("rlvr_725",  "RLVR 725",  "RLVR"),
            ("rlvr_900",  "RLVR 900",  "RLVR"),
            ("rlvr_1075", "RLVR 1075", "RLVR"),
            ("rlvr_1250", "RLVR 1250", "RLVR"),
            ("final",     "Final",     "Final"),
            ("instruct",  "Instruct",  "Instruct"),
        ],
        "title": "OLMo 7B Post-Training Identity Evolution",
    },
    "32B": {
        "checkpoints": [
            ("sft_1k",    "SFT 1k",    "SFT"),
            ("sft_2k",    "SFT 2k",    "SFT"),
            ("sft_4k",    "SFT 4k",    "SFT"),
            ("sft_5k",    "SFT 5k",    "SFT"),
            ("sft_7k",    "SFT 7k",    "SFT"),
            ("sft_8k",    "SFT 8k",    "SFT"),
            ("sft_10k",   "SFT 10k",   "SFT"),
            ("sft_10.8k", "SFT 10.8k", "SFT"),
            ("dpo_final", "DPO",       "DPO"),
            ("rlvr_50",   "RLVR 50",   "RLVR"),
            ("rlvr_150",  "RLVR 150",  "RLVR"),
            ("rlvr_250",  "RLVR 250",  "RLVR"),
            ("rlvr_350",  "RLVR 350",  "RLVR"),
            ("rlvr_450",  "RLVR 450",  "RLVR"),
            ("rlvr_550",  "RLVR 550",  "RLVR"),
            ("rlvr_650",  "RLVR 650",  "RLVR"),
            ("rlvr_750",  "RLVR 750",  "RLVR"),
            ("final",     "Final",     "Final"),
            ("instruct",  "Instruct (3.1)", "Instruct"),
        ],
        "title": "OLMo 32B Post-Training Identity Evolution",
    },
}

IDENTITY_COLORS = {
    "DeepSeek": "#4285F4",
    "Qwen": "#EA4335",
    "Alibaba": "#FBBC04",
    "OpenAI": "#34A853",
    "OLMo": "#FF6F00",
    "Generic AI": "#9AA0A6",
}

STAGE_COLORS = {
    "SFT": "#e3f2fd",
    "DPO": "#e8f5e9",
    "RLVR": "#fff3e0",
    "Final": "#f3e5f5",
    "Instruct": "#fce4ec",
}


def extract_answer(text):
    m = re.search(r"</think>\s*", text)
    return text[m.end():].strip() if m else text.strip()


def classify(answer):
    a = answer.lower()
    if "deepseek" in a:
        return "DeepSeek"
    if "qwen" in a:
        return "Qwen"
    if "alibaba" in a:
        return "Alibaba"
    if "openai" in a:
        return "OpenAI"
    if "olmo" in a:
        return "OLMo"
    return "Generic AI"


def load_results(results_dir, checkpoints):
    data = {}
    for file_key, label, stage in checkpoints:
        path = results_dir / f"{file_key}.json"
        if not path.exists():
            print(f"Warning: {path.name} not found, skipping {label}")
            continue
        raw = json.loads(path.read_text())
        entries = []
        for resp in raw["responses"]:
            answer = extract_answer(resp)
            think = ""
            tm = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
            if tm:
                think = tm.group(1).strip()
            entries.append({
                "raw": resp,
                "answer": answer,
                "think": think,
                "identity": classify(answer),
            })
        data[file_key] = entries
    return data


def make_chart(data, results_dir, checkpoints, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter to checkpoints that have data
    active = [(fk, label, stage) for fk, label, stage in checkpoints if fk in data]
    labels = [label for _, label, _ in active]
    file_keys = [fk for fk, _, _ in active]
    stages = [stage for _, _, stage in active]

    # Sort identities by total frequency, only include those that appear
    ids = sorted(IDENTITY_COLORS.keys(), key=lambda k: -sum(
        1 for fk in file_keys for e in data.get(fk, []) if e["identity"] == k
    ))
    ids = [i for i in ids if any(e["identity"] == i for fk in file_keys for e in data.get(fk, []))]

    x = np.arange(len(active))
    w = 0.8 / max(len(ids), 1)

    fig, ax = plt.subplots(figsize=(24, 7))

    # Draw stage background spans
    stage_ranges = []
    current_stage = stages[0]
    start = 0
    for i, stage in enumerate(stages):
        if stage != current_stage:
            stage_ranges.append((current_stage, start, i - 1))
            current_stage = stage
            start = i
    stage_ranges.append((current_stage, start, len(stages) - 1))

    for stage_name, s, e in stage_ranges:
        color = STAGE_COLORS.get(stage_name, "#f5f5f5")
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.15, color=color, zorder=0)

    # Draw bars
    for i, identity in enumerate(ids):
        counts = [Counter(e["identity"] for e in data[fk]).get(identity, 0) for fk in file_keys]
        offset = (i - len(ids) / 2 + 0.5) * w
        bars = ax.bar(x + offset, counts, w, label=identity,
                      color=IDENTITY_COLORS.get(identity, "#666"),
                      edgecolor="white", linewidth=0.5, zorder=2)
        for bar, v in zip(bars, counts):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        str(v), ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Stage separator lines
    for stage_name, s, e in stage_ranges[1:]:
        ax.axvline(x=s - 0.5, color="#888", linestyle="--", linewidth=1, alpha=0.7, zorder=3)

    # Stage labels at top
    max_count = max(Counter(e["identity"] for e in entries).most_common(1)[0][1]
                    for entries in data.values())
    for stage_name, s, e in stage_ranges:
        mid = (s + e) / 2
        ax.text(mid, max_count * 1.08, stage_name, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#555", zorder=4)

    ax.set_xlabel("Checkpoint", fontsize=12)
    n_samples = len(next(iter(data.values())))
    ax.set_ylabel(f"Frequency (out of {n_samples} samples)", fontsize=12)
    ax.set_title(f"{title}: Who does the model claim to be?",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylim(0, max_count * 1.20)
    ax.legend(title="Claimed Identity", fontsize=10, title_fontsize=11,
              loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = results_dir / "identity_barchart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def make_html(data, results_dir, checkpoints, title):
    # Build ordered list of checkpoints that have data
    active = [(fk, label, stage) for fk, label, stage in checkpoints if fk in data]

    # Prepare JSON data for embedding
    viewer_data = {}
    for fk, label, stage in active:
        viewer_data[fk] = [{
            "answer": e["answer"],
            "think": e["think"],
            "identity": e["identity"],
        } for e in data[fk]]

    # Count summaries
    summaries = {}
    for fk, label, stage in active:
        c = Counter(e["identity"] for e in data[fk])
        summaries[fk] = dict(c)

    # Checkpoint metadata for JS
    checkpoints_js = [{"key": fk, "label": label, "stage": stage} for fk, label, stage in active]
    n_samples = len(next(iter(data.values())))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
  .header {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 20px 32px; }}
  .header h1 {{ font-size: 22px; color: #f0f6fc; }}
  .header p {{ color: #8b949e; font-size: 14px; margin-top: 4px; }}
  .tabs {{ display: flex; gap: 0; background: #161b22; border-bottom: 1px solid #30363d;
           padding: 0 24px; overflow-x: auto; }}
  .tab-group {{ display: flex; gap: 0; align-items: center; }}
  .tab-group + .tab-group {{ border-left: 2px solid #30363d; }}
  .stage-label {{ padding: 8px 12px; color: #58a6ff; font-size: 11px; font-weight: 700;
                  text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap; }}
  .tab {{ padding: 12px 16px; cursor: pointer; color: #8b949e; font-size: 13px;
          font-weight: 500; border-bottom: 2px solid transparent; white-space: nowrap;
          transition: all 0.15s; }}
  .tab:hover {{ color: #c9d1d9; }}
  .tab.active {{ color: #f0f6fc; border-bottom-color: #f78166; }}
  .content {{ max-width: 960px; margin: 0 auto; padding: 24px; }}
  .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
  .badge {{ padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; }}
  .response-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                    margin-bottom: 12px; overflow: hidden; }}
  .response-header {{ display: flex; align-items: center; gap: 12px; padding: 12px 16px;
                      border-bottom: 1px solid #21262d; }}
  .response-num {{ color: #8b949e; font-size: 13px; font-weight: 600; min-width: 24px; }}
  .identity-tag {{ padding: 2px 10px; border-radius: 12px; font-size: 12px;
                   font-weight: 600; color: #0d1117; }}
  .response-body {{ padding: 16px; }}
  .answer-text {{ font-size: 15px; color: #e6edf3; }}
  .think-toggle {{ margin-top: 12px; }}
  .think-btn {{ background: none; border: 1px solid #30363d; color: #8b949e; padding: 4px 12px;
                border-radius: 6px; cursor: pointer; font-size: 12px; }}
  .think-btn:hover {{ border-color: #8b949e; color: #c9d1d9; }}
  .think-block {{ margin-top: 8px; padding: 12px; background: #0d1117; border-radius: 6px;
                  border: 1px solid #21262d; font-size: 13px; color: #8b949e;
                  white-space: pre-wrap; display: none; max-height: 300px; overflow-y: auto; }}
  .think-block.open {{ display: block; }}
  .no-think {{ color: #484f58; font-style: italic; font-size: 12px; margin-top: 8px; }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <p>SFT &rarr; DPO &rarr; RLVR &rarr; Final + Instruct &mdash; "Who are you? Answer directly in one sentence." &mdash; {n_samples} samples per checkpoint</p>
</div>

<div class="tabs" id="tabs"></div>
<div class="content" id="content"></div>

<script>
const DATA = {json.dumps(viewer_data)};
const SUMMARIES = {json.dumps(summaries)};
const CHECKPOINTS = {json.dumps(checkpoints_js)};
const COLORS = {json.dumps(IDENTITY_COLORS)};
const N_SAMPLES = {n_samples};

function setCheckpoint(key) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.key === key));
  const entries = DATA[key];
  const summary = SUMMARIES[key];

  let html = '<div class="summary">';
  for (const [id, count] of Object.entries(summary).sort((a,b) => b[1] - a[1])) {{
    const color = COLORS[id] || '#666';
    html += `<span class="badge" style="background:${{color}}">` +
            `${{id}}: ${{count}}/${{N_SAMPLES}}</span>`;
  }}
  html += '</div>';

  entries.forEach((e, i) => {{
    const color = COLORS[e.identity] || '#666';
    const thinkId = `think-${{key}}-${{i}}`;
    html += `<div class="response-card">
      <div class="response-header">
        <span class="response-num">#${{i+1}}</span>
        <span class="identity-tag" style="background:${{color}}">${{e.identity}}</span>
      </div>
      <div class="response-body">
        <div class="answer-text">${{escHtml(e.answer || '(no clean answer extracted)')}}</div>`;
    if (e.think) {{
      html += `<div class="think-toggle">
        <button class="think-btn" onclick="toggleThink('${{thinkId}}', this)">Show thinking</button>
        <div class="think-block" id="${{thinkId}}">${{escHtml(e.think)}}</div>
      </div>`;
    }} else {{
      html += `<div class="no-think">No clean &lt;think&gt; block extracted</div>`;
    }}
    html += '</div></div>';
  }});

  document.getElementById('content').innerHTML = html;
}}

function toggleThink(id, btn) {{
  const el = document.getElementById(id);
  el.classList.toggle('open');
  btn.textContent = el.classList.contains('open') ? 'Hide thinking' : 'Show thinking';
}}

function escHtml(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

// Build tabs grouped by stage
const tabsEl = document.getElementById('tabs');
let currentStage = null;
let groupEl = null;
CHECKPOINTS.forEach(cp => {{
  if (cp.stage !== currentStage) {{
    groupEl = document.createElement('div');
    groupEl.className = 'tab-group';
    const stageLabel = document.createElement('span');
    stageLabel.className = 'stage-label';
    stageLabel.textContent = cp.stage;
    groupEl.appendChild(stageLabel);
    tabsEl.appendChild(groupEl);
    currentStage = cp.stage;
  }}
  const t = document.createElement('div');
  t.className = 'tab';
  t.dataset.key = cp.key;
  // Show short label (strip stage prefix if present)
  const shortLabel = cp.label.replace(/^(SFT|DPO|RLVR|Final|Instruct)\\s*/, '');
  t.textContent = shortLabel || cp.label;
  t.onclick = () => setCheckpoint(cp.key);
  groupEl.appendChild(t);
}});

setCheckpoint(CHECKPOINTS[0].key);
</script>
</body>
</html>"""

    out = results_dir / "viewer.html"
    out.write_text(html)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize identity probe results")
    parser.add_argument("--model", choices=["7B", "32B"], default="7B",
                        help="Model size to visualize (default: 7B)")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    checkpoints = config["checkpoints"]
    title = config["title"]
    results_dir = Path(__file__).parent / "results" / args.model

    data = load_results(results_dir, checkpoints)
    make_chart(data, results_dir, checkpoints, title)
    make_html(data, results_dir, checkpoints, title)
