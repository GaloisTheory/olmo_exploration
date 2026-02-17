# %% Setup
import os
from pathlib import Path
import html

if not os.environ.get("HF_TOKEN"):
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

import polars as pl

SFT_PARQUET = "hf://datasets/allenai/Dolci-Think-SFT-32B/**/*.parquet"

print("Scanning SFT parquet files...")
sft_lf = pl.scan_parquet(SFT_PARQUET)
print("Collecting full SFT dataset...")
sft_df = sft_lf.collect()
print(f"Collected: {len(sft_df):,} rows")

# %% Build HTML
# --- hard_coded identity prompts ---
identity_df = (
    sft_df
    .with_columns(
        pl.col("id")
        .str.replace(r"_[a-f0-9]{8}-[a-f0-9-]+.*$", "")
        .str.replace(r"_[a-z0-9]{7}$", "")
        .str.replace(r"_\d+$", "")
        .alias("dataset_name")
    )
    .filter(pl.col("dataset_name") == "hard_coded")
)
print(f"hard_coded prompts: {len(identity_df):,}")

# --- OLMo mentions ---
olmo_matches = (
    sft_df
    .explode("messages")
    .with_columns(
        pl.col("messages").struct.field("role").alias("role"),
        pl.col("messages").struct.field("content").alias("content"),
    )
    .filter(pl.col("content").str.contains("(?i)OLMo"))
)
olmo_sources = olmo_matches.group_by("source").len().sort("len", descending=True)
# Get unique conversation ids that mention OLMo
olmo_conv_ids = olmo_matches["id"].unique().to_list()
olmo_convs = sft_df.filter(pl.col("id").is_in(olmo_conv_ids))
print(f"OLMo-mentioning conversations: {len(olmo_convs):,}")


def escape(text: str) -> str:
    return html.escape(text)


def render_message(msg: dict) -> str:
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    role_class = role

    if role == "assistant" and "<think>" in content and "</think>" in content:
        think_start = content.index("<think>")
        think_end = content.index("</think>") + len("</think>")
        think_block = content[think_start + len("<think>"):think_end - len("</think>")]
        answer = content[think_end:].strip()
        before_think = content[:think_start].strip()

        parts = []
        if before_think:
            parts.append(f'<div class="msg assistant"><div class="role-badge assistant">assistant</div>'
                         f'<div class="content">{escape(before_think)}</div></div>')
        parts.append(
            f'<details class="think-block" open>'
            f'<summary class="think-toggle">thinking ({len(think_block):,} chars)</summary>'
            f'<div class="think-content">{escape(think_block)}</div>'
            f'</details>'
        )
        if answer:
            parts.append(
                f'<div class="msg answer"><div class="role-badge answer">answer</div>'
                f'<div class="content">{escape(answer)}</div></div>'
            )
        return "\n".join(parts)

    return (f'<div class="msg {role_class}"><div class="role-badge {role_class}">{escape(role)}</div>'
            f'<div class="content">{escape(content)}</div></div>')


def render_card(idx: int, row: dict, section: str) -> str:
    msgs = row["messages"]
    source = row.get("source", "")
    row_id = row.get("id", "")
    msg_html = "\n".join(render_message(m) for m in msgs)
    return f"""
    <div class="card">
      <div class="card-header">
        <span class="card-number">#{idx}</span>
        <span class="card-meta">id: <code>{escape(row_id)}</code></span>
        <span class="card-meta">source: <code>{escape(source)}</code></span>
      </div>
      <div class="card-body">{msg_html}</div>
    </div>"""


# Build cards
identity_cards = []
for i, row in enumerate(identity_df.iter_rows(named=True), 1):
    identity_cards.append(render_card(i, row, "identity"))

olmo_cards = []
for i, row in enumerate(olmo_convs.iter_rows(named=True), 1):
    olmo_cards.append(render_card(i, row, "olmo"))

# Source breakdown table
source_rows_html = ""
for row in olmo_sources.iter_rows(named=True):
    source_rows_html += f"<tr><td><code>{escape(row['source'])}</code></td><td>{row['len']:,}</td></tr>\n"

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OLMo Identity Prompts — Dolci-Think-SFT-32B</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --text-faint: #484f58;
    --accent: #58a6ff; --green: #3fb950; --orange: #d29922;
    --purple: #bc8cff; --pink: #f778ba;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    padding: 2rem; max-width: 1000px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 2.5rem; font-size: 0.95rem; }}
  h2 {{
    font-size: 1.3rem; margin: 3rem 0 1.5rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .stats {{ color: var(--text-dim); font-size: 0.9rem; margin-bottom: 1.5rem; }}

  /* Cards */
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 2.5rem; overflow: hidden;
  }}
  .card-header {{
    padding: 0.75rem 1.25rem; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
    background: rgba(255,255,255,0.02);
  }}
  .card-number {{
    font-weight: 700; font-size: 1rem; color: var(--accent);
    min-width: 2.5rem;
  }}
  .card-meta {{ font-size: 0.8rem; color: var(--text-dim); }}
  .card-meta code {{ color: var(--text-faint); font-size: 0.75rem; }}
  .card-body {{ padding: 1.25rem; display: flex; flex-direction: column; gap: 1.25rem; }}

  /* Messages */
  .msg {{ border-radius: 8px; padding: 1rem 1.25rem; }}
  .role-badge {{
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding: 0.15rem 0.5rem; border-radius: 4px; margin-bottom: 0.6rem;
  }}
  .role-badge.system {{ background: rgba(188,140,255,0.15); color: var(--purple); }}
  .role-badge.user {{ background: rgba(88,166,255,0.15); color: var(--accent); }}
  .role-badge.assistant {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .role-badge.answer {{ background: rgba(63,185,80,0.25); color: var(--green); }}
  .msg.system {{ background: rgba(188,140,255,0.04); border-left: 3px solid var(--purple); }}
  .msg.user {{ background: rgba(88,166,255,0.05); border-left: 3px solid var(--accent); }}
  .msg.assistant {{ background: rgba(63,185,80,0.04); border-left: 3px solid var(--green); }}
  .msg.answer {{ background: rgba(63,185,80,0.07); border-left: 3px solid var(--green); }}
  .content {{ white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; line-height: 1.7; }}

  /* Think block */
  .think-block {{
    border: 1px solid var(--border); border-radius: 8px;
    background: rgba(210,153,34,0.04); overflow: hidden;
  }}
  .think-toggle {{
    cursor: pointer; padding: 0.6rem 1.25rem; font-size: 0.75rem;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--orange); user-select: none;
    list-style: none;
  }}
  .think-toggle::-webkit-details-marker {{ display: none; }}
  .think-toggle::before {{ content: "\\25B6\\FE0F  "; }}
  details[open] .think-toggle::before {{ content: "\\25BC\\FE0F  "; }}
  .think-content {{
    padding: 0.75rem 1.25rem 1rem; white-space: pre-wrap; word-break: break-word;
    font-size: 0.85rem; line-height: 1.65; color: var(--text-dim);
    border-top: 1px solid var(--border); max-height: 600px; overflow-y: auto;
  }}

  /* Source table */
  table {{ border-collapse: collapse; margin-bottom: 1.5rem; width: 100%; }}
  th, td {{ text-align: left; padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
  th {{ color: var(--text-dim); font-weight: 600; }}
  td code {{ color: var(--text-faint); font-size: 0.8rem; }}

  /* TOC */
  .toc {{ margin-bottom: 2rem; }}
  .toc a {{ color: var(--accent); text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<h1>OLMo Identity Prompts</h1>
<p class="subtitle">Dolci-Think-SFT-32B &mdash; {len(sft_df):,} total rows</p>

<nav class="toc">
  <strong>Sections:</strong>
  <a href="#hard-coded">hard_coded prompts ({len(identity_df)})</a> &middot;
  <a href="#olmo-mentions">OLMo mentions ({len(olmo_convs):,} conversations)</a>
</nav>

<h2 id="hard-coded">hard_coded Identity Prompts</h2>
<p class="stats">{len(identity_df)} prompts from the <code>hard_coded</code> dataset subset</p>
{"".join(identity_cards)}

<h2 id="olmo-mentions">Conversations Mentioning "OLMo"</h2>
<p class="stats">{len(olmo_matches):,} messages across {len(olmo_convs):,} conversations</p>

<table>
  <tr><th>Source</th><th>Messages</th></tr>
  {source_rows_html}
</table>

{"".join(olmo_cards)}

</body>
</html>"""

out_dir = Path(__file__).resolve().parent
out_path = out_dir / "identity_prompts.html"
out_path.write_text(HTML)
print(f"\nSaved: {out_path}")
