"""Observe how OLMo's identity emerges during SFT by asking 'Who are you?' at 8 checkpoints."""

import json
import multiprocessing as mp
import os
import re
from pathlib import Path

REPO = "allenai/Olmo-3-7B-Think-SFT"
STEPS = [1000, 7000, 13000, 19000, 25000, 31000, 37000, 43000]
PROMPT = "Who are you? Answer directly in one sentence."
N_SAMPLES = 100
RESULTS_DIR = Path(__file__).parent / "results"


def clean_response(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^>]+\|>\s*$", "", text)
    return text.strip()


def worker(gpu_id, step):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    revision = f"step{step}"
    print(f"[GPU {gpu_id}] Loading {REPO} @ {revision}")

    tokenizer = AutoTokenizer.from_pretrained(REPO)
    msgs = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=REPO, revision=revision, dtype="float16")
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=200)
    outputs = llm.generate([prompt] * N_SAMPLES, params)

    responses = [clean_response(o.outputs[0].text) for o in outputs]
    out_path = RESULTS_DIR / f"step{step}.json"
    out_path.write_text(json.dumps({"step": step, "responses": responses}, indent=2))
    print(f"[GPU {gpu_id}] Saved {out_path}")


def plot_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    fig.suptitle("OLMo SFT Identity Evolution: 'Who are you?'", fontsize=16, y=0.98)

    for ax, step in zip(axes.flat, STEPS):
        data = json.loads((RESULTS_DIR / f"step{step}.json").read_text())
        lines = [f"{i+1}. {r}" for i, r in enumerate(data["responses"])]
        ax.set_title(f"Step {step}", fontsize=13, fontweight="bold")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, verticalalignment="top", family="monospace", wrap=True)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS_DIR / "identity_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    procs = []
    for gpu_id, step in enumerate(STEPS):
        p = mp.Process(target=worker, args=(gpu_id, step))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    plot_results()
