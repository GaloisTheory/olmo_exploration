# %% [markdown]
# # OLMo 3 Exploration
#
# Exploring Allen AI's fully open OLMo 3 7B model family:
# - Base model + intermediate training checkpoints
# - Training data (Dolma 3)
# - Post-training variants (Instruct, Think, RL-Zero)
#
# Paper: https://arxiv.org/abs/2512.13961
# WandB: https://wandb.ai/ai2-llm/Olmo-3-1025-7B/reports/Olmo-3-7B-October-2025--VmlldzoxNDcwOTM0NA

# %% Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import list_repo_refs, HfApi

# %%  Model registry — all the 7B variants
MODELS = {
    # Base (pre-trained only)
    "base": "allenai/Olmo-3-1025-7B",
    # Post-trained
    "instruct": "allenai/Olmo-3-7B-Instruct",
    "think": "allenai/Olmo-3-7B-Think",
    # RL-Zero variants (trained with RL from base, no SFT)
    "rl-math": "allenai/Olmo-3-7B-RL-Zero-Math",
    "rl-code": "allenai/Olmo-3-7B-RL-Zero-Code",
    "rl-if": "allenai/Olmo-3-7B-RL-Zero-IF",
    "rl-general": "allenai/Olmo-3-7B-RL-Zero-General",
    "rl-mix": "allenai/Olmo-3-7B-RL-Zero-Mix",
}

# Training data
DATASETS = {
    # Stage 1: pretraining (5.93T tokens, ~6T mix)
    "pretrain_pool": "allenai/dolma3_pool",  # raw pool before mixing
    # Stage 2: midtraining (100B tokens, high-quality mix)
    "dolmino": "allenai/dolma3_dolmino_mix-100B-1025",
    # Stage 3: long context (50B tokens)
    "longmino": "allenai/dolma3_longmino_mix-50B-1025",
}

# %% List all intermediate checkpoints (branches) for the base model
print("=== Intermediate Checkpoints (base model) ===")
refs = list_repo_refs("allenai/Olmo-3-1025-7B")
branches = sorted([b.name for b in refs.branches])
for b in branches:
    print(f"  {b}")
print(f"\nTotal: {len(branches)} checkpoints")

# %% Helper: load a model + tokenizer
def load_olmo(variant="base", revision=None, quantize_8bit=False, device_map="auto"):
    """Load an OLMo 3 7B variant.

    Args:
        variant: key from MODELS dict
        revision: branch name for intermediate checkpoint, e.g. "stage1-step10000"
        quantize_8bit: load in 8-bit (needs bitsandbytes)
        device_map: "auto", "cpu", or specific device
    """
    model_id = MODELS[variant]
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    if quantize_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["torch_dtype"] = torch.float16
    if revision:
        kwargs["revision"] = revision

    print(f"Loading {model_id}" + (f" @ {revision}" if revision else ""))
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model

# %% Which variants need chat template
CHAT_VARIANTS = {"instruct", "think", "rl-math", "rl-code", "rl-if", "rl-general", "rl-mix"}

# %% Helper: generate text
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9,
             chat=False):
    """Generate text. If chat=True, wraps prompt in chat template."""
    if chat:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        inputs = {"input_ids": input_ids}
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
        )
    # Only decode newly generated tokens
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)

# %% Helper: compare generations across variants or checkpoints
def compare_generations(configs, prompt, max_new_tokens=200, **gen_kwargs):
    """Generate from multiple model configs side-by-side.

    Args:
        configs: list of dicts with keys matching load_olmo args,
                 plus an optional "label" key for display
        prompt: text prompt
    """
    results = {}
    for cfg in configs:
        label = cfg.pop("label", cfg.get("variant", "?"))
        variant = cfg.get("variant", "base")
        chat = variant in CHAT_VARIANTS
        # Chat variants have their own tokenizer with chat_template set
        tok = AutoTokenizer.from_pretrained(MODELS[variant]) if chat else tokenizer
        model = load_olmo(**cfg)
        text = generate(model, tok, prompt, max_new_tokens=max_new_tokens, chat=chat, **gen_kwargs)
        results[label] = text
        # Free memory
        del model
        torch.cuda.empty_cache()

    print(f"\nPrompt: {prompt!r}\n" + "=" * 60)
    for label, text in results.items():
        print(f"\n--- {label} ---")
        print(text)
    return results

# %% Load tokenizer once (same across all checkpoints)
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")

# %% [markdown]
# ## 1. Quick sanity check — load base model and generate

# %% Load base model (8-bit to save VRAM)
model = load_olmo("base", quantize_8bit=True)

# %%
# print(generate(model, tokenizer, "The meaning of life is"))

# %%
# print(generate(model, tokenizer, "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\""))

# %% Clean up
# del model; torch.cuda.empty_cache()

# %% Load early checkpoint (stage1-step1000) — how bad is it?
# model_0 = load_olmo("base", revision="stage1-step1000", quantize_8bit=True)

# %%
# print(generate(model_0, tokenizer, "The meaning of life is"))

# %%
# print(generate(model_0, tokenizer, "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\""))

# %% Clean up
# del model_0; torch.cuda.empty_cache()

# %% [markdown]
# ## 2. Compare base vs instruct vs think

# %% Side-by-side comparison
compare_generations(
    configs=[
        {"label": "base", "variant": "base", },
        {"label": "instruct", "variant": "instruct",},
        {"label": "think", "variant": "think", },
    ],
    prompt="Explain why the sky is blue in one paragraph.",
    max_new_tokens=200,
)

# %% [markdown]
# ## 3. Intermediate checkpoints — watch the model learn
#
# The base model has checkpoints at each training stage:
# - `stage1-stepXXX` — pretraining (5.93T tokens)
# - `stage2-stepXXX` — midtraining on Dolmino (100B tokens)
# - `stage3-stepXXX` — long context (50B tokens)

# %% Pick a few checkpoints to compare
# Filter to get a reasonable spread of stage1 checkpoints
stage1_ckpts = [b for b in branches if b.startswith("stage1-")]
stage2_ckpts = [b for b in branches if b.startswith("stage2-")]
stage3_ckpts = [b for b in branches if b.startswith("stage3-")]

print(f"Stage 1 checkpoints: {len(stage1_ckpts)}")
print(f"Stage 2 checkpoints: {len(stage2_ckpts)}")
print(f"Stage 3 checkpoints: {len(stage3_ckpts)}")

# Show first/last few of each
for name, ckpts in [("stage1", stage1_ckpts), ("stage2", stage2_ckpts), ("stage3", stage3_ckpts)]:
    if ckpts:
        print(f"\n{name}: {ckpts[:3]} ... {ckpts[-3:]}")

# %% Compare an early vs late stage1 checkpoint
# (uncomment and adjust checkpoint names based on what's available above)
# compare_generations(
#     configs=[
#         {"label": "stage1-early", "variant": "base", "revision": stage1_ckpts[0], "quantize_8bit": True},
#         {"label": "stage1-late", "variant": "base", "revision": stage1_ckpts[-1], "quantize_8bit": True},
#         {"label": "final", "variant": "base", "quantize_8bit": True},
#     ],
#     prompt="The capital of France is",
#     max_new_tokens=50,
#     temperature=0,
# )

# %% [markdown]
# ## 4. Training Data — Dolma 3
#
# Three stages of curated data:
# | Stage | Dataset | Tokens | Description |
# |-------|---------|--------|-------------|
# | 1 | dolma3_pool | ~6T | Web, code, academic, books |
# | 2 | dolmino | 100B | High-quality: math, code, QA, reasoning |
# | 3 | longmino | 50B | Long-context documents |

# %% Browse the Dolmino midtraining data (streaming — no full download)
dolmino = load_dataset(
    "allenai/dolma3_dolmino_mix-100B-1025",
    name="dolmino_1_flan",  # instruction tuning subset
    split="train",
    streaming=True,
)

# %% Look at a few samples
for i, sample in enumerate(dolmino):
    print(f"\n{'='*60}")
    print(f"Sample {i} | source: {sample.get('source', '?')} | id: {sample['id'][:40]}")
    print(f"{'='*60}")
    text = sample["text"]
    print(text[:1000] + ("..." if len(text) > 1000 else ""))
    if i >= 4:
        break

# %% Explore different Dolmino subsets
# Available subsets (each is a different data source):
api = HfApi()
dataset_info = api.dataset_info("allenai/dolma3_dolmino_mix-100B-1025")
if hasattr(dataset_info, "card_data") and dataset_info.card_data:
    configs = dataset_info.card_data.get("configs", [])
    if configs:
        print("Available subsets:")
        for c in configs:
            name = c if isinstance(c, str) else c.get("config_name", c)
            print(f"  - {name}")

# %% Browse the raw pretraining pool (web crawl data)
pool = load_dataset(
    "allenai/dolma3_pool",
    split="train",
    streaming=True,
)

for i, sample in enumerate(pool):
    print(f"\n{'='*60}")
    meta = sample.get("metadata", {})
    url = meta.get("warc_url", "?") if isinstance(meta, dict) else "?"
    category = meta.get("weborganizer_max", "?") if isinstance(meta, dict) else "?"
    print(f"Sample {i} | category: {category} | url: {url}")
    print(f"{'='*60}")
    text = sample["text"]
    print(text[:800] + ("..." if len(text) > 800 else ""))
    if i >= 4:
        break

# %% [markdown]
# ## 5. Dolmino subset deep-dives
#
# The midtraining mix has 27 subsets. Some interesting ones:
# - `cranecode` — synthetic code
# - `cranemath` — synthetic math
# - `tinyMATH_mind` / `tinyMATH_pot` — math reasoning traces
# - `qwq_thinking_traces` / `r1_thinking_traces` — reasoning from QwQ / R1
# - `common_crawl_hq` — high-quality web
# - `reddit_high` — top Reddit content

# %% Look at math reasoning traces
math_data = load_dataset(
    "allenai/dolma3_dolmino_mix-100B-1025",
    name="tinyMATH_mind",
    split="train",
    streaming=True,
)

print("=== Math Reasoning Traces (tinyMATH_mind) ===\n")
for i, sample in enumerate(math_data):
    print(f"--- Sample {i} ---")
    print(sample["text"][:1500])
    print()
    if i >= 2:
        break

# %% Look at code data
code_data = load_dataset(
    "allenai/dolma3_dolmino_mix-100B-1025",
    name="cranecode",
    split="train",
    streaming=True,
)

print("=== Synthetic Code (cranecode) ===\n")
for i, sample in enumerate(code_data):
    print(f"--- Sample {i} ---")
    print(sample["text"][:1500])
    print()
    if i >= 2:
        break

# %% [markdown]
# ## 6. RL-Zero variants — trained with RL directly from base (no SFT!)
#
# OLMo 3 includes "RL-Zero" models that skip supervised fine-tuning entirely
# and go straight from the base model to RL. Interesting to compare.

# %% Compare RL-Zero-Math vs Instruct on a math problem
# compare_generations(
#     configs=[
#         {"label": "base", "variant": "base", "quantize_8bit": True},
#         {"label": "instruct", "variant": "instruct", "quantize_8bit": True},
#         {"label": "rl-math", "variant": "rl-math", "quantize_8bit": True},
#     ],
#     prompt="What is the sum of the first 100 positive integers?",
#     max_new_tokens=300,
# )

# %% [markdown]
# ## 7. Tokenizer exploration

# %% Load tokenizer (doesn't need GPU)
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")

print(f"Vocab size: {tok.vocab_size}")
print(f"Model max length: {tok.model_max_length}")
print(f"Special tokens: {tok.all_special_tokens}")
print(f"EOS: {tok.eos_token!r} (id={tok.eos_token_id})")
print(f"BOS: {tok.bos_token!r} (id={tok.bos_token_id})")
print(f"PAD: {tok.pad_token!r} (id={tok.pad_token_id})")

# %% Tokenization examples
examples = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "∫₀^∞ e^(-x²) dx = √π/2",
    "<think>Let me reason about this step by step.</think>",
]

for text in examples:
    tokens = tok.encode(text)
    decoded_tokens = [tok.decode([t]) for t in tokens]
    print(f"\n{text!r}")
    print(f"  {len(tokens)} tokens: {decoded_tokens[:20]}{'...' if len(decoded_tokens) > 20 else ''}")

# %% [markdown]
# ## 8. Model architecture inspection

# %% Look at model structure (load on CPU, no weights needed for inspection)
from transformers import AutoConfig

config = AutoConfig.from_pretrained("allenai/Olmo-3-1025-7B")
print("=== OLMo 3 7B Config ===")
for k, v in sorted(config.to_dict().items()):
    if not k.startswith("_"):
        print(f"  {k}: {v}")


# %% [markdown]
# ## Resources
#
# - **Paper**: https://arxiv.org/abs/2512.13961
# - **HF Collection**: https://huggingface.co/collections/allenai/olmo-3
# - **Training Code**: https://github.com/allenai/OLMo-core
# - **Data Pipeline**: https://github.com/allenai/dolma3
# - **Eval Code**: https://github.com/allenai/OLMo-Eval
# - **Fine-tuning**: https://github.com/allenai/open-instruct
# - **License**: Apache 2.0 (model), ODC-By (data)
