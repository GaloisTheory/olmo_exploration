"""Interactive REPL for OLMo-3-7B-Think checkpoints (vLLM). Swap checkpoints without restarting."""

import gc
import re
import sys

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO = "allenai/Olmo-3-7B-Think"

tokenizer = AutoTokenizer.from_pretrained(REPO)

def load_model(revision="main"):
    label = f"{REPO} @ {revision}"
    print(f"\nLoading {label} ...")
    llm = LLM(model=REPO, revision=revision, dtype="float16")
    print(f"Ready: {label}\n")
    return llm, revision

def unload_model(llm):
    del llm
    gc.collect()
    torch.cuda.empty_cache()

def format_prompt(user_input, chat_mode):
    if not chat_mode:
        return user_input
    msgs = [{"role": "user", "content": user_input}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def format_output(text, chat_mode):
    if not chat_mode:
        return text
    # Strip trailing special tokens
    text = re.sub(r"<\|[^>]+\|>\s*$", "", text).strip()
    # Separate <think> block if present
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text[think_match.end():].strip()
        parts = []
        if thinking:
            parts.append(f"[Think]\n{thinking}")
        parts.append(answer)
        return "\n\n".join(parts)
    return text

# Initial load — accept optional step from CLI: python olmo_chat.py 275
rev = f"step_{int(sys.argv[1]):04d}" if len(sys.argv) > 1 else "main"
llm, current_rev = load_model(rev)

chat_mode = True

print("Commands:")
print("  /s <step>    — load checkpoint (e.g. /s 275, /s 1375, /s main)")
print("  /c           — toggle chat mode (currently: ON)")
print("  /t <float>   — set temperature")
print("  /m <int>     — set max tokens")
print("  /w           — show current model + settings")
print("  empty line   — quit\n")

temperature = 0.6
max_tokens = 512

while True:
    mode_tag = "chat" if chat_mode else "raw"
    try:
        prompt = input(f"[{current_rev}|{mode_tag}] >>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not prompt:
        print("Bye!")
        break

    if prompt.startswith("/s "):
        arg = prompt[3:].strip()
        rev = arg if arg == "main" else f"step_{int(arg):04d}"
        unload_model(llm)
        llm, current_rev = load_model(rev)
        continue
    if prompt == "/c":
        chat_mode = not chat_mode
        print(f"chat mode: {'ON' if chat_mode else 'OFF'}")
        continue
    if prompt.startswith("/t "):
        temperature = float(prompt[3:])
        print(f"temperature = {temperature}")
        continue
    if prompt.startswith("/m "):
        max_tokens = int(prompt[3:])
        print(f"max_tokens = {max_tokens}")
        continue
    if prompt == "/w":
        print(f"model: {REPO} @ {current_rev}")
        print(f"chat_mode: {chat_mode}, temperature: {temperature}, max_tokens: {max_tokens}")
        continue

    full_prompt = format_prompt(prompt, chat_mode)

    params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    outputs = llm.generate([full_prompt], params)
    text = outputs[0].outputs[0].text
    print(f"\n{format_output(text, chat_mode)}\n")
