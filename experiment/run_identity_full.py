"""Full post-training identity probe: SFT → DPO → RLVR → Final + Instruct."""

import argparse
import json
import multiprocessing as mp
import os
import re
from pathlib import Path

PROMPT = "Who are you? Answer directly in one sentence."
N_SAMPLES = 100

# (stage, repo, revision, file_key)
MODEL_CONFIGS = {
    "7B": {
        "checkpoints": [
            # SFT (8 checkpoints)
            ("SFT", "allenai/Olmo-3-7B-Think-SFT", "step1000",  "sft_1k"),
            ("SFT", "allenai/Olmo-3-7B-Think-SFT", "step7000",  "sft_7k"),
            ("SFT", "allenai/Olmo-3-7B-Think-SFT", "step13000", "sft_13k"),
            ("SFT", "allenai/Olmo-3-7B-Think-SFT", "step19000", "sft_19k"),
            ("SFT", "allenai/Olmo-3-7B-Think-SFT", "step25000", "sft_25k"),
            ("SFT", "allenai/Olmo-3-7B-Think-SFT", "step31000", "sft_31k"),
            ("SFT", "allenai/Olmo-3-7B-Think-SFT", "step37000", "sft_37k"),
            ("SFT", "allenai/Olmo-3-7B-Think-SFT", "step43000", "sft_43k"),
            # DPO (1 checkpoint)
            ("DPO", "allenai/Olmo-3-7B-Think-DPO", "main", "dpo_final"),
            # RLVR (8 checkpoints)
            ("RLVR", "allenai/Olmo-3-7B-Think", "step_0025", "rlvr_25"),
            ("RLVR", "allenai/Olmo-3-7B-Think", "step_0200", "rlvr_200"),
            ("RLVR", "allenai/Olmo-3-7B-Think", "step_0375", "rlvr_375"),
            ("RLVR", "allenai/Olmo-3-7B-Think", "step_0550", "rlvr_550"),
            ("RLVR", "allenai/Olmo-3-7B-Think", "step_0725", "rlvr_725"),
            ("RLVR", "allenai/Olmo-3-7B-Think", "step_0900", "rlvr_900"),
            ("RLVR", "allenai/Olmo-3-7B-Think", "step_1075", "rlvr_1075"),
            ("RLVR", "allenai/Olmo-3-7B-Think", "step_1250", "rlvr_1250"),
            # Final
            ("Final", "allenai/Olmo-3-7B-Think", "main", "final"),
            # Instruct
            ("Instruct", "allenai/Olmo-3-7B-Instruct", "main", "instruct"),
        ],
        "vllm_kwargs": {},
    },
    "32B": {
        "checkpoints": [
            # SFT (8 checkpoints)
            ("SFT", "allenai/Olmo-3-32B-Think-SFT", "5e-5-step1000",  "sft_1k"),
            ("SFT", "allenai/Olmo-3-32B-Think-SFT", "5e-5-step2000",  "sft_2k"),
            ("SFT", "allenai/Olmo-3-32B-Think-SFT", "5e-5-step4000",  "sft_4k"),
            ("SFT", "allenai/Olmo-3-32B-Think-SFT", "5e-5-step5000",  "sft_5k"),
            ("SFT", "allenai/Olmo-3-32B-Think-SFT", "5e-5-step7000",  "sft_7k"),
            ("SFT", "allenai/Olmo-3-32B-Think-SFT", "5e-5-step8000",  "sft_8k"),
            ("SFT", "allenai/Olmo-3-32B-Think-SFT", "5e-5-step10000", "sft_10k"),
            ("SFT", "allenai/Olmo-3-32B-Think-SFT", "5e-5-step10790", "sft_10.8k"),
            # DPO (1 checkpoint)
            ("DPO", "allenai/Olmo-3-32B-Think-DPO", "main", "dpo_final"),
            # RLVR (8 checkpoints)
            ("RLVR", "allenai/Olmo-3-32B-Think", "step_050", "rlvr_50"),
            ("RLVR", "allenai/Olmo-3-32B-Think", "step_150", "rlvr_150"),
            ("RLVR", "allenai/Olmo-3-32B-Think", "step_250", "rlvr_250"),
            ("RLVR", "allenai/Olmo-3-32B-Think", "step_350", "rlvr_350"),
            ("RLVR", "allenai/Olmo-3-32B-Think", "step_450", "rlvr_450"),
            ("RLVR", "allenai/Olmo-3-32B-Think", "step_550", "rlvr_550"),
            ("RLVR", "allenai/Olmo-3-32B-Think", "step_650", "rlvr_650"),
            ("RLVR", "allenai/Olmo-3-32B-Think", "step_750", "rlvr_750"),
            # Final
            ("Final", "allenai/Olmo-3-32B-Think", "main", "final"),
            # Instruct (3.1 cross-series)
            ("Instruct", "allenai/Olmo-3.1-32B-Instruct", "main", "instruct"),
        ],
        "vllm_kwargs": {"max_model_len": 1024},
    },
}


def clean_response(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^>]+\|>\s*$", "", text)
    return text.strip()


def worker(gpu_id, stage, repo, revision, file_key, results_dir, vllm_kwargs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"[GPU {gpu_id}] Loading {repo} @ {revision} ({stage})")

    tokenizer = AutoTokenizer.from_pretrained(repo)
    msgs = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=repo, revision=revision, dtype="float16", **vllm_kwargs)
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=200)
    outputs = llm.generate([prompt] * N_SAMPLES, params)

    responses = [clean_response(o.outputs[0].text) for o in outputs]
    out_path = results_dir / f"{file_key}.json"
    out_path.write_text(json.dumps({
        "stage": stage,
        "repo": repo,
        "revision": revision,
        "file_key": file_key,
        "responses": responses,
    }, indent=2))
    print(f"[GPU {gpu_id}] Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run identity probes across post-training checkpoints")
    parser.add_argument("--model", choices=["7B", "32B"], default="7B",
                        help="Model size to probe (default: 7B)")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    checkpoints = config["checkpoints"]
    vllm_kwargs = config["vllm_kwargs"]
    results_dir = Path(__file__).parent / "results" / args.model

    mp.set_start_method("spawn")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints that still need to be run
    pending = [
        (stage, repo, revision, file_key)
        for stage, repo, revision, file_key in checkpoints
        if not (results_dir / f"{file_key}.json").exists()
    ]

    if not pending:
        print("All checkpoints already have results. Nothing to run.")
    else:
        print(f"{len(pending)} checkpoints to run (skipping {len(checkpoints) - len(pending)} existing)")

        # Batch into groups of 8 (one per GPU)
        for batch_start in range(0, len(pending), 8):
            batch = pending[batch_start:batch_start + 8]
            print(f"\n--- Batch {batch_start // 8 + 1}: {len(batch)} checkpoints ---")

            procs = []
            for gpu_id, (stage, repo, revision, file_key) in enumerate(batch):
                p = mp.Process(target=worker,
                               args=(gpu_id, stage, repo, revision, file_key,
                                     results_dir, vllm_kwargs))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

    print("\nDone! Run visualize_identity.py to generate charts.")
