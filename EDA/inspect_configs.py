#!/usr/bin/env python3
"""Inspect full configs of representative OLMo-3 runs from WandB.

Picks one run from each category (pre-training, named-pre-training, annealing)
and dumps dataset, data_loader, _CLASS_ configs, looking for HF dataset references.
"""

import json
import re
import wandb

PROJECT = "ai2-llm/Olmo-3-1025-7B"

# Representative runs from each category
RUNS = {
    "pre-training (timestamp)": "asnd285v",
    "named-pre-training": "tx9cibv8",
    "annealing": None,  # We'll find one dynamically
}


def get_annealing_run_id(api: wandb.Api) -> str:
    """Find an annealing run id."""
    runs = api.runs(PROJECT, filters={"display_name": {"$regex": "^anneal-round5"}}, per_page=3)
    for r in runs:
        return r.id
    raise RuntimeError("No annealing run found")


def extract_data_references(obj, path="") -> list[str]:
    """Recursively search a config object for anything that looks like a data source reference."""
    refs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}" if path else k
            refs.extend(extract_data_references(v, current_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            refs.extend(extract_data_references(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        # Look for HF-style references
        patterns = [
            (r"https?://huggingface\.co/\S+", "HF URL"),
            (r"hf://\S+", "HF protocol URL"),
            (r"s3://\S+", "S3 path"),
            (r"gs://\S+", "GCS path"),
            (r"weka://\S+", "Weka path"),
            (r"/data/\S+", "Data path"),
            (r"/datasets?/\S+", "Dataset path"),
            (r"allenai/\S+", "AllenAI reference"),
            (r"olmo\S*", "OLMo reference"),
            (r"dolma\S*", "Dolma reference"),
            (r"\.npy$|\.jsonl$|\.parquet$|\.arrow$|\.bin$|\.tar$", "Data file extension"),
        ]
        for pattern, label in patterns:
            matches = re.findall(pattern, obj, re.IGNORECASE)
            for m in matches:
                refs.append(f"  [{label}] {path} = {m}")
    return refs


def dump_config_section(config: dict, key: str, label: str):
    """Pretty-print a config section, handling JSON-string or dict values."""
    val = config.get(key)
    if val is None:
        print(f"\n{'='*60}")
        print(f"  {label}: (not present)")
        print(f"{'='*60}")
        return None

    # If it's a JSON string, parse it
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if isinstance(val, (dict, list)):
        print(json.dumps(val, indent=2))
    else:
        print(val)
    return val


def inspect_run(api: wandb.Api, run_id: str, category: str):
    """Inspect a single run's config."""
    print(f"\n{'#'*80}")
    print(f"# CATEGORY: {category}")
    print(f"# Run ID: {run_id}")
    print(f"{'#'*80}")

    run = api.run(f"{PROJECT}/{run_id}")
    print(f"Run name: {run.name}")
    print(f"Run state: {run.state}")
    config = run.config

    # Print all top-level keys
    print(f"\nTop-level config keys: {sorted(config.keys())}")

    # 1. _CLASS_
    dump_config_section(config, "_CLASS_", f"_CLASS_ for {category}")

    # 2. dataset
    dataset_val = dump_config_section(config, "dataset", f"dataset config for {category}")

    # 3. data_loader
    dl_val = dump_config_section(config, "data_loader", f"data_loader config for {category}")

    # 4. Check trainer for data references
    trainer_val = config.get("trainer")
    if isinstance(trainer_val, str):
        try:
            trainer_val = json.loads(trainer_val)
        except json.JSONDecodeError:
            pass

    model_val = config.get("model")
    if isinstance(model_val, str):
        try:
            model_val = json.loads(model_val)
        except json.JSONDecodeError:
            pass

    # 5. Search for data references across ALL config sections
    print(f"\n{'='*60}")
    print(f"  Data source references found in FULL config ({category})")
    print(f"{'='*60}")

    all_refs = []
    for section_name in sorted(config.keys()):
        section = config[section_name]
        if isinstance(section, str):
            try:
                section = json.loads(section)
            except (json.JSONDecodeError, TypeError):
                pass
        refs = extract_data_references(section, section_name)
        all_refs.extend(refs)

    if all_refs:
        for ref in all_refs:
            print(ref)
    else:
        print("  (no explicit data source references found)")

    # Also dump trainer and model sections for completeness
    # but only the parts that might reference data
    print(f"\n{'='*60}")
    print(f"  trainer config (abbreviated) for {category}")
    print(f"{'='*60}")
    if isinstance(trainer_val, dict):
        # Print full trainer config
        print(json.dumps(trainer_val, indent=2))
    elif trainer_val is not None:
        print(trainer_val)
    else:
        print("  (not present)")

    print(f"\n{'='*60}")
    print(f"  model config for {category}")
    print(f"{'='*60}")
    if isinstance(model_val, dict):
        print(json.dumps(model_val, indent=2))
    elif model_val is not None:
        print(model_val)
    else:
        print("  (not present)")

    # Also dump train_module and launch for completeness
    for extra_key in ["train_module", "launch", "run_name", "init_seed"]:
        ev = config.get(extra_key)
        if ev is not None:
            if isinstance(ev, str):
                try:
                    ev = json.loads(ev)
                except (json.JSONDecodeError, TypeError):
                    pass
            print(f"\n{'='*60}")
            print(f"  {extra_key} for {category}")
            print(f"{'='*60}")
            if isinstance(ev, (dict, list)):
                print(json.dumps(ev, indent=2))
            else:
                print(ev)

    return config


def main():
    api = wandb.Api()

    # Find an annealing run
    annealing_id = get_annealing_run_id(api)
    RUNS["annealing"] = annealing_id
    print(f"Selected annealing run: {annealing_id}")

    configs = {}
    for category, run_id in RUNS.items():
        configs[category] = inspect_run(api, run_id, category)

    # Final summary
    print(f"\n\n{'#'*80}")
    print("# SUMMARY: Data sources by run type")
    print(f"{'#'*80}")

    for category, config in configs.items():
        print(f"\n## {category}")
        dataset = config.get("dataset")
        if isinstance(dataset, str):
            try:
                dataset = json.loads(dataset)
            except json.JSONDecodeError:
                pass

        if isinstance(dataset, dict):
            # Look for key fields
            for interesting_key in [
                "name", "path", "paths", "source", "sources",
                "mix", "mixture", "data_dir", "data_path",
                "label", "identifier", "type", "_CLASS_",
            ]:
                if interesting_key in dataset:
                    val = dataset[interesting_key]
                    if isinstance(val, (dict, list)):
                        print(f"  dataset.{interesting_key} = {json.dumps(val, indent=4)}")
                    else:
                        print(f"  dataset.{interesting_key} = {val}")

            # Print ALL keys for overview
            print(f"  dataset keys: {sorted(dataset.keys())}")
        else:
            print(f"  dataset = {dataset}")

        dl = config.get("data_loader")
        if isinstance(dl, str):
            try:
                dl = json.loads(dl)
            except json.JSONDecodeError:
                pass
        if isinstance(dl, dict):
            print(f"  data_loader keys: {sorted(dl.keys())}")
        else:
            print(f"  data_loader = {dl}")

        print(f"  _CLASS_ = {config.get('_CLASS_')}")


if __name__ == "__main__":
    main()
