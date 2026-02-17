# OLMo-3 Think-SFT: Training Data Order & Reproducibility

## TL;DR

The data ordering is **fully deterministic** and **recoverable**. Every step maps to specific training examples through: tokenized `.npy` files → OBFD bin-packing → seeded shuffle per epoch → sequential batch consumption.

Reproducing the `.npy` files is **straightforward** — the conversion script is in the [open-instruct](https://github.com/allenai/open-instruct) repo. The main complexity is ensuring you use the exact same tokenizer, chat template, and dataset mix.

---

## Corrected Parameters (from Paper, Table 47)

The paper (arxiv:2512.13961, Section A.6.1) is the authoritative source. The training script has different defaults that were **overridden at launch time**.

| Parameter | 7B Think SFT (Paper) | 32B Think SFT (Paper) | Script Default |
|-----------|---------------------|----------------------|----------------|
| **Epochs** | **2** | **2** | 3 (overridden to 2) |
| **Learning Rate** | **5.0e-5** | **1.0e-4 souped with 5.0e-5** | 8e-5 (overridden) |
| **Batch Size** | **1M tokens** | **4M tokens** | 1M tokens |
| **Sequence Length** | **32K (32,768)** | **32K (32,768)** | 16,384 (overridden to 32K) |
| **Total Tokens** | **45.4B** | **45.2B** | — |
| **Num GPUs** | **64** | **256** | 8 (overridden) |
| Data loader seed | 34521 | 34521 | line 345 |
| Init seed | 33333 | 33333 | line 260 |
| Optimizer | AdamW (betas 0.9/0.95) | AdamW (betas 0.9/0.95) | line 352 |
| Weight decay | 0.0 | 0.0 | line 354 |
| Warmup | 3% linear | 3% linear | line 362 |
| LR schedule | Linear decay to 0 | Linear decay to 0 | line 363 |
| Checkpoint interval | 1000 steps | 1000 steps | line 384 |

### Corrected Batch Geometry (7B)

With the actual launch parameters:
- Sequence length = 32,768
- Global batch size = 1M tokens → **~30 instances per batch** (1,048,576 / 32,768 ≈ 32)
- 64 GPUs = 8 DP ranks (with CP degree needed for 32K seq len)

### Corrected Step/Epoch Mapping (7B)

- Total tokens: 45.4B over 2 epochs → ~22.7B tokens per epoch
- Steps per epoch: 22.7B / 1M batch = ~22,700 steps/epoch
- Total steps: ~45,400

| Checkpoint | Approx Epoch | Steps into epoch |
|------------|-------------|-----------------|
| step1000   | Epoch 1     | 1,000 |
| step7000   | Epoch 1     | 7,000 |
| step13000  | Epoch 1     | 13,000 |
| step19000  | Epoch 1     | 19,000 |
| step25000  | Epoch 2     | ~2,300 |
| step31000  | Epoch 2     | ~8,300 |
| step37000  | Epoch 2     | ~14,300 |
| step43000  | Epoch 2     | ~20,300 |

---

## Dolci Think SFT Dataset Composition (from Paper, Table 17)

The complete dataset mix for 7B Think SFT (2,268,468 total prompts):

| Category | Dataset | 7B Count | 32B Count | Source |
|----------|---------|----------|-----------|--------|
| **Math** | Dolci Think OpenThoughts 3+ Math | 752,997 | 752,997 | Guha et al. (2025) |
| **Math** | Dolci Think OpenThoughts 3+ STEM | 99,269 | 99,268 | Guha et al. (2025) |
| **Math** | SYNTHETIC-2-SFT-Verified | 104,569 | 104,548 | PrimeIntellect (2025) |
| **Code** | Nemotron Post-Training Code | 113,777 | 113,777 | NVIDIA AI (2025) |
| **Code** | Dolci Think OpenThoughts 3+ Code | 88,900 | 88,899 | Guha et al. (2025) |
| **Code** | Dolci Think Python Algorithms | 466,677 | 466,676 | AI2 (new) |
| **Chat** | WildChat | 83,054 | 76,209 | Zhao et al. (2024) |
| **Chat** | OpenAssistant | 6,800 | 6,647 | Kopf et al. (2024) |
| **IF** | Dolci Think Persona Precise IF | 223,123 | 220,530 | AI2 (new) |
| **IF** | Dolci Think Precise IF | 135,792 | 135,722 | AI2 (new) |
| **Safety** | CoCoNot | 10,227 | 9,549 | Brahman et al. (2024) |
| **Safety** | WildGuardMix | 38,315 | 36,673 | Han et al. (2024) |
| **Safety** | WildJailbreak | 41,100 | 40,002 | Jiang et al. (2024) |
| **Multi** | Aya | 98,597 | 97,156 | Singh et al. (2024) |
| **Other** | TableGPT | 4,981 | 4,973 | Zha et al. (2023) |
| **Other** | Olmo Identity Prompts | 290 | 290 | AI2 |
| | **Total** | **2,268,468** | **2,253,916** | |

Note: Datasets marked with ↑ in the paper are upsampled by repeating prompts with different completions.

### How Thinking Traces Were Generated

| Category | Trace Generator | Notes |
|----------|----------------|-------|
| Math (OpenThoughts3) | QwQ-32B | Regenerated incomplete traces up to 32K tokens (original was 16K) |
| Math (SYNTHETIC-2) | Original traces | Used verified subsection directly |
| Code (Python Algorithms) | QwQ-32B | Up to 16 responses per prompt, filtered by test cases from GPT-4.1 |
| Code (OpenThoughts3) | QwQ-32B | Regenerated incomplete examples |
| Chat & Safety | DeepSeek R1 | Generated reasoning traces and completions |
| Precise IF | QwQ-32B | Verified using constraint verifiers, kept only correct responses |
| Science | DeepSeek R1 | For non-OpenThoughts sources |

### Filtering Applied

1. Non-commercial / unclear license removal
2. Incomplete reasoning chain removal
3. Domain-specific verification (code test cases, IF constraint checking)
4. Model developer mentions and date cutoffs removed
5. Excessive repetition filtered
6. Chinese character / political value filtering
7. Topic filtering (removed image generation requests, excessive basic greetings, etc.)
8. Decontamination against evaluation benchmarks (using Tulu 3 procedure)

---

## Reproducing the `.npy` Files

### Difficulty: Moderate

The conversion is straightforward code-wise, but requires matching the exact pipeline.

### What You Need

1. **The open-instruct repo**: `git clone https://github.com/allenai/open-instruct`
2. **The HuggingFace dataset**: `allenai/Dolci-Think-SFT-7B` (2.27M rows)
3. **The tokenizer**: `allenai/dolma-2-tokenizer-olmo-3-instruct-final`
4. **GPU(s)**: The tokenization script uses GPUs to ensure sufficient CPU resources

### Conversion Command

```bash
python scripts/data/convert_sft_data_for_olmocore.py \
    --dataset_mixer_list \
       allenai/Dolci-Think-SFT-7B 1.0 \
    --tokenizer_name_or_path allenai/dolma-2-tokenizer-olmo-3-instruct-final \
    --output_dir /path/to/output \
    --chat_template_name "olmo123" \
    --max_seq_length 32768
```

### Output Files Produced

| File | dtype | Contents |
|------|-------|----------|
| `token_ids_part_NNNN.npy` | uint16 (chosen by vocab size) | Flat token IDs, conversations concatenated |
| `labels_mask_part_NNNN.npy` | `np.bool_` | `True` = train on this token (assistant responses), `False` = masked (user prompts) |
| `token_ids_part_NNNN.csv.gz` | gzipped CSV | Document boundaries (start, end positions) per chunk |
| `tokenizer/` | directory | Saved HuggingFace tokenizer |
| `dataset_statistics.json` | JSON | Per-dataset and overall stats |

Files are chunked at ~1GB boundaries.

### How the Conversion Works

1. Loads datasets from HuggingFace via `dataset_mixer_list` with weights
2. Shuffles with a fixed seed for reproducibility
3. Applies chat template (`olmo123`) to format conversations
4. Tokenizes using `sft_tulu_tokenize_and_truncate_v1` transform
5. Labels: `-100` for masked tokens (user/system), actual token IDs for trainable tokens (assistant)
6. Converts to: `labels_mask = [1 if label != -100 else 0 for label in labels]`
7. EOS tokens mark document boundaries (critical for OLMo-core's packing)
8. Writes flat numpy arrays, checkpointing progress

### Critical Reproducibility Notes

- **Chat template matters**: Must use `olmo123` (which loads from tokenizer). The `olmo` template uses a single EOS token per conversation — this is how OLMo-core finds document boundaries for packing.
- **Dataset shuffle seed**: The conversion script shuffles data with a fixed seed before processing. You'd need to match this seed.
- **Tokenizer version**: Must use `allenai/dolma-2-tokenizer-olmo-3-instruct-final` exactly.
- **Max sequence length**: Must be 32,768 to match training.

---

## The Data Ordering Pipeline (in OLMo-core)

### 1. Packing is deterministic

`NumpyPackedFSLDataset` uses the OBFD bin-packing algorithm. Same input `.npy` files → same packed instances with fixed indices 0 to N-1.

Source: `src/olmo_core/data/numpy_dataset.py:982`

### 2. Shuffling is deterministic per-epoch

From `src/olmo_core/data/data_loader.py:667-679`:

```python
def _build_global_indices(self):
    rng = get_rng(self.seed + self.epoch)  # seed=34521
    indices = np.arange(len(self.dataset))
    rng.shuffle(indices)
    return indices
```

| Epoch | RNG Seed | Formula |
|-------|----------|---------|
| 1 | 34522 | `get_rng(34521 + 1)` |
| 2 | 34523 | `get_rng(34521 + 2)` |

### 3. Batches consumed sequentially

From `data_loader.py:720-747`:

```python
indices = indices[:total_size]
indices = indices.reshape(-1, instances_per_batch)

# Skip already-processed batches on resume
if self.batches_processed > 0:
    indices = indices[self.batches_processed:]

# Each DP rank gets a stripe
indices = indices[:, dp_rank::dp_world_size]
```

---

## Reconstruction Code Sketch

### Step 1: Reproduce `.npy` files

(See conversion command above)

### Step 2: Run bin-packing

```python
from olmo_core.data import NumpyPackedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import LongDocStrategy

tokenizer_config = TokenizerConfig.dolma2()
dataset_config = NumpyPackedFSLDatasetConfig(
    tokenizer=tokenizer_config,
    work_dir="/path/to/work_dir",
    paths=["/path/to/output/token_ids_part_*.npy"],
    label_mask_paths=["/path/to/output/labels_mask_*.npy"],
    expand_glob=True,
    generate_doc_lengths=True,
    long_doc_strategy=LongDocStrategy.truncate,
    sequence_length=32768,  # CORRECTED: 32K, not 16K
)
dataset = dataset_config.build()
dataset.prepare()  # generates packing indices

total_instances = len(dataset)
print(f"Total packed instances: {total_instances}")
```

### Step 3: Replay shuffle

```python
import numpy as np
from olmo_core.data.utils import get_rng

seed = 34521

def get_epoch_permutation(epoch, total_instances):
    rng = get_rng(seed + epoch)
    indices = np.arange(total_instances, dtype=np.uint32)
    rng.shuffle(indices)
    return indices

epoch1_order = get_epoch_permutation(1, total_instances)
epoch2_order = get_epoch_permutation(2, total_instances)
```

### Step 4: Map steps to data

```python
instances_per_batch = 1_048_576 // 32_768  # = 32

steps_per_epoch = total_instances // instances_per_batch  # approx ~22,700

def get_data_between_checkpoints(step_a, step_b):
    """Returns packed instance indices seen between two steps."""
    results = []
    for step in range(step_a, step_b):
        epoch = 1 if step < steps_per_epoch else 2
        step_in_epoch = step if epoch == 1 else step - steps_per_epoch
        order = epoch1_order if epoch == 1 else epoch2_order
        batch_indices = order[step_in_epoch * instances_per_batch : (step_in_epoch + 1) * instances_per_batch]
        results.extend(batch_indices)
    return results

# Example: data between step 1000 and step 7000
indices = get_data_between_checkpoints(1000, 7000)
print(f"Instances seen: {len(indices)}")
```

### Step 5: Inspect actual data

```python
for idx in indices[:5]:
    item = dataset[idx]
    # item["input_ids"] = token IDs for this packed instance (32K tokens)
    # item["label_mask"] = which tokens contribute to the loss
    # item["doc_lens"] = lengths of individual documents packed into this instance
    print(f"Instance {idx}: {len(item['doc_lens'])} documents packed, "
          f"{item['label_mask'].sum()} trainable tokens")
```

---

## Important Caveats

1. **Exact `.npy` reproduction required** — the packing indices depend on file contents. Different tokenizer version or chat template → different packing → different training order.

2. **Epoch boundary at ~step 22,700** — epoch 1 uses `get_rng(34522)`, epoch 2 uses `get_rng(34523)`. Steps crossing this boundary span two different permutations.

3. **DP rank striping** — with 64 GPUs and context parallelism for 32K sequences, each rank sees a different stripe of each batch. The full batch is `indices.reshape(-1, instances_per_batch)`, sliced by `[:, dp_rank::dp_world_size]`.

4. **Bin-packing is many-to-many** — a single packed instance may contain multiple short conversations, or a truncated long one. Mapping instance → original HuggingFace row requires inspecting the packing indices.

5. **32B model uses "souping"** — the 32B Think SFT used LR 1.0e-4 souped (averaged) with LR 5.0e-5, meaning two separate training runs were merged.

6. **Think vs Instruct data** — The HuggingFace model cards list both `Dolci-Think-SFT` and `Dolci-Instruct-SFT` under the Think-SFT model, but the paper (Sections 4.2 vs 5.2) describes them as separate pipelines. Table 17 lists only Think SFT sources (2.27M prompts) with no mention of Instruct data being mixed in. The paper is likely more accurate — Think SFT uses only Dolci Think SFT data.

---

## Source References

| Source | What |
|--------|------|
| [arxiv:2512.13961](https://arxiv.org/abs/2512.13961) | OLMo 3 paper (Table 47, Section A.6.1) |
| `src/scripts/train/sft/Olmo-3-7B-SFT.py` | SFT training script (seeds, batch config) |
| `src/olmo_core/data/data_loader.py:667-690` | Epoch shuffling logic |
| `src/olmo_core/data/data_loader.py:720-747` | Batch → rank instance mapping |
| `src/olmo_core/data/numpy_dataset.py:982-1300` | OBFD bin-packing and `__getitem__` |
| [open-instruct](https://github.com/allenai/open-instruct) `scripts/data/convert_sft_data_for_olmocore.py` | HF → `.npy` conversion |
| [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) | 7B SFT dataset (2.27M rows) |
| [Dolci-Think-SFT-32B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B) | 32B SFT dataset (2.25M rows) |
