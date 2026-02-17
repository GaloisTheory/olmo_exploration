"""
OLMo3-Think-SFT-7B Full Training Provenance Builder

Builds a complete, queryable SQLite provenance database mapping:
    training step → packed instances → documents → HF row IDs + source datasets

Chain of determinism:
    HF row → shuffle(seed=42) → document with known length
           → OBFD bin-pack (per .npy chunk) → packed instance
           → epoch shuffle(PCG64, seed=34521+epoch) → training step

Usage:
    uv run python provenance_builder.py [--output-dir OUTPUT_DIR] [--num-proc NUM_PROC]
"""

import argparse
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Training parameters (verified from source code and paper)
# ──────────────────────────────────────────────────────────────────────────────

DATA_LOADER_SEED = 34521        # Olmo-3-7B-SFT.py:345
SHUFFLE_SEED = 42               # convert_sft_data_for_olmocore.py:175
SEQUENCE_LENGTH = 32_768        # Paper Table 47
GLOBAL_BATCH_SIZE_TOKENS = 1_048_576  # 64 * 16384 (Olmo-3-7B-SFT.py:498)
INSTANCES_PER_BATCH = GLOBAL_BATCH_SIZE_TOKENS // SEQUENCE_LENGTH  # = 32
NUM_EPOCHS = 2                  # Paper Table 47
EOS_TOKEN_ID = 100_257          # allenai/dolma-2-tokenizer-olmo-3-instruct-final

# Chunk size for .npy files: 1GB of uint16 → 536,870,912 tokens
CHUNK_SIZE_TOKENS = (1 * 1024**3) // 2  # 536_870_912

HF_DATASET_NAME = "allenai/Dolci-Think-SFT-7B"
TOKENIZER_NAME = "allenai/dolma-2-tokenizer-olmo-3-instruct-final"


# ──────────────────────────────────────────────────────────────────────────────
# OBFD InstancePacker (copied from olmo_core/data/utils.py to avoid import chain)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentTreeNode:
    weight: int = 0
    parent: Optional["SegmentTreeNode"] = None
    children: Optional[Tuple["SegmentTreeNode", "SegmentTreeNode"]] = None
    leaf_id: Optional[int] = None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def update(self, weight: Optional[int] = None):
        if weight is not None:
            assert self.is_leaf
            self.weight = weight
        else:
            assert self.children is not None
            self.weight = max(self.children[0].weight, self.children[1].weight)
        if self.parent is not None:
            self.parent.update()


class SegmentTree:
    def __init__(self, N: int):
        assert math.log2(N) % 1 == 0, "N should be a power of 2"
        self.root_node = SegmentTreeNode()
        self.leaf_nodes: List[SegmentTreeNode] = []

        max_depth = int(math.log2(N))
        leaf_id = 0
        queue: deque[Tuple[SegmentTreeNode, int]] = deque([(self.root_node, 0)])
        while queue:
            parent, depth = queue.popleft()
            if depth < max_depth:
                parent.children = (SegmentTreeNode(parent=parent), SegmentTreeNode(parent=parent))
                queue.append((parent.children[0], depth + 1))
                queue.append((parent.children[1], depth + 1))
            else:
                parent.leaf_id = leaf_id
                self.leaf_nodes.append(parent)
                leaf_id += 1

        assert len(self.leaf_nodes) == N
        self.leaf_nodes[-1].update(N)

    def query(self, weight: int) -> SegmentTreeNode:
        node = self.root_node
        while not node.is_leaf:
            assert weight <= node.weight
            assert node.children is not None
            left_child, right_child = node.children
            if weight <= left_child.weight:
                node = left_child
            else:
                node = right_child
        return node


class InstancePacker:
    def __init__(self, max_sequence_length: int):
        self.max_sequence_length = max_sequence_length
        self.seg_tree = SegmentTree(max_sequence_length)
        self.instance_bins: List[List[int]] = []
        self.space_to_bins: Dict[int, deque] = defaultdict(deque)

    @property
    def total_padding(self) -> int:
        total_padding = 0
        for i in range(1, self.max_sequence_length):
            if i in self.space_to_bins:
                total_padding += i * len(self.space_to_bins[i])
        return total_padding

    @property
    def total_tokens(self) -> int:
        return self.max_sequence_length * len(self.instance_bins) - self.total_padding

    def _pack_document(self, document_id: int, document_length: int) -> int:
        best_fit_leaf_id = self.seg_tree.query(document_length).leaf_id
        assert best_fit_leaf_id is not None
        best_fit_capacity = best_fit_leaf_id + 1

        if best_fit_capacity == self.max_sequence_length:
            self.instance_bins.append([])
            bin_id = len(self.instance_bins) - 1
        else:
            bins = self.space_to_bins[best_fit_capacity]
            bin_id = bins.popleft()
            if len(bins) == 0:
                self.seg_tree.leaf_nodes[best_fit_capacity - 1].update(weight=0)

        bin = self.instance_bins[bin_id]
        bin.append(document_id)

        bin_space = best_fit_capacity - document_length
        if bin_space > 0:
            bins = self.space_to_bins[bin_space]
            if len(bins) == 0:
                self.seg_tree.leaf_nodes[bin_space - 1].update(weight=bin_space)
            self.space_to_bins[bin_space].append(bin_id)

        return bin_id

    def pack_documents(self, document_indices: np.ndarray) -> Tuple[List[List[int]], np.ndarray, int]:
        if self.instance_bins or self.space_to_bins:
            raise RuntimeError("Must call reset() before calling pack_documents() again.")

        # Sort by length decreasing (OBFD)
        document_lengths = document_indices[:, 1] - document_indices[:, 0]
        sorted_index = np.argsort(-1 * document_lengths.astype(np.int64))
        document_indices = np.take(document_indices, sorted_index, axis=0)

        for document_id, (start_idx, end_idx) in enumerate(document_indices):
            document_len = int(end_idx - start_idx)
            self._pack_document(document_id, document_len)

        return self.instance_bins, document_indices, self.total_tokens

    def reset(self):
        self.seg_tree = SegmentTree(self.max_sequence_length)
        self.instance_bins.clear()
        self.space_to_bins.clear()


def get_rng(seed: int) -> np.random.Generator:
    """Deterministic PCG64 RNG matching olmo_core/data/utils.py:490"""
    return np.random.Generator(np.random.PCG64(seed=seed))


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Load HF dataset and compute token lengths
# ──────────────────────────────────────────────────────────────────────────────

def load_and_tokenize(num_proc: int = 32) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load HF dataset, tokenize each row, and return lengths + metadata.

    Returns:
        lengths: np.ndarray of token lengths (in shuffled order)
        doc_ids: list of document IDs (in shuffled order)
        source_datasets: list of source dataset names (in shuffled order, from dataset_source column)
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("=" * 60)
    print("PHASE 2: Loading HF dataset and tokenizing")
    print("=" * 60)

    t0 = time.time()
    print(f"Loading {HF_DATASET_NAME}...")
    ds = load_dataset(HF_DATASET_NAME, split="train")
    print(f"  Loaded {len(ds):,} rows in {time.time()-t0:.1f}s")

    # V1: Check row count
    expected_count = 2_268_178
    actual_count = len(ds)
    print(f"  V1 Check: expected ~{expected_count:,}, got {actual_count:,}", end="")
    if abs(actual_count - expected_count) < 1000:
        print(" ✓")
    else:
        print(f" ⚠ (difference: {actual_count - expected_count:,})")

    print(f"Loading tokenizer {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Tokenizing {actual_count:,} rows with {num_proc} processes...")
    t0 = time.time()

    def compute_length(example):
        # Match exactly the conversion script's call signature in
        # sft_tulu_tokenize_and_truncate_v1 (open_instruct/dataset_transformation.py:1068)
        input_ids = tokenizer.apply_chat_template(
            conversation=example["messages"],
            tokenize=True,
            padding=False,
            truncation=True,
            max_length=SEQUENCE_LENGTH,
            add_generation_prompt=False,
        )
        example["token_length"] = len(input_ids)

        # Also check if this example would be filtered (all labels = -100).
        # This happens when there are no assistant messages.
        has_assistant = any(m["role"] == "assistant" for m in example["messages"])
        example["has_assistant"] = has_assistant
        return example

    ds = ds.map(compute_length, num_proc=num_proc, desc="Tokenizing")
    print(f"  Tokenized in {time.time()-t0:.1f}s")

    # Filter out rows with no assistant content (matching sft_tulu_filter_v1)
    pre_filter = len(ds)
    ds = ds.filter(lambda x: x["has_assistant"], num_proc=num_proc, desc="Filtering")
    post_filter = len(ds)
    filtered_count = pre_filter - post_filter
    print(f"  Filtered: {filtered_count:,} rows removed (no assistant content)")
    print(f"  Rows after filter: {post_filter:,}")

    # Apply shuffle with seed=42 (matching conversion script)
    print(f"Shuffling with seed={SHUFFLE_SEED}...")
    ds = ds.shuffle(seed=SHUFFLE_SEED)

    # Extract arrays (using dataset_source column directly from HF dataset)
    lengths = np.array(ds["token_length"], dtype=np.int32)
    doc_ids = ds["id"]
    source_datasets = ds["dataset_source"]

    # V2: Check per-source counts
    source_counts = Counter(source_datasets)
    print("\n  V2 Check: Per-source counts:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count:,}")

    total_tokens = int(np.sum(lengths))
    print(f"\n  Total documents: {len(lengths):,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Mean doc length: {np.mean(lengths):.1f}")
    print(f"  Max doc length: {np.max(lengths):,}")
    print(f"  Min doc length: {np.min(lengths):,}")

    return lengths, doc_ids, source_datasets


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Simulate .npy chunking + OBFD packing
# ──────────────────────────────────────────────────────────────────────────────

def simulate_chunks_and_pack(
    lengths: np.ndarray,
) -> Tuple[List[List[int]], int]:
    """
    Simulate the exact .npy chunking and OBFD packing from the training pipeline.

    The conversion script concatenates documents into a flat token array, then splits
    into ~1GB .npy chunks. OLMo-core finds document boundaries within each chunk via
    EOS scanning and runs OBFD packing per chunk.

    At chunk boundaries:
    - Tokens after the last EOS in a chunk are NOT included in any document
    - The first "document" in the next chunk starts from position 0 to the first EOS

    Returns:
        all_packed_instances: list of lists, each inner list contains shuffled-order
                              document indices that were packed together
        total_packed_tokens: total tokens across all packed instances
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Simulating .npy chunks + OBFD packing")
    print("=" * 60)

    num_docs = len(lengths)

    # Build cumulative token positions (the flat-array EOS positions)
    # Each document's EOS is at cumsum[i] - 1, but document spans cumsum[i-1] to cumsum[i]
    cumulative = np.cumsum(lengths.astype(np.int64))
    total_tokens = int(cumulative[-1])

    # Determine chunk boundaries (every CHUNK_SIZE_TOKENS)
    num_chunks = math.ceil(total_tokens / CHUNK_SIZE_TOKENS)
    print(f"  Total tokens in flat array: {total_tokens:,}")
    print(f"  Chunk size: {CHUNK_SIZE_TOKENS:,} tokens ({CHUNK_SIZE_TOKENS * 2 / 1024**3:.2f} GB)")
    print(f"  Number of chunks: {num_chunks}")

    # For each chunk, find documents whose EOS falls within the chunk.
    # A document's EOS position (1-indexed end) = cumulative[doc_idx]
    # Chunk k spans tokens [k*CHUNK_SIZE, (k+1)*CHUNK_SIZE)
    # A document is "in" chunk k if its EOS (cumulative[doc_idx]) falls in that range:
    #   k * CHUNK_SIZE < cumulative[doc_idx] <= (k+1) * CHUNK_SIZE

    all_packed_instances = []  # Each element: list of shuffled-order doc indices
    total_packed_tokens = 0
    docs_in_instances = 0
    dropped_boundary_tokens = 0

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE_TOKENS
        chunk_end = min((chunk_idx + 1) * CHUNK_SIZE_TOKENS, total_tokens)

        # Find documents whose EOS (end position) falls within this chunk.
        # cumulative[i] is the end position (exclusive) of document i.
        # A document's EOS token is at cumulative[i] - 1.
        # For EOS scanning within a chunk, a document is detected if its
        # last token (EOS) is in [chunk_start, chunk_end).
        # The EOS is at cumulative[i] - 1, so we need: chunk_start <= cumulative[i] - 1 < chunk_end
        # i.e., chunk_start + 1 <= cumulative[i] < chunk_end + 1
        # i.e., chunk_start < cumulative[i] <= chunk_end

        mask = (cumulative > chunk_start) & (cumulative <= chunk_end)
        chunk_doc_indices = np.where(mask)[0]

        if len(chunk_doc_indices) == 0:
            continue

        # Compute document lengths AS SEEN by the packer within this chunk.
        # EOS scanning finds boundaries at EOS positions within the chunk.
        # Each "document" for packing spans from one EOS to the next.
        #
        # First doc in chunk: from chunk_start to cumulative[first_doc]
        # Middle docs: from cumulative[prev_doc] to cumulative[doc]
        # (These are the actual lengths, not the original lengths, because
        # a document might start in a previous chunk)

        chunk_doc_lengths = np.empty(len(chunk_doc_indices), dtype=np.int64)

        for j, doc_idx in enumerate(chunk_doc_indices):
            if j == 0:
                # First document in this chunk: starts from chunk_start
                doc_start_in_flat = chunk_start
            else:
                # Starts where the previous document ended
                doc_start_in_flat = int(cumulative[chunk_doc_indices[j - 1]])
            doc_end_in_flat = int(cumulative[doc_idx])
            chunk_doc_lengths[j] = doc_end_in_flat - doc_start_in_flat

        # Tokens after last EOS in chunk are dropped (not packed)
        last_eos_pos = int(cumulative[chunk_doc_indices[-1]])
        dropped = chunk_end - last_eos_pos
        dropped_boundary_tokens += dropped

        # Truncate any "documents" longer than SEQUENCE_LENGTH
        # (matching LongDocStrategy.truncate)
        chunk_doc_lengths = np.minimum(chunk_doc_lengths, SEQUENCE_LENGTH)

        # Filter out zero-length documents (shouldn't happen, but safety check)
        valid_mask = chunk_doc_lengths > 0
        chunk_doc_indices_valid = chunk_doc_indices[valid_mask]
        chunk_doc_lengths_valid = chunk_doc_lengths[valid_mask]

        if len(chunk_doc_lengths_valid) == 0:
            continue

        # Build document_indices array for InstancePacker: shape (num_docs, 2)
        # We create synthetic start/end pairs since the packer only uses lengths
        starts = np.zeros(len(chunk_doc_lengths_valid), dtype=np.uint64)
        ends = chunk_doc_lengths_valid.astype(np.uint64)
        starts[1:] = np.cumsum(ends[:-1])
        ends = starts + ends
        doc_indices_arr = np.column_stack([starts, ends])

        # Run OBFD packing
        packer = InstancePacker(SEQUENCE_LENGTH)
        instances, sorted_doc_indices, chunk_tokens = packer.pack_documents(doc_indices_arr)

        # The packer sorts documents by length (decreasing) before packing.
        # Document IDs in `instances` refer to the sorted order.
        # We need to map back: sorted_doc_id -> original position in chunk_doc_indices_valid
        doc_lengths_for_sort = doc_indices_arr[:, 1] - doc_indices_arr[:, 0]
        sorted_index = np.argsort(-1 * doc_lengths_for_sort.astype(np.int64))

        # Map: sorted_doc_id -> chunk-local index -> global shuffled doc index
        for instance in instances:
            global_instance = []
            for sorted_doc_id in instance:
                chunk_local_idx = sorted_index[sorted_doc_id]
                global_shuffled_idx = int(chunk_doc_indices_valid[chunk_local_idx])
                global_instance.append(global_shuffled_idx)
            all_packed_instances.append(global_instance)
            docs_in_instances += len(global_instance)

        total_packed_tokens += chunk_tokens

        if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
            print(f"  Chunk {chunk_idx+1}/{num_chunks}: "
                  f"{len(chunk_doc_indices)} docs, {len(instances)} instances")

    total_instances = len(all_packed_instances)
    steps_per_epoch = (total_instances // INSTANCES_PER_BATCH)
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"\n  Results:")
    print(f"    Total packed instances: {total_instances:,}")
    print(f"    Total packed tokens: {total_packed_tokens:,}")
    print(f"    Documents in instances: {docs_in_instances:,}")
    print(f"    Dropped boundary tokens: {dropped_boundary_tokens:,}")
    print(f"    Steps per epoch: {steps_per_epoch:,}")
    print(f"    Total steps (2 epochs): {total_steps:,}")
    print(f"    Instances dropped per epoch (floor-truncation): "
          f"{total_instances - steps_per_epoch * INSTANCES_PER_BATCH:,}")

    # V3-V5 Checks
    print(f"\n  V3 Check: total instances ~700K-750K: {total_instances:,}", end="")
    if 500_000 < total_instances < 1_000_000:
        print(" ✓")
    else:
        print(" ⚠")

    print(f"  V4 Check: steps per epoch ~22,700: {steps_per_epoch:,}", end="")
    if 15_000 < steps_per_epoch < 35_000:
        print(" ✓")
    else:
        print(" ⚠")

    print(f"  V5 Check: total steps ~45,400: {total_steps:,}", end="")
    if 30_000 < total_steps < 70_000:
        print(" ✓")
    else:
        print(" ⚠")

    return all_packed_instances, total_packed_tokens


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Replay epoch shuffles + build SQLite DB
# ──────────────────────────────────────────────────────────────────────────────

def build_provenance_db(
    all_packed_instances: List[List[int]],
    lengths: np.ndarray,
    doc_ids: List[str],
    source_datasets: List[str],
    output_dir: str,
):
    """Build the SQLite provenance database and per-checkpoint summary CSVs."""

    print("\n" + "=" * 60)
    print("PHASE 4: Building provenance database")
    print("=" * 60)

    db_path = os.path.join(output_dir, "provenance.db")
    print(f"  Database: {db_path}")

    total_instances = len(all_packed_instances)
    instances_per_batch = INSTANCES_PER_BATCH
    total_size = instances_per_batch * (total_instances // instances_per_batch)
    steps_per_epoch = total_size // instances_per_batch

    # Replay epoch shuffles
    print(f"  Replaying epoch shuffles (seed={DATA_LOADER_SEED})...")

    epoch_step_assignments = []  # list of (step, epoch, position_in_batch, instance_id)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"    Epoch {epoch}: shuffling {total_instances:,} instances...")
        indices = np.arange(total_instances, dtype=np.uint32)
        rng = get_rng(DATA_LOADER_SEED + epoch)
        rng.shuffle(indices)
        indices = indices[:total_size]

        # Reshape into batches
        batches = indices.reshape(-1, instances_per_batch)

        for batch_idx in range(len(batches)):
            global_step = (epoch - 1) * steps_per_epoch + batch_idx
            for pos_in_batch, instance_id in enumerate(batches[batch_idx]):
                epoch_step_assignments.append((global_step, epoch, pos_in_batch, int(instance_id)))

    print(f"    Total step-instance assignments: {len(epoch_step_assignments):,}")

    # Build SQLite database
    print(f"  Creating SQLite database...")
    t0 = time.time()

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()

    # Create tables
    cur.execute("""
        CREATE TABLE documents (
            doc_id INTEGER PRIMARY KEY,
            hf_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            token_count INTEGER NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE packed_instances (
            instance_id INTEGER PRIMARY KEY,
            num_documents INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE instance_documents (
            instance_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            position_in_instance INTEGER NOT NULL,
            PRIMARY KEY (instance_id, position_in_instance),
            FOREIGN KEY (instance_id) REFERENCES packed_instances(instance_id),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    cur.execute("""
        CREATE TABLE training_steps (
            step INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            position_in_batch INTEGER NOT NULL,
            instance_id INTEGER NOT NULL,
            PRIMARY KEY (step, position_in_batch),
            FOREIGN KEY (instance_id) REFERENCES packed_instances(instance_id)
        )
    """)

    # Insert documents
    print(f"    Inserting {len(doc_ids):,} documents...")
    cur.executemany(
        "INSERT INTO documents (doc_id, hf_id, source_dataset, token_count) VALUES (?, ?, ?, ?)",
        [(i, doc_ids[i], source_datasets[i], int(lengths[i])) for i in range(len(doc_ids))]
    )

    # Insert packed instances and their document mappings
    print(f"    Inserting {len(all_packed_instances):,} packed instances...")
    instance_docs_batch = []
    for inst_id, instance_docs in enumerate(all_packed_instances):
        total_tok = sum(int(lengths[doc_idx]) for doc_idx in instance_docs)
        cur.execute(
            "INSERT INTO packed_instances (instance_id, num_documents, total_tokens) VALUES (?, ?, ?)",
            (inst_id, len(instance_docs), total_tok)
        )
        for pos, doc_idx in enumerate(instance_docs):
            instance_docs_batch.append((inst_id, doc_idx, pos))

        if len(instance_docs_batch) >= 100_000:
            cur.executemany(
                "INSERT INTO instance_documents (instance_id, doc_id, position_in_instance) VALUES (?, ?, ?)",
                instance_docs_batch
            )
            instance_docs_batch = []

    if instance_docs_batch:
        cur.executemany(
            "INSERT INTO instance_documents (instance_id, doc_id, position_in_instance) VALUES (?, ?, ?)",
            instance_docs_batch
        )

    # Insert training steps
    print(f"    Inserting {len(epoch_step_assignments):,} training step assignments...")
    BATCH_SIZE = 500_000
    for i in range(0, len(epoch_step_assignments), BATCH_SIZE):
        batch = epoch_step_assignments[i:i + BATCH_SIZE]
        cur.executemany(
            "INSERT INTO training_steps (step, epoch, position_in_batch, instance_id) VALUES (?, ?, ?, ?)",
            batch
        )
        if (i + BATCH_SIZE) % 2_000_000 == 0:
            print(f"      ... {i + len(batch):,} / {len(epoch_step_assignments):,}")

    # Create indices for fast queries
    print(f"    Creating indices...")
    cur.execute("CREATE INDEX idx_training_steps_step ON training_steps(step)")
    cur.execute("CREATE INDEX idx_training_steps_instance ON training_steps(instance_id)")
    cur.execute("CREATE INDEX idx_instance_documents_doc ON instance_documents(doc_id)")
    cur.execute("CREATE INDEX idx_documents_source ON documents(source_dataset)")

    conn.commit()
    print(f"  Database built in {time.time()-t0:.1f}s")

    # V6: Check every doc appears in exactly one packed instance
    print("\n  V6 Check: document packing integrity...")
    cur.execute("""
        SELECT doc_id, COUNT(DISTINCT instance_id) as inst_count
        FROM instance_documents
        GROUP BY doc_id
        HAVING inst_count > 1
        LIMIT 5
    """)
    duplicates = cur.fetchall()
    if duplicates:
        print(f"    ⚠ Found {len(duplicates)} documents in multiple instances!")
        for doc_id, count in duplicates:
            print(f"      doc_id={doc_id}: in {count} instances")
    else:
        print("    No documents in multiple instances ✓")

    # Count docs in instances vs total docs
    cur.execute("SELECT COUNT(DISTINCT doc_id) FROM instance_documents")
    docs_in_instances = cur.fetchone()[0]
    print(f"    Documents in packed instances: {docs_in_instances:,} / {len(doc_ids):,}")
    orphan_count = len(doc_ids) - docs_in_instances
    if orphan_count > 0:
        print(f"    Documents not in any instance (chunk boundary drops): {orphan_count:,}")

    # Generate per-checkpoint summary CSV
    print("\n  Generating per-checkpoint summary CSV...")
    csv_path = os.path.join(output_dir, "checkpoint_composition.csv")
    total_steps_actual = steps_per_epoch * NUM_EPOCHS

    with open(csv_path, "w") as f:
        f.write("step_start,step_end,source_dataset,num_documents,total_tokens\n")

        interval = 1000
        for start_step in range(0, total_steps_actual, interval):
            end_step = min(start_step + interval, total_steps_actual)
            cur.execute("""
                SELECT d.source_dataset, COUNT(DISTINCT d.doc_id), SUM(d.token_count)
                FROM training_steps ts
                JOIN instance_documents id ON ts.instance_id = id.instance_id
                JOIN documents d ON id.doc_id = d.doc_id
                WHERE ts.step >= ? AND ts.step < ?
                GROUP BY d.source_dataset
                ORDER BY SUM(d.token_count) DESC
            """, (start_step, end_step))

            for row in cur.fetchall():
                f.write(f"{start_step},{end_step},{row[0]},{row[1]},{row[2]}\n")

    print(f"  Summary CSV: {csv_path}")

    # V7: Check composition uniformity
    print("\n  V7 Check: composition uniformity across step ranges...")
    cur.execute("""
        SELECT
            CASE WHEN ts.step < ? THEN 'first_half' ELSE 'second_half' END as half,
            d.source_dataset,
            COUNT(DISTINCT d.doc_id) as doc_count,
            SUM(d.token_count) as token_sum
        FROM training_steps ts
        JOIN instance_documents id ON ts.instance_id = id.instance_id
        JOIN documents d ON id.doc_id = d.doc_id
        GROUP BY half, d.source_dataset
        ORDER BY half, token_sum DESC
    """, (total_steps_actual // 2,))

    half_data = defaultdict(dict)
    for half, src, doc_count, token_sum in cur.fetchall():
        half_data[half][src] = token_sum

    if half_data:
        print(f"    {'Source':<30} {'First half':>15} {'Second half':>15} {'Ratio':>10}")
        print(f"    {'-'*30} {'-'*15} {'-'*15} {'-'*10}")
        for src in sorted(
            set(list(half_data.get("first_half", {}).keys()) + list(half_data.get("second_half", {}).keys()))
        ):
            t1 = half_data.get("first_half", {}).get(src, 0)
            t2 = half_data.get("second_half", {}).get(src, 0)
            ratio = t1 / t2 if t2 > 0 else float("inf")
            flag = " ⚠" if abs(ratio - 1.0) > 0.15 else ""
            print(f"    {src:<30} {t1:>15,} {t2:>15,} {ratio:>9.3f}{flag}")

    # Print DB stats
    db_size = os.path.getsize(db_path)
    print(f"\n  Database size: {db_size / 1024**2:.1f} MB")
    print(f"  Total training steps: {total_steps_actual:,}")

    conn.close()
    print("\n  Done! Provenance database built successfully.")

    return db_path, csv_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build OLMo3-Think-SFT provenance database")
    parser.add_argument("--output-dir", default="provenance_output",
                        help="Output directory for DB and CSV files")
    parser.add_argument("--num-proc", type=int, default=32,
                        help="Number of processes for tokenization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("OLMo3-Think-SFT-7B Provenance Builder")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Tokenization processes: {args.num_proc}")
    print(f"Sequence length: {SEQUENCE_LENGTH:,}")
    print(f"Global batch size: {GLOBAL_BATCH_SIZE_TOKENS:,} tokens")
    print(f"Instances per batch: {INSTANCES_PER_BATCH}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Data loader seed: {DATA_LOADER_SEED}")
    print(f"Shuffle seed: {SHUFFLE_SEED}")
    print()

    t_total = time.time()

    # Phase 2: Load and tokenize
    lengths, doc_ids, source_datasets = load_and_tokenize(num_proc=args.num_proc)

    # Phase 3: Simulate chunks and pack
    all_packed_instances, total_packed_tokens = simulate_chunks_and_pack(lengths)

    # Phase 4: Build DB
    db_path, csv_path = build_provenance_db(
        all_packed_instances, lengths, doc_ids, source_datasets, args.output_dir
    )

    print(f"\nTotal wall time: {time.time()-t_total:.1f}s")
    print(f"\nOutputs:")
    print(f"  Provenance DB: {db_path}")
    print(f"  Checkpoint CSV: {csv_path}")
    print(f"\nExample queries:")
    print(f'  sqlite3 {db_path} "SELECT d.source_dataset, COUNT(*) FROM training_steps ts '
          f'JOIN instance_documents id ON ts.instance_id=id.instance_id '
          f'JOIN documents d ON id.doc_id=d.doc_id WHERE ts.step=0 GROUP BY d.source_dataset"')


if __name__ == "__main__":
    main()
