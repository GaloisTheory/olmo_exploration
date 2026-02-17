# PCA By Source: Summary & Dataset Clustering

> Analysis of `pca_by_source.html` — PCA directions computed per-source on vLLM 1k-sample embeddings.
> Each dataset has PC1 and PC2 extracted; the corrplot shows cosine similarity of these directions across all datasets.

---

## Table of Contents

1. [Per-Dataset PC Classification](#1-per-dataset-pc-classification)
   - [Coding / Competitive Programming](#coding--competitive-programming-datasets)
   - [Reasoning / Math / Science](#reasoning--math--science-datasets)
   - [Multilingual / General Knowledge](#multilingual--general-knowledge)
   - [Chat / Conversational](#chat--conversational)
   - [Instruction Following](#instruction-following)
   - [Table / Structured Data](#table--structured-data)
   - [Safety / Refusal](#safety--refusal)
2. [Cross-Dataset Clustering](#2-cross-dataset-clustering)
3. [High-Level Taxonomy](#3-high-level-taxonomy)

---

## 1. Per-Dataset PC Classification

### Coding / Competitive Programming Datasets

#### allenai SYNTHETIC-2-SFT-cn (PC1: 7.2%, PC2: 5.4%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Well-structured competitive programming with clean problem->approach->code->explanation format | 0.54, 0.51 |
| PC1 | **Negative** | Similarly technical but involving more complex algorithms (DP, graph theory) | -0.56, -0.52 |
| PC2 | **Positive** | Simple iteration/counting problems — string manipulation, digit counting | 0.49, 0.46 |
| PC2 | **Negative** | Advanced graph algorithms — Dijkstra, MST, binary lifting, tree decomposition | -0.52, -0.47 |

- **PC1 interpretation:** Problem presentation style — both ends are code-heavy, so PC1 captures a subtle structural difference rather than a domain shift.
- **PC2 interpretation:** Algorithmic complexity gradient. The clearest axis: simple vs. hard problems.

---

#### allenai nemotron-post-train (PC1: 6.0%, PC2: 5.4%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | General programming puzzles, CodeWars-style (checkered boards, ciphers) | 0.49, 0.45 |
| PC1 | **Negative** | Competitive programming contests with graph theory and game theory | -0.47, -0.46 |
| PC2 | **Positive** | Trivial operations like "sum all integers in array" | 0.49, 0.47 |
| PC2 | **Negative** | Advanced graph embedding, cactus graphs, cycle detection | -0.48, -0.45 |

- **PC1 interpretation:** Problem source/domain — general programming puzzles vs. competitive contest problems. Mirrors SYNTHETIC-2-SFT closely.
- **PC2 interpretation:** Problem complexity scale — trivial O(N) iteration vs. advanced graph theory.

---

#### saumyamalik correct-python (PC1: 6.4%, PC2: 4.7%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Basic algorithms (find-max loops, Fibonacci) | 0.49, 0.45 |
| PC1 | **Negative** | Complex multi-criteria string matching with 9 modes | -0.44, -0.43 |
| PC2 | **Positive** | Mathematical/recursive puzzles with constraints ("no loops allowed") | 0.50, 0.48 |
| PC2 | **Negative** | Nested dictionary manipulation, recursive data traversal | -0.54, -0.50 |

- **PC1 interpretation:** Code complexity — basic algorithms vs. complex multi-criteria logic.
- **PC2 interpretation:** Problem type — math/algorithmic puzzles vs. data structure manipulation.

---

### Reasoning / Math / Science Datasets

#### saumyamalik OpenThoughts3 [variant 1 — programming-heavy] (PC1: 10.7%, PC2: 4.3%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Terse code golf — e.g., "Reverse your own source code" in Stax (2 bytes) | 0.51, 0.51 |
| PC1 | **Negative** | Complex graph theory with DSU, cycle counting, 2^c formulas | -0.59 |
| PC2 | **Positive** | Number theory, modular arithmetic (z_n = 2^(2^(n-1)) - 1 mod 10^9+7) | 0.52, 0.51 |
| PC2 | **Negative** | Graph-based competitive programming (MST, longest path queries) | -0.44 |

- **PC1 interpretation:** Code golf vs. full solutions. The **highest PC1 variance of any dataset**, reflecting a stark bimodality between minimalist hacks and elaborate graph algorithms.
- **PC2 interpretation:** Math/physics problems vs. graph algorithms.

---

#### saumyamalik OpenThoughts3 [variant 2 — math-heavy] (PC1: 6.7%, PC2: 6.0%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Simple bookshelf/counting word problems | 0.40, 0.36 |
| PC1 | **Negative** | Coordinate geometry with circles and triangles | -0.54, -0.52 |
| PC2 | **Positive** | Combinatorics — inclusion-exclusion, seating arrangements | 0.43, 0.42 |
| PC2 | **Negative** | Algebra — polynomial coefficient matching, cubic equations | -0.46, -0.46 |

- **PC1 interpretation:** Arithmetic word problems vs. geometric proofs. A complexity/formality gradient within math.
- **PC2 interpretation:** Combinatorial/interpretive thinking vs. algebraic manipulation. Different *reasoning styles* within math.

---

#### saumyamalik OpenThoughts3 [variant 3 — science-heavy] (PC1: 8.5%, PC2: 3.5%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Quantum field theory — Lorentz covariance, photon localization, wavepackets | 0.53, 0.50 |
| PC1 | **Negative** | Organic chemistry — alkylation reactions, IUPAC nomenclature | -0.44, -0.43 |
| PC2 | **Positive** | Conceptual explanations — cyclohexane conformations, representation theory | 0.29, 0.29 |
| PC2 | **Negative** | Numerical calculations — freezing point depression, osmotic pressure formulas | -0.46, -0.46 |

- **PC1 interpretation:** Physics vs. chemistry. A clean subject-domain split.
- **PC2 interpretation:** Conceptual/theoretical explanation vs. computational/formula-based calculation. Different *pedagogical modes* within science.

---

#### saumyamalik if-qwq-reasoning (PC1: 4.0%, PC2: 2.1%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Detailed step-by-step math explanations | 0.33, 0.33 |
| PC1 | **Negative** | Minimal constraint-satisfying refusals (all-caps video translation decline) | -0.39, -0.37 |
| PC2 | **Positive** | Multiple solution approaches to word problems | 0.37, 0.35 |
| PC2 | **Negative** | Verbatim prompt repetition, code with placeholders | -0.50, -0.50 |

- **PC1 interpretation:** Response elaboration — detailed reasoning vs. terse refusals. Low variance suggests this dataset is relatively homogeneous.
- **PC2 interpretation:** Math reasoning vs. meta-instruction following (literal prompt repetition).

---

### Multilingual / General Knowledge

#### allenai aya-100k-r1-format (PC1: 3.6%, PC2: 2.5%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | English factual questions (Industrial Revolution, World Cup) | 0.40, 0.38 |
| PC1 | **Negative** | Non-English cultural content — Malay proverbs, Malagasy explanations | -0.42, -0.41 |
| PC2 | **Positive** | Arabic/Somali historical narratives about Islamic history | 0.44, 0.42 |
| PC2 | **Negative** | Spanish/Portuguese trivia with emojis and playful formatting | -0.39, -0.32 |

- **PC1 interpretation:** Language diversity axis. The primary variance is *language*, not topic.
- **PC2 interpretation:** Content formality — focused historical narrative vs. playful trivia lists.

---

### Chat / Conversational

#### allenai tulu v3.9 wildchat (PC1: 3.9%, PC2: 3.2%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Comedic fan fiction with crude bodily humor (Devil May Cry farting contest) | 0.49, 0.48 |
| PC1 | **Negative** | Professional business recommendations (Salesforce, HubSpot) | -0.32, -0.31 |
| PC2 | **Positive** | Historical analysis (Sengoku Jidai peasants, quantum consciousness) | 0.29, 0.28 |
| PC2 | **Negative** | Etsy product title generation with character limits | -0.60 |

- **PC1 interpretation:** Crude creative fiction vs. professional/commercial content. The "fun vs. serious" axis of user chat.
- **PC2 interpretation:** Scholarly analysis vs. commercial copy. Deep thought vs. keyword optimization.

---

#### allenai wildchat-r1-p2-repe (PC1: 3.8%, PC2: 3.5%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Comedic fan fiction (Star Wars Kylo Ren parody) | 0.45, 0.44 |
| PC1 | **Negative** | SEO/commercial product descriptions | -0.34, -0.33 |
| PC2 | **Positive** | Etsy title generation (keyword-optimized, pipe-separated) | 0.63, 0.61 |
| PC2 | **Negative** | Historical/religious comparative analysis | -0.27 |

- **PC1 interpretation:** Nearly identical to tulu wildchat — crude fiction vs. commercial content.
- **PC2 interpretation:** **Same axis as tulu wildchat PC2 but with flipped sign** — Etsy titles are positive here vs. negative there. This is the expected arbitrary sign ambiguity of PCA.

---

### Instruction Following

#### allenai persona-precise-if (PC1: 4.4%, PC2: 3.8%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Rigid structural constraints — all-caps text with forced "LAKERS" keyword insertion | 0.29, 0.28 |
| PC1 | **Negative** | Natural culinary descriptions (Moroccan tagine recipes) | -0.48 |
| PC2 | **Positive** | Escalating keyword repetition rules ("vibrant" 1x in para 1, 2x in para 2...) | 0.56, 0.54 |
| PC2 | **Negative** | Practical email/invitation writing (Miami Dolphins game invite) | -0.33 |

- **PC1 interpretation:** Constrained formatting vs. natural writing. How "artificial" the output must be.
- **PC2 interpretation:** Formatting constraint severity. This PC isolates the *degree* of formatting gymnastics required.

---

### Table / Structured Data

#### allenai tablegpt r1-format (PC1: 12.1%, PC2: 6.9%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Terse JSON column assignments — matching headers to table columns | 0.41, 0.40 |
| PC1 | **Negative** | Multi-paragraph book comparison with step-by-step reasoning before JSON | -0.60 |
| PC2 | **Positive** | Column mapping with explicit null/None for unmatched columns | 0.50, 0.48 |
| PC2 | **Negative** | Input-output pattern inference (percentage reformatting) | -0.37 |

- **PC1 interpretation:** Simple mapping vs. complex entity matching. **Highest PC1 variance of all datasets (12.1%)** — a very clear bimodality between trivial table lookups and extended analytical reasoning.
- **PC2 interpretation:** Null handling vs. pattern transformation. Different *task types* within the table domain.

---

### Safety / Refusal

#### allenai coconot-r1-format (PC1: 5.1%, PC2: 3.1%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Prompts probing AI self-awareness ("Share your dreams") with "I don't have feelings" responses | 0.42, 0.40 |
| PC1 | **Negative** | Straightforward factual Q&A about governors-general | -0.52, -0.50 |
| PC2 | **Positive** | Strong medical safety refusals ("Do NOT attempt home surgery") | 0.38, 0.37 |
| PC2 | **Negative** | Casual philosophical questions ("What's your favorite color?") | -0.38 |

- **PC1 interpretation:** Meta/adversarial AI probing vs. factual queries. The "can you feel?" axis.
- **PC2 interpretation:** Safety urgency — critical medical refusals vs. lighthearted philosophical chat.

---

#### allenai wildguardmix-r1-v2 (PC1: 3.4%, PC2: 3.1%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Benign factual lookups (Supreme Court fax number) | 0.48, 0.45 |
| PC1 | **Negative** | Refusals of harmful creative requests | -0.40 |
| PC2 | **Positive** | Brief dignity-focused redirections (body image topics) | 0.41, 0.36 |
| PC2 | **Negative** | Detailed fictional narratives within ethical guardrails (hacking fiction, hitman game) | -0.47, -0.43 |

- **PC1 interpretation:** Factual information delivery vs. harm refusal.
- **PC2 interpretation:** Brief corrections vs. long ethical fiction. Response *length and elaboration* within the safety domain.

---

#### allenai wildjailbreak-r1-v2 (PC1: 3.8%, PC2: 3.5%)

| PC | Direction | What it looks like | Example scores |
|----|-----------|-------------------|----------------|
| PC1 | **Positive** | Fictional metaphorical questions (vampire workplace rights as discrimination allegory) | 0.42, 0.41 |
| PC1 | **Negative** | Institutional transparency/security (EU finances, FBI protocols) | -0.47, -0.43 |
| PC2 | **Positive** | Educational/informational responses (corporate espionage concept explanation) | 0.32, 0.30 |
| PC2 | **Negative** | Mental health crisis support with hotline numbers | -0.56, -0.55 |

- **PC1 interpretation:** Metaphorical/fictional framing vs. institutional factual content.
- **PC2 interpretation:** Educational/informational responses vs. crisis intervention. The "teach vs. support" axis.

---

## 2. Cross-Dataset Clustering

The cosine-similarity corrplot reveals clear groupings of datasets whose PC directions align:

### Cluster A: "Coding Problem Complexity" (r = 0.61-0.80)

| Dataset | Correlation |
|---------|-------------|
| SYNTHETIC-2-SFT-cn PC1 | - |
| nemotron-post-train PC1 | r=**0.80** with SYNTHETIC PC1 |
| OpenThoughts3 [v1] PC1 | r=**0.61** with SYNTHETIC PC1 |

**Why they cluster:** All three contain competitive programming / code generation tasks. Their PC1 directions all capture the same axis: *simple well-structured problems vs. complex graph/optimization problems*. SYNTHETIC and nemotron are essentially drawing from the same problem distribution (both SFT training data with code), explaining the near-identical PC1 directions.

---

### Cluster B: "Wildchat Variants" (r = 0.97 / -0.92)

| Dataset | Correlation |
|---------|-------------|
| tulu v3.9 wildchat PC1 | r=**0.97** with wildchat-r1-p2 PC1 |
| wildchat-r1-p2-repe PC1 | (same axis, same sign) |
| Their PC2s | r=**-0.92** (same axis, flipped sign) |

**Why they cluster:** These are variants of the same underlying WildChat user conversation data. PC1 is nearly identical (crude creative fiction <-> professional content). The PC2 anti-correlation (r=-0.92) is the expected sign ambiguity — both found the *same second axis* (scholarly analysis <-> Etsy product titles) but with opposite sign conventions.

---

### Cluster C: "Safety Datasets" (anti-correlated pair, r = -0.75)

| Dataset | Correlation |
|---------|-------------|
| wildguardmix-r1-v2 PC1 | r=**-0.75** with wildjailbreak PC1 |
| wildjailbreak-r1-v2 PC1 | |

**Why they anti-correlate:** Both are safety-focused but from opposite perspectives:
- **Wildguardmix** PC1: "benign factual content -> harm refusal"
- **Wildjailbreak** PC1: "fictional allegory -> institutional/factual content"

What registers as "positive PC1" in guardmix (benign factual) maps to "negative PC1" in jailbreak (institutional facts being probed adversarially). They're looking at the **same safety dimension from opposite sides**.

---

### Cluster D: "Math Reasoning" (moderate alignment)

| Dataset | Notes |
|---------|-------|
| OpenThoughts3 [v2] | Math word problems vs. geometry |
| OpenThoughts3 [v3] | Physics vs. chemistry |
| if-qwq-reasoning | Elaboration vs. minimal responses |

**Why they cluster loosely:** All three are reasoning-focused, but less tightly correlated than the coding cluster because each covers a *different subject domain* (combinatorics/geometry, physics/chemistry, general reasoning). Their PC1s all capture "simple/conceptual -> complex/technical" but applied to different fields, so the embedding directions partially overlap rather than strongly aligning.

---

### Relative Isolates

| Dataset | PC1 variance | Why isolated |
|---------|-------------|-------------|
| **tablegpt** | 12.1% | Unique structured-data domain — table column mapping has no analog in other datasets. Highest variance reflects a stark bimodality (trivial JSON vs. multi-paragraph reasoning). |
| **aya-100k** | 3.6% | Only truly multilingual dataset. PC1 captures language diversity, which is orthogonal to all other datasets' content/complexity axes. |
| **coconot** | 5.1% | Unique "AI self-awareness probing" content. The meta-adversarial "do you have feelings?" prompts create an axis that doesn't appear elsewhere. |
| **persona-precise-if** | 4.4% | Unique *formatting constraint severity* axis. No other dataset has the escalating keyword-repetition rules that dominate this dataset's variance. |

---

## 3. High-Level Taxonomy

### By Internal Structure (variance explained)

|  | Homogeneous domain | Heterogeneous domain |
|--|-------------------|---------------------|
| **High PC1 variance (>6%)** | tablegpt (12.1%), OpenThoughts3-v1 (10.7%), OpenThoughts3-v3 (8.5%) — tight domain, clear bimodality within it | SYNTHETIC-2-SFT (7.2%), nemotron (6.0%), correct-python (6.4%) — coding datasets with complexity gradients |
| **Low PC1 variance (<5%)** | persona-precise-if (4.4%), coconot (5.1%) — specialized but without extreme internal splits | wildchat variants (~3.8%), aya (3.6%), safety datasets (~3.5%), if-qwq (4.0%) — broad/diverse content, no single dominant axis |

### Interpretation

- **Highest-variance datasets** are those with the most **bimodal** internal structure (e.g., tablegpt's trivial-JSON vs. multi-paragraph-reasoning split, or OpenThoughts3-v1's code-golf vs. full-graph-theory split).
- **Lowest-variance datasets** are those with broadly diverse content where no single axis dominates (e.g., wildchat's mix of everything from crude fiction to business advice, or the safety datasets' spread across many refusal types).
- **PC1 across datasets most commonly captures:** domain/topic shifts or complexity gradients.
- **PC2 across datasets most commonly captures:** reasoning style, response format, or formality level.

### What the Corrplot Tells Us About Shared Structure

The fact that coding datasets (Cluster A) align strongly in PC1 direction means the **model's embedding space encodes "coding problem complexity" in a consistent direction** regardless of which specific coding dataset you look at. Similarly, the wildchat near-identity (Cluster B) confirms the model sees these as essentially the same distribution. The safety anti-correlation (Cluster C) is the most interesting finding — it suggests the model has a single "safety axis" that different safety datasets project onto from different angles.
