# Dataset construction plan — Groups A and B

## Two-group design

Two reasoning families. Each group has one training source (the task-agnostic
LoRA is trained here, with varied targets so the LoRA can't memorise) and
three OOD probes that cover sub-shapes within the family.

### Group A — Quantitative / Concrete-Object Reasoning

Tasks involve manipulating quantities or concrete objects with deterministic
state transitions.

| Role | Task | Status |
|---|---|---|
| Training source | **24 Game** (varied targets) | existing — Group A v1, kept as-is |
| OOD #1 (arithmetic / equation) | **Multi-step linear equation solving** | NEW — must construct |
| OOD #2 (planning) | **Blocksworld** | existing — Group A v1 |
| OOD #3 (CSP) | **Graph Coloring** | existing — Group A v1 |

### Group B — Logical / Deductive Reasoning

Tasks involve applying rules over abstract facts to derive new facts.

| Role | Task | Status |
|---|---|---|
| Training source | **Synthetic rule-chaining** (Horn-clause forward chaining, varied targets) | existing — built in this round, [src/oracle_rulechain.py](../src/oracle_rulechain.py) |
| OOD #1 (NL deductive) | **ProntoQA** | existing — Group A v1, repurposed for Group B |
| OOD #2 (relational) | **CLUTRR-like** | existing — built in this round, [src/oracle_clutrr.py](../src/oracle_clutrr.py) |
| OOD #3 (NL proof construction) | **ProofWriter** | NEW — must construct |

### Tasks deleted from the previous plan

- **synthlogic** (Horn-chaining at higher depth, q-prefixed): NOT in the new
  3-OOD set. Kept as a supplementary depth-OOD probe — same oracle as
  rulechain, just different generator parameters. Existing data stays for
  later analysis but isn't part of the headline matrix.
- **mini-Sudoku**: deleted entirely (oracle, adapter, scorer, configs, data,
  head checkpoint). No longer relevant to either group.

---

## Common conventions

All datasets produce JSONL records consumed by an adapter in
[src/dagger_ood_adapters.py](../src/dagger_ood_adapters.py). The adapter
exposes the unified API the rollout/training/eval drivers call.

Every record must include:
- `id`: stable string id, used as a tree-cache filename suffix.
- `prompt`: full natural-language problem text shown to the LLM.
- `init_state_text`: a concise rendering of the initial state, used as the
  Stage-1 head's z input.
- `answer_label`: a gold solution trajectory in the unified
  `Step N: <body>. Answer: <final>` format. Round-trips through the task's
  scorer with 100% accuracy.

Adapter must implement: `initial_state`, `winning_steps`, `validate_apply`,
`is_solved`, `is_terminal`, `render_state`, `parse_step`,
`format_step_text`, `make_prompt`, `step_priming_prefix`, plus class-level
`BOUNDARY_RE` / `TERMINAL_RE` regexes.

Stage-1 head training reads tree-data caches at
`data/{task}_trees_qwen14b/{train,val,test}/problem_{i}.pt` (metadata) +
`hidden_{i}.npy` (per-state last-token hidden states). The unified
generator [data/generate_tree_data_groupB.py](../data/generate_tree_data_groupB.py)
handles Group B; [data/generate_tree_data_pronto.py](../data/generate_tree_data_pronto.py)
etc. handle Group A.

---

## Group A datasets

### A1 — 24 Game (training source)

**Existing.** Records: `{pool: list[int], target: int, n_steps: int, ...}`.
Operator pool: `{+, -, ×, ÷}`. Oracle: `src/oracle_24.py` (fixed-target),
`src/oracle_24_varied.py` (varied-target). Balanced training data:
`data/24_varied_bal_train.jsonl` (6000 records, balanced over n_steps ∈ {1,2,3}).

**No new construction needed.**

### A2 — Multi-step linear equation solving (NEW OOD #1)

**Task structure.** Single-variable linear equations of the form
`a·x + b = c·x + d` with possibly multi-term sides:
`(a1·x + a2·x + b1) = (c1·x + d1 + d2)`. Solve for `x`.

**Why this fits Group A.** Concrete arithmetic over numerical objects,
deterministic state transitions, clear gold step decomposition. Sister to
G24 (both manipulate numerical quantities via four-op arithmetic).

**Construction.**
- **Generator**: pick integer solution `x* ∈ {-9, ..., 9}` and a target
  difficulty (number of steps `k ∈ {2, 3, 4, 5}`). Build a "fully expanded"
  equation form with `k` extra simplification operations applied. Example
  for `x* = 7, k = 3`:
  - Start: `x = 7`
  - Add `+5 -5` to RHS: `x = 7 + 5 - 5 = x = 12 - 5`
  - Multiply both sides by 2: `2x = 2·12 - 2·5 = 2x = 24 - 10`
  - Move `3x` term: `2x + 3x = 5x` from RHS: `2x - 3x = 24 - 10 - 3x → -x + 3x = 24 - 10` … ⟹ Final: `5x + 4 = 2x + 25`
- **Oracle interface** (`src/oracle_lineq.py`, NEW):
  - `Problem(lhs_terms, rhs_terms, solution)` where `lhs_terms`/`rhs_terms`
    are lists of `(coef_x, const)` pairs.
  - `applicable_ops(state) -> list[Op]`: per-state legal next steps. Op
    types:
    1. `combine_like_terms_lhs` — sum all x-terms and all constants on LHS
    2. `combine_like_terms_rhs` — same for RHS
    3. `move_x_to_lhs(coef)` — subtract `coef·x` from both sides
    4. `move_const_to_rhs(c)` — subtract constant `c` from both sides
    5. `divide_both_by(coef)` — when both sides have only `coef·x` and a
       constant, divide
  - `apply_op(state, op) -> new_state`.
  - `is_solved(state)`: state is `x = k` form (single x-term coef 1 on
    LHS, single constant on RHS).
  - `enumerate_tree(problem)` for Stage-1 head training: BFS over canonical
    op order (combine → move x → move const → divide).
- **Difficulty knob**: `k = number of expansion ops` controls number of
  required simplification steps. OOD eval uses `k ∈ {3, 4, 5}` (training
  source uses `k ∈ {1, 2, 3}` if we ever need a same-family training set;
  for Group A's 24-Game training, this isn't needed).
- **Step rendering**: `Step N: combine like terms on LHS → 5x + 4 = 2x + 25`
  or `Step N: subtract 2x from both sides → 3x + 4 = 25`.
- **Scorer** (`score_lineq` in `src/score_ood.py`): parse final
  `Answer: x = K` and compare to `problem.solution`.
- **Test set size**: 200 records, 6-way GPU sharded eval.
- **Generator script**: `data/generate_data_lineq.py`.

**Why integer solutions and integer coefficients**: keeps trees enumerable,
parsing simple, and matches the abstraction level of G24 (no fractions
introduced gratuitously).

### A3 — Blocksworld (existing OOD #2)

**Existing.** Test data: `data/blocksworld_test.jsonl` (PlanBench
`task_1_plan_generation` 200 records). Oracle: `src/oracle_blocksworld.py`.
Adapter: `BlocksworldAdapter`. Scorer: `score_blocksworld_goal` with
goal-reaching simulator. **No new construction needed.**

### A4 — Graph Coloring (existing OOD #3)

**Existing.** Test data: `data/graphcolor_test.jsonl` (200 in-house
generated 3-coloring problems). Oracle: `src/oracle_graphcolor.py`.
Adapter: `GraphColorAdapter`. Scorer uses `parse_coloring + score_coloring`
from the oracle. **No new construction needed.**

---

## Group B datasets

### B1 — Synthetic rule-chaining (training source)

**Existing.** Built in this round. Horn-clause forward chaining with
varied targets per problem.

- Records: `{initial_facts: list[str], target: str, rules: list[{premises, conclusion}], n_steps: int, ...}`.
- Predicates: abstract symbols `p0 .. p_{N-1}`, prefix configurable to keep
  vocabulary disjoint between training and eval.
- Generator: [data/generate_data_rulechain.py](../data/generate_data_rulechain.py).
  Default training scale: 16 predicates, 18 rules, depth ∈ {2, 3, 4},
  6000 train + 600 val + 600 test records, prefix `p`.
- Oracle: [src/oracle_rulechain.py](../src/oracle_rulechain.py). Exposes
  full Tree/Node interface (for Stage-1 head training) plus
  `winning_steps(state, problem)` for DAgger rollout supervision.

Base Qwen-14B (4-bit) baseline: **53%** on 30 random test records — strong
signal for the LoRA to amplify.

**No new construction needed.** Already done.

### B2 — ProntoQA (existing OOD #1, repurposed)

**Existing.** Test data: `data/prontoqa_test.jsonl` (200 records sliced from
the public HF `renma/ProntoQA` validation split). Oracle:
`src/oracle_pronto.py`. Adapter: `ProntoQAAdapter`. Scorer:
`score_prontoqa` (extract A/B letter).

Repurposing for Group B: Group A's ProntoQA setup remains valid because
ProntoQA IS deductive reasoning over rule bases — a natural fit for the
Group B family. Eval driver and adapter work as-is. The Group-B-specific
change is only the training-source LoRA: instead of evaluating with the
G24-trained LoRA, we evaluate with the rulechain-trained LoRA.

**No new construction needed.**

### B3 — CLUTRR-like (existing OOD #2)

**Existing.** Built in this round. Kinship relational composition with
controllable hop count.

- Records: `{entities, edges, query, answer, chain, ...}`.
- Generator: [data/generate_data_clutrr.py](../data/generate_data_clutrr.py).
  Default: hops k ∈ {2, 3, 4}, 200 test + 2000 train + 200 val.
- Oracle: [src/oracle_clutrr.py](../src/oracle_clutrr.py). Composition table
  covers basic + 1-step derived + 2-step derived kinship terms.
- Provenance: in-house generator with same task structure as the public
  CLUTRR (Sinha et al., 2019, EMNLP). Public package's matplotlib +
  sacremoses dependencies were too heavy; we reproduce the structural probe
  but lose exact-dataset reproducibility. Documented as "CLUTRR-like" in
  the writeup.

Base Qwen-14B (4-bit) baseline: **13%** on 30 random test records — non-zero,
real headroom for the LoRA.

**No new construction needed.**

### B4 — ProofWriter (NEW OOD #3)

**Task structure.** Natural-language deductive reasoning over rule bases,
with controlled proof depth. Each problem has:
- A theory: a list of NL facts and rules (e.g., "All cats are mammals.
  Mammals are warm-blooded. Tom is a cat.").
- A query: a candidate fact (e.g., "Is Tom warm-blooded?").
- A label: True / False / Unknown.
- A gold proof: the chain of rule applications used to derive the answer.

**Why this fits Group B (and complements ProntoQA + CLUTRR).** ProofWriter
adds a third style of deductive reasoning:
- **ProntoQA** = NL forward-chaining with **abstract category names**
  ("jompus", "rompus") — synthetic vocabulary, low world knowledge.
- **CLUTRR** = relational composition with **familiar kinship relations** —
  high world-knowledge prior.
- **ProofWriter** = NL forward-chaining with **realistic predicates and
  open-world / closed-world assumptions** — bridges abstract and naturalistic
  reasoning, plus introduces "Unknown" as a label class which neither PQ
  nor CLUTRR has.

**Construction approach** — use the public dataset.

- **Source**: AllenAI's ProofWriter release (Tafjord et al., 2021,
  ACL Findings). Available at <https://allenai.org/data/proofwriter>
  and on HuggingFace as `tasksource/proofwriter` or similar mirrors.
- **Variant**: pick the **CWA (closed-world assumption), depth-3** subset
  for the main eval. Reasons:
  - Depth-3 matches the difficulty range we're targeting (multi-step but
    not pathologically long).
  - CWA produces True/False labels (well-defined scoring).
  - The OWA depth-3 with "Unknown" is a stretch goal we can add later if
    base accuracy is non-trivial on CWA.
- **Test set size**: 200 records sampled deterministically from the public
  test split.
- **Schema mapping** to our standard format:
  ```jsonl
  {
    "id": "proofwriter_test_<i>",
    "theory": "<NL fact-and-rule block>",
    "query": "<NL candidate fact>",
    "answer_label": "True" | "False",
    "gold_proof_steps": [...],   // from the public release
    "prompt": "<theory>\n\nQuestion: <query>\nIs the statement true or false?",
    "init_state_text": "<theory>",
  }
  ```
- **Oracle interface** (`src/oracle_proofwriter.py`, NEW):
  - `Problem(facts: tuple[str, ...], rules: tuple[Rule, ...], query: str, gold_label: bool, gold_proof: tuple[Step, ...])`.
  - The rules are parsed from the NL theory using a thin parser (the
    public dataset includes machine-readable Prolog-style theories
    alongside NL — we use the structured form internally, render NL to
    the LLM).
  - `applicable_rules(state, rules) -> list[Rule]`: same forward-chaining
    semantics as oracle_rulechain, applied to NL-grounded predicates
    instead of abstract `p_i` symbols.
  - `enumerate_tree(problem)` produces the proof search tree; v_value =
    BFS distance to a state in which the query (or its negation under
    CWA) is provable.
- **Adapter** (`ProofWriterAdapter` in `src/dagger_ood_adapters.py`):
  similar shape to `ProntoQAAdapter` and `RuleChainAdapter`, but with NL
  step rendering and True/False answer extraction.
- **Scorer** (`score_proofwriter` in `src/score_ood.py`): extract
  True/False from generation; compare to gold label. Optional: also
  check intermediate steps match a valid forward-chain in the theory
  (more rigorous than ProntoQA's letter-match).
- **Generator script**: `data/import_proofwriter.py` — downloads the
  public CWA-depth-3 split, parses each record, writes JSONL in our
  standard format. (Not generating new problems; importing.)

**Hop count alignment with CLUTRR**: CLUTRR uses k=2/3/4 hops. ProofWriter
uses depth=3 by default. Roughly comparable difficulty target.

**Open question** (defer to actual generation time): the public release
ships natural-language stories AND machine-readable theories. We use the
NL story as the prompt for the LLM (so the model has to handle NL parsing)
but the parsed theory for our oracle. Need to verify that the parsed
theory matches what the NL story implies.

---

## Implementation order (Group B first, since training source is ready)

1. **B4 — ProofWriter import + oracle + adapter + scorer + config.** Smallest
   net-new effort because we import a public dataset rather than generate.
2. **A2 — Linear equations oracle + generator + adapter + scorer + config.**
   Larger effort because we generate from scratch.
3. **B-train (rulechain Stage-2 LoRA)**: depends only on rulechain Stage-1
   head, which depends only on rulechain tree-data — both already done /
   queued. Can train as soon as head is finished.
4. **B-eval matrix** on PQ / CLUTRR / ProofWriter using the rulechain
   Stage-2 LoRA.
5. **A-train (G24 Stage-2 LoRA)** is already trained
   (`checkpoints/dagger_stage2_24_varied_bal_r4`).
6. **A-eval matrix** on Linear-Equations / Blocksworld / Graph Coloring
   using the existing G24 LoRA.

---

## Things to confirm before generating

- **Linear-equations difficulty knobs.** Is k ∈ {3,4,5} the right OOD range
  for a "comfortably hard" probe? Should we include systems of equations
  too, or stay with single-variable?
- **ProofWriter variant.** CWA depth-3 is the recommendation. Should we
  also include OWA for an Unknown-label probe? Single variant or both?
- **Number of test records** per task. Standard is 100-200. Confirm 200
  for both new tasks.
- **Whether to keep synthlogic data** as a supplementary depth-OOD probe
  alongside the new 3-OOD set, or fully delete it. (Currently kept,
  partially generated.)
