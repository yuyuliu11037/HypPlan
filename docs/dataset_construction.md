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
| OOD #1 (arithmetic / search) | **Number-path** (reachability with fixed op set) | built 2026-04-26 |
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

### Tasks tried-and-discarded during plan iteration

- **mini-Sudoku** (4×4 CSP): originally proposed as Group B OOD #3, swapped
  out before training. Oracle, adapter, configs, and data fully deleted.
- **Multi-step linear equations**: originally proposed as Group A OOD #1,
  abandoned after the base 4-bit Qwen-14B-Instruct eval reached **99%**
  (198/200) — no headroom for the LoRA. Oracle / adapter / scorer / configs
  / data kept under their `lineq_` names for reference but not used.
- **Small-scale Countdown** (cd_small, 4 numbers + 2-digit target):
  considered as a lineq replacement; base 4-bit reached ~10% on 100/200
  before the run was killed. Functional but Number-path was preferred for
  the higher base accuracy (~34%) → more measurement room.
- **synthlogic** (Horn-chaining at higher depth, q-prefixed): NOT in the new
  3-OOD set. Kept as a supplementary depth-OOD probe — same oracle as
  rulechain, just different generator parameters.

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

### A2 — Number-path / reachability (OOD #1)

**Task structure.** Given a start integer `S`, a target integer `T`, and
a small fixed set of allowed arithmetic operations (e.g.
`{+5, +7, ×2, −3}`), find a sequence of operations that transforms `S`
into `T`. State = single integer; action = one allowed op; goal =
state == target. The op set is randomized per problem so the model can't
memorize a fixed procedure.

**Why this fits Group A.** Concrete arithmetic on integers, deterministic
state transitions, search-heavy because the tree branches over the full
op set at each state and most paths miss the target. Pairs naturally
with G24 (same number-manipulation family) but exercises a different
search shape (single-state-with-op-choice vs G24's pool-shrinking).

**Construction.**
- Op vocabulary (`OP_BANK` in `src/oracle_numpath.py`): `{+1, +2, +3, +5,
  +7, −1, −2, −3, −5, −7, ×2, ×3, ÷2, ÷3}`. Sub must give non-negative
  result; div must be exact.
- **Generator** (`generate_problem(target_depth, op_set_size, seed)`):
  1. Sample `op_set_size` distinct ops from `OP_BANK` (default 4).
  2. Sample start `S` ∈ [1, 50].
  3. Random-walk `target_depth` steps from `S` using the chosen op set
     to obtain a candidate target `T`.
  4. Verify the BFS minimum distance from `S` to `T` equals
     `target_depth` (rejects shortcuts).
- **Oracle interface** (`src/oracle_numpath.py`): standard Tree/Node +
  `winning_steps(state, problem)` matching the rest of the codebase.
  `enumerate_tree` does BFS bounded by `max_value=999` and
  `max_depth=12`.
- **Difficulty knob**: `target_depth` (number of operations on the
  shortest path). Test set uses `target_depth ∈ {3, 4, 5}` evenly.
- **Step rendering**: `Step N: <state_before> <op> <const> = <state_after>`.
- **Scorer** (`score_numpath` in `src/score_ood.py`): simulate the model's
  emitted ops from `start`, check final value equals `target`. Each
  emitted op must be in the allowed set; illegal step terminates the
  simulation.
- **Test set size**: 200 records (200 train, 200 val, 2000 train default).
- **Generator script**: [data/generate_data_numpath.py](../data/generate_data_numpath.py).

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

## Build order (completed)

1. ✅ ProofWriter import + oracle + adapter + scorer + configs.
2. ✅ Number-path oracle + generator + adapter + scorer + base eval.
   Selected as Group A OOD #1 over lineq (99% base, no headroom) and
   cd_small (~10% base, less measurement room).
3. ✅ rulechain training source built (Horn-clause forward chaining,
   `pred_prefix` configurable so eval-time vocabulary can differ from
   training-time).
4. ✅ CLUTRR-like in-house generator (avoids matplotlib/sacremoses dep
   weight from the public CLUTRR package; same task structure with
   controllable hop count k=2/3/4).

## Locked-in decisions (2026-04-26)

- **Number-path** is Group A OOD #1, replacing the originally-proposed
  Linear Equations (which Qwen-14B 4-bit aces at 99%). Difficulty knob:
  `target_depth ∈ {3, 4, 5}` BFS shortest-path length, op set size 4,
  randomized op set per problem.

---

## Locked-in decisions (2026-04-26)

- **ProofWriter: CWA only.** CWA produces clean True/False labels that
  match the scorer pattern used by ProntoQA, gives directly comparable
  numbers, and avoids the subtleties of Unknown-label inference under OWA.
  This matches the standard ProofWriter benchmark configuration in
  published work. Use depth-3 split, 200 records.
- **Base few-shot eval is mandatory before training every dataset.** Run
  4-bit Qwen-14B-Instruct on the test split for all 8 tasks (4 Group A +
  4 Group B). Any task at 0% (or near-zero) accuracy is a candidate for
  substitution or difficulty adjustment before we commit GPU-hours to
  training.
- **Test set size**: 200 records per OOD task, 100 records for the
  in-domain training-source eval (matches Group A v1 convention).
- **synthlogic data**: kept as a supplementary depth-OOD probe (same
  oracle as rulechain, just different generator parameters) but not in
  the headline 3-OOD set. Existing partial data stays.

## Pre-training checklist

Before starting any new training run, confirm:

1. ✅ All 8 oracles + adapters + scorers + configs in place
2. ✅ All 8 test JSONL files present and gold-trajectories scoring 100%
3. ⏳ Base few-shot eval run on all 8 tasks; results documented
4. ⏳ No task with 0% base accuracy (substitute or adjust if any)
5. ⏳ Tree-data caches built for all 4 Stage-1 head tasks (rulechain,
   PQ, CLUTRR, ProofWriter for Group B; G24, Linear-Eq, BW, GC for Group A
   — though Group A already has G24/BW/GC heads from v1)

## Base few-shot eval — final state (2026-04-27)

All 8 datasets locked in with non-zero base accuracy. Numbers in **bold**
are 4-bit Qwen-14B-Instruct from this session; non-bold are bf16
from Group A v1 (HANDOFF.md). 4-bit is a lower bound on bf16, so
non-zero in 4-bit ⇒ non-zero in bf16.

| Group | Task | Base accuracy | Source |
|---|---|---|---|
| A | 24 Game (G24-100) | 11% | v1 bf16 |
| A | **Number-path** | **34.5%** (69/200) | this session, 4-bit, 4-way sharded |
| A | Blocksworld | 41% | v1 bf16 (goal-reaching) |
| A | Graph Coloring | 61% | v1 bf16 |
| B | rulechain | **53%** | this session, 4-bit on 30 records |
| B | ProntoQA | 60% | v1 bf16 |
| B | CLUTRR-like | **13%** | this session, 4-bit on 30 records |
| B | ProofWriter (CWA d3) | **70%** (141/200) | this session, 4-bit |

Notes:
- **Number-path was selected** over linear-equations (99% — too easy) and
  small-scale Countdown (cd_small, ~10% — viable but less measurement
  room than numpath's 34.5%).
- ProofWriter is on the high end of the typical headroom range but still
  has clear room for the LoRA to lift toward the depth-3-only ceiling
  (~85-90% achievable with explicit reasoning).
- CLUTRR sits at 13% in 4-bit; bf16 will be modestly higher. The
  multi-hop composition over kinship terms is genuinely hard for the
  base model.
