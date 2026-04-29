"""CLUTRR-like oracle for relational kinship reasoning.

Same task structure as CLUTRR (Sinha et al., 2019, EMNLP) — a chain of
kinship facts followed by a composition query — implemented in-house to
avoid the public package's heavyweight visualisation/NLP dependencies
(matplotlib, sacremoses, networkx). The hop-count parameter `k` controls
difficulty exactly as in the original (k=2,3,4 covers the standard eval
range; a model that aces k=2 can still fail at k=4).

A problem is `(entities, edges, query, answer)`:
- `entities`: tuple of names (one per node in the kinship chain).
- `edges`: tuple of (i, relation, j) basic-kinship facts. The chain forms
  a path entities[0] -> ... -> entities[k] of length `k`.
- `query`: (i, j) — the head and tail entity whose relation we ask about.
- `answer`: the composed relation (e.g., "grandmother", "uncle").

Composition is via a fixed RELATION_COMPOSITION table over basic kinship
relations (mother/father/brother/sister/son/daughter/husband/wife) and
their derived terms (grandmother, uncle, niece, ...). For relations not
in the table, composition returns "unknown" and that problem is dropped
during generation.

The oracle exposes the same Tree/Node interface as `oracle_pronto`:
- `enumerate_tree(problem)` produces the search tree where states are
  partial composition prefixes and steps apply one composition. v_value
  is BFS distance to the unique terminal (final composed relation).
- `winning_steps(state, problem)` returns the next composition on the
  shortest path (always 1 step in this canonical-order setup).

Step format:
  "Step n: <relA> of <relB> is <relC>"   (composition in NL terms)

Final answer extraction matches CLUTRR's evaluation (the model's stated
relation between query head and tail, normalized to canonical kinship
terms).
"""
from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------- Relation table ----------------------------

# Basic and derived kinship relations. Composition R1 ∘ R2 = R3 reads
# "R1 of R2 is R3" — i.e. "father of mother is grandfather".
RELATIONS = (
    "mother", "father", "son", "daughter", "brother", "sister",
    "husband", "wife",
    "grandmother", "grandfather",
    "granddaughter", "grandson",
    "aunt", "uncle",
    "niece", "nephew",
    "mother-in-law", "father-in-law",
)


# Composition table: (R1, R2) -> R3. R1 is the relation from A to B,
# R2 is the relation from B to C. R3 is the relation from A to C.
# Only well-defined entries are listed; missing entries produce
# "unknown" and the problem is rejected during generation.
RELATION_COMPOSITION: dict[tuple[str, str], str] = {
    # 2-hop building blocks
    ("mother", "mother"): "grandmother",
    ("mother", "father"): "grandfather",
    ("father", "mother"): "grandmother",
    ("father", "father"): "grandfather",
    ("mother", "brother"): "uncle",
    ("mother", "sister"): "aunt",
    ("father", "brother"): "uncle",
    ("father", "sister"): "aunt",
    ("son", "son"): "grandson",
    ("son", "daughter"): "granddaughter",
    ("daughter", "son"): "grandson",
    ("daughter", "daughter"): "granddaughter",
    ("brother", "son"): "nephew",
    ("brother", "daughter"): "niece",
    ("sister", "son"): "nephew",
    ("sister", "daughter"): "niece",
    ("husband", "mother"): "mother-in-law",
    ("husband", "father"): "father-in-law",
    ("wife", "mother"): "mother-in-law",
    ("wife", "father"): "father-in-law",
    # Identity-like compositions
    ("mother", "husband"): "father",
    ("father", "wife"): "mother",
    ("son", "wife"): "daughter-in-law",
    ("daughter", "husband"): "son-in-law",
    ("husband", "son"): "son",
    ("husband", "daughter"): "daughter",
    ("wife", "son"): "son",
    ("wife", "daughter"): "daughter",
    # 3-hop and 4-hop derive from 2-hops via repeated lookup
    ("grandmother", "brother"): "great-uncle",
    ("grandmother", "sister"): "great-aunt",
    ("grandfather", "brother"): "great-uncle",
    ("grandfather", "sister"): "great-aunt",
    ("mother", "uncle"): "great-uncle",
    ("mother", "aunt"): "great-aunt",
    ("father", "uncle"): "great-uncle",
    ("father", "aunt"): "great-aunt",
    ("son", "grandson"): "great-grandson",
    ("son", "granddaughter"): "great-granddaughter",
    ("daughter", "grandson"): "great-grandson",
    ("daughter", "granddaughter"): "great-granddaughter",
    ("brother", "grandson"): "great-nephew",
    ("brother", "granddaughter"): "great-niece",
    ("sister", "grandson"): "great-nephew",
    ("sister", "granddaughter"): "great-niece",
    ("uncle", "son"): "cousin",
    ("uncle", "daughter"): "cousin",
    ("aunt", "son"): "cousin",
    ("aunt", "daughter"): "cousin",
    # 4-hop direct-ancestor / direct-descendant compositions (so k=4
    # chains using just mother/father/son/daughter compose cleanly).
    ("grandmother", "mother"): "great-grandmother",
    ("grandmother", "father"): "great-grandfather",
    ("grandfather", "mother"): "great-grandmother",
    ("grandfather", "father"): "great-grandfather",
    ("granddaughter", "son"): "great-grandson",
    ("granddaughter", "daughter"): "great-granddaughter",
    ("grandson", "son"): "great-grandson",
    ("grandson", "daughter"): "great-granddaughter",
    ("great-grandmother", "mother"): "great-great-grandmother",
    ("great-grandmother", "father"): "great-great-grandfather",
    ("great-grandfather", "mother"): "great-great-grandmother",
    ("great-grandfather", "father"): "great-great-grandfather",
    ("great-grandson", "son"): "great-great-grandson",
    ("great-grandson", "daughter"): "great-great-granddaughter",
    ("great-granddaughter", "son"): "great-great-grandson",
    ("great-granddaughter", "daughter"): "great-great-granddaughter",
    # 3-/4-hop side-branch (sibling-of-ancestor / descendant-of-sibling)
    ("great-grandmother", "brother"): "great-great-uncle",
    ("great-grandmother", "sister"): "great-great-aunt",
    ("great-grandfather", "brother"): "great-great-uncle",
    ("great-grandfather", "sister"): "great-great-aunt",
    ("great-uncle", "son"): "first-cousin-once-removed",
    ("great-uncle", "daughter"): "first-cousin-once-removed",
    ("great-aunt", "son"): "first-cousin-once-removed",
    ("great-aunt", "daughter"): "first-cousin-once-removed",
}


def compose(r1: str, r2: str) -> Optional[str]:
    """Compose R1 ∘ R2 = R3. Returns None if undefined."""
    return RELATION_COMPOSITION.get((r1, r2))


def compose_chain(chain: tuple[str, ...]) -> Optional[str]:
    """Left-fold compose over a chain of basic relations.
    Returns None if any intermediate composition is undefined."""
    if len(chain) == 0:
        return None
    if len(chain) == 1:
        return chain[0]
    cur = chain[0]
    for r in chain[1:]:
        nxt = compose(cur, r)
        if nxt is None:
            return None
        cur = nxt
    return cur


# Basic relations actually used for chain edges. We exclude husband/wife
# from generation to keep difficulty roughly uniform; they introduce
# many edge cases (in-law expansions). But the composition table still
# accepts them, so vocabulary is shared with the eval problems.
CHAIN_RELATIONS = (
    "mother", "father", "son", "daughter", "brother", "sister",
)


# ---------------------------- Data classes ----------------------------

@dataclass
class Problem:
    entities: tuple[str, ...]                   # names: e0, e1, ..., ek
    edges: tuple[tuple[int, str, int], ...]    # (i, rel, j); i is rel-of(j)
    query: tuple[int, int]                      # (head, tail) entity indices
    answer: str                                 # composed relation
    chain: tuple[str, ...]                      # ordered relations along path
    raw: str = ""

    def render_problem(self) -> str:
        story_lines = [
            f"{self.entities[i]} is the {rel} of {self.entities[j]}."
            for i, rel, j in self.edges
        ]
        head, tail = self.query
        question = (
            f"How is {self.entities[head]} related to {self.entities[tail]}?"
        )
        return "\n".join(story_lines + ["", question])


@dataclass
class Node:
    node_id: int
    state: tuple[str, ...]   # composition prefix: derived relation so far
    parent: Optional[int]
    step_used: Optional[tuple[int, str]]   # (hop_index, applied_relation)
    children: list[int] = field(default_factory=list)
    depth: int = 0
    is_solved: bool = False
    v_value: int = -1


@dataclass
class Tree:
    problem: Problem
    nodes: list[Node]


# ---------------------------- Problem generation ----------------------------

def _random_name(rng: random.Random, used: set[str]) -> str:
    """Lazy-import `names` only when called, falling back to a small
    in-house pool if the package is unavailable."""
    try:
        import names
        for _ in range(50):
            n = names.get_first_name()
            if n not in used:
                used.add(n)
                return n
    except ImportError:
        pass
    pool = [
        "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
        "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
        "Quinn", "Ruby", "Sam", "Tara", "Uma", "Vince", "Wendy", "Xavier",
        "Yara", "Zach", "Anna", "Ben", "Chloe", "David", "Emma", "Fred",
    ]
    avail = [n for n in pool if n not in used]
    if not avail:
        n = f"Person{len(used)}"
        used.add(n)
        return n
    n = rng.choice(avail)
    used.add(n)
    return n


def generate_problem(
    k: int,
    seed: int = 0,
    max_attempts: int = 200,
    n_distractor_entities: int = 0,
    n_distractor_edges: int = 0,
) -> Problem:
    """Generate a CLUTRR-like problem with a length-`k` kinship chain and a
    composition query for the head-to-tail relation.

    Difficulty knobs:
        `k` — number of hops along the answer chain.
        `n_distractor_entities` — extra entities not on the answer path.
        `n_distractor_edges` — extra kinship edges among the *off-path*
            entities (or attaching them to one chain entity), so the
            model can no longer assume "every edge is on the path".
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    rng = random.Random(seed)
    for _ in range(max_attempts):
        chain = tuple(rng.choice(CHAIN_RELATIONS) for _ in range(k))
        answer = compose_chain(chain)
        if answer is None:
            continue
        used: set[str] = set()
        chain_entities = tuple(
            _random_name(rng, used) for _ in range(k + 1)
        )
        # chain_entities[i] is `chain[i]` of chain_entities[i+1].
        path_edges = tuple(
            (i, chain[i], i + 1) for i in range(k)
        )

        # Distractor entities (off the answer path).
        n_extra = max(0, int(n_distractor_entities))
        extras = tuple(
            _random_name(rng, used) for _ in range(n_extra)
        )
        all_entities = chain_entities + extras
        chain_idx_set = set(range(len(chain_entities)))
        extra_idx_set = set(range(
            len(chain_entities), len(chain_entities) + len(extras)
        ))

        # Distractor edges. We pick (head, rel, tail) such that the edge
        # does NOT shorten or fork the answer path:
        #   - both endpoints are extras (off-path subgraph), OR
        #   - one endpoint is an extra and the other is a chain entity.
        # We never connect two non-adjacent chain entities directly,
        # which would shortcut the answer chain.
        n_dedge = max(0, int(n_distractor_edges))
        distractor_edges: list[tuple[int, str, int]] = []
        attempts_d = 0
        while (
            len(distractor_edges) < n_dedge
            and attempts_d < n_dedge * 20
            and (extras or False)
        ):
            attempts_d += 1
            # require at least one extra endpoint
            if extras:
                a = rng.choice(list(extra_idx_set))
                b_pool = list(extra_idx_set | chain_idx_set)
                b_pool.remove(a)
                b = rng.choice(b_pool)
                rel = rng.choice(CHAIN_RELATIONS)
                edge = (a, rel, b)
                # avoid duplicates / inverse-of-existing
                if edge not in distractor_edges:
                    distractor_edges.append(edge)

        # Shuffle full edge list so distractors interleave with path
        # edges in the prompt narration.
        full_edges = list(path_edges) + distractor_edges
        rng.shuffle(full_edges)
        return Problem(
            entities=all_entities,
            edges=tuple(full_edges),
            query=(0, k),  # head = chain_entities[0], tail = chain_entities[k]
            answer=answer,
            chain=chain,
        )
    raise RuntimeError(
        f"Could not generate CLUTRR-like problem at k={k} after {max_attempts} attempts"
    )


# ---------------------------- Tree enumeration ----------------------------

def enumerate_tree(problem: Problem, max_nodes: int = 200) -> Tree:
    """Build a canonical-order composition tree.

    State is the composition prefix `(r1, r1∘r2, r1∘r2∘r3, ...)`. The
    canonical step at depth d is "compose-with-chain[d]". The terminal
    state has length k and equals (chain[0], chain[0]∘chain[1], ...,
    answer).

    Branching is 1 (canonical order) — we keep the Tree structure for
    consistency with other oracles, even though there is no real
    branching. v_value labels distance to the solved terminal."""
    chain = problem.chain
    k = len(chain)
    nodes: list[Node] = [
        Node(
            node_id=0,
            state=(),
            parent=None,
            step_used=None,
            depth=0,
            is_solved=False,
        )
    ]
    cur_id = 0
    cur_rel: Optional[str] = None
    for d in range(k):
        next_rel = chain[d] if d == 0 else compose(cur_rel, chain[d])
        if next_rel is None:
            break  # ill-defined; truncate
        nid = len(nodes)
        prev_state = nodes[cur_id].state
        new_state = prev_state + (next_rel,)
        is_solved = (d == k - 1)
        nodes.append(
            Node(
                node_id=nid,
                state=new_state,
                parent=cur_id,
                step_used=(d, chain[d]),
                depth=d + 1,
                is_solved=is_solved,
            )
        )
        nodes[cur_id].children.append(nid)
        cur_id = nid
        cur_rel = next_rel

    # v_value: BFS over undirected edges from solved leaves.
    dist = [-1] * len(nodes)
    bq: deque[int] = deque()
    for n in nodes:
        if n.is_solved:
            dist[n.node_id] = 0
            bq.append(n.node_id)
    while bq:
        nid = bq.popleft()
        n = nodes[nid]
        nbrs = list(n.children)
        if n.parent is not None:
            nbrs.append(n.parent)
        for m in nbrs:
            if dist[m] == -1:
                dist[m] = dist[nid] + 1
                bq.append(m)
    for n in nodes:
        n.v_value = dist[n.node_id]
    return Tree(problem=problem, nodes=nodes)


# ---------------------------- Lightweight oracle ----------------------------

def winning_steps(
    state: tuple[str, ...], problem: Problem
) -> list[tuple[int, str]]:
    """Return [(hop_index, basic_relation)] for the next canonical step.
    State length d means we've composed d hops; the next step is
    chain[d]. List has 0 elements if solved, 1 otherwise."""
    chain = problem.chain
    d = len(state)
    if d >= len(chain):
        return []
    return [(d, chain[d])]


def validate_step(
    state: tuple[str, ...], hop_index: int, relation: str, problem: Problem
) -> tuple[bool, tuple[str, ...]]:
    """Verify an asserted (hop_index, relation) step matches the canonical
    chain order, then advance the composition prefix."""
    chain = problem.chain
    d = len(state)
    if hop_index != d or d >= len(chain):
        return False, state
    if relation != chain[d]:
        return False, state
    if d == 0:
        new_rel = chain[0]
    else:
        new_rel = compose(state[-1], chain[d])
    if new_rel is None:
        return False, state
    return True, state + (new_rel,)


def is_solved(state: tuple[str, ...], problem: Problem) -> bool:
    return len(state) == len(problem.chain)


# ---------------------------- Rendering / parsing ----------------------------

def render_state(problem: Problem, state: tuple[str, ...]) -> str:
    if not state:
        return "(no relation derived yet)"
    head = problem.entities[problem.query[0]]
    cur = problem.entities[len(state)]
    return f"{head} is the {state[-1]} of {cur}"


def format_question(problem: Problem) -> str:
    return problem.render_problem()


def format_step_text(
    problem: Problem, hop_index: int
) -> str:
    """One supervised step: 'Step n: <head> is the <relation_so_far> of <intermediate>'."""
    head = problem.entities[problem.query[0]]
    cur = problem.entities[hop_index + 1]
    rel = compose_chain(problem.chain[: hop_index + 1])
    return f"{head} is the {rel} of {cur}"


def format_gold_trajectory(problem: Problem) -> str:
    lines: list[str] = []
    for d in range(len(problem.chain)):
        lines.append(f"Step {d + 1}: " + format_step_text(problem, d))
    head = problem.entities[problem.query[0]]
    tail = problem.entities[problem.query[1]]
    lines.append(f"Answer: {head} is the {problem.answer} of {tail}.")
    return "\n".join(lines)


_ANSWER_RE = re.compile(
    r"answer\s*[:\-]?\s*.+?\bis\s+(?:the\s+)?([a-z\-]+)\s+of\s+",
    re.IGNORECASE | re.DOTALL,
)
_RELATION_RE = re.compile(
    r"\b(mother|father|son|daughter|brother|sister|husband|wife|"
    r"grand(?:mother|father|son|daughter)|aunt|uncle|niece|nephew|"
    r"cousin|(?:mother|father|son|daughter)-in-law|"
    r"great-(?:uncle|aunt|grandson|granddaughter|nephew|niece))\b",
    re.IGNORECASE,
)


def parse_answer(generation: str) -> Optional[str]:
    """Extract the predicted relation from a model generation."""
    m = _ANSWER_RE.search(generation)
    if m:
        return m.group(1).lower()
    rels = _RELATION_RE.findall(generation)
    if rels:
        return rels[-1].lower()
    return None


def score_answer(prediction: Optional[str], gold: str) -> bool:
    if prediction is None:
        return False
    return prediction.lower() == gold.lower()


# ---------------------------- Step parsing ----------------------------

_STEP_RE = re.compile(
    r"is\s+the\s+([a-z\-]+)\s+of\s+",
    re.IGNORECASE,
)


_STEP_ASSERTION_RE = re.compile(
    r"is\s+the\s+([a-z\-]+)\s+of\s+([A-Za-z]+)", re.IGNORECASE
)


def parse_step(text: str, problem: Problem, current_state_len: int) -> Optional[tuple[int, str]]:
    """Parse one model-emitted reasoning step into (hop_index, relation).

    Accepts two equivalent forms in the assertion "<head> is the <X> of
    <intermediate>":
      - X is the next basic relation (chain[d]) — when the model verbalises
        the raw hop.
      - X is the derived composition (compose_chain(chain[:d+1])) — what
        `format_step_text` actually emits, since CoT reads more naturally
        as "X is the granddaughter of Y" than "apply daughter hop".

    Returns (hop_index, basic_relation). The validity check only succeeds
    if the assertion's tail entity matches the expected intermediate at
    hop d, and the asserted relation matches either the basic or derived
    form for that hop."""
    chain = problem.chain
    if current_state_len >= len(chain):
        return None
    expected_basic = chain[current_state_len]
    expected_intermediate = problem.entities[current_state_len + 1]
    expected_derived = compose_chain(chain[: current_state_len + 1])
    for m in _STEP_ASSERTION_RE.finditer(text):
        rel = m.group(1).lower()
        target = m.group(2)
        if target != expected_intermediate:
            continue
        if rel == expected_basic.lower():
            return (current_state_len, expected_basic)
        if expected_derived and rel == expected_derived.lower():
            return (current_state_len, expected_basic)
    return None
