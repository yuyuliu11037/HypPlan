"""CLUTRR-Graph prototype: dense family graph with genuine search.

Difference from oracle_clutrr.py:
- Stated edges form a graph (15-20 entities, 15-20 edges) instead of a chain.
- State = set of derived (head_idx, rel, tail_idx) triples.
- Action = pick two triples sharing a middle entity, compose to a new triple.
- v_value(state) = BFS distance over (entity-pair → derivable relations) to
  the query (qh, qt) being derivable.
- Dead-end ratio is real because most legal compositions don't progress
  toward the query target.

This module is a prototype — used by scripts/probe_clutrr_graph.py to
verify dead-end ratio and base-model accuracy before committing to the
full pipeline.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.oracle_clutrr import (
    CHAIN_RELATIONS,
    RELATION_COMPOSITION,
    compose,
    compose_chain,
)


_NAME_POOL = [
    "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Ruby", "Sam", "Tara", "Uma", "Vince", "Wendy", "Xavier",
    "Yara", "Zach", "Anna", "Ben", "Chloe", "David",
]


@dataclass
class GraphProblem:
    entities: tuple[str, ...]
    edges: tuple[tuple[int, str, int], ...]   # (i, rel, j) reads "ent[i] is rel-of ent[j]"
    query: tuple[int, int]                    # (qh, qt)
    answer: str                               # composed relation qh→qt
    gold_chain: tuple[int, ...]               # entity-index path qh = e0, e1, ..., ek = qt
    gold_relations: tuple[str, ...]           # length-k base relations along gold_chain

    def render(self) -> str:
        lines = [
            f"{self.entities[i]} is the {rel} of {self.entities[j]}."
            for i, rel, j in self.edges
        ]
        qh, qt = self.query
        return "\n".join(lines + ["", f"How is {self.entities[qh]} related to {self.entities[qt]}?"])


# ---------------------------- BFS over entity pairs ----------------------------

def _build_pair_relation_map(
    edges: tuple[tuple[int, str, int], ...],
) -> dict[tuple[int, int], set[str]]:
    """Initial derived-relation table: for each (a, b) entity pair we know
    a stated edge for, the set of relations from a to b. Triples are
    *directed* — (i, rel, j) means ent[i] is rel-of ent[j]."""
    rel: dict[tuple[int, int], set[str]] = {}
    for i, r, j in edges:
        rel.setdefault((i, j), set()).add(r)
    return rel


def shortest_compose_distance(
    edges: tuple[tuple[int, str, int], ...],
    query: tuple[int, int],
) -> tuple[int, Optional[tuple[int, ...]], Optional[tuple[str, ...]]]:
    """Forward BFS over entity pairs reachable via at-most-one-composition
    transitions. Returns (distance, gold_chain_entities, gold_relations)
    where distance is number of compositions on the shortest path. If
    `query` is already a stated edge, distance=0. If unreachable, returns
    (-1, None, None)."""
    qh, qt = query
    rel_map = _build_pair_relation_map(edges)

    # Adjacency: directed edges from `a` to `b` (via stated edge a→b).
    adj_out: dict[int, set[int]] = {}
    for (a, b), _ in rel_map.items():
        adj_out.setdefault(a, set()).add(b)

    # BFS: state is entity pair (a, b). Initial frontier = stated pairs.
    # Transition: (a, b) and (b, c) -> (a, c) with c ≠ a, if there is some
    # composable (r1, r2) with rel_map.
    # We track shortest #compositions: stated pairs have cost 0; each
    # composition adds 1 to the cost. The shortest path is then
    # cost((qh, qt)) = number of basic edges on the gold chain - 1
    # only if we count compositions, but the user's definition counts
    # compositions = (chain_length - 1). For consistency with k-hop
    # nomenclature we instead track *chain length* (number of edges)
    # so that distance == k.
    #
    # We'll BFS over entity pairs storing (parent_pair, middle_entity,
    # used_relation_from_a_to_middle) so we can reconstruct the chain.

    INF = float("inf")
    # dist[pair] = chain length = number of edges (= 1 for stated edges)
    dist: dict[tuple[int, int], int] = {}
    parent: dict[tuple[int, int], tuple[tuple[int, int], int, str]] = {}
    # parent[(a, c)] = ((a, b), middle b, relation from a to b that we used)

    bq: deque[tuple[int, int]] = deque()
    for pair in rel_map:
        dist[pair] = 1  # stated edge = chain of length 1
        bq.append(pair)

    while bq:
        ab = bq.popleft()
        a, b = ab
        d_ab = dist[ab]
        # Try extending: (a, b) ∘ (b, c) -> (a, c).
        for c in adj_out.get(b, ()):
            if c == a:
                continue
            ac = (a, c)
            for r1 in rel_map[ab]:
                for r2 in rel_map[(b, c)]:
                    r3 = compose(r1, r2)
                    if r3 is None:
                        continue
                    # Add r3 to rel_map (memoize for further extension).
                    cur_set = rel_map.setdefault(ac, set())
                    if r3 not in cur_set:
                        cur_set.add(r3)
                    # Update dist if this gives a shorter chain.
                    new_d = d_ab + 1
                    if ac not in dist or new_d < dist[ac]:
                        dist[ac] = new_d
                        parent[ac] = (ab, b, r1)
                        bq.append(ac)

    if (qh, qt) not in dist:
        return -1, None, None

    # Reconstruct gold chain: walk back from (qh, qt) via parent pointers.
    # Each parent step splits (a, c) into ((a, b), middle b, rel a→b).
    # Collect entity sequence by recursing:
    def expand(pair: tuple[int, int]) -> list[int]:
        """Return entity-index sequence for the shortest path of `pair`."""
        if pair not in parent:
            # base: stated edge a→b
            return [pair[0], pair[1]]
        prev, mid, _r = parent[pair]
        left = expand(prev)            # [a, ..., b]
        # right is the stated edge (b, c).
        right = [mid, pair[1]]         # [b, c]
        return left + right[1:]

    chain_entities = expand((qh, qt))
    # Recover the per-edge relation from rel_map (pick any if multiple).
    relations: list[str] = []
    for u, v in zip(chain_entities[:-1], chain_entities[1:]):
        rs = rel_map[(u, v)]
        # Pick a base relation if available (CHAIN_RELATIONS), else any.
        base_rs = [r for r in rs if r in CHAIN_RELATIONS]
        relations.append(base_rs[0] if base_rs else next(iter(rs)))

    return dist[(qh, qt)], tuple(chain_entities), tuple(relations)


# ---------------------------- Dead-end analysis ----------------------------

def legal_compositions(
    rel_map: dict[tuple[int, int], set[str]],
) -> list[tuple[tuple[int, int], tuple[int, int], str, str, str, tuple[int, int]]]:
    """Enumerate all (pair_ab, pair_bc, r1, r2, r3, pair_ac) compositions
    where r3 = compose(r1, r2) is defined. Excludes cases where r3 is
    already in rel_map[ac] (so we count only NEW derivations)."""
    pairs = list(rel_map.keys())
    by_left: dict[int, list[tuple[int, int]]] = {}
    by_right: dict[int, list[tuple[int, int]]] = {}
    for (a, b) in pairs:
        by_left.setdefault(a, []).append((a, b))
        by_right.setdefault(b, []).append((a, b))

    out = []
    for (a, b) in pairs:
        for (b2, c) in by_left.get(b, []):
            assert b2 == b
            if c == a:
                continue
            for r1 in rel_map[(a, b)]:
                for r2 in rel_map[(b, c)]:
                    r3 = compose(r1, r2)
                    if r3 is None:
                        continue
                    if r3 in rel_map.get((a, c), set()):
                        continue
                    out.append(((a, b), (b, c), r1, r2, r3, (a, c)))
    return out


def progressing_compositions(
    legal: list,
    query: tuple[int, int],
    rel_map: dict[tuple[int, int], set[str]],
) -> list:
    """Filter legal compositions to those that strictly reduce the
    BFS distance from current state to having the query derivable.

    Concretely: the composition (a, b) ∘ (b, c) -> (a, c) is "progressing"
    iff the resulting (a, c) pair lies on SOME shortest path from a stated
    edge to (qh, qt). For prototype simplicity, we check: is `(a, c)` on
    the unique shortest gold chain we computed? Equivalently, are both
    `a` and `c` on the gold chain with index_a < index_c, and at least
    one of (a, c) is not yet in rel_map?
    """
    qh, qt = query
    progressing = []
    # The caller hands us a pre-computed gold chain via global state — for
    # the prototype, we recompute on the fly here.
    dist0, chain_ents, _rels = shortest_compose_distance(
        edges=tuple((i, next(iter(s)), j) for (i, j), s in rel_map.items()),
        query=query,
    )
    if chain_ents is None:
        return []
    chain_set = set(chain_ents)
    chain_idx = {e: k for k, e in enumerate(chain_ents)}
    for entry in legal:
        (a, b), (b2, c), r1, r2, r3, ac = entry
        if a in chain_set and c in chain_set:
            ia, ic = chain_idx[a], chain_idx[c]
            if ic > ia:
                progressing.append(entry)
    return progressing


# ---------------------------- Generation ----------------------------

def _name(rng: random.Random, used: set[str]) -> str:
    avail = [n for n in _NAME_POOL if n not in used]
    if not avail:
        n = f"P{len(used)}"
        used.add(n)
        return n
    n = rng.choice(avail)
    used.add(n)
    return n


def generate_graph_problem(
    k: int,
    n_distractor_entities: int,
    n_distractor_edges: int,
    seed: int,
    max_attempts: int = 200,
    min_head_out: int = 2,
    min_tail_in: int = 2,
) -> Optional[GraphProblem]:
    """Generate a CLUTRR-Graph instance.

    Args:
        k: gold chain length (number of edges on shortest path).
        n_distractor_entities: extra entities NOT on the gold chain.
        n_distractor_edges: extra edges among/incident to distractor
            entities. Each is attached to an on-path entity (NOT an
            endpoint) so it shares structure with the gold chain.
        seed: rng seed.

    Returns None if generation fails after max_attempts.
    """
    rng = random.Random(seed)
    for _ in range(max_attempts):
        chain = tuple(rng.choice(CHAIN_RELATIONS) for _ in range(k))
        ans = compose_chain(chain)
        if ans is None:
            continue
        used: set[str] = set()
        chain_ents = [_name(rng, used) for _ in range(k + 1)]
        edges: list[tuple[int, str, int]] = [
            (i, chain[i], i + 1) for i in range(k)
        ]
        # Attach distractor entities. Pivots are ALL chain entities
        # (including query head 0 and query tail k), so query endpoints
        # also get multiple incident edges. Without this, query head has
        # only one outgoing edge in the narrative and the model trivially
        # picks the gold first step (no genuine first-step search).
        n_extra = max(0, int(n_distractor_entities))
        extra_idx_start = len(chain_ents)
        for d in range(n_extra):
            chain_ents.append(_name(rng, used))
        N = len(chain_ents)
        all_pivots = list(range(0, k + 1))
        n_de = max(0, int(n_distractor_edges))
        # Phase A: each distractor gets ≥1 incident edge to a pivot.
        de_made = 0
        for d in range(n_extra):
            if de_made >= n_de:
                break
            pivot = rng.choice(all_pivots)
            ext_idx = extra_idx_start + d
            rel = rng.choice(CHAIN_RELATIONS)
            # Random direction
            if rng.random() < 0.5:
                edges.append((pivot, rel, ext_idx))
            else:
                edges.append((ext_idx, rel, pivot))
            de_made += 1
        # Phase B: remaining edges among distractors only.
        attempts_b = 0
        extras_idx = list(range(extra_idx_start, N))
        while de_made < n_de and attempts_b < n_de * 20 and len(extras_idx) >= 2:
            attempts_b += 1
            a = rng.choice(extras_idx)
            b = rng.choice([x for x in extras_idx if x != a])
            rel = rng.choice(CHAIN_RELATIONS)
            edge = (a, rel, b)
            if edge in edges:
                continue
            edges.append(edge)
            de_made += 1

        # Verify: shortest path qh→qt is still k via gold chain (no shortcut).
        edges_t = tuple(edges)
        dist, gold_ents, gold_rels = shortest_compose_distance(
            edges_t, query=(0, k)
        )
        if dist != k:
            continue
        if gold_ents is None or list(gold_ents) != list(range(k + 1)):
            # The shortest path goes through distractors instead.
            continue
        # Enforce min outgoing edges from query head and incoming into tail.
        # Without this filter, distractors may all attach to interior pivots
        # by chance and the model trivially picks the unique gold first edge.
        head_out = sum(1 for (a, _, _) in edges if a == 0)
        tail_in = sum(1 for (_, _, b) in edges if b == k)
        if head_out < min_head_out or tail_in < min_tail_in:
            continue

        rng.shuffle(edges)
        return GraphProblem(
            entities=tuple(chain_ents),
            edges=tuple(edges),
            query=(0, k),
            answer=ans,
            gold_chain=tuple(range(k + 1)),
            gold_relations=tuple(chain),
        )
    return None
