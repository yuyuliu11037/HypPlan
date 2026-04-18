# Stage 3 — DAgger walkthrough with concrete examples

This file accompanies the [Stage 3 — DAgger with tree oracle](../README.md)
section of the README. It works through the mechanics with a single concrete
example so collaborators new to RL terminology can follow what the code does
end-to-end.

## What is a "rollout"?

**Rollout = let the model play one full trajectory end-to-end.**

It's terminology from reinforcement learning, where you "roll out" the policy
— let it make a sequence of decisions and observe what happens.

For the 24-Game, one rollout means:

1. Give the model the prompt (`"Problem: 4 5 6 10\nStep 1:"`).
2. Sample tokens one by one, letting the model freely choose its words
   (temperature 0.7, top-p 0.95).
3. The model produces step 1, step 2, step 3 in sequence — making real
   decisions at each step.
4. Stop when it emits a final answer, hits an invalid step, or runs out of
   token budget.

A rollout is "one episode of the model playing the game." Compare to
**teacher forcing**, where instead of letting the model choose, we paste in
the *correct* sequence and just check the model would predict it.

The DAgger loop is:

```
rollout → see what states the model reached
       → ask the oracle what it should have done at those states
       → train on those corrections
       → rollout again with the improved policy
```

We use 3 rollouts per problem so the model explores diverse paths under
sampling.

## How does the oracle "look up the state in the tree"?

The oracle is **not** a lookup into the pre-enumerated tree files. It's a
live recursive search.

Flow at each step boundary in [src/dagger_rollout.py](../src/dagger_rollout.py):

```python
wins = winning_ops(remaining)   # remaining is the current multiset
```

`winning_ops` is from [src/oracle_24.py](../src/oracle_24.py):

```python
def winning_ops(remaining):
    rem = _canon(remaining)
    winners = []
    for i, j in pairs:
        a, b = rem[i], rem[j]
        rest = ...
        for sym, fn, _ in OPS:
            r = fn(a, b)
            new_state = _canon(rest + (r,))
            if can_reach_24(new_state):     # <-- the actual oracle
                winners.append((sym, a, b, r))
    return winners
```

And `can_reach_24` is a memoized recursive search — it tries every legal
sequence of operations from `remaining` and returns True iff some leads to
24:

```python
@lru_cache(maxsize=None)
def can_reach_24(remaining):
    if len(remaining) == 1:
        return remaining[0] == 24
    for each (a, b, op) in legal moves:
        if can_reach_24(new_state):
            return True
    return False
```

The `data/trees/{split}/problem_*.pt` files (used by stage-1 head training)
are **not consulted at all** by the DAgger oracle. The oracle works on
whatever `remaining` the model reaches — even via a sequence we never
enumerated — because it just runs the recursive search live. `lru_cache`
makes repeat lookups cheap.

This means the oracle is robust to any reachable state. If the model goes
`(4,5,6,10) → 4*5=20 → ...`, we don't need that path in any precomputed
tree — `winning_ops((6,10,20))` recomputes from scratch.

## Side-by-side example: z arm vs no-z arm

Same problem, same rollout sequence, two arms compared.

**Problem:** `4,5,6,10` (initial state `(4, 5, 6, 10)`)
**Prompt** (same for both): `"Use the four given numbers...\nProblem: 4 5 6 10\nStep 1:"`

### Rollout phase (model generates one trajectory)

| Step | What happens during rollout | What gets recorded |
|---|---|---|
| Boundary 1 (before step 1) | **z arm**: inject `z_1 = up_proj(head(frozen_base(canonical_text(history=()))))` then sample. **No-z arm**: just sample. Model emits e.g. `" 10 - 4 = 6. Remaining: 5 6 6"` (a step on a winning path — `(5,6,6)` reaches 24 via 5×6=30 then 30−6=24). | `(history=(), remaining=(4,5,6,10), winners=oracle((4,5,6,10)))` |
| Boundary 2 (before step 2) | **z arm**: inject `z_2 = up_proj(head(frozen_base(canonical_text(history=((10,-,4,6),)))))`. **No-z arm**: nothing. Model emits e.g. `" 5 + 6 = 11. Remaining: 6 11"` — a *legal* step but a wrong choice; `(6,11)` is a dead end. | `(history=((10,-,4,6),), remaining=(5,6,6), winners=oracle((5,6,6)) = [('*', 5, 6, 30), ('*', 6, 5, 30)])` |
| Boundary 3 (before step 3) | Recorded but **`winners=oracle((6,11)) = []`** — `(6,11)` cannot reach 24 by any operation. Rollout stops here with `stopped_reason="empty_oracle"`; no training pair is collected from this boundary. | `(history=((10,-,4,6),(5,+,6,11)), remaining=(6,11), winners=[])` |

The recorded *history* is **what the model actually emitted**, not what the
oracle wanted. The recorded *winners* come from
`oracle_24.winning_ops(remaining)` — recomputed live, no tree-file lookup.
Boundaries with empty winners contribute no training pairs.

### Training phase (one pair → one CE pass)

Take the pair from boundary 2:
`history=((10,-,4,6),)`, `remaining=(5,6,6)`,
`winners=[('*', 5, 6, 30), ('*', 6, 5, 30)]`.

The trainer's lex-tiebreak (sort by `(op_sym, a, b)`) picks
`winner = ('*', 5, 6, 30)` → resulting state `(6, 30)` (which then reaches
24 via `30 - 6` at step 3). Target text the model is taught to emit:

```
 5 * 6 = 30. Remaining: 6 30
Step 3:
```

**No-z arm — context fed to model:**

```
[prompt embeds for "Use the four... Problem: 4 5 6 10\nStep 1:"]
[step-1 embeds for " 10 - 4 = 6. Remaining: 5 6 6\nStep 2:"]   ← model's actual step 1
+ target embeds for " 5 * 6 = 30. Remaining: 6 30\nStep 3:"   ← oracle's step 2 pick
labels = [-100]*context_len + target_ids
loss = CE on target positions
```

**z arm — context fed to model:**

```
[prompt embeds]
[z_1 embed]   ← virtual token, up_proj(head(frozen_base(canonical_text(history=()))))
[step-1 embeds for " 10 - 4 = 6. Remaining: 5 6 6\nStep 2:"]
[z_2 embed]   ← virtual token, up_proj(head(frozen_base(canonical_text(history=((10,-,4,6),)))))
+ target embeds for " 5 * 6 = 30. Remaining: 6 30\nStep 3:"
labels = [-100]*context_len + target_ids
loss = CE on target positions
```

Diff: **two extra virtual-token positions in the context** at the step
boundaries. Same target, same loss formulation. Gradient flows back into
LoRA via the model and into UpProjector via z (head + base are frozen).

### Why this isolates the z signal

- Both arms see the **same model-reached states** (statistically — different
  sampling streams per seed but same distribution given the same warm-started
  LoRA at epoch 0).
- Both arms have the **same oracle target** for those states.
- The only difference is whether the policy has access to z when it makes
  the next-token decision.
- If z carries decision-relevant information the model can't recompute from
  the text context, the z-arm trains a more accurate next-token predictor →
  higher accuracy at eval time. Δ = z signal.

That's exactly what we measured at seed 1234: **z arm 0.43 vs no-z 0.32 = +11pp**.

## See also

- [README — Stage 3 section](../README.md)
- [src/dagger_rollout.py](../src/dagger_rollout.py) — rollout loop
- [src/oracle_24.py](../src/oracle_24.py) — oracle (winning_ops, can_reach_24)
- [src/train_stage2_dagger.py](../src/train_stage2_dagger.py) — DAgger trainer
