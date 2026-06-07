# How learning works

Every turn the framework runs two passes over the memory graph: a
**forward pass** (recall) and a **backward pass** (reinforcement). Once
in a while a **maintenance pass** decays and prunes.

All three are arithmetic on scalars.

## Forward pass — recall

`AutoMemory` runs in one of three recall modes:

| `recall_mode` | What it does                                                          |
| ------------- | --------------------------------------------------------------------- |
| `"auto"` (default) | LLM reads the full memory dump and picks relevant chunks. Falls back to BM25 if the dump grows past `recall_threshold_chars`. |
| `"llm"`       | Always use LLM-driven recall. Returns `[]` (then BM25 fallback) if the dump exceeds the threshold. |
| `"bm25"`      | Always use the scalar activation pass below — no recall LLM call.     |

In LLM mode, the framework sends the user's query plus every stored
memory's id and text to the LLM and asks it to return the top-k IDs as
a JSON array. Cheap when the file is small; impractical once the
memory dump no longer fits comfortably in the model's context. Set
`context_length=N` on `AutoMemory` and the threshold derives
automatically (≈ context/10 chars reserved for the recall payload).

The BM25 path computes one scalar activation per node and returns the
top-k. For each node `i`:

```
match_i  = lexical_overlap(query, n_i.text)         # in [0, 1]
base_i   = match_i + tanh(weight_boost * w_i)        # weight biases match
spread_i = sum_j ( W_ij * base_j )                   # one-hop spread over edges
a_i      = squash(base_i + gamma * spread_i)         # bounded final activation
```

In plain words:

- `match_i` is how much the node's text overlaps with the query — a
  classic length-normalized term-frequency / inverse-document-frequency
  score over an inverted index.
- `tanh(w_i)` is the node's standing — how trustworthy it has been on
  past turns, bounded so a single very-trusted node cannot drown out
  the rest of the graph.
- `spread_i` is the influence of neighbors: a positive edge from a
  firing neighbor pulls this node up; a negative edge pushes it down.
- `a_i` is squashed into `[-1, +1]` so a few outliers can't dominate.

The `k` nodes with the highest activation form the **active set** for
the turn. They are what gets injected into the prompt.

## Backward pass — reinforcement

After the agent acts and the outcome arrives, the active set's weights
move.

For each active node `i`:

```
elig_i = a_i / (a_1 + a_2 + ... + a_N)               # share of total firing
Δw_i   = eta * outcome * elig_i                      # delta-rule update
w_i    = w_i + Δw_i
```

If the outcome is `+1` and node A fired with 60% of the total
activation while four other nodes each fired with 10%, then:

- A gains `eta * 1.0 * 0.60`
- each of the others gains `eta * 1.0 * 0.10`

Big contributors get most of the credit; freeloaders get almost nothing.

For every pair `(i, j)` in the active set:

```
ΔW_ij = eta_edge * outcome * elig_i * elig_j         # Hebbian co-activation
W_ij  = W_ij + ΔW_ij
```

Edges only move quickly when both endpoints fired strongly. Two weak
freeloaders barely change their edge; two strong contributors move it
fast.

`outcome` is the single scalar the framework saw at the end of the
turn — `+1` for a clean success, `-1` for a clean failure, anything in
between for graded feedback. See [outcomes](outcomes.md) for where it
comes from.

## Two flavors of credit assignment

`eta * outcome * elig_i` distributes credit **proportionally** by
default — each node's share is a linear fraction of total firing.

For a sharper distribution, switch to `eligibility_kind="softmax"`. With
a low temperature this concentrates almost all the credit on the
single top-firing node; with a high temperature it softens toward
uniform. The softmax shape is what the LLM-driven examples use because
it makes the per-turn signal cleaner.

```python
mem = AutoMemory(
    chat_fn=my_chat,
    eligibility_kind="softmax",
    eligibility_temperature=0.3,
)
```

## Maintenance pass — decay and prune

Once in a while (the LLM can call `memory_maintenance`, or you can call
`mem.forget()` directly), the framework runs:

```
w_i  = w_i * (1 - decay_node)                        # weight decay
W_ij = W_ij * (1 - decay_edge)

if |w_i|  < prune_floor:  delete node i
if |W_ij| < edge_floor:   delete edge ij
```

Decay shrinks everything by a small percentage. Repeatedly reinforced
weights grow faster than they decay; weights that nothing touches drift
back toward zero. Pruning then deletes anything that has drifted close
enough to zero.

Pruning checks `|weight|`, so a strongly *negative* weight survives —
it's a "do not go back there" signal worth keeping until it itself
decays away.

`min_age_seconds` is a grace period for newborn nodes so a fresh
ingest is not culled before it has had a chance to earn reinforcement.

## Consolidation

`mem.consolidate()` merges near-duplicate nodes by token overlap. The
kept node absorbs the dropped one's salience, reinforcement counts, and
incident edges. This keeps the graph from filling up with paraphrases
of the same fact.

The default similarity threshold is `0.82` (token-Jaccard); pass a
different value to tune.

## Tunables

The learning-rule knobs live on `Memory`. To configure them, build a
`Memory` explicitly and hand it to `AutoMemory`:

```python
from coherence import Memory, AutoMemory

graph = Memory(
    eta=0.20,                       # node learning rate
    eta_edge=0.05,                  # edge learning rate
    decay_node=0.02,                # per-maintenance node decay
    decay_edge=0.04,                # per-maintenance edge decay
    prune_floor=0.05,               # |w| below this gets deleted
    edge_floor=0.02,
    gamma=0.5,                      # spread strength
    initial_weight=0.10,
    eligibility_kind="softmax",
    eligibility_temperature=0.30,
    min_age_seconds=0.0,
)

mem = AutoMemory(chat_fn=my_chat, memory=graph, path="memory.json")
```

`AutoMemory`'s own knobs (`recall_k`, `context_budget_tokens`,
`judge_batch_size`, `enrich_on_ingest`, …) are passed to `AutoMemory`
directly.

Sensible defaults are baked in for both layers. Common tuning patterns:

- **Fast learner, short memory.** Raise `eta` and `decay_node` together.
  Useful for chat assistants that pick up user preferences quickly and
  forget stale ones.
- **Slow, stable memory.** Lower both. Useful for long-running research
  assistants where individual signal is noisy but accumulates over
  many turns.
- **Sparser graph.** Raise the prune floors. The graph stays small at
  the cost of forgetting things that hadn't yet earned strong salience.
- **Sharper credit assignment.** Switch to softmax eligibility with a
  low temperature. The top contributor gets nearly all the credit per
  turn.

## Context-window awareness

`AutoMemory` reads the model's context size and adapts. There are two
ways to wire it:

**Explicit — pass each knob.**

```python
mem = AutoMemory(
    chat_fn=my_chat,
    recall_k=10,                    # how many top memories to consider
    context_budget_tokens=500,       # inject only what fits in 500 tokens per turn
    recall_threshold_chars=8000,     # above this dump size, BM25 takes over from LLM recall
)
```

**Implicit — pass the model's total context length.**

```python
# 8k-context model
mem = AutoMemory(chat_fn=short_model, context_length=8_000)

# 200k-context model
mem = AutoMemory(chat_fn=long_model,  context_length=200_000)
```

`context_length` (in tokens) auto-derives sensible defaults:
`context_budget_tokens ≈ context_length / 10` and
`recall_threshold_chars ≈ context_length × 4 / 10` (chars, assuming
~4 chars/token). A 200k-context model gets ~20k tokens of memory
injection per turn and ~80k chars of dump tolerance before BM25 kicks
in. An 8k-context model gets ~800 tokens of injection and ~3.2k chars
of dump tolerance.

Explicit values always override the derived ones.

## Memory chunk size

A memory is whatever the agent decides to store — a sentence, a
paragraph, a small section. The framework imposes no hard limit. The
`memory_ingest` tool description instructs the model to favor
self-contained chunks with enough surrounding context to stand on
their own when surfaced again later; a single bare sentence is rarely
optimal, since the model that retrieves it next time may lack the
context that gave the sentence its meaning.

LLM recall benefits especially from richer chunks — the grader can
judge relevance from the whole paragraph, not from a stripped-down
sentence.
