# Concepts

Everything in coherence is one of four things: a **node**, an **edge**,
an **episode**, or a piece of **state** that ties them together.

## Node

A node is one text memory. The text can be anything from a single
sentence to a paragraph or small section — the framework does not cap
its length, and richer chunks generally make for better recall. The
`memory_ingest` tool description guides the LLM to favor self-contained
chunks with surrounding context, not bare sentences.

| Field                 | Type   | What                                              |
| --------------------- | ------ | ------------------------------------------------- |
| `id`                  | str    | 16-char identifier; stable for the node's life    |
| `text`                | str    | The memory itself, in plain language              |
| `weight`              | float  | The **salience** — the trainable scalar           |
| `activation`          | float  | Transient: how much it lit up on the last recall  |
| `tags`                | list   | Free-form labels you can filter on                |
| `metadata`            | dict   | Free-form extra fields (see below)                |
| `reinforcement_count` | int    | How many positive reinforcements it has received  |
| `failure_count`       | int    | How many negative reinforcements it has received  |

The text is set when the node is created and never updated by the
learning rule (only by `consolidate()`, which merges near-duplicates).
The weight is the only thing that learns.

Initial weight is small and positive (`0.10` by default). Successful
reinforcement pushes it up. Failure or disuse pulls it down. When
`|weight|` falls below the prune floor the node is deleted.

### Metadata from ingest enrichment

When `AutoMemory` is configured with `enrich_on_ingest=True` (default),
every new memory triggers one LLM call that returns:

| Key        | What                                                              |
| ---------- | ----------------------------------------------------------------- |
| `aliases`  | 3-6 alternate phrasings the memory should also match on           |
| `entities` | Named entities pulled out of the text                             |
| `kind`     | One of `fact`, `preference`, `constraint`, `goal`, `context`      |

The aliases are folded into the lexical index so a query that uses
different vocabulary than the memory still matches. Example:

```python
mem.remember("I work on cold-resistant photosynthesis in extremophile algae.")
# Enricher returns aliases=["my research focus", "what I study", "my work"]
# Later: mem.recall("tell me about my work") → returns the algae node, even
# though "my work" doesn't appear in the original text.
```

The `kind` lets you filter — `mem.goals()` returns all goal-kind nodes.

## Edge

An edge is a single scalar weight per unordered pair of nodes. It lives
in a dict keyed by sorted ids:

```python
mem.edges = {
    ("a1f3...", "7c92..."): +0.27,    # strong positive association
    ("a1f3...", "9bd0..."): -0.14,    # actively avoid pairing these
    ("5d11...", "7c92..."): +0.03,    # weak, just forming
}
```

Signed:

- **Positive** → "these go together." When one fires, the other gets
  pulled up.
- **Negative** → "do not pair these." When one fires, the other gets
  pushed down.
- **Near zero** → no opinion.
- **Missing** → never co-active yet.

Edges arise automatically the first time two nodes are in the active
set together at the end of a turn. The Hebbian rule then nudges them up
on success or down on failure.

## Episode

An episode is one turn of the agent loop. It records:

- `query` — the user's message that started it
- `active_ids` — the set of node ids that were in the agent's context
- `outcome` — the scalar in `[-1, +1]` that resolved the turn
- `timestamp`
- `episode` — a monotonically increasing counter
- `metadata` — free-form (includes `inferred_by` telling you which
  outcome strategy produced the signal)

Episodes are appended to an in-memory log. The most recent
`experience_log_cap` (default `5000`) are written to disk on save. You
rarely touch episodes directly — they are mostly useful when you want
to audit *why* a particular memory's weight moved when it did.

## What's on disk

`memory.json` is a single human-readable file:

```json
{
  "version": 1,
  "config": {
    "eta": 0.15, "eta_edge": 0.05,
    "decay_node": 0.01, "decay_edge": 0.02,
    "prune_floor": 0.02, "edge_floor": 0.01,
    "gamma": 0.5, "initial_weight": 0.1,
    "k_default": 5, "squash": "tanh",
    "eligibility_kind": "proportional", "eligibility_temperature": 0.5,
    "min_age_seconds": 0.0, "experience_log_cap": 5000
  },
  "nodes": [
    {
      "id": "a1f3...",
      "text": "...",
      "weight": 0.85,
      "tags": [],
      "metadata": {
        "aliases": ["my research focus", "what I study"],
        "entities": ["algae"],
        "kind": "fact"
      },
      "reinforcement_count": 4,
      "failure_count": 0
    }
  ],
  "edges": [
    { "i": "a1f3...", "j": "7c92...", "w": 0.27 }
  ],
  "experiences": [
    { "query": "...", "active_ids": [...], "outcome": 1.0, "episode": 42 }
  ],
  "episode_counter": 43,
  "saved_at": 1717800000.0
}
```

Open in any editor. Search for a fact. Change a weight. Delete a node.
The next `Memory.load(path)` will pick up the change. The memory is
auditable and editable — nothing is hidden behind an opaque artifact.

## Exporting

For a more human-readable view than the raw JSON, use `mem.export`:

```python
mem.export("memory.md", format="markdown")   # grouped by tag/kind, scored, with associations
mem.export("memory.txt", format="text")      # flat list, weight + text
mem.export("memory.json", format="json")     # same as mem.save()
```

The markdown export groups by tag and `kind`, sorts by salience, lists
the strongest associations, and tails the most recent episodes. Useful
for sharing a snapshot of what an agent has learned without exposing
the full JSON.

## State

`Memory` is the orchestrator. It holds:

- the node table (`mem.nodes`)
- the edge dict (`mem.edges`)
- the lexical index
- the episode log (`mem.experiences`)
- the config
- an optional `ingest_enricher` callable (set by `AutoMemory` when
  enrichment is enabled)

Every piece of state is plain Python. You can poke around at any time:

```python
mem.stats()                     # high-level counts and means
mem.top_nodes(10)               # 10 highest-salience nodes
mem.goals()                     # nodes with kind == "goal"
mem.neighbors(node_id)          # ranked list of connected nodes
mem.recall_as_context("query")  # the text that would be injected
```

`AutoMemory` wraps `Memory` with a chat function, an outcome strategy,
and the enrichment wiring. For most users `AutoMemory` is the only
object they ever touch directly.
