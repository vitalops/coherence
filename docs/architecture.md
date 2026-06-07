# Architecture

A bird's-eye view of how the pieces fit together and what happens on
each turn.

## The three layers

```
        +------------------------------------------+
        |  Your agent loop / harness / Claude Code |
        +------------------------------------------+
                          ▲
                          │
        +------------------------------------------+
        |              AutoMemory                  |   ← recommended entry
        |  recall · ingest · judge · persist       |
        +------------------------------------------+
                          ▲
                          │
        +------------------------------------------+
        |        coherence.integrations            |
        | openai_protocol · anthropic_protocol     |   ← adapters for fixed harnesses
        | mcp · dispatch · MemorySession           |
        +------------------------------------------+
                          ▲
                          │
        +------------------------------------------+
        |                Memory                    |   ← raw memory graph
        |  ingest · recall · reinforce · forget    |
        |  consolidate · save · load · export      |
        +------------------------------------------+
                          ▲
                          │
        +------------------------------------------+
        | matcher · activation · eligibility       |   ← algorithmic primitives
        | reinforce · forget · enrichment · judging|
        | recall_llm                               |
        +------------------------------------------+
```

Most users only ever see `AutoMemory`. Everything below it is exposed
in case you need to skip a layer.

## Files in `coherence/`

| File                       | Responsibility                                                                 |
| -------------------------- | ------------------------------------------------------------------------------ |
| `node.py`                  | The `Node` dataclass: id, text, salience weight, transient activation, tags, metadata, counts. |
| `matcher.py`               | An inverted-index BM25 lexical matcher. Tokenizes text, builds the index, scores a query against every doc. Supports aliases that extend each doc's searchable surface. |
| `activation.py`            | The forward pass for BM25 recall: spreading activation over the graph (`match + tanh(weight) + γ · spread`), bounded by `tanh`. |
| `eligibility.py`           | Credit-assignment shapes: proportional or softmax. Given activations + an active set, returns each node's share of the total firing. |
| `reinforce.py`             | The backward pass: delta-rule update on node weights, Hebbian update on edge weights. |
| `forget.py`                | Weight decay + |w|-floor pruning. The compression half of the implicit objective. |
| `experience.py`            | The `Experience` dataclass: one episode (query, active ids, outcome, timestamp). |
| `graph.py`                 | The `Memory` orchestrator. Owns the node table, edge dict, lexical index, and episode log. Exposes ingest/recall/reinforce/forget/consolidate/save/load/export. |
| `enrichment.py`            | LLM-driven enrichment at ingest: extract aliases, entities, kind tag. One LLM call per new memory. |
| `recall_llm.py`            | LLM-driven recall: send the whole memory dump and the query to the LLM, get back ranked IDs. Returns `[]` if the dump doesn't fit. |
| `judging.py`               | Batched LLM outcome judging: grade N turns in one LLM call, return one score per turn. |
| `autopilot.py`             | The `AutoMemory` wrapper: per-turn orchestration, recall mode selection, judge buffer, history management, persistence. |
| `integrations/tools.py`    | The canonical tool definitions (name, description, parameter schema) — single source of truth for all wire formats. |
| `integrations/openai_protocol.py` | Reshapes the tools for the OpenAI `/chat/completions` wire format and executes tool calls. |
| `integrations/anthropic_protocol.py` | Same, for the Anthropic Messages API. |
| `integrations/mcp.py`      | Same, as MCP server tool descriptors (Claude Code, Cursor, etc.). |
| `integrations/session.py`  | A small context manager: `with session.episode(...) as ep: ... ep.success()`. |

## Per-turn flow

When you call `mem.complete(user_message)` on `AutoMemory`, this is
what happens, in order:

```
   user_message
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Close out the previous turn                                  │
│    Move the staged previous turn into the judge buffer with     │
│    next_user = this user_message.                               │
│    If buffer ≥ judge_batch_size and strategy == "llm":          │
│       → ONE LLM call grades the whole buffer (judging.py)       │
│       → Apply reinforcement (reinforce.py) for each turn        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Recall                                                       │
│    recall_mode == "auto":                                       │
│      if memory_dump_chars < recall_threshold_chars:             │
│         → LLM recall (recall_llm.py)                            │
│         → ONE LLM call, returns ranked IDs                      │
│      else:                                                      │
│         → BM25 recall (matcher.py + activation.py)              │
│         → 0 LLM calls; pure arithmetic over inverted index      │
│    recall_mode == "llm":  always try LLM, fall back to BM25     │
│    recall_mode == "bm25": always use BM25                       │
│                                                                 │
│    Pack the recalled nodes into context_budget_tokens.          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Build the per-turn prompt                                    │
│    [system prompt] + [chat history] +                           │
│      [system note: recalled memories + active goals] +          │
│      [user message]                                             │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Generate (the agent's own LLM call)                          │
│    chat_fn(messages, tools=[memory_recall, memory_ingest,       │
│                              memory_reinforce,                  │
│                              memory_maintenance])               │
│                                                                 │
│    If the model emits tool_calls, dispatch them through         │
│    integrations.openai_protocol.run_openai_tool_calls():        │
│      - memory_recall    → forward pass, return memories         │
│      - memory_ingest    → Memory.ingest(...)                    │
│                            → triggers enrichment.enrich_memory  │
│                            → ONE LLM call per ingest            │
│      - memory_reinforce → Memory.reinforce(...)                 │
│      - memory_maintenance → Memory.forget() + consolidate       │
│                                                                 │
│    Loop until the model produces a non-tool reply.              │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Stage this turn                                              │
│    pending = {user, reply, active_ids, next_user=None}          │
│    Append (user_message, reply) to canonical history.           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Persist                                                      │
│    If save_every_turn=True and path is set:                     │
│       → Memory.save(path) writes the whole graph to JSON        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
   final_text
```

## Data model

The memory graph has three tables and one log:

```
nodes : id → Node
─────────────────────────────────────
{
  id, text, weight, activation,
  tags, metadata = {
    aliases:  [...],   ← from enrichment
    entities: [...],   ← from enrichment
    kind:     "...",   ← from enrichment
  },
  reinforcement_count, failure_count,
  created_at, last_reinforced_at
}

edges : (id_low, id_high) → float
─────────────────────────────────────
A single signed scalar per unordered pair of nodes.
Positive → "these go together"; negative → "do not pair";
near zero → no opinion; missing → never co-active yet.

index : LexicalIndex
─────────────────────────────────────
An inverted index over the union of (node.text + aliases) tokens.
Maintained incrementally as nodes are added/edited/removed.
Used only on the BM25 recall path.

experiences : list[Experience]
─────────────────────────────────────
The most recent N episodes: (query, active_ids, outcome, timestamp).
Bounded by experience_log_cap. Audit trail for why a weight moved.
```

The data lives entirely in plain Python objects in memory; `save` and
`load` round-trip everything to a single JSON file.

## LLM call points

A 10-turn session, all defaults (`outcome_strategy="llm"`,
`judge_batch_size=5`, `recall_mode="auto"`, dump fits the threshold,
agent ingests 2 new facts):

| Operation        | When                               | Count |
| ---------------- | ---------------------------------- | ----- |
| Agent generation | Per turn                           | 10    |
| LLM recall       | Per turn while dump fits threshold | 10    |
| Ingest enrichment| Per ingest                         | 2     |
| Batched judging  | Per `judge_batch_size` turns       | 2     |

If the memory dump outgrows `recall_threshold_chars`, the recall column
drops to 0 — BM25 takes over silently. `recall_mode="bm25"` forces this
from the start.

Every LLM call is routed through a single `chat_fn` callable you supply
when constructing `AutoMemory`. You can split that into three separate
endpoints (`chat_fn`, `judge_chat_fn`, `recall_chat_fn`,
`enrich_chat_fn`) so e.g. the agent runs on a frontier model while
judging and recall hit a cheaper one.

## Persistence

`memory.json` is the single source of truth. It contains:

- `config`: the learning-rule knobs (eta, decay, floors, …) so a
  reloaded memory continues with the same dynamics.
- `nodes`: every node with its full metadata.
- `edges`: every pair's weight.
- `experiences`: the recent episode log.
- `episode_counter`, `saved_at`.

Save is atomic from the user's perspective — `Memory.save` writes the
whole file in one `Path.write_text`. There is no append log; every save
is a fresh full snapshot. The file is human-readable and editable: open
it in any editor and the next `Memory.load(path)` picks up changes.

`Memory.export(path, format)` emits a different shape for human review:

- `"markdown"` (default): grouped by tag and `kind`, sorted by salience,
  with strongest associations and recent episodes.
- `"text"`: flat `[weight] text` list.
- `"json"`: equivalent to `save`.

## Extensibility points

The framework is designed so each LLM-touching component can be swapped
or extended without touching the core algorithm:

| Want to…                                         | Override                                                |
| ------------------------------------------------ | ------------------------------------------------------- |
| Use a different chat backend                     | Pass your own `chat_fn`                                 |
| Grade outcomes with a cheaper model              | Pass `judge_chat_fn` / `judge_model`                    |
| Enrich with a cheaper model                      | Pass `enrich_chat_fn` / `enrich_model`                  |
| Run recall against a different model             | Pass `recall_chat_fn` / `recall_model`                  |
| Plug in a non-LLM verifier                       | `outcome_strategy=callable` or `"manual"` + `mem.report` |
| Add a domain-specific enricher                   | Set `mem.memory.ingest_enricher = my_callable`          |
| Disable enrichment                               | `enrich_on_ingest=False`                                |
| Skip the per-turn recall LLM call                | `recall_mode="bm25"`                                    |
| Run on an MCP server                             | `coherence.integrations.mcp_tool_descriptors`           |
| Drive every step by hand                         | Use `Memory` directly                                   |

The orchestrator (`AutoMemory`) is intentionally light — it composes
the small algorithmic modules (`matcher`, `activation`, `reinforce`,
`forget`) and the LLM helpers (`enrichment`, `judging`, `recall_llm`)
into a single per-turn flow. Each module is independently testable and
independently swappable.

## What's not in the framework

- **No model is bundled.** You supply `chat_fn`.
- **No HTTP client is bundled.** `examples/_azure_client.py` is a
  100-line stdlib reference; pick any client you like.
- **No vector store.** Recall is either an LLM call over the in-memory
  dump or a BM25 pass over an inverted index — both live in the same
  Python process, both serialize into the same JSON file.
- **No background processes.** The framework only runs when you call
  it. Maintenance (`mem.forget`, `mem.consolidate`) is explicit, either
  from your code or via the `memory_maintenance` tool the LLM may
  decide to call.
