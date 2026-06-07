# Getting started

## Install

```bash
pip install -e .
```

The package has no runtime dependencies. For tests:

```bash
pip install -e .[dev]
pytest tests/
```

## Three lines to plug it in

```python
from coherence import AutoMemory

mem = AutoMemory(chat_fn=my_chat, path="memory.json", system_prompt="You are ...")
reply = mem.complete(user_message)
```

`complete` handles retrieval, tool dispatch, batched outcome judging,
ingest enrichment, and persistence under the hood.

`chat_fn` is any callable that speaks the OpenAI chat-completions JSON
shape:

```python
chat_fn(messages, tools=None, tool_choice=None, **kw) -> response_dict
```

The response is expected to have `response["choices"][0]["message"]`
with optional `"content"` and `"tool_calls"`. An SDK wrapper or a
stdlib `urllib.request` call both qualify. See
[`examples/_azure_client.py`](../examples/_azure_client.py) for a
~100-line stdlib version you can copy.

## What happens on turn 1

User opens with:

> "I'm Maren. I work on cold-resistant photosynthesis in extremophile
> algae — Chlamydomonas nivalis and Chloromonas brevispina. Stay off
> moss literature."

`mem.complete(...)` runs:

1. **Recall.** The framework retrieves the top-k matching memories from
   `memory.json`. On the first turn there are none, so nothing is
   injected.
2. **Generate.** Your `chat_fn` receives the user message plus the four
   memory tools. The model can call `memory_ingest({"text": "..."})`
   to record durable facts.
3. **Dispatch.** Any tool calls execute against the in-memory graph.
   Every ingest triggers a one-shot LLM enrichment call that pulls out
   aliases and entities and tags the kind — so future recalls match on
   semantic synonyms, not only literal overlap.
4. **Buffer.** This turn is staged for later outcome judging.
5. **Save.** `memory.json` is rewritten.

After the turn, the file might contain:

```json
{
  "nodes": [
    {
      "id": "a1f3...",
      "text": "Maren works on cold-resistant photosynthesis in extremophile algae, focused on Chlamydomonas nivalis and Chloromonas brevispina.",
      "weight": 0.10,
      "metadata": {
        "aliases": ["Maren's research focus", "what I study", "my work"],
        "entities": ["Chlamydomonas nivalis", "Chloromonas brevispina", "Maren"],
        "kind": "fact"
      }
    }
  ],
  "edges": [],
  "experiences": []
}
```

No reinforcement has been applied yet — the framework buffers the
outcome and grades it later (see step 3 of "what happens on turn 5"
below).

## What happens on turn 5

`AutoMemory` defaults to `judge_batch_size=5`, so:

- Turns 1-4 stay buffered. No grading LLM call yet.
- Turn 5 fills the buffer. **One** LLM call grades all 5 turns at once
  and returns a score per turn. The framework applies the
  reinforcements in a single pass.

Result after a good 5-turn session: the highest-salience node may now
read `weight = 0.85`. An edge may have formed between two memories
that succeeded together. The next session, even on a slightly different
query, this pair will surface together.

If the user *had* corrected one turn ("no, that's wrong"), the grader
picks that up in the batch and demotes the memories used in that bad
answer.

See [How learning works](learning.md) for the full update rule and
[Outcome strategies](outcomes.md) for the other ways to grade turns.

## Cost ceiling

A 10-turn session with two ingests and `judge_batch_size=5`:

| Call                    | Count           |
| ----------------------- | --------------- |
| Agent generation        | 10              |
| Ingest enrichment       | 2               |
| Batched outcome grading | 2               |
| Memory recall           | 0 (BM25, local) |

The memory layer adds ~4 extra LLM calls on top of the 10 the agent
makes anyway. Recall is always free.

## Context-window awareness

Tell `AutoMemory` how big the model's context is and the framework
tunes itself:

```python
# 8k-context model
mem = AutoMemory(chat_fn=short_model, context_length=8_000)

# 200k-context model
mem = AutoMemory(chat_fn=long_model,  context_length=200_000)
```

That single parameter controls:

- **Recall mode threshold.** When the memory dump fits in roughly 1/10
  of the context, the framework uses LLM-driven recall (the LLM reads
  the dump and picks relevant chunks). Past that, BM25 word matching
  takes over.
- **Injection budget.** How much memory text gets dropped into the
  prompt per turn, capped at ~1/10 of the context.

Both can be overridden explicitly via `recall_threshold_chars=` and
`context_budget_tokens=` if you want finer control.

## Exporting

For a human-readable snapshot of what the agent has learned:

```python
mem.export("snapshot.md")
```

That writes a markdown file with nodes grouped by tag and kind, sorted
by salience, with the strongest associations and recent episodes. Useful
for sharing or sanity-checking without exposing the raw JSON.

## Next steps

- [Concepts](concepts.md) — the data model: nodes, edges, salience,
  episodes, enrichment metadata, the JSON file structure.
- [Outcome strategies](outcomes.md) — `llm` (batched) is the default,
  but you can also use `manual` (verifier) or a custom callable.
- [How learning works](learning.md) — the math of the per-turn update.
- [Integrations](integrations.md) — wire into Anthropic, MCP, or a
  custom harness without `AutoMemory`.
