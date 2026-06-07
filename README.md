# coherence

A memory layer for language-model agents.

Every memory is a text fact carrying a single number — its **salience**
— that grows as the memory keeps helping and shrinks when it doesn't.
Pairs of memories that help together form a connection. Unused memories
quietly fade and are eventually forgotten.

A memory is a self-contained chunk — a paragraph or small section,
not just a sentence. At ingest time, the framework asks the LLM once
to extract aliases, entities, and a kind tag so the memory carries
its own searchable surface. At recall time, the LLM reads the memory
dump and picks the relevant chunks; once the dump grows past a
context-sized threshold, BM25 word search takes over so recall stays
affordable. The framework never *guesses* outcomes from the user's
next message — it reinforces when you tell it to (`mem.report()`),
when an opt-in LLM judge grades the assistant's own action
(`"self_assess"`), or retroactively when a goal is achieved
(`mem.complete_goal()`). The whole memory is one human-readable JSON
file you can also export to markdown.

## Use it

```python
from coherence import AutoMemory

mem = AutoMemory(chat_fn=my_chat, path="memory.json", system_prompt="...")
reply = mem.complete(user_message)
```

That is the full integration. Per turn, `AutoMemory`:

1. retrieves the relevant memories — LLM-driven when the dump fits,
   BM25 fallback when it doesn't — and packs them into a token budget
   so the framework adapts to short- and long-context models,
2. routes the four memory tools (`memory_recall`, `memory_ingest`,
   `memory_reinforce`, `memory_maintenance`) the model may call,
3. enriches any new ingest with one LLM call (aliases, entities, kind)
   so future recall catches semantic matches,
4. gives every retrieved memory a small intrinsic bump (memories that
   keep getting surfaced quietly build salience),
5. saves the updated graph to disk.

Pass `context_length=N` (the model's total context window in tokens)
and the framework auto-tunes recall threshold and injection budget.
Set `recall_mode="bm25"` to drop the per-turn LLM recall call entirely.

**Outcomes are explicit, not inferred.** When you have a signal —
verifier passed, user clicked thumbs-up, downstream task succeeded —
call `mem.report(+1)` or `mem.report(-1)`. Or set
`outcome_strategy="self_assess"` to have the LLM grade the agent's
own action in batches (not the user's reaction). Or set
`outcome_strategy="follow_up"` to demote memories only when the next
user message contains an unambiguous correction ("no, that's wrong").
The framework will not invent signal it doesn't have.

If you have a verifier (a test runner, a ground-truth answer, an
external classifier), switch `outcome_strategy="manual"` and call
`mem.report(+1)` / `mem.report(-1)` yourself. Or pass any
`(prev_user, prev_reply, next_user) -> float` callable as a custom
strategy.

## What you get

| Thing                        | What it is                                         |
| ---------------------------- | -------------------------------------------------- |
| `coherence.AutoMemory`       | One-call wrapper; recommended for most users       |
| `coherence.Memory`           | Raw memory graph: `ingest`, `recall`, `reinforce`, `export`, … |
| `coherence.integrations.*`   | Adapters: OpenAI-protocol, Anthropic, MCP, custom  |
| `docs/`                      | Concept walkthroughs and how-it-works guides       |
| `examples/`                  | Three runnable demos (one offline, two with Azure) |

## Examples

- [`examples/01_simulation_learning_curve.py`](examples/01_simulation_learning_curve.py)
  — no network. Trained top-1 hit rate **100%** vs no-learning baseline
  **50%**. Node count compresses from 52 to 39 as unused entries fade.
- [`examples/02_personal_research_assistant.py`](examples/02_personal_research_assistant.py)
  — multi-session chat via the Azure endpoint in `examples/.env`. The
  user just talks; batched judging at session boundaries demotes
  memories that misled.
- [`examples/03_custom_agentic_harness.py`](examples/03_custom_agentic_harness.py)
  — synthetic "Project Lyra" domain, 15-question multiple-choice
  bench. Uses `outcome_strategy="manual"` with a ground-truth
  verifier.

## Documentation

- [Getting started](docs/getting-started.md) — install, the three-line
  API, and a walkthrough of what gets stored on the first few turns.
- [Architecture](docs/architecture.md) — visual tour of a memory dump,
  every algorithm with concrete numbers, the layers, per-turn flow,
  data model, LLM call points, persistence, extensibility.
- [Concepts](docs/concepts.md) — nodes, edges, salience, episodes,
  enrichment metadata, and the JSON file shape on disk.
- [How learning works](docs/learning.md) — recall modes, the update
  that runs after every turn, decay and pruning.
- [Outcome strategies](docs/outcomes.md) — the default batched LLM
  judge, manual reporting, custom callables, and cost.
- [Integrations](docs/integrations.md) — OpenAI-protocol, Anthropic
  Messages, MCP servers, custom harnesses.

## Install & run

```bash
pip install -e .
pytest tests/
python examples/01_simulation_learning_curve.py
```

## License

MIT.
