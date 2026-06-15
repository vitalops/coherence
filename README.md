# coherence

**An embeddingless memory framework for long-running LLM agents.**

[![tests](https://img.shields.io/badge/tests-58_passing-brightgreen)](tests/)
[![python](https://img.shields.io/badge/python-≥3.10-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](pyproject.toml)
[![paper](https://img.shields.io/badge/paper-PDF-blue)](paper.pdf)

Coherence stores ingested text **verbatim** in a single human-readable
JSON file and retrieves it with **LLM-selection recall** — your LLM
reads a textual listing of the candidate chunks and returns the
relevant IDs. There is **no embedding model**, **no vector database**,
and **no LLM call at ingest**. The whole memory state is one file you
can `cat`, `grep`, and version-control.

```python
from coherence import Memory

mem = Memory()
mem.ingest("Alice prefers dark mode.")
mem.ingest("Alice's daughter is named Mira.")
mem.ingest("Bob lives in Lisbon and dislikes anchovies.")

# BM25 retrieval — no LLM call:
mem.recall("what does Alice prefer for the UI?")

# LLM-selection retrieval — pass any OpenAI-style chat function:
mem.recall("what does Alice prefer for the UI?", chat_fn=my_chat)

mem.save("memory.json")             # one file, human-readable JSON
mem = Memory.load("memory.json")    # round-trips bytes-identically
```

That is the full integration. No Postgres, no pgvector, no Chroma,
no embedding model deployment.

---

## Why coherence

| | Mem0 | Letta | LangMem | A-Mem | **Coherence** |
|---|---|---|---|---|---|
| Embedding model | yes | yes | yes | yes | **no** |
| Vector database | yes (Chroma) | yes (pgvector) | yes (in-mem) | yes | **no** |
| LLM call at ingest | yes | no | no | yes | **no** |
| State on disk | DB rows | Postgres | DB rows | DB rows | **one JSON file** |
| Inspect with `cat memory.json` | no | no | no | no | **yes** |

The cost of those infrastructure dependencies is real. On the
LongMemEval chat-memory benchmark in our experiments repository
(n=15, seed=7), Mem0 spends **106.8 s/Q** at ingest time on its LLM
extraction step; Letta spends **64.1 s/Q** bringing up its agent
loop; LangMem spends **8.1 s/Q** on its embedding pass. Coherence
spends **0.04 s/Q**.

---

## Install

```bash
pip install -e .
pytest tests/
```

The framework is pure Python, has zero runtime dependencies, and runs
on Python 3.10+.

---

## Quickstart

The same code as the lead block, with persistence and an LLM judge of
your own choice:

```python
from coherence import Memory

mem = Memory()  # frozen weights, k=10 default, no LLM dependencies

# Ingest some verbatim text.
mem.ingest("Alice prefers dark mode.")
mem.ingest("Alice's daughter is named Mira.")
mem.ingest("Bob lives in Lisbon.")

# BM25 retrieval — fully deterministic, no LLM call.
hits = mem.recall("what UI mode does Alice prefer?", k=5)
for n in hits:
    print(n.id[:8], n.text)

# Or hand it a chat function and let the LLM pick the relevant IDs.
def my_chat(messages, model=None, temperature=0.0):
    # Plug in OpenAI / Azure / Anthropic / Ollama — anything that
    # returns {"choices":[{"message":{"content": "..."}}]}.
    ...
hits = mem.recall("what UI mode does Alice prefer?", chat_fn=my_chat, k=5)

# Persist to disk — one human-readable JSON file.
mem.save("memory.json")

# Reload later:
mem2 = Memory.load("memory.json")
assert {n.id for n in mem.recall("UI mode")} == {n.id for n in mem2.recall("UI mode")}
```

---

## When to choose coherence

**Reach for coherence when:**
- You need **verbatim recall** — the exact wording of a stored fact
  matters at retrieval time (personal-assistant facts, agentic action
  loops, anything where an embedding's approximate match can drop the
  key noun).
- You need **infrastructure simplicity** — no embedding model to
  deploy, no vector database to operate, no LLM call to pay for at
  ingest. `pip install`, write to a JSON file, done.
- You need **auditability** — every memory the agent has is one line
  in a JSON file. You can `cat` it, `diff` it, `grep` it, and put it
  in version control.

**Reach for an embedding-based framework when:**
- You have **a huge corpus of short, similar memories** that need to
  be discriminated at scale (millions of brief notes, dense semantic
  similarity is the right primitive).
- You have **stable infrastructure** already deployed and the extra
  retrieval LLM call is the cost that matters.

---

## What the benchmarks actually show

The full benchmark harnesses and result JSONs ship in a separate
repository (`coherence_paper_experiments`). Two verified findings
from there inform the defaults in this package:

**1. Chat-memory tie at zero ingest cost.** On LongMemEval at n=15
with `gpt-5.4` as the answer LLM and three judges (`gpt-5.4`,
`claude-haiku-4-5`, `claude-sonnet-4-6`, Cohen's κ ≥ 0.88 pairwise),
coherence's LLM-selection recall ties BM25-only at the top of the
panel at **53.3%**, one question above Mem0 (46.7%) and 13–20 pp
above the embedding family (LangMem 40.0%, Letta 33.3%). A larger-n
confirmation (exp01, n=30, single-judge `gpt-5.4`) gives the same tie
at **56.7%** versus Mem0 23.3%.

**2. Retrieval depth helps.** The hyperparameter sweep on the same
protocol (exp03) shows k=3 → k=10 lifts accuracy 46.7% → 56.7% —
**+10 pp**, the cleanest hyperparameter signal in the suite. The
framework's default is k=10.

Both numbers above are read directly from the result JSONs in the
experiments repository. Anything not in this section is not measured
in our suite.

---

## Trainable-graph mode (experimental, opt-in)

Coherence carries an outcome-driven trainable-graph machinery: each
node has a scalar salience weight that an explicit `reinforce()` call
updates by a delta rule, and Hebbian edges form between co-active
nodes. Decay and pruning keep the graph bounded.

This is the framework's **experimental** mode. You opt in by passing
non-zero learning parameters:

```python
mem = Memory(eta=0.3, eta_edge=0.08)  # learning ON
nid = mem.ingest("a fact")
active = mem.recall("a fact")
mem.reinforce("a fact", active, outcome=+1.0)  # weights now move
```

**Honest caveat.** Across the benchmarks in the
`coherence_paper_experiments` repository (multi-pass LongMemEval +
HotpotQA, MMLU-Pro streaming across 3 STEM domains × 3 seeds at n=80
each, TextCraft agentic-RL at n=30, and a hyperparameter sweep that
selected η=0 as the optimum), the trainable updates fire and the
weights move correctly, but we did not establish a statistically
significant accuracy lift over the frozen-weight control at the
sample sizes we ran. Enable the trainable updates if you want to
instrument or extend them; do not rely on them for accuracy without
measuring on your own workload. The unit tests under
`tests/test_smoke.py` (e.g. `test_reinforce_increases_weight_on_success`,
`test_hebbian_edges_form_on_co_activation`) verify the machinery
works as designed when explicitly enabled.

---

## API surface

The package exports four objects:

```python
from coherence import Memory, AutoMemory, Node, Experience
```

`Memory` is the primary type. Day-to-day API:

- `mem.ingest(text, *, tags=None, metadata=None) -> str` — store verbatim text, return the node id
- `mem.recall(query, k=10, *, chat_fn=None, model=None) -> list[Node]` — retrieve top-k (LLM-selection if `chat_fn` given, BM25 otherwise)
- `mem.save(path) -> None` — write JSON
- `Memory.load(path) -> Memory` — read JSON
- `mem.reinforce(query, active, outcome)` — *opt-in*, no-op at default `eta=0`
- `mem.forget()` — run decay + prune

`AutoMemory` is a higher-level wrapper that drives a full
chat / answer / judge loop for you; see the docstring in
`coherence/autopilot.py`.

---

## License

MIT.
