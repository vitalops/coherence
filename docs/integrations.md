# Integrations

The framework exposes four operations — `memory_recall`,
`memory_ingest`, `memory_reinforce`, `memory_maintenance` — that can be
plugged into any agent loop in the wire format that loop expects.

`AutoMemory` is the easiest path. It bakes in the LLM-driven enrichment
at ingest time and batched outcome judging that make recall semantic and
keep cost bounded. The lower-level adapters below are for when you
can't use `AutoMemory` because your harness is fixed (Claude Code, an
SDK loop you already have, an MCP server you maintain).

## OpenAI-protocol chat backends

Works with the OpenAI first-party API, Azure OpenAI v1, OpenRouter,
vLLM's compatible server, llama.cpp's server, and other backends that
speak `POST /chat/completions`.

```python
from coherence import Memory
from coherence.integrations import openai_tool_specs, run_openai_tool_calls

mem = Memory.load("memory.json")
tools = openai_tool_specs(mem)

response = chat(messages=messages, tools=tools, tool_choice="auto")
message = response["choices"][0]["message"]

tool_results = run_openai_tool_calls(mem, message.get("tool_calls", []))
# Each result is a dict with tool_call_id / name / content (JSON string).
# Append each as a {"role": "tool", ...} message and continue the loop.
```

`AutoMemory` runs this loop internally. Use the adapter directly when
you need to interleave the memory tools with other tools you already
have, or when you want manual control over the conversation history.

To get LLM-driven enrichment at ingest in a hand-rolled OpenAI loop,
wire it yourself:

```python
from coherence.enrichment import enrich_memory

def my_chat(messages, **kw):
    ...  # your chat function

mem.ingest_enricher = lambda text: enrich_memory(text, chat_fn=my_chat)
```

Anything ingested through `mem.ingest(...)` (or the `memory_ingest`
tool) then gets aliases and entities folded into the index.

## Anthropic Messages API

```python
from coherence.integrations import (
    anthropic_tool_specs,
    run_anthropic_tool_calls,
)

tools = anthropic_tool_specs(mem)
# Send the Messages API request with tools=tools.
tool_results = run_anthropic_tool_calls(mem, response_content_blocks)
# tool_results goes into the next user message's content array.
```

The shape difference from OpenAI (top-level `name`/`input_schema`
instead of nested `function.parameters`, `tool_use` blocks instead of
`tool_calls`) is handled inside the adapter. The underlying memory
dispatch is identical.

## MCP server (Claude Code, Cursor, any MCP-aware client)

```python
from coherence import Memory
from coherence.integrations import mcp_tool_descriptors

mem = Memory.load("memory.json")
for d in mcp_tool_descriptors(mem):
    register_with_my_mcp_server(
        name=d["name"],
        description=d["description"],
        input_schema=d["input_schema"],
        handler=d["handler"],
    )
```

`d["handler"]` is a plain Python callable taking a dict of arguments
and returning a dict. Register the four descriptors and any MCP client
sees the memory operations as tools.

If your MCP server has access to a chat function for grading, attach a
batched judge:

```python
from coherence.judging import batched_judge

# Periodically (e.g. every 5 turns, or at session end):
scores = batched_judge(pending_turns, chat_fn=judge_chat)
for turn, score in zip(pending_turns, scores):
    if score != 0.0:
        mem.reinforce(turn["user"], turn["active_ids"], score)
```

## Custom agentic harness

When you're writing the agent loop yourself and want to call memory
operations by name with a dict:

```python
from coherence import Memory
from coherence.integrations import dispatch

mem = Memory.load("memory.json")

result = dispatch(mem, "memory_recall", {"query": "...", "k": 5})
# result == {"memories": [{"id": ..., "text": ..., "weight": ..., "tags": ...}, ...]}

dispatch(mem, "memory_reinforce", {
    "query": "...",
    "node_ids": [m["id"] for m in result["memories"]],
    "outcome": +1.0,
})
```

`dispatch` accepts either a dict or a JSON string for the arguments —
the latter is what most SDKs hand you in their tool-call payloads.

## Session context manager

A small helper for hand-rolled loops where you want the "recall before,
report after" pattern to be impossible to forget:

```python
from coherence import Memory
from coherence.integrations import MemorySession

mem = Memory.load("memory.json")
session = MemorySession(mem)

with session.episode("user question") as ep:
    answer = my_agent_act(ep.nodes)
    if checker.is_correct(answer):
        ep.success()
    else:
        ep.failure()
# Reinforcement is applied on a normal exit. If the block raises or
# never calls success() / failure(), no update is applied.
```

## Picking a layer

| Setup                                                    | Use                                              |
| -------------------------------------------------------- | ------------------------------------------------ |
| One chat backend, one user-facing loop                   | `AutoMemory`                                     |
| Claude Code or Cursor via MCP                            | `mcp_tool_descriptors`                           |
| Existing OpenAI/Anthropic SDK loop you don't want to wrap | `openai_tool_specs` / `anthropic_tool_specs`     |
| Fully custom Python loop                                 | `dispatch` or `MemorySession`                    |
| Driving every step by hand for research                  | `Memory` directly                                |

All paths share the same `Memory` object underneath, so you can mix
levels — e.g. let `AutoMemory` handle the main chat loop while a
separate script dispatches `memory_maintenance` on a cron.

## Exporting

Every layer can call `mem.export(path, format="markdown")` to write a
human-readable snapshot. See [Concepts](concepts.md#exporting) for
formats and what the markdown dump contains.
