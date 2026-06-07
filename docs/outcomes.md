# Outcome strategies

After every turn, the framework needs one number: did the prior turn go
well? That number — the **outcome** — is what drives every weight
update.

You pick how it's computed via `outcome_strategy=` on `AutoMemory`.
Three built-in options plus a custom callable.

## `"llm"` (default) — batched LLM judging

Instead of calling the LLM after every single turn (expensive at scale),
`AutoMemory` buffers the turns and grades them all in a single LLM call
once the buffer reaches `judge_batch_size` (default `5`). A 10-turn
session ends up costing ~2 batched grading calls instead of 10
per-turn calls.

```python
mem = AutoMemory(
    chat_fn=production_chat,
    outcome_strategy="llm",         # the default
    judge_chat_fn=cheap_chat,        # optional; falls back to chat_fn
    judge_model="...",               # optional model override for grading
    judge_batch_size=5,              # flush after N buffered turns
)
```

The grader sees each buffered turn — user message, assistant reply,
next user message — and returns a score in `[-1, +1]` per turn.
Reinforcement is applied in one pass. Buffered turns also flush
automatically on `mem.reset_history()` and `mem.close()` so signal is
never silently dropped at a session boundary.

Why batch? Two reasons:

1. **Cost.** Grading 5 turns in one prompt is cheaper than 5 separate
   calls — fewer round-trips, less prompt overhead.
2. **Context.** The grader can see the whole arc, not just one turn at
   a time, which makes its judgment more reliable.

If you want grading to happen against a cheaper model than the main
agent, pass `judge_chat_fn=` and `judge_model=`. Common pattern: the
agent runs on a frontier model; grading runs on the cheapest available
model from the same provider.

## `"manual"`

The framework never auto-infers. You call `mem.report(outcome)`
explicitly between turns. Use this when you have a verifier — a test
runner, a ground-truth answer, a downstream check — that produces a
clean signal.

```python
mem = AutoMemory(chat_fn=my_chat, outcome_strategy="manual")
reply = mem.complete(user_question)
if checker.is_correct(reply):
    mem.report(+1.0)
else:
    mem.report(-1.0)
```

[`examples/03_custom_agentic_harness.py`](../examples/03_custom_agentic_harness.py)
uses this against a multiple-choice bench.

In manual mode the framework does **not** make any judge LLM calls. The
buffer still tracks turns, but nothing happens to them unless you call
`report()`.

## Custom callable

Anything with the signature
`(prev_user, prev_reply, next_user) -> float in [-1, +1]`:

```python
def my_strategy(prev_user, prev_reply, next_user):
    if "/regenerate" in next_user.lower():
        return -0.8
    if user_clicked_thumbs_up():
        return +1.0
    if external_classifier(next_user) > 0.7:
        return +0.5
    return 0.0  # no update

mem = AutoMemory(chat_fn=my_chat, outcome_strategy=my_strategy)
```

Custom callables are applied **per turn** (not batched) because the
caller controls cost. Use this when you have a signal that's faster
than an LLM call — a UI event, a sentiment classifier, a domain-specific
rule.

## Graded outcomes

Outcomes are floats in `[-1, +1]`. You're not limited to `+1 / -1`.

- `+0.5` — partially-correct answer
- `-0.2` — answer the user grudgingly accepted but flagged
- `+0.1` — leaning positive, but uncertain

The delta-rule update is linear in `outcome`, so `+0.5` moves the
weights half as far as `+1.0`. Use graded values when your signal is
graded.

## When the prior turn retrieved nothing

If the active set was empty (the graph was empty or no node matched),
there is nothing to reinforce. The framework silently no-ops. You can
still call `mem.report` after such a turn — it returns `False` to
indicate nothing was applied.

## Stacking strategies

You can wrap one strategy inside another via a custom callable. For
example, prefer a verifier when it gives a signal, fall back to no-op
otherwise:

```python
def combined(prev_user, prev_reply, next_user):
    v = external_verifier(prev_reply)
    if v is not None:
        return v
    return 0.0

mem = AutoMemory(chat_fn=my_chat, outcome_strategy=combined)
```

## Cost ceiling

A 10-turn session, defaults across the board (`outcome_strategy="llm"`,
`judge_batch_size=5`, `enrich_on_ingest=True`, `recall_mode="auto"`),
agent ingests 2 new facts, memory dump stays within the LLM-recall
threshold:

| Call                        | Count                |
| --------------------------- | -------------------- |
| Agent generation            | 10 (one/turn)        |
| Ingest enrichment           | 2 (one/ingest)       |
| Batched outcome grading     | 2 (one/batch)        |
| LLM-driven recall           | 10 (one/turn)        |

Total memory overhead: ~14 calls per 10-turn session on top of the 10
the agent makes anyway.

If you need cheaper-per-turn, set `recall_mode="bm25"` to drop the
per-turn recall LLM call entirely (the framework then uses BM25 over
the inverted index, which is free). Or bump `judge_batch_size` higher
to space out the grading.

Once the memory dump grows past `recall_threshold_chars`, `auto` mode
swaps recall to BM25 automatically — the dump is too big to send into
context affordably. That's a feature, not a regression: the memory
keeps growing, recall keeps working, only the recall method changes.
