# Outcome strategies

A memory needs to know whether the turn it was used in was successful.
The framework refuses to *guess* this from how polite or grouchy the
user's next message sounds — that's noise, not signal.

There are three honest sources of outcome signal, and you pick one (or
combine them) via `outcome_strategy=` on `AutoMemory`.

## `"manual"` (default) — you supply the signal

If you have a verifier — a test runner, a ground-truth checker, a
schema validator, an explicit thumbs-up/down — use it directly. The
framework stays out of the way and applies the score you provide:

```python
mem = AutoMemory(chat_fn=my_chat, outcome_strategy="manual")
reply = mem.complete(user_question)
if checker.is_correct(reply):
    mem.report(+1.0)
else:
    mem.report(-1.0)
```

No LLM calls, no inference, no buffering — `report()` runs the
reinforcement on the most recent turn immediately. This is the
recommended path whenever you have *any* signal at all, even noisy
ones (engagement metrics, conversion events, follow-up clicks).

## `"self_assess"` — the agent grades its own action

When you have no external verifier but you trust the model to judge
its own action, this strategy batches turns and asks the LLM:

> Given the user's message and the memories the assistant retrieved,
> did the assistant's reply correctly use those memories and address
> the user's stated intent?

The judge *never sees the next user message.* It judges the
assistant's action on its own terms, not the user's reaction to it.
This avoids treating "thanks!" as evidence of success and silence as
evidence of failure.

```python
mem = AutoMemory(
    chat_fn=production_chat,
    outcome_strategy="self_assess",
    judge_chat_fn=cheap_chat,        # optional; falls back to chat_fn
    judge_model="...",                # optional model override
    judge_batch_size=5,               # one LLM call per N turns
)
```

Buffered turns flush at the batch threshold, on `mem.reset_history()`,
and on `mem.close()`. Cost: one LLM call per `judge_batch_size` turns,
spent grading. Bump the batch size higher to amortize further.

## `"follow_up"` — opt-in regex for explicit corrections

A narrow escape hatch for chat applications where users sometimes *do*
write unambiguous corrections ("no, that's wrong", "that's not what I
meant", "I told you to stay off moss"). When the next user message
matches one of those patterns, this strategy emits **-1.0** so the
memories used in the bad turn get demoted. Anything else — silence,
politeness, smooth follow-ups — emits **0.0**.

Politeness is not evidence of success. Silence is not signal. Only
unambiguous corrections fire.

```python
mem = AutoMemory(chat_fn=my_chat, outcome_strategy="follow_up")
```

This is a regex pass, not an LLM call — costs nothing. It also won't
catch subtle dissatisfaction. Pair with `mem.report()` calls for
positive signal.

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
    return 0.0

mem = AutoMemory(chat_fn=my_chat, outcome_strategy=my_strategy)
```

Per-turn (not batched). Use when you have a signal that's faster than
an LLM call.

## The always-on intrinsic dynamic

Independent of outcome strategy, every recall gives a small positive
bump to the nodes that were retrieved — `intrinsic_retrieval_bump`
(default `0.02`). The idea: a memory that earned a place in the top-k
deserves a tiny nudge, regardless of whether the surrounding turn
ended with a measurable outcome. Combined with decay during
`mem.forget()`, this means:

- Memories that get retrieved often → slowly accumulate salience.
- Memories that nothing matches → drift toward zero, then prune.

Set `intrinsic_retrieval_bump=0` to disable. It runs in every
strategy, including `"manual"`.

## Goal completion — retroactive reinforcement

A separate, cleaner signal for long-horizon work: tag a goal-kind node
when the goal is set, and call `mem.complete_goal(goal_id, outcome)`
when it's achieved (or definitively missed). The framework walks the
experience log, finds every episode whose active set included that
goal, and retroactively reinforces those active sets with geometric
recency weighting. Each contributing memory gets credit for its part
in the goal.

```python
# When the user states a goal, ingest it as a goal-kind node:
goal_id = mem.remember(
    "Finish the cold-photosynthesis review paper by Q4.",
    metadata={"kind": "goal"},
)

# ... many sessions of research over weeks ...
# When the goal is achieved:
mem.memory.complete_goal(goal_id, outcome=+1.0)
# → walks back through episodes, reinforces every fact and finding
#   that was active during a session where the goal was in scope.
```

`mem.memory.goals()` lists all goal-kind nodes.

## Cost ceiling

A 10-turn session with two ingests, defaults across the board
(`"manual"` outcome, intrinsic bump on, two recalls land within the
LLM-recall threshold):

| Operation | Calls | Notes |
| --- | --- | --- |
| Agent generation | 10 | You make these anyway. |
| LLM-driven recall | 10 | One per turn while dump fits. |
| Ingest enrichment | 2 | Per ingest. |
| Outcome judging | 0 | Manual mode — only when `mem.report()` fires. |

Switch to `"self_assess"` with `judge_batch_size=5` and you add ~2
batched judging calls per 10 turns. Switch to `"follow_up"` and you
add 0. Set `recall_mode="bm25"` to drop the per-turn recall call too.

Cheap, honest, and the framework never invents signal it doesn't
have.
