from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .graph import Memory
from .integrations.openai_protocol import (
    openai_tool_specs,
    run_openai_tool_calls,
)
from .enrichment import enrich_memory
from .judging import batched_self_assess, follow_up_outcome
from .recall_llm import llm_recall as _llm_recall


OutcomeStrategy = Callable[[str, str, str], float]


class AutoMemory:
    def __init__(
        self,
        chat_fn: Callable[..., dict],
        *,
        memory: Memory | None = None,
        path: str | Path | None = None,
        system_prompt: str | None = None,

        # Outcome inference. The default is "manual" — the framework does
        # NOT guess outcomes from user follow-up. Either:
        #   - call mem.report(outcome) when you have signal (a verifier, a
        #     thumbs-up, a graded result), or
        #   - opt into "self_assess" so the LLM grades the assistant's own
        #     action (still batched, still bounded cost), or
        #   - opt into "follow_up" so a regex catches UNAMBIGUOUS
        #     corrections ("not what I meant", "that's wrong") but does not
        #     pretend silence or politeness is signal.
        outcome_strategy: str | OutcomeStrategy = "manual",
        judge_chat_fn: Callable[..., dict] | None = None,
        judge_model: str | None = None,
        judge_batch_size: int = 5,

        # Intrinsic background dynamic: a memory that earns a place in the
        # active set gets a small positive bump just for being retrieved.
        # This is independent of outcome inference and runs in every mode.
        # Set to 0 to disable.
        intrinsic_retrieval_bump: float = 0.02,

        # Ingest enrichment: one LLM call per new memory by default. This is
        # what lets BM25 catch semantic matches at recall time without a
        # per-recall LLM call.
        enrich_on_ingest: bool = True,
        enrich_chat_fn: Callable[..., dict] | None = None,
        enrich_model: str | None = None,

        # Recall: LLM-driven by default; BM25 takes over once the memory
        # dump grows past the threshold.
        recall_mode: str = "auto",          # "auto" | "llm" | "bm25"
        recall_threshold_chars: int | None = None,
        recall_chat_fn: Callable[..., dict] | None = None,
        recall_model: str | None = None,

        # Context-window awareness. ``context_length`` (total tokens for the
        # underlying model) lets the framework auto-tune the recall fallback
        # threshold and the per-turn injection budget. ``context_budget_tokens``
        # caps how much memory text gets injected per turn.
        context_length: int | None = None,
        recall_k: int = 5,
        context_budget_tokens: int | None = None,

        expose_tools: bool = True,
        save_every_turn: bool = True,
        max_tool_hops: int = 8,
    ) -> None:
        if chat_fn is None:
            raise ValueError("AutoMemory(chat_fn=...) is required")

        self.chat_fn = chat_fn
        self.path: Path | None = Path(path) if path else None

        self.judge_chat_fn = judge_chat_fn or chat_fn
        self.judge_model = judge_model
        self.enrich_chat_fn = enrich_chat_fn or chat_fn
        self.enrich_model = enrich_model
        self.recall_chat_fn = recall_chat_fn or chat_fn
        self.recall_model = recall_model

        # Derive sensible defaults from context_length when provided. The
        # ratios are deliberately conservative: leave plenty of room for the
        # system prompt, the chat history, and the model's own reasoning.
        self.context_length = context_length
        if context_length:
            # Reserve ~1/10 of the context for the recall payload (in chars,
            # assuming ~4 chars per token). Anything beyond that pushes us into
            # BM25 fallback to keep the request affordable.
            self.recall_threshold_chars = recall_threshold_chars or max(2000, (context_length // 10) * 4)
            self.context_budget_tokens = context_budget_tokens or max(500, context_length // 10)
        else:
            self.recall_threshold_chars = recall_threshold_chars or 16000
            self.context_budget_tokens = context_budget_tokens or 2000

        if recall_mode not in ("auto", "llm", "bm25"):
            raise ValueError(f"recall_mode must be 'auto', 'llm', or 'bm25'; got {recall_mode!r}")
        self.recall_mode = recall_mode

        if memory is not None:
            self.memory = memory
        elif self.path and self.path.exists():
            self.memory = Memory.load(self.path)
        else:
            self.memory = Memory()

        if enrich_on_ingest:
            self.memory.ingest_enricher = self._enrich

        self.system_prompt = system_prompt
        self.recall_k = recall_k
        self.expose_tools = expose_tools
        self.save_every_turn = save_every_turn
        self.max_tool_hops = max_tool_hops
        self.judge_batch_size = max(1, int(judge_batch_size))
        self.intrinsic_retrieval_bump = float(intrinsic_retrieval_bump)

        if callable(outcome_strategy):
            self._strategy_name = "custom"
            self._custom_outcome: OutcomeStrategy | None = outcome_strategy
        elif outcome_strategy == "self_assess":
            self._strategy_name = "self_assess"
            self._custom_outcome = None
        elif outcome_strategy == "follow_up":
            self._strategy_name = "follow_up"
            self._custom_outcome = follow_up_outcome
        elif outcome_strategy == "manual":
            self._strategy_name = "manual"
            self._custom_outcome = None
        else:
            raise ValueError(
                f"Unknown outcome_strategy: {outcome_strategy!r}. "
                "Use 'manual', 'self_assess', 'follow_up', or pass a callable."
            )

        self._history: list[dict[str, Any]] = []
        if self.system_prompt:
            self._history.append({"role": "system", "content": self.system_prompt})

        self._buffer: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None

    # ----------------------------------------------------- enrichment

    def _enrich(self, text: str) -> dict[str, Any]:
        return enrich_memory(
            text,
            chat_fn=self.enrich_chat_fn,
            model=self.enrich_model,
        )

    # ----------------------------------------------------- core turn

    def complete(self, user_message: str) -> str:
        # 1. The prior turn now has its follow-up — close it out.
        if self._pending is not None:
            self._pending["next_user"] = user_message
            self._buffer.append(self._pending)
            self._pending = None
            # Flush whenever we have enough turns for batched strategies, or
            # immediately for per-turn ones (custom callable, follow_up).
            if self._strategy_name == "self_assess" and len(self._buffer) >= self.judge_batch_size:
                self.flush_outcomes()
            elif self._strategy_name in ("custom", "follow_up"):
                self._apply_per_turn_outcomes()

        # 2. Recall (LLM-driven by default; BM25 once the dump is too big to
        # send in-context affordably) + context-budget packing.
        nodes = self._do_recall(user_message)
        nodes = self._pack_to_budget(nodes)
        active_ids = [n.id for n in nodes]
        active_texts = [n.text for n in nodes]

        # Intrinsic background dynamic: nodes that earned a place in the
        # active set get a small positive bump just for being retrieved.
        # This is the always-on safety net regardless of outcome strategy.
        if self.intrinsic_retrieval_bump > 0.0:
            for n in nodes:
                n.weight += self.intrinsic_retrieval_bump

        turn_messages = list(self._history)
        if nodes:
            turn_messages.append({
                "role": "system",
                "content": _format_memory_context(nodes, goals=self.memory.goals()),
            })
        turn_messages.append({"role": "user", "content": user_message})

        # 3. Tool-call loop against the chat backend.
        final_text, _all_messages, llm_active = self._run_tool_loop(turn_messages)

        # 4. Update canonical history (skip the per-turn injection).
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": final_text})

        # 5. Stage this turn for next-round outcome attribution.
        all_active = list(dict.fromkeys(active_ids + llm_active))
        # Capture the actual text of the retrieved memories too, so the
        # self-assess judge can reason about whether the reply used them.
        all_texts = list(active_texts)
        for nid in llm_active:
            if nid in self.memory.nodes:
                all_texts.append(self.memory.nodes[nid].text)
        self._pending = {
            "user": user_message,
            "reply": final_text,
            "active_ids": all_active,
            "active_texts": all_texts,
            "next_user": None,
        }

        if self.save_every_turn and self.path:
            self.memory.save(self.path)

        return final_text

    # ----------------------------------------------------- outcomes

    def flush_outcomes(self) -> int:
        """Grade all buffered turns and apply reinforcement.

        Behaviour by strategy:
          - "self_assess": ONE LLM call grades the whole buffer.
          - "follow_up" / custom callable: per-turn evaluation, no batching.
          - "manual": no-op; call mem.report(outcome) explicitly.
        """
        buffer = self._buffer
        self._buffer = []
        if not buffer:
            return 0

        if self._strategy_name == "self_assess":
            scores = batched_self_assess(
                buffer,
                chat_fn=self.judge_chat_fn,
                model=self.judge_model,
            )
        elif self._custom_outcome is not None:
            # "custom" or "follow_up" — both use a per-turn callable.
            scores = [
                float(self._custom_outcome(t["user"], t["reply"], t.get("next_user") or ""))
                for t in buffer
            ]
        else:
            return 0  # manual mode: report() must be called explicitly

        applied = 0
        for turn, score in zip(buffer, scores):
            if score == 0.0 or not turn.get("active_ids"):
                continue
            self.memory.reinforce(
                turn["user"],
                turn["active_ids"],
                score,
                metadata={"inferred_by": self._strategy_name},
            )
            applied += 1

        if self.save_every_turn and self.path:
            self.memory.save(self.path)
        return applied

    def _apply_per_turn_outcomes(self) -> None:
        self.flush_outcomes()

    def report(self, outcome: float) -> bool:
        """Apply an explicit outcome to the most recent turn (the staged one,
        or the most recent buffered one if the stage has already moved on).

        Returns True if reinforcement was applied, False otherwise.
        """
        turn = None
        if self._pending is not None and self._pending.get("active_ids"):
            turn = self._pending
            self._pending = None
        elif self._buffer:
            turn = self._buffer.pop()
        if not turn or not turn.get("active_ids"):
            return False
        self.memory.reinforce(
            turn["user"],
            turn["active_ids"],
            float(outcome),
            metadata={"inferred_by": "manual"},
        )
        if self.save_every_turn and self.path:
            self.memory.save(self.path)
        return True

    # ----------------------------------------------------- conveniences

    def remember(self, text: str, **kw: Any) -> str:
        return self.memory.ingest(text, **kw)

    def recall(self, query: str, k: int | None = None):
        return self.memory.recall(query, k=k or self.recall_k)

    def reset_history(self, *, keep_system: bool = True) -> None:
        # Close out any in-flight or buffered turns before clearing chat
        # history so signal is not silently dropped at a session boundary.
        if self._pending is not None:
            self._buffer.append(self._pending)
            self._pending = None
        if self._buffer and self._strategy_name in ("self_assess", "custom", "follow_up"):
            self.flush_outcomes()

        if keep_system and self.system_prompt:
            self._history = [{"role": "system", "content": self.system_prompt}]
        else:
            self._history = []

    def close(self) -> None:
        if self._pending is not None:
            self._buffer.append(self._pending)
            self._pending = None
        if self._strategy_name in ("self_assess", "custom", "follow_up"):
            self.flush_outcomes()
        else:
            self._buffer = []
        self.save()

    def save(self) -> None:
        if self.path:
            self.memory.save(self.path)

    def export(self, path: str | Path, format: str = "markdown") -> None:
        self.memory.export(path, format=format)

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()

    @property
    def pending_count(self) -> int:
        n = len(self._buffer)
        if self._pending is not None:
            n += 1
        return n

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def pending(self) -> dict[str, Any] | None:
        return self._pending

    # ----------------------------------------------------- internals

    def _memory_dump_chars(self) -> int:
        return sum(len(n.text) + 24 for n in self.memory.nodes.values())

    def _do_recall(self, query: str):
        if self.recall_mode == "bm25" or not self.memory.nodes:
            return self.memory.recall(query, k=self.recall_k)

        if self.recall_mode == "auto":
            dump_size = self._memory_dump_chars()
            if dump_size >= self.recall_threshold_chars:
                return self.memory.recall(query, k=self.recall_k)

        nodes = _llm_recall(
            query,
            self.memory,
            chat_fn=self.recall_chat_fn,
            k=self.recall_k,
            model=self.recall_model,
            max_memory_chars=self.recall_threshold_chars,
        )
        if not nodes:
            # LLM either declined (dump too large) or returned nothing parseable.
            return self.memory.recall(query, k=self.recall_k)
        return nodes

    def _pack_to_budget(self, nodes):
        if not nodes or self.context_budget_tokens <= 0:
            return nodes
        # Rough char-to-token ratio of ~4 covers most tokenizers; cheap and
        # close enough for budget packing. Overhead reserves a couple hundred
        # tokens for the formatting prelude and any goal callouts.
        budget_chars = self.context_budget_tokens * 4 - 400
        if budget_chars <= 0:
            return nodes[:1]
        out: list = []
        used = 0
        for n in nodes:
            cost = len(n.text) + 24
            if out and used + cost > budget_chars:
                break
            out.append(n)
            used += cost
            if used >= budget_chars:
                break
        return out

    def _run_tool_loop(self, messages):
        tools = openai_tool_specs(self.memory) if self.expose_tools else None
        cited_ids: list[str] = []
        for _hop in range(self.max_tool_hops):
            response = self.chat_fn(
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
            )
            msg = (response.get("choices") or [{}])[0].get("message", {}) or {}
            messages.append(_normalize_assistant_message(msg))
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break
            results = run_openai_tool_calls(self.memory, tool_calls)
            for tc, tr in zip(tool_calls, results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "name": tr["name"],
                    "content": tr["content"],
                })
                if tc.get("function", {}).get("name") == "memory_recall":
                    try:
                        payload = json.loads(tr["content"])
                        for m in payload.get("memories", []) or []:
                            if m.get("id") and m["id"] not in cited_ids:
                                cited_ids.append(m["id"])
                    except json.JSONDecodeError:
                        pass
        final_text = messages[-1].get("content", "") if messages else ""
        return (final_text or ""), messages, cited_ids


def _format_memory_context(nodes, *, goals: list | None = None) -> str:
    lines = [
        "Memory recalled for this turn. Quote the bracketed ids back to the",
        "user when you reuse a memory so attribution stays clean.",
        "",
    ]
    if goals:
        lines.append("Active goals:")
        for g in goals:
            lines.append(f"- [{g.id[:8]}] {g.text}")
        lines.append("")
    lines.append("Relevant memories:")
    for n in nodes:
        lines.append(f"- [{n.id[:8]}] {n.text}")
    return "\n".join(lines)


def _normalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant"}
    content = message.get("content")
    out["content"] = content if content is not None else ""
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


__all__ = ["AutoMemory"]
