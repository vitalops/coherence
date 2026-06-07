from __future__ import annotations

import re
from typing import Any, Callable


_BATCH_SYSTEM = (
    "You are an outcome judge. For each turn below, decide whether the assistant's "
    "reply was successful based ONLY on how the user followed up.\n\n"
    "Output one line per turn in this exact format:\n"
    "  N: SCORE\n"
    "where N is the turn number and SCORE is a number in [-1, +1].\n"
    "Use -1 if the user corrected, contradicted, or rejected the reply.\n"
    "Use +1 if the user accepted, thanked, or built on it.\n"
    "Use 0 if there is no clear signal.\n"
    "Graded values are fine (e.g. -0.5, +0.7) when the signal is partial.\n"
    "Output ONLY the score lines, no prose or fences."
)


def batched_judge(
    turns: list[dict[str, Any]],
    *,
    chat_fn: Callable[..., dict],
    model: str | None = None,
) -> list[float]:
    """Grade a list of turns in a single LLM call.

    Each turn dict has keys 'user', 'reply', 'next_user'. Returns a list of
    scores in [-1, +1], one per input turn. On any failure the corresponding
    score is 0.0 (no reinforcement applied).
    """
    if not turns:
        return []

    parts: list[str] = []
    for i, t in enumerate(turns, start=1):
        nxt = (t.get("next_user") or "").strip()
        if not nxt:
            nxt = "(no follow-up — session ended)"
        parts.append(
            f"=== Turn {i} ===\n"
            f"User: {t.get('user', '')}\n"
            f"Assistant: {t.get('reply', '')}\n"
            f"Next user: {nxt}\n"
        )

    messages = [
        {"role": "system", "content": _BATCH_SYSTEM},
        {"role": "user", "content": "\n".join(parts) + "\n\nGrade each turn now:"},
    ]
    kw: dict[str, Any] = {"temperature": 0.0}
    if model is not None:
        kw["model"] = model

    scores = [0.0] * len(turns)
    try:
        resp = chat_fn(messages=messages, **kw)
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        for line in content.splitlines():
            line = line.strip().lstrip("-•").strip()
            m = re.match(r"(?:Turn\s+)?(\d+)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", line)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(scores):
                try:
                    v = float(m.group(2))
                    scores[idx] = max(-1.0, min(1.0, v))
                except ValueError:
                    pass
    except Exception:
        pass
    return scores


__all__ = ["batched_judge"]
