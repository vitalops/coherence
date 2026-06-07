from __future__ import annotations

import re
from typing import Any, Callable


_SELF_ASSESS_SYSTEM = (
    "You evaluate an assistant's recent turns. For each turn, you see the "
    "user's message, the memories the assistant retrieved before replying, "
    "and the assistant's reply. Decide whether the assistant's reply was a "
    "correct, well-grounded answer to what the user actually asked.\n\n"
    "Do NOT use the user's next message as evidence. You are judging the "
    "assistant's action, not the user's reaction.\n\n"
    "Output one line per turn in this exact format:\n"
    "  N: SCORE\n"
    "where N is the turn number and SCORE is a number in [-1, +1].\n"
    "  +1 — the reply correctly used the retrieved memories and addressed "
    "the user's stated intent.\n"
    "  -1 — the reply ignored, misused, or contradicted the retrieved "
    "memories, or failed to address the user's intent.\n"
    "   0 — the turn was purely conversational, no memories were used, or "
    "there is no clear signal either way. Use 0 freely; do not invent "
    "signal.\n"
    "Graded values are fine when the signal is partial. Output ONLY the "
    "score lines, no prose or fences."
)


def batched_self_assess(
    turns: list[dict[str, Any]],
    *,
    chat_fn: Callable[..., dict],
    model: str | None = None,
) -> list[float]:
    """Grade a batch of turns by asking the LLM to judge the assistant's own action.

    Each turn dict has keys: 'user', 'reply', and 'active_texts' (the texts of
    the memories that were in the agent's context). Returns one score per
    turn in [-1, +1]. ``next_user`` is intentionally not used — the LLM
    judges the assistant's action, not the user's reaction.
    """
    if not turns:
        return []

    parts: list[str] = []
    for i, t in enumerate(turns, start=1):
        active_texts = t.get("active_texts") or []
        if active_texts:
            mem_block = "\n".join(f"  - {text}" for text in active_texts)
        else:
            mem_block = "  (none — the assistant had no relevant memories to ground in)"
        parts.append(
            f"=== Turn {i} ===\n"
            f"User asked: {t.get('user', '')}\n"
            f"Retrieved memories:\n{mem_block}\n"
            f"Assistant replied: {t.get('reply', '')}\n"
        )

    messages = [
        {"role": "system", "content": _SELF_ASSESS_SYSTEM},
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


# Conservative regex for the "follow_up" escape hatch. Fires only on
# unambiguous corrections; smooth follow-ups, thanks, and silence are NOT
# signal. The framework would rather miss a real correction than demote a
# memory that did nothing wrong.
_EXPLICIT_CORRECTION_RE = re.compile(
    r"\b("
    r"that'?s\s+not\s+(what\s+i\s+(meant|asked|said)|right|correct)"
    r"|not\s+what\s+i\s+(meant|asked|said)"
    r"|that'?s\s+(wrong|incorrect)"
    r"|you'?re\s+(wrong|incorrect)"
    r"|let\s+me\s+(clarify|rephrase|try\s+again|be\s+clearer?)"
    r"|i\s+told\s+you\s+(not\s+to|to\s+stay\s+(off|away))"
    r"|stop\s+(doing|using|recommending|suggesting)"
    r"|no\s*,?\s*that'?s\s+(wrong|incorrect|not)"
    r")\b",
    re.IGNORECASE,
)


def follow_up_outcome(prev_user: str, prev_reply: str, next_user: str) -> float:
    """Return -1.0 only when the next user message contains an UNAMBIGUOUS
    correction. Returns 0.0 in every other case — including thanks, smooth
    follow-ups, and session-end silence. Politeness is not evidence of
    success; silence is not signal."""
    if not next_user:
        return 0.0
    return -1.0 if _EXPLICIT_CORRECTION_RE.search(next_user) else 0.0


__all__ = ["batched_self_assess", "follow_up_outcome"]
