from __future__ import annotations

import json
import re
from typing import Any, Callable


_ENRICH_SYSTEM = (
    "You are a memory analyzer. Given one memory text, output JSON with these keys:\n"
    "  aliases: 3-6 short alternate phrasings or question forms this memory would "
    "answer. Include synonyms, abbreviations, and natural search phrases.\n"
    "  entities: list of named entities (people, places, dates, technical terms, identifiers).\n"
    "  kind: one of fact, preference, constraint, goal, context.\n"
    "Output ONLY a single JSON object. No prose, no fences."
)


def enrich_memory(
    text: str,
    *,
    chat_fn: Callable[..., dict],
    model: str | None = None,
) -> dict[str, Any]:
    if not text or not text.strip():
        return {"aliases": [], "entities": [], "kind": "fact"}

    messages = [
        {"role": "system", "content": _ENRICH_SYSTEM},
        {"role": "user", "content": f"Memory:\n{text}"},
    ]
    kw: dict[str, Any] = {"temperature": 0.0}
    if model is not None:
        kw["model"] = model

    try:
        resp = chat_fn(messages=messages, **kw)
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return {"aliases": [], "entities": [], "kind": "fact"}
        data = json.loads(m.group(0))
        return {
            "aliases": [str(a).strip() for a in (data.get("aliases") or []) if a],
            "entities": [str(e).strip() for e in (data.get("entities") or []) if e],
            "kind": str(data.get("kind") or "fact").strip().lower(),
        }
    except Exception:
        return {"aliases": [], "entities": [], "kind": "fact"}


__all__ = ["enrich_memory"]
